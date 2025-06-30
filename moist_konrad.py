import numpy as np
from scipy import integrate
from scipy import optimize
import konrad

cp_air = 1004.64 #J kg-1 K-1
g = 9.80665 #m s-2
Rd = 287.0570048852906 #J kg-1 K-1 air
Rv = 461.52280830495 #J kg-1 K-1 water vapor
Lv = 2501000 #J kg-1
eps = Rd/Rv #dimensionless
e_sat_ref = 610.94 #Pa, saturation water vapor pressure for Tref = 273.15 K
T_ref = 273.15 #K
seconds_day = 24*60*60 #seconds in a day

lapse_dry = -g/cp_air #K/m

convection = konrad.convection.HardAdjustment()
lapserate = konrad.lapserate.MoistLapseRate()


#AIR
def moist_adiabat(T_s,T_rad,atmosphere):
    T_moist_ad = convection.get_moist_adiabat(atmosphere, T_s, lapserate)

    T_con = np.maximum(T_moist_ad,T_rad)
    
    return T_con

def convective_top(T_con,T_rad):
    itop = np.where(T_rad - T_con == 0)[0][0]
    if itop == 0:
        itop = np.where(T_rad - T_con == 0)[0][1]
    return itop

def coldpoint(T):
    itop = np.where(np.diff(T)>=0)[0][0]
    
    return itop

def height(p,T,ps,Ts): #m
    p = np.append(ps,p)
    T = np.append(Ts,T)
    rho = p/(Rd*T)
    z = integrate.cumtrapz(-1/(g*rho),p,initial = 0)
    
    return z


#WATER
def vmr_to_mmr(vmr):
    mass_mix = vmr*eps
    
    return mass_mix

def column_water_mass(vmr,ph): #kg m-2
    mmr = vmr_to_mmr(vmr) #kg kg-1
    dp = np.diff(ph)
    M_water = np.sum(-dp*mmr/g)
    
    return M_water

def manabe_rh(rhs, p):
#    rh = np.maximum(0,rhs*(p/p[0]-0.02)/(1-0.02))
    rh = rhs*(p/p[0]-0.02)/(1-0.02)
    return rh

def rh_to_vmr(rh,T,p,itop):
    mixing_ratio = np.ones_like(T)
    vmr_itop = konrad.physics.relative_humidity2vmr(rh[itop],p[itop],T[itop])
    mixing_ratio[:itop] = konrad.physics.relative_humidity2vmr(rh[:itop],p[:itop],T[:itop])
    mixing_ratio[itop:] = vmr_itop
    
    return mixing_ratio

def opt_column_water_to_rh(M_water,T_atm,p,ph,itop):
    
    def fun(rhs):
        rh = manabe_rh(rhs, p)
        vmr = rh_to_vmr(rh,T_atm,p,itop)
        rh_w_mass = column_water_mass(vmr,ph)
        res = rh_w_mass - M_water
        return res
    
    rhs_opt = optimize.brentq(fun, 0., 5)
    rh_opt = manabe_rh(rhs_opt, p)
    return rh_opt

#FIRE
def TE(T,p):
    energy = integrate.trapz(-cp_air*T/g,p) #J m-2
    return energy #J m-2

def DSE(T,z,p):
    energy = integrate.trapz(-cp_air*T/g-z[1:],p) #J m-2
    return energy #J m-2

def T_convection_strong(p,T_rad,T_surface,HF,atmosphere): 
    #find atmospheric T after convective adjustment with the T surface as lowest level T, i.e., STRONG atmsophere-surface coupling
    #note that here HF is the heat provided to the atmosphere by the surface (sensible) and precipitation for the whole timestep (J m-2) and not the flux itself (J m-2 s-1)
    thermal_energy = TE(T_rad,p) + HF
    
    T_con = moist_adiabat(T_surface,T_rad,atmosphere)
    
    TE_moist = TE(T_con,p)
    
    ds = TE_moist - thermal_energy

    return T_con, T_surface, ds

def T_convection_strong_DSE(p,T_atm_ini,T_atm_low,T_rad,T_surface,HF,atmosphere): 
    #find atmospheric T after convective adjustment with the T surface as lowest level T, i.e., STRONG atmsophere-surface coupling
    #note that here HF is the heat provided to the atmosphere by the surface (sensible) and precipitation for the whole timestep (J m-2) and not the flux itself (J m-2 s-1)
    z = height(p,T_atm_ini,atmosphere['phlev'][0],T_atm_low)
    dry_static_energy = DSE(T_atm_ini,z,p) + HF
    
    T_con = moist_adiabat(T_surface,T_rad,atmosphere)
    
    z_con = height(p,T_con,atmosphere['phlev'][0],T_atm_low)
    DSE_moist = DSE(T_con,z_con,p)
    
    ds = DSE_moist - dry_static_energy

    return T_con, T_surface, ds
    
def T_convection_weak(p, T_rad, T_surface, HF, atmosphere): #find T after convective adjustment conserving thermal energy
    #note that here HF is the heat provided to the atmosphere by the surface (sensible) and precipitation for the whole timestep (J m-2) and not the flux itself (J m-2 s-1)
    thermal_energy = TE(T_rad,p) + HF
    tol = 100

    def fun(T_s):
        T_atm = moist_adiabat(T_s,T_rad,atmosphere)
        TE_moist = TE(T_atm,p)
        res = TE_moist - thermal_energy
        return res
    
    Tc = T_rad[0]
    Th = T_surface + 10
    
    dpos = fun(Th)
    dneg = fun(Tc)
    print(dpos,dneg)
    
    if dneg > 0 or np.abs(dneg)<tol:
#        print('Trad high already', dneg)
        ds = dneg
        dneg = 0
        Ts = Tc
    
    if dpos < tol:
#        print('Tsurf low already', dpos)
        ds = dpos
        dpos = 0
        Ts = Th
     
    ''' 
    Ts = Tc + (Th-Tc) * -dneg/(-dneg+dpos)
    ds = fun(Ts)
#    print(ds)

    if ds > 0:
        dpos = ds
        Th = Ts
    else:
        dneg = ds
        Tc = Ts
    '''
    
    maxiter = 100
    k = 0
    while dpos>tol and np.abs(dneg)>tol and k < maxiter:
        Ts = Tc + (Th-Tc) * -dneg/(-dneg+dpos)
        ds = fun(Ts)
    #    print(ds)
        
        if ds > 0:
            dpos = ds
            Th = Ts
        else:
            dneg = ds
            Tc = Ts
        k += 1
        
    if k == 100:
        print(dpos, dneg, ds, Th, Tc, Ts, T_rad[0])
        print('reached max iter')
    T_bl = Ts
    T_con = moist_adiabat(T_bl,T_rad,atmosphere)

    return T_con, T_bl, ds

def T_convection_weak_DSE(p,T_atm_ini,T_atm_low,T_rad,T_surface,HF,atmosphere): #find T after convective adjustment conserving thermal energy
    #note that here HF is the heat provided to the atmosphere by the surface (sensible) and precipitation for the whole timestep (J m-2) and not the flux itself (J m-2 s-1)
    z = height(p,T_atm_ini,atmosphere['phlev'][0],T_atm_low)
    dry_static_energy = DSE(T_atm_ini,z,p) + HF
    tol = 100

    def fun(T_s):
        T_con = moist_adiabat(T_s,T_rad,atmosphere)
        z_con = height(p,T_con,atmosphere['phlev'][0],T_s)
        DSE_moist = DSE(T_con,z_con,p)
        res = DSE_moist - dry_static_energy
        return res
    
    Tc = T_rad[0]
    Th = T_surface + 10
    
    dpos = fun(Th)
    dneg = fun(Tc)
    print(dpos,dneg)
    
    if dneg > 0 or np.abs(dneg)<tol:
#        print('Trad high already', dneg)
        ds = dneg
        dneg = 0
        Ts = Tc
    
    if dpos < tol:
#        print('Tsurf low already', dpos)
        ds = dpos
        dpos = 0
        Ts = Th
     
    ''' 
    Ts = Tc + (Th-Tc) * -dneg/(-dneg+dpos)
    ds = fun(Ts)
#    print(ds)

    if ds > 0:
        dpos = ds
        Th = Ts
    else:
        dneg = ds
        Tc = Ts
    '''
    
    maxiter = 100
    k = 0
    while dpos>tol and np.abs(dneg)>tol and k < maxiter:
        Ts = Tc + (Th-Tc) * -dneg/(-dneg+dpos)
        ds = fun(Ts)
    #    print(ds)
        
        if ds > 0:
            dpos = ds
            Th = Ts
        else:
            dneg = ds
            Tc = Ts
        k += 1
        
    if k == 100:
        print(dpos, dneg, ds, Th, Tc, Ts, T_rad[0])
        print('reached max iter')
    T_atm_low = Ts
    T_con = moist_adiabat(T_atm_low,T_rad,atmosphere)

    return T_con, T_atm_low, ds

#EARTH
wind = 10 # wind speed m s-1
Cd = 0.002 # Drag coefficient unitless

def sensible_heat(T_bl,T_s,p_s,ratio_ls): #sensible heat flux into the atmosphere (positive for heat trasnport into the atm)
    rho_s = p_s/(Rd*T_bl) #kg m-3
    sh = wind*Cd*rho_s*cp_air*(T_s-T_bl) #J s-1 m-2
    
    return sh/ratio_ls

def latent_heat(vmr, T_bl, p_s, ratio_ls): #sensible heat flux into the atmosphere (positive for heat trasnport into the atm)
    rho_s = p_s/(Rd*T_bl) #kg m-3
    
    rh = konrad.physics.vmr2relative_humidity(vmr,p_s,T_bl)
    mmr = vmr_to_mmr(vmr)
    mmr_sat = mmr/rh
    
    lh = wind*Cd*rho_s*Lv*(mmr_sat-mmr) #J s-1 m-2
    return lh*ratio_ls

def T_vmr_z(Ts,Tatm,vmr0,vmr1,z,z_low):
    T_low = (z[1]-z_low)*(Ts-Tatm)/z[1] + Tatm
    vmr_low = (vmr0-vmr1)*(z[2]-z_low)/(z[2]-z[1]) + vmr1
    
    return T_low,vmr_low

#THE AVATAR
def RCPE_step(timestep,
              atmosphere,surface,radiation,clearsky,
              SH,LH,albedo,T_atm_low,
              conv_top, 
              strong_coupling = False, constrain_RH = True):
    
    
    surface.albedo = albedo
    
    #update heating rates
    
    radiation.update_heatingrates(atmosphere = atmosphere,surface = surface,cloud=clearsky)
    
    rad_heat_atm = np.ones_like(radiation['net_htngrt'][0])
    
    rad_heat_atm[:] = - np.diff(radiation['lw_flxu'][0] + radiation['sw_flxu'][0]-
                             (radiation['lw_flxd'][0] + radiation['sw_flxd'][0]))
    
    troposphere_radiation = np.sum(rad_heat_atm[:conv_top])
     
    heating_rates = np.ones_like(rad_heat_atm)
    heating_rates[:] = ((rad_heat_atm[:])
                                 /cp_air * g/-np.diff(atmosphere['phlev'])[:] * seconds_day)

    #update net radiaton at surface

    net_rad_surface = (radiation['lw_flxd'][0,0] + radiation['sw_flxd'][0,0] - 
                    (radiation['lw_flxu'][0,0] + radiation['sw_flxu'][0,0]))
                    
    troposphere_radiation = np.sum(rad_heat_atm[:conv_top])
    atm_rad = np.sum(rad_heat_atm[:])                       
        
    Flux = np.maximum(net_rad_surface,np.maximum(abs(atm_rad),SH+LH))
    Flux = net_rad_surface
    Flux = (abs(atm_rad)+SH+LH+net_rad_surface)/3
    Flux = abs(atm_rad)
    prec_eff = np.maximum(0.,np.minimum(1.,LH/Flux))
  #  prec_eff = LH/Flux
          
    #temperature of atmosphere after radiative update

    atmosphere['T'] += heating_rates * timestep
    T_radiation = atmosphere['T'][0].copy()
    
    #convective adjustment of atmosphere (conserves thermal energy)
    prec_heating = - prec_eff * atm_rad
    prec_heating = - prec_eff * troposphere_radiation
    #prec_heating = - prec_eff * troposphere_radiation #amount of energy invested in precipitation
    prec_mass = prec_heating/Lv * seconds_day * timestep
    
    if strong_coupling == True:
        atmosphere['T'][0],T_atm_low,E_dif = T_convection_strong(atmosphere['plev'], T_radiation, surface['temperature'],
                                         (SH + prec_heating) * seconds_day * timestep, atmosphere)
    else:
        atmosphere['T'][0],T_atm_low,E_dif = T_convection_weak(atmosphere['plev'], T_radiation, surface['temperature'],
                                         (SH + prec_heating) * seconds_day * timestep, atmosphere)

    E_imbalance = E_dif.copy()/(timestep*seconds_day)
    #print(E_imbalance,SH,LH)

    
    #water adjustment
    conv_top = convective_top(atmosphere['T'][0],T_radiation)
    cold_point = coldpoint(atmosphere['T'][0])
    M_w = column_water_mass(atmosphere['H2O'][0],atmosphere['phlev']) - prec_mass + LH/Lv*seconds_day*timestep
    
    RH = opt_column_water_to_rh(M_w,atmosphere['T'][0],
                                atmosphere['plev'],atmosphere['phlev'],cold_point)
    if constrain_RH == True:
        if RH[0] > 0.95:
            RH = manabe_rh(0.95, atmosphere['plev'])
            
        if RH[0] < 0.3:
            RH = manabe_rh(0.3, atmosphere['plev'])
        
    atmosphere['H2O'][0] = rh_to_vmr(RH,atmosphere['T'][0],atmosphere['plev'],cold_point)
    
    return atmosphere,surface,radiation,net_rad_surface,atm_rad,T_atm_low,E_imbalance,prec_mass,RH,cold_point,conv_top


def RCPE_step_DSE(timestep,
              atmosphere,surface,radiation,clearsky,
              SH,LH,albedo,T_atm_low, 
              strong_coupling = False, constrain_RH = True):
    
    T_atm_ini = atmosphere['T'][0].copy()
    surface.albedo = albedo
    
    #update heating rates
    
    radiation.update_heatingrates(atmosphere = atmosphere,surface = surface,cloud=clearsky)
    
    rad_heat_atm = np.ones_like(radiation['net_htngrt'][0])
    
    rad_heat_atm[:] = - np.diff(radiation['lw_flxu'][0] + radiation['sw_flxu'][0]-
                             (radiation['lw_flxd'][0] + radiation['sw_flxd'][0]))
    
    # troposphere_radiation = np.sum(rad_heat_atm[:conv_top])
     
    heating_rates = np.ones_like(rad_heat_atm)
    heating_rates[:] = ((rad_heat_atm[:])
                                 /cp_air * g/-np.diff(atmosphere['phlev'])[:] * seconds_day)

    #update net radiaton at surface

    net_rad_surface = (radiation['lw_flxd'][0,0] + radiation['sw_flxd'][0,0] - 
                    (radiation['lw_flxu'][0,0] + radiation['sw_flxu'][0,0]))
                    
    atm_rad = np.sum(rad_heat_atm[:])                       
        
    Flux = np.maximum(net_rad_surface,np.maximum(abs(atm_rad),SH+LH))
    Flux = (abs(atm_rad)+SH+LH+net_rad_surface)/3
    Flux = net_rad_surface
    Flux = LH/(LH+SH)
    Flux = abs(atm_rad)
    prec_eff = np.maximum(0.,np.minimum(1.,LH/Flux))
 #   prec_eff = LH/Flux
          
    #temperature of atmosphere after radiative update

    atmosphere['T'] += heating_rates * timestep
    T_radiation = atmosphere['T'][0].copy()
    
    #convective adjustment of atmosphere (conserves thermal energy)
    prec_heating = - prec_eff * atm_rad
    #prec_heating = - prec_eff * troposphere_radiation #amount of energy invested in precipitation
    prec_mass = prec_heating/Lv * seconds_day * timestep
    
    if strong_coupling == True:
        atmosphere['T'][0],T_atm_low,E_dif = T_convection_strong_DSE(atmosphere['plev'], T_atm_ini, T_atm_low, T_radiation, surface['temperature'],
                                         (SH + prec_heating + atm_rad) * seconds_day * timestep, atmosphere)
    else:
        atmosphere['T'][0],T_atm_low,E_dif = T_convection_weak_DSE(atmosphere['plev'], T_atm_ini, T_atm_low, T_radiation, surface['temperature'],
                                         (SH + prec_heating + atm_rad) * seconds_day * timestep, atmosphere)

    E_imbalance = E_dif.copy()/(timestep*seconds_day)
    #print(E_imbalance,SH,LH)

    
    #water adjustment
    conv_top = convective_top(atmosphere['T'][0],T_radiation)
    cold_point = coldpoint(atmosphere['T'][0])
    M_w = column_water_mass(atmosphere['H2O'][0],atmosphere['phlev']) - prec_mass + LH/Lv*seconds_day*timestep
    
    RH = opt_column_water_to_rh(M_w,atmosphere['T'][0],
                                atmosphere['plev'],atmosphere['phlev'],cold_point)
    if constrain_RH == True:
        if RH[0] > 0.95:
            RH = manabe_rh(0.95, atmosphere['plev'])
            
        if RH[0] < 0.3:
            RH = manabe_rh(0.3, atmosphere['plev'])
        
    atmosphere['H2O'][0] = rh_to_vmr(RH,atmosphere['T'][0],atmosphere['plev'],cold_point)
    
    return atmosphere,surface,radiation,net_rad_surface,atm_rad,T_atm_low,E_imbalance,prec_mass,RH,cold_point
