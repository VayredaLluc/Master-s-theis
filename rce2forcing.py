import numpy as np

day_len = 24
year_len = 365
day_time = np.linspace(0,day_len-1,day_len)

def write_forcing_files(lrad_rce, 
                        srad_rce, 
                        rain_rce, 
                        rhum_rce, 
                        tair_rce, 
                        wind_rce = 10., 
                        snow_rce = 0.):

    lrad_day = np.ones((day_len))
    srad_day = np.zeros((day_len))
    rain_day = np.zeros((day_len))
    snow_day = np.zeros((day_len))
    rhum_day = np.zeros((day_len))
    tair_day = np.zeros((day_len))
    wind_day = np.ones((day_len))

    #Longwave radiation
    lrad = np.tile(lrad_day,year_len)
    lrad = lrad/np.mean(lrad)*lrad_rce
    np.savetxt('lrad.txt', lrad)

    #Shortwave radiation
    srad_day = np.maximum(-np.cos(day_time*2*np.pi/day_len),0)
    srad = np.tile(srad_day,year_len)
    srad = srad/np.mean(srad)*srad_rce
    np.savetxt('srad.txt', srad)

    #Rain
    rain_day = 1 - np.sin(day_time*2*np.pi/day_len+4*np.pi/day_len)
    daily_precip = np.random.poisson(1, year_len)
    rain = np.zeros((0))
    for r in daily_precip:
        rain = np.append(rain,rain_day*r)
    rain = rain/np.sum(rain)*year_len*rain_rce/1000
    np.savetxt('rain.txt', rain)

    #Relative Humidity
    #rhum_day = 1-(1-rhum_rce)*2*np.sin(day_time*np.pi/24-np.pi/4)**2
    rhum_day = rhum_rce+(1-rhum_rce)/2*np.sin(day_time*2*np.pi/day_len)
    rhum = np.tile(rhum_day,year_len)
    np.savetxt('rhum.txt', rhum)

    #Air temperature
    tair_day = -3*np.sin(day_time*2*np.pi/day_len)
    tair = np.tile(tair_day,year_len)
    tair = tair-np.mean(tair)+tair_rce
    np.savetxt('tair.txt', tair)

    #Wind speed
    wind = np.tile(wind_day*wind_rce,year_len)
    np.savetxt('wind.txt', wind)

    #Snow
    snow = np.tile(snow_day,year_len)
    np.savetxt('snow.txt', snow)

    return


