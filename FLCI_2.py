import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs, cartopy.feature as feature
import sys, time

#=====================================================

def main():
    
    date_str, coordinate_box, region_name = setVariables()
    gfs_region = getGfsData(date_str, coordinate_box)
    datetime_str = createDatetime(gfs_region)
    u = createMassDensityTable(gfs_region)
    optical_mass_ds = createOpticalMassTable(u, gfs_region)
    mass_ext_df_13, mass_ext_df_14, mass_ext_df_07 = openMassExtinctionTables()
    optical_thickness_07, optical_thickness_13, optical_thickness_14 = createOpticalThicknessTable(date_str, optical_mass_ds, mass_ext_df_07, mass_ext_df_13, mass_ext_df_14)
    sst_ds = getSstData(date_str)
    sst_padded = matchSstDataToGfs(sst_ds, coordinate_box, optical_thickness_07)
    first_wl, second_wl = setWavelengths()
    first_I_tot, second_I_tot, BTD = runRadianceAndBrightnessTemp(sst_padded, optical_thickness_07, first_wl, optical_thickness_14, second_wl, optical_mass_ds)
    plotFigure(gfs_region, BTD, first_wl, second_wl, datetime_str, region_name, date_str)
    saveAsNetCDF(BTD, gfs_region, region_name, date_str)


#=====================================================

def setVariables():
    date_str = "20250612"
    coordinate_box = [50, 24, -125, -66] # lat north, lat south, lon west, lon east
    region_name = "global"

    return date_str, coordinate_box, region_name

#-----------------------------------------------------

def getGfsData(date_str, coordinate_box):
    #--- Open the GFS data
    gfs_file = 'model_data/gfs_'+date_str
    gfs_ds = xr.open_dataset(gfs_file, engine="cfgrib",backend_kwargs={'filter_by_keys': {'typeOfLevel':'isobaricInhPa'}})
    gfs_region = gfs_ds.sel(latitude=slice(coordinate_box[0], coordinate_box[1]), longitude=slice(360+coordinate_box[2], 360+coordinate_box[3]))

    return gfs_region

#-----------------------------------------------------
    
def createDatetime(gfs_region):
    datetime_str = np.datetime_as_string(gfs_region.time.values, unit='h')
    datetime_str = datetime_str.replace('T', ' ')+"Z"

    return datetime_str

#-----------------------------------------------------

def createMassDensityTable(gfs_region):
    g = 9.807 #m s-2
    u = []
    for i in range(len(gfs_region.isobaricInhPa.values)-1):
        p1 = gfs_region.isobaricInhPa.values[i]*100 #kg s-2 m-1
        p2 = gfs_region.isobaricInhPa.values[i+1]*100 #kg s-2 m-1
        dp = p1-p2
        r_g = (gfs_region.q.values[i] + gfs_region.q.values[i+1]) / 2 #kg kg-1
        u.append((1/g)*r_g*dp) #kg m-2

    return u

#-----------------------------------------------------

def createOpticalMassTable(u, gfs_region):
    optical_mass_da = xr.DataArray(u, dims=('hPa', 'lat', 'lon'),
                        coords={'hPa': gfs_region.isobaricInhPa.values[0:-1], 'lat': gfs_region.latitude.values, 'lon': gfs_region.longitude.values})
    temperature_da = xr.DataArray(gfs_region.t[0:-1], dims=('hPa', 'lat', 'lon'),
                        coords={'hPa': gfs_region.isobaricInhPa.values[0:-1], 'lat': gfs_region.latitude.values, 'lon': gfs_region.longitude.values})
    optical_mass_ds = xr.Dataset({'u': optical_mass_da,'T': temperature_da})

    return optical_mass_ds

#-----------------------------------------------------

def openMassExtinctionTables():
    mass_ext_df_13 = pd.read_pickle('tables/mass_ext_band13')
    mass_ext_df_14 = pd.read_pickle('tables/mass_ext_band14')
    mass_ext_df_07 = pd.read_pickle('tables/mass_ext_band07')

    return mass_ext_df_13, mass_ext_df_14, mass_ext_df_07

#-----------------------------------------------------

def getDimensions(optical_mass_ds):

    pressure_profile = optical_mass_ds.hPa[:21].values
    lat_len = len(optical_mass_ds.lat)
    lon_len = len(optical_mass_ds.lon)

    return pressure_profile, lat_len, lon_len

#-----------------------------------------------------

def initializeArrays(pressure_profile, lat_len, lon_len):

    optical_thickness = np.zeros([len(pressure_profile), lat_len, lon_len])

    return optical_thickness

#-----------------------------------------------------

def extractSlices(optical_mass_ds):
    temperatures = optical_mass_ds['T'].isel(hPa=slice(0, 21)).values
    optical_masses = optical_mass_ds['u'].isel(hPa=slice(0, 21)).values

    return temperatures, optical_masses

#-----------------------------------------------------

def nearestTempPress(mass_ext_df, temperatures, pressure_profile, lat_len, lon_len):
    
    # Pre-calculate nearest temperature and pressure indices for all bands
    nearest_temp_indices = np.argmin((mass_ext_df.index.values[:, None] - temperatures.flatten())**2, axis=0)
    nearest_pressure_indices = np.argmin((mass_ext_df.columns.values[:, None] - pressure_profile)**2, axis=0)

    # Reshape indices to match the dimensions of temperatures
    nearest_temp_indices = nearest_temp_indices.reshape((len(pressure_profile), lat_len, lon_len))
    nearest_pressure_indices = nearest_pressure_indices.reshape(len(pressure_profile))

    return nearest_temp_indices, nearest_pressure_indices

#-----------------------------------------------------

def calcOpticalThickness(x, y, z, optical_masses, mass_ext_df, nearest_temp_indices, nearest_pressure_indices, optical_thickness):

    optical_mass_value = optical_masses[z, y, x]
    # Lookup the mass extinction values using pre-calculated indices
    mass_ext_value = mass_ext_df.iloc[nearest_temp_indices[z, y, x], nearest_pressure_indices[z]]
    optical_thickness[z, y, x] = optical_mass_value * mass_ext_value

    return optical_thickness

#-----------------------------------------------------

def iterateGridPoints(pressure_profile, lat_len, lon_len, optical_masses, temperatures, 
                      mass_ext_df_07, mass_ext_df_13, mass_ext_df_14):
    
    optical_thickness_07 = initializeArrays(pressure_profile, lat_len, lon_len)
    optical_thickness_13 = initializeArrays(pressure_profile, lat_len, lon_len)
    optical_thickness_14 = initializeArrays(pressure_profile, lat_len, lon_len)
    
    nearest_temp_indices_07, nearest_pressure_indices_07 = nearestTempPress(mass_ext_df_07, temperatures, pressure_profile, lat_len, lon_len)
    nearest_temp_indices_13, nearest_pressure_indices_13 = nearestTempPress(mass_ext_df_13, temperatures, pressure_profile, lat_len, lon_len)
    nearest_temp_indices_14, nearest_pressure_indices_14 = nearestTempPress(mass_ext_df_14, temperatures, pressure_profile, lat_len, lon_len)
    
    # Iterate through the grid points
    # Only doing up to 550hPa, in order to speed things up
    for z in range(len(pressure_profile[:12])):

        print(f"Processing pressure_profile[{z}] = {pressure_profile[z]}")
        for y in range(lat_len):
            for x in range(lon_len):

                optical_thickness_07 = calcOpticalThickness(x, y, z, optical_masses, 
                                                            mass_ext_df_07, nearest_temp_indices_07, 
                                                            nearest_pressure_indices_07, optical_thickness_07)
                
                optical_thickness_13 = calcOpticalThickness(x, y, z, optical_masses, 
                                                            mass_ext_df_13, nearest_temp_indices_13, 
                                                            nearest_pressure_indices_13, optical_thickness_13)
                
                optical_thickness_14 = calcOpticalThickness(x, y, z, optical_masses, 
                                                            mass_ext_df_14, nearest_temp_indices_14, 
                                                            nearest_pressure_indices_14, optical_thickness_14)
            
        
        print("Calculation progress: "+str(z)+"/12")

    return optical_thickness_07, optical_thickness_13, optical_thickness_14

#-----------------------------------------------------

def createOpticalThicknessTable(date_str, optical_mass_ds, mass_ext_df_07, mass_ext_df_13, mass_ext_df_14):
    
    # Create optical thickness table (optical mass * optical thickness)
    # Takes 30 minutes for global run
    print("Starting calculation for ", date_str)

    pressure_profile, lat_len, lon_len = getDimensions(optical_mass_ds)
    temperatures, optical_masses = extractSlices(optical_mass_ds)

    print("Calculation progress: 0/12")

    start_time = time.time()

    optical_thickness_07, optical_thickness_13, optical_thickness_14 = iterateGridPoints(
                      pressure_profile, lat_len, lon_len, optical_masses, temperatures,
                      mass_ext_df_07, mass_ext_df_13, mass_ext_df_14)
                

    loop_time = time.time() - start_time
    print(f"Total loop time: {loop_time:.2f} seconds")

    return optical_thickness_07, optical_thickness_13, optical_thickness_14

#-----------------------------------------------------

# Function for blackbody radiance
def blackbody_radiance(T, wl):
    h = 6.626e-34
    c = 3e8
    k = 1.380e-23
    B = (2*h*c**2)/(wl**5 * (np.exp((h*c)/(k*wl*T))-1))
    return B

#-----------------------------------------------------

# Function for expected radiance from surface
def I_sfc(T_sfc, optical_thickness, wl):
    mu = 1
    tau_star = np.sum(optical_thickness, axis=0)
    I_sfc = blackbody_radiance(T_sfc, wl)*np.exp(-tau_star/mu)
    return I_sfc

#-----------------------------------------------------

# Function for expected radiance from atmosphere
def I_atm(optical_thickness, optical_mass_ds, wl):
    p_len = np.shape(optical_thickness)[0] - 1
    I_levels = []
    mu = 1
    for i in range(p_len):
        T = optical_mass_ds.isel(hPa=i)['T']
        B = blackbody_radiance(T, wl)
        tau_above = np.sum(optical_thickness[i+1:], axis=0)
        tau_level = np.sum(optical_thickness[i:], axis=0)
        press_levels = optical_mass_ds['hPa'][:21].values
        dp = press_levels[i+1] - press_levels[i]
        dT_dp = ((np.exp(-tau_above/mu)) - (np.exp(-tau_level/mu))) / dp
        I_level = B*dT_dp*dp
        I_levels.append(I_level)

    I_atm = np.sum(I_levels, axis=0)
    return I_atm

#-----------------------------------------------------

# Function for brightness temperature
def brightness_temperature(I, wl):
    h = 6.626e-34
    c = 3e8
    k = 1.380e-23
    Tb = (h*c)/(k*wl * np.log(1 + ((2*h*c**2)/(I*wl**5))))
    return Tb

#-----------------------------------------------------

def getSstData(date_str): 

    sst_file = "sst_data/sst_"+date_str
    sst_ds = xr.open_dataset(sst_file)
    sst_ds =  sst_ds.squeeze()
    sst_ds.sst.values = sst_ds.sst.values+273.15
    sst_ds = sst_ds.sst.fillna(0)
    
    return sst_ds

#-----------------------------------------------------

def matchSstDataToGfs(sst_ds, coordinate_box, optical_thickness_07):
    # Match SST shape to the GFS shape
    sst_ds = sst_ds.sel(lat=slice(coordinate_box[0], coordinate_box[1], -1), lon=slice(coordinate_box[2]+360, coordinate_box[3]+360))
    sst_padded = sst_ds
    if np.shape(sst_ds)[0] != np.shape(optical_thickness_07)[1]:
        sst_padded = np.pad(sst_ds, ((1, 0), (0, 0)), mode='edge')
    if np.shape(sst_padded)[1] != np.shape(optical_thickness_07)[2]:
        sst_padded = np.pad(sst_padded, ((0, 0), (0, 1)), mode='edge')
    
    return sst_padded

#-----------------------------------------------------

def setWavelengths():
        
    # Setting wavelengths for BTD
    first_wl = 11.2e-6
    first_wl_str = str(first_wl*1e6).replace(".", "_")
    second_wl = 3.9e-6
    second_wl_str = str(second_wl*1e6).replace(".", "_")

    return first_wl, second_wl

#-----------------------------------------------------

def runRadianceAndBrightnessTemp(sst_padded, first_optical_thickness, first_wl, second_optical_thickness, second_wl, optical_mass_ds):

    first_I_tot = I_sfc(sst_padded, first_optical_thickness, first_wl) + I_atm(first_optical_thickness, optical_mass_ds, first_wl)
    second_I_tot = I_sfc(sst_padded, second_optical_thickness, second_wl) + I_atm(second_optical_thickness, optical_mass_ds, second_wl)

    BTD = brightness_temperature(first_I_tot, first_wl) - brightness_temperature(second_I_tot, second_wl)

    return first_I_tot, second_I_tot, BTD

#-----------------------------------------------------

def plotFigure(gfs_region, BTD, first_wl, second_wl, datetime_str, region_name, date_str):

    projection=ccrs.PlateCarree(central_longitude=0)
    fig,ax=plt.subplots(1, figsize=(12,12),subplot_kw={'projection': projection})

    from matplotlib.colors import LinearSegmentedColormap
    colors = [(0, '#A9A9A9'), (0.5, 'white'), (1, '#1167b1')]  # +3 = blueish teal, 0 = white, -3 = grey
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
    levels = np.linspace(-3, 3, 31)
    c=ax.contourf(gfs_region.longitude, gfs_region.latitude, BTD, cmap=cmap, extend='both', levels=levels)
    clb = plt.colorbar(c, shrink=0.4, pad=0.02, ax=ax)
    clb.ax.tick_params(labelsize=15)
    clb.set_label('(K)', fontsize=15)
    ax.set_title("Simulated BTD ("+ str(round(first_wl*1e6, 1)) + " μm - " + str(round(second_wl*1e6, 1)) +" μm) \n("+datetime_str+")", fontsize=20, pad=10)
    ax.add_feature(feature.LAND, zorder=100, edgecolor='#000', facecolor='tan')
    fig.set_dpi(200)
    fig.savefig("composite/images/"+region_name+"_"+date_str, dpi=200, bbox_inches='tight')

    return

#-----------------------------------------------------

def saveAsNetCDF(BTD, gfs_region, region_name, date_str):
    btd_ds = xr.Dataset(
        {
            "BTD": (["latitude", "longitude"], BTD)
        },
        coords={
            "latitude": gfs_region.latitude,
            "longitude": gfs_region.longitude
        }
    )
    btd_ds.to_netcdf("composite/"+region_name+"/"+region_name+"_"+date_str+".nc")
    print("Successfully saved netCDF for "+date_str)

    return

#=====================================================

if __name__ == '__main__':
    main()