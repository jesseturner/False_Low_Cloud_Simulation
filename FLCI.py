import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import cartopy.feature as feature
import sys
import time

#--- Pass in date variable from FLCI_bulk.sh
date_str = sys.argv[1]

#--- Open the GFS data
gfs_file = 'model_data/gfs_'+date_str
gfs_ds = xr.open_dataset(gfs_file, engine="cfgrib",backend_kwargs={'filter_by_keys': {'typeOfLevel':'isobaricInhPa'}})

#--- Define the region
#------ Global
latitude_north = 90
latitude_south = -90
longitude_west = -360
longitude_east = 0
region = gfs_ds.sel(latitude=slice(latitude_north, latitude_south), longitude=slice(360+longitude_west, 360+longitude_east))
region_name = "global"

#--- Create the datetime string
datetime_str = np.datetime_as_string(region.time.values, unit='h')
datetime_str = datetime_str.replace('T', ' ')+"Z"


#--- Create the mass density table
g = 9.807 #m s-2
u = []
for i in range(len(region.isobaricInhPa.values)-1):
    p1 = region.isobaricInhPa.values[i]*100 #kg s-2 m-1
    p2 = region.isobaricInhPa.values[i+1]*100 #kg s-2 m-1
    dp = p1-p2
    r_g = (region.q.values[i] + region.q.values[i+1]) / 2 #kg kg-1
    u.append((1/g)*r_g*dp) #kg m-2

#--- Create the optical mass table
optical_mass_da = xr.DataArray(u, dims=('hPa', 'lat', 'lon'),
                    coords={'hPa': region.isobaricInhPa.values[0:-1], 'lat': region.latitude.values, 'lon': region.longitude.values})
temperature_da = xr.DataArray(region.t[0:-1], dims=('hPa', 'lat', 'lon'),
                    coords={'hPa': region.isobaricInhPa.values[0:-1], 'lat': region.latitude.values, 'lon': region.longitude.values})
optical_mass_ds = xr.Dataset({'u': optical_mass_da,'T': temperature_da})

#--- Open mass extinction look-up tables
mass_ext_df_13 = pd.read_pickle('tables/mass_ext_band13')
mass_ext_df_14 = pd.read_pickle('tables/mass_ext_band14')
mass_ext_df_07 = pd.read_pickle('tables/mass_ext_band07')

#--- Create optical thickness table (optical mass * optical thickness)
#------ Takes 30 minutes for global run

print("Starting calculation for ", date_str)

start_time = time.time()

pressure_profile = optical_mass_ds.hPa[:21].values
lat_len = len(optical_mass_ds.lat)
lon_len = len(optical_mass_ds.lon)

#--------- Pre-allocate arrays
optical_thickness_07 = np.zeros([len(pressure_profile), lat_len, lon_len])
optical_thickness_13 = np.zeros([len(pressure_profile), lat_len, lon_len])
optical_thickness_14 = np.zeros([len(pressure_profile), lat_len, lon_len])

#--------- Extract the relevant slices once
temperatures = optical_mass_ds['T'].isel(hPa=slice(0, 21)).values
optical_masses = optical_mass_ds['u'].isel(hPa=slice(0, 21)).values

#--------- Pre-calculate nearest temperature and pressure indices for all bands
nearest_temp_indices_07 = np.argmin((mass_ext_df_07.index.values[:, None] - temperatures.flatten())**2, axis=0)
nearest_pressure_indices_07 = np.argmin((mass_ext_df_07.columns.values[:, None] - pressure_profile)**2, axis=0)

nearest_temp_indices_13 = np.argmin((mass_ext_df_13.index.values[:, None] - temperatures.flatten())**2, axis=0)
nearest_pressure_indices_13 = np.argmin((mass_ext_df_13.columns.values[:, None] - pressure_profile)**2, axis=0)

nearest_temp_indices_14 = np.argmin((mass_ext_df_14.index.values[:, None] - temperatures.flatten())**2, axis=0)
nearest_pressure_indices_14 = np.argmin((mass_ext_df_14.columns.values[:, None] - pressure_profile)**2, axis=0)

#--------- Reshape indices to match the dimensions of temperatures
nearest_temp_indices_07 = nearest_temp_indices_07.reshape((len(pressure_profile), lat_len, lon_len))
nearest_pressure_indices_07 = nearest_pressure_indices_07.reshape(len(pressure_profile))

nearest_temp_indices_13 = nearest_temp_indices_13.reshape((len(pressure_profile), lat_len, lon_len))
nearest_pressure_indices_13 = nearest_pressure_indices_13.reshape(len(pressure_profile))

nearest_temp_indices_14 = nearest_temp_indices_14.reshape((len(pressure_profile), lat_len, lon_len))
nearest_pressure_indices_14 = nearest_pressure_indices_14.reshape(len(pressure_profile))

setup_time = time.time() - start_time
print("Calculation progress: 0/12")

start_time = time.time()

#--------- Iterate through the grid points
#------------ Only doing up to 550hPa, in order to speed things up
for z in range(len(pressure_profile[:12])):

    start_inner_time = time.time()  # Start timing the outer loop
    print(f"Processing pressure_profile[{z}] = {pressure_profile[z]}")
    for y in range(lat_len):
        for x in range(lon_len):            
            optical_mass_value = optical_masses[z, y, x]

            # Lookup the mass extinction values using pre-calculated indices
            mass_ext_value_07 = mass_ext_df_07.iloc[nearest_temp_indices_07[z, y, x], nearest_pressure_indices_07[z]]
            optical_thickness_07[z, y, x] = optical_mass_value * mass_ext_value_07

            mass_ext_value_13 = mass_ext_df_13.iloc[nearest_temp_indices_13[z, y, x], nearest_pressure_indices_13[z]]
            optical_thickness_13[z, y, x] = optical_mass_value * mass_ext_value_13

            mass_ext_value_14 = mass_ext_df_14.iloc[nearest_temp_indices_14[z, y, x], nearest_pressure_indices_14[z]]
            optical_thickness_14[z, y, x] = optical_mass_value * mass_ext_value_14
            
        
    
    inner_time = time.time() - start_inner_time
    print("Calculation progress: "+str(z)+"/12")

loop_time = time.time() - start_time
print(f"Total loop time: {loop_time:.2f} seconds")


#--- Function for blackbody radiance
def blackbody_radiance(T, wl):
    h = 6.626e-34
    c = 3e8
    k = 1.380e-23
    B = (2*h*c**2)/(wl**5 * (np.exp((h*c)/(k*wl*T))-1))
    return B

#--- Function for expected radiance from surface
def I_sfc(T_sfc, optical_thickness, wl):
    mu = 1
    tau_star = np.sum(optical_thickness, axis=0)
    I_sfc = blackbody_radiance(T_sfc, wl)*np.exp(-tau_star/mu)
    return I_sfc


#--- Function for expected radiance from atmosphere
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


#--- Function for brightness temperature
def brightness_temperature(I, wl):
    h = 6.626e-34
    c = 3e8
    k = 1.380e-23
    Tb = (h*c)/(k*wl * np.log(1 + ((2*h*c**2)/(I*wl**5))))
    return Tb

#--- Brightness temperature using SST as surface
sst_file = "sst_data/sst_"+date_str
sst_ds = xr.open_dataset(sst_file)
sst_ds =  sst_ds.squeeze()
sst_ds.sst.values = sst_ds.sst.values+273.15
sst_ds = sst_ds.sst.fillna(0)

#--- Match SST shape to the GFS shape
sst_ds = sst_ds.sel(lat=slice(latitude_north,latitude_south,-1), lon=slice(longitude_west+360,longitude_east+360))
sst_padded = sst_ds
if np.shape(sst_ds)[0] != np.shape(optical_thickness_07)[1]:
    sst_padded = np.pad(sst_ds, ((1, 0), (0, 0)), mode='edge')
if np.shape(sst_padded)[1] != np.shape(optical_thickness_07)[2]:
    sst_padded = np.pad(sst_padded, ((0, 0), (0, 1)), mode='edge')

#--- Setting wavelengths for BTD
first_wl = 11.2e-6
first_optical_thickness = optical_thickness_14
first_wl_str = str(first_wl*1e6).replace(".", "_")
second_wl = 3.9e-6
second_optical_thickness = optical_thickness_07
second_wl_str = str(second_wl*1e6).replace(".", "_")

first_I_tot = I_sfc(sst_padded, first_optical_thickness, first_wl) + I_atm(first_optical_thickness, optical_mass_ds, first_wl)
second_I_tot = I_sfc(sst_padded, second_optical_thickness, second_wl) + I_atm(second_optical_thickness, optical_mass_ds, second_wl)

BTD = brightness_temperature(first_I_tot, first_wl) - brightness_temperature(second_I_tot, second_wl)
projection=ccrs.PlateCarree(central_longitude=0)
fig,ax=plt.subplots(1, figsize=(12,12),subplot_kw={'projection': projection})

#--- Plot the BTD figure
from matplotlib.colors import LinearSegmentedColormap
colors = [(0, '#A9A9A9'), (0.5, 'white'), (1, '#1167b1')]  # +3 = blueish teal, 0 = white, -3 = grey
cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
levels = np.linspace(-3, 3, 31)
c=ax.contourf(region.longitude, region.latitude, BTD, cmap=cmap, extend='both', levels=levels)
clb = plt.colorbar(c, shrink=0.4, pad=0.02, ax=ax)
clb.ax.tick_params(labelsize=15)
clb.set_label('(K)', fontsize=15)
ax.set_title("Simulated BTD ("+ str(round(first_wl*1e6, 1)) + " μm - " + str(round(second_wl*1e6, 1)) +" μm) \n("+datetime_str+")", fontsize=20, pad=10)
ax.add_feature(feature.LAND, zorder=100, edgecolor='#000', facecolor='tan')
fig.set_dpi(200)
fig.savefig("composite/images/"+region_name+"_"+date_str, dpi=200, bbox_inches='tight')

#--- Save as a netCDF
btd_ds = xr.Dataset(
    {
        "BTD": (["latitude", "longitude"], BTD)
    },
    coords={
        "latitude": region.latitude,
        "longitude": region.longitude
    }
)
btd_ds.to_netcdf("composite/"+region_name+"/"+region_name+"_"+date_str+".nc")
print("Successfully saved netCDF for "+date_str)