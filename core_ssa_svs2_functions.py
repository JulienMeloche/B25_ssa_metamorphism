import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime
import matplotlib.pyplot as plt


#local import
from snowmatching.DTW import usefull

def import_crocus(file_path, str_begin, str_end, keep_empty_layers = False):
    mod = xr.open_dataset(file_path)
    df_s = mod[['SNODEN_ML','SNOMA_ML','TSNOW_ML','SNODOPT_ML', 'SNOTYPE_ML','SNODP', 'SNOSPH_ML', 'SNOAGE_ML', 'WSNO', 'TSNO_SURF', 'SNOWRATE', 'RAINRATE']].to_dataframe()
    begin_datetime = datetime.strptime(str_begin, '%Y-%m-%d')
    end_datetime = datetime.strptime(str_end, '%Y-%m-%d')
    df = df_s.loc[begin_datetime:end_datetime,:]

    #create thickness columns
    df['thickness'] = df[['SNODEN_ML','SNOMA_ML']].apply(lambda x : x[1] / x[0], axis = 1) 
    df['ssa'] = df['SNODOPT_ML'].apply(lambda x: 6/( x * 917) if x>0 else 0)
    #df['corr_length'] = df[['SNODEN_ML','ssa']].apply(lambda x: debye_eqn(x[1], x[0]), axis = 1)
    #df.loc[df.ssa >= 30, 'ssa'] = 30

    #filter out low snowdepth 

    if keep_empty_layers:
        #filter out low snowdepth 
        df = df[(df.SNODP > 0.05) ]
    else:
        #filter out low snowdepth and empty layers
        df = df[(df.SNODP > 0.05) & (df.thickness > 0.00005) ]

    #get list of date
    dates = df.groupby(level = 'time').mean().index.get_level_values(0)


    #get height
    height= [list(np.cumsum(df.loc[date].thickness.values[::-1])[::-1]) for date in dates]
    flat_height = [ x for row in height for x in row]
    df['height'] = flat_height

    df.rename(columns = {'SNODEN_ML' : 'density'}, inplace = True)
    
    return df, dates

def get_mean_profile_obs(dict_site, res = 0.05, plot = False, plot_param = 'density'):
    """
    intepolate profile on the same grid and get the mean
        return: profiles on the same height grid
                mean profile
    """
    # mean total depth per date of field data, used to  at the end
    depth_date = np.mean([dict_site[site].thickness.sum() for site in dict_site.keys()])
    #norm height
    height_grid = np.arange(0, 1 + res, res)[::-1]
    #set all field profile to same grid
    interp_profile_density = np.array([interpolate_profile(dict_site[site], height_grid, param = 'density') for site in dict_site.keys()])
    interp_profile_ssa = np.array([interpolate_profile(dict_site[site], height_grid, param = 'ssa') for site in dict_site.keys()])
    interp_profile_temp = np.array([interpolate_profile(dict_site[site], height_grid, param = 'temperature') for site in dict_site.keys()])
    interp_profile = {'density' : interp_profile_density, 'ssa' : interp_profile_ssa, 'temperature' : interp_profile_temp}

    coeffs = np.array([[1, 1, 0]])
    M_density, _, _ = usefull.autofit_set(np.expand_dims(interp_profile_density, axis = 1), height_grid, coeffs)
    M_ssa, _, _ = usefull.autofit_set(np.expand_dims(interp_profile_ssa, axis = 1), height_grid, coeffs)
    M_temperature, _, _ = usefull.autofit_set(np.expand_dims(interp_profile_temp, axis = 1), height_grid, coeffs)

    #mean_profile = np.mean([interp_profile['density'], interp_profile['ssa'], interp_profile['temperature']], axis = 1)
    mean_site = pd.DataFrame({'height': height_grid * depth_date,
                              'density' : M_density[0],
                              'ssa' : M_ssa[0],
                              'temperature' : M_temperature[0]})
    if plot:
        #plot mean and matching of obs
        fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
        for site in dict_site.keys():
            ax1.step(dict_site[site][plot_param].to_numpy(), dict_site[site].l_height_top.to_numpy(), label = 'r', alpha =0.6)

        for profile in interp_profile[plot_param]:
            ax2.step(profile, height_grid*depth_date, alpha =0.6)
        ax2.step(mean_site[plot_param], mean_site['height'], label = 'mean', color = 'k')

        ax2.legend()
        ax1.set_ylabel('Height (m)')
        ax1.set_xlabel(f'{plot_param}')
        ax2.set_xlabel(f'{plot_param}')

    return interp_profile, mean_site

def interpolate_profile(profile, height_grid, param):
    
    profile_param = profile[param].to_numpy()
    profile_height = profile.h_norm.to_numpy()
    interp_profile = usefull.oversample_profile(profile_height, profile_param, height_grid, kind = 'next')

    return interp_profile


def norm_simu_obs(simu_df, obs_df, param = 'ssa'):

    profile_simu = simu_df[param].dropna().to_numpy()
    #norm h
    height_simu = simu_df.height.to_numpy()/simu_df.height[0]
    profile_obs = obs_df[param].to_numpy()
    #norm h
    height_obs = obs_df.height.to_numpy()/obs_df.height[0]
    #resample height to supply high res for DTW
    height_grid = np.arange(0,  1, 0.01)
    profile_obs_s = usefull.oversample_profile(height_obs, profile_obs, height_grid)
    profile_simu_s = usefull.oversample_profile(height_simu, profile_simu, height_grid)

    return profile_obs_s, profile_simu_s, height_grid,
