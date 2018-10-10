import dcpy.oceans
import numpy as np
import pandas as pd
import tqdm

import xarray as xr

region = dict(lon=slice(80, 94), lat=slice(4, 24))
ebob_region = dict(lon=slice(85.5, 88.5), lat=slice(5, 8))
seasons = ['NE', 'NESW', 'SW', 'SWNE']
moor_names = {'ra12': 'RAMA 12N',
              'ra15': 'RAMA 15N',
              'nrl1': 'NRL 1',
              'nrl3': 'NRL 3',
              'nrl4': 'NRL 4',
              'nrl5': 'NRL 5'}


def make_merged_nc(moorings):
    ''' Makes merged netCDF files with turbulence info.

    Inputs
    ------
    A list of mooring objects (moor)

    Outputs
    -------
    None
    '''

    Turb = xr.Dataset()

    for m in tqdm.tqdm(moorings):
        subset = (m.turb.reset_coords(['ρ', 'S', 'T', 'z', 'mld', 'ild'])
                  .expand_dims(['lat', 'lon'])
                  .drop(['χ', 'ε', 'N2', 'Tz', 'Sz']))
        subset['season'] = subset.time.monsoon.labels

        depth_season = np.round(subset.z.groupby(subset['season'])
                                .median(dim='time')).astype('int64')

        seas = xr.broadcast(subset['KT'], subset['season'])[1]

        mean_depth = xr.zeros_like(subset['KT'])
        for ss in ['NE', 'NESW', 'SW', 'SWNE']:
            mask = seas.values == ss
            if np.all(~mask):
                continue

            zz = xr.broadcast(seas == ss, depth_season.sel(season=ss))[1]
            mean_depth.values[mask] = zz.values[mask]

        # get a reasonable depth for the subset
        subset['mean_depth'] = mean_depth

        Turb = xr.merge([Turb, subset.reset_coords()])

    del subset
    del mask

    # print('\t\t merging ...')
    # vards.append(xr.merge(varlist))

    # make sure the merging worked
    # for m in moorings:
    #     for var in Turb.data_vars:
    #         merged = (Turb.reset_coords()[var]
    #                   .sel(lon=m.lon, lat=m.lat)
    #                   .dropna(dim='depth', how='all')
    #                   .dropna(dim='time', how='all'))
    #         orig = (m.turb.reset_coords()[var]
    #                 .dropna(dim='time', how='all'))
    #         xr.testing.assert_equal(merged, orig)
    #         del orig

    # Turb = Turb.rename({'ε': 'epsilon',
    #                     'χ': 'chi_t'})

    # Turb.epsilon.attrs['long_name'] = (
    #     'Ocean turbulence kinetic energy dissipation rate')
    # Turb.epsilon.attrs['standard_name'] = (
    #     'Ocean_turbulence_kinetic_energy_dissipation_rate')
    # Turb.epsilon.attrs['units'] = 'W kg^-1'

    # Turb['chi-t'].attrs['long_name'] = (
    #     'Ocean dissipation rate of thermal variance from microtemperature')
    # Turb['chi-t'].attrs['standard_name'] = (
    #     'Ocean_dissipation_rate_of_thermal_variance_from_microtemperature')
    # Turb['chi-t'].attrs['units'] = 'C^2 s^-1'

    Turb.lat.attrs['long_name'] = 'latitude'
    Turb.lat.attrs['units'] = 'degrees_north'
    Turb.lon.attrs['long_name'] = 'longitude'
    Turb.lon.attrs['units'] = 'degrees_east'

    Turb['T'].attrs['standard_name'] = 'sea_water_potential_temperature'
    Turb['ρ'].attrs['standard_name'] = 'sea_water_potential_density'
    Turb['S'].attrs['standard_name'] = 'sea_water_practical_salinity'

    # Turb.attrs['Conventions'] = 'CF-1.6'
    Turb.attrs['netcdf_version'] = '4'
    Turb.attrs['title'] = (
        'Merged and processed χpod data from the Bay of Bengal')
    Turb.attrs['institution'] = 'Oregon State University'
    Turb.attrs['data_originator'] = 'Shroyer and Moum'
    Turb.attrs['chief_scientist'] = 'Emily L. Shroyer'

    print('Writing to file.')
    Turb.to_netcdf('bay_merged_10min.nc')
    (Turb.resample(time='1H').mean(dim='time')
     .to_netcdf('bay_merged_hourly.nc'))
    (Turb.resample(time='6H').mean(dim='time')
     .to_netcdf('bay_merged_6hourly.nc'))


def bin_and_to_dataframe(KTm, ρbins=None, Sbins=None):
    def bin(var, bins):
        binned = bins[np.digitize(var, bins)-1]
        return xr.DataArray(binned, dims=var.dims, coords=var.coords)

    if ρbins is not None:
        KTm['ρbinned'] = bin(KTm.ρ, ρbins)
    if Sbins is not None:
        KTm['Sbinned'] = bin(KTm.S, Sbins)

    if np.all(KTm['KT'].values > 0):
        KTm['KT'].values = np.log10(KTm['KT'].values)

    KTdf = (KTm[['KT', 'ρbinned', 'z', 'season']]
            .to_dataframe()
            .dropna(axis=0, subset=['KT'])
            .reset_index())

    KTdf['latlon'] = (KTdf['lat'].astype('str') + 'N, '
                      + KTdf['lon'].astype('str') + 'E').astype('category')

    moornames = {'RAMA12': '12.0N, 90.0E',
                 'RAMA15': '15.0N, 90.0E',
                 'NRL1': '5.0N, 85.5E',
                 'NRL2': '6.5N, 85.5E',
                 'NRL3': '8.0N, 85.5E',
                 'NRL4': '8.0N, 87.0E',
                 'NRL5': '8.0N, 88.5E'}
    KTdf['moor'] = (KTdf['latlon'].map(
        dict(zip(moornames.values(), moornames.keys())))
                    .astype('category'))
    return KTdf


def remake_summaries(moorings=None):
    ''' Remake all summary images '''

    from .read_data import read_all_moorings

    if moorings is None:
        moorings = read_all_moorings()

    print('making all summaries')
    for m in moorings:
        m.Summarize(savefig=True)


def calc_wind_input():
    taux = xr.open_dataset('~/datasets/tropflux/taux_tropflux_1d_2014.nc')
    tauy = xr.open_dataset('~/datasets/tropflux/tauy_tropflux_1d_2014.nc')

    tropflux = xr.Dataset()
    tropflux['taux'] = taux.taux
    tropflux['tauy'] = tauy.tauy

    tropflux = (tropflux.rename({'latitude': 'lat', 'longitude': 'lon'})
                .sel(**region))

    mimoc = xr.open_mfdataset('/home/deepak/datasets/mimoc/MIMOC_ML_*.nc',
                              concat_dim='month')
    mimoc['LATITUDE'] = mimoc.LATITUDE.isel(month=1)
    mimoc['LONGITUDE'] = mimoc.LONGITUDE.isel(month=1)
    mimoc = (mimoc.swap_dims({'LAT': 'LATITUDE', 'LONG': 'LONGITUDE'})
             .rename({'LATITUDE': 'lat',
                      'LONGITUDE': 'lon'}))
    mimoc['month'] = pd.date_range('2014-01-01', '2014-12-31', freq='SM')[::2]
    mimoc = mimoc.rename({'month': 'time'})

    mld = (mimoc.DEPTH_MIXED_LAYER
           .sel(**region).load())

    wind_input, _ = dcpy.oceans.calc_wind_power_input(
        (tropflux.taux.interpolate_na('time')
         + 1j * tropflux.tauy.interpolate_na('time')),
        mld=mld.interp_like(tropflux.taux),
        f0=dcpy.oceans.coriolis(tropflux.lat))

    wind_input.attrs['description'] = (
        'Near-inertial power input calculated using Tropflux winds and the '
        'MIMOC mixed layer climatology using the method of Alford (2003) to '
        'solve the Pollard & Millard slab model.')
    wind_input.to_netcdf('~/bay/estimates/wind-power-input-2014.nc')
