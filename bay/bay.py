import airsea
import cycler
import numpy as np
import pandas as pd
import scipy as sp
import tqdm

import dcpy.oceans
import dcpy.util
import xarray as xr

region = dict(lon=slice(80, 94), lat=slice(4, 24))
default_density_bins = [1018, 1021.7, 1022.5, 1023, 1023.5, 1024.25, 1029]
ebob_region = dict(lon=slice(85.5, 88.5), lat=slice(5, 8))
seasons = ['NE', 'NESW', 'SW', 'SWNE']
season_months = {'NE': 3, 'NESW': 2, 'SW': 5, 'SWNE': 2}
splitseasons = ['NE', 'Mar', 'Apr', 'SW', 'SWNE']
moor_names = {'ra12': 'RAMA 12N',
              'ra15': 'RAMA 15N',
              'nrl1': 'NRL 1',
              'nrl3': 'NRL 3',
              'nrl4': 'NRL 4',
              'nrl5': 'NRL 5'}
monsoon_cycler = ((cycler.cycler(color=['darkorange', 'darkorange', 'teal', 'teal']))
                  + (cycler.cycler(linestyle=['-', '--', '-', '--'])))
loc = {
    'NRL1': dict(lat=5, lon=85.5),
    'nrl1': dict(lat=5, lon=85.5),
    'NRL2': dict(lat=6.5, lon=85.5),
    'nrl2': dict(lat=6.5, lon=85.5),
    'NRL3': dict(lat=8, lon=85.5),
    'nrl3': dict(lat=8, lon=85.5),
    'NRL4': dict(lat=8, lon=87.0),
    'nrl4': dict(lat=8, lon=87.0),
    'NRL5': dict(lat=8, lon=88.5),
    'nrl5': dict(lat=8, lon=88.5)
}


pods = {
        'RAMA12':
            {'lon': 90, 'lat': 12, 'label': 'RAMA',
             'ha': 'right', 'va': 'center',
             'depths': {
                 '15 m': '2014-15',
                 '30 m': '2014-15',
                 '45 m': '2015'}},
        'RAMA15':
            {'lon': 90, 'lat': 15, 'label': 'RAMA',
             'ha': 'right', 'va': 'center',
             'depths': {
                 '15 m': '2015',
                 # '30 m': '2015'
             }},
        'WHOI':
            {'lon': 90, 'lat': 18, 'label': 'OMM/WHOI',
             'ha': 'left', 'va': 'top',
             'depths': {
                 '22 m': '2014-15',
                 '30 m': '2014-15',
                 '46 m': '2014-15',
                 '55 m': '2014-15',
                 '65 m': '2014-15'}},
        'NRL1':
            {'lon': 85.5, 'lat': 5, 'label': 'NRL',
             'ha': 'center', 'va': 'top',
             'depths': {
                 '60 m (55-100)': '2014',
                 '80 m (75-115)': '2014'}},
        'NRL3':
            {'lon': 85.5, 'lat': 8, 'label': 'NRL',
             'ha': 'right', 'va': 'top',
             'depths': {
                 '32 m (28-78)': '2014',
                 '52 m (48-100)': '2014'}},
        'NRL4':
            {'lon': 87, 'lat': 8, 'label': 'NRL',
             'ha': 'center', 'va': 'bottom',
             'depths': {
                 '63 m (60-85)': '2014',
                 '83 m (80-105)': '2014'}},
        'NRL5':
            {'lon': 88.5, 'lat': 8, 'label': 'NRL',
             'ha': 'left', 'va': 'top',
             'depths': {
                 '  85 m': '2014',
                 '105 m': '2014'}},
    }


def make_merged_nc(moorings, fileprefix='bay_merged'):
    ''' Makes merged netCDF files with turbulence info.

    Inputs
    ------
    moorings: A list of mooring objects
    filename: output filename

    Outputs
    -------
    None
    '''

    Turb = xr.Dataset()

    for m in tqdm.tqdm(moorings):
        subset = (m.turb.reset_coords())

        if m.name == 'NRL1':
            subset.depth.values = [55.0, 75.0]

        if m.name == 'NRL3':
            subset.depth.values = np.array([30, 45])

        Turb = xr.merge([Turb, (subset.expand_dims(['lat', 'lon'])
                                .set_coords(['lat', 'lon']))])

    del subset

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
    #         del merged orig

    Turb = Turb.rename({'ε': 'epsilon',
                        'χ': 'chi-t'})

    Turb.epsilon.attrs['long_name'] = (
        'Ocean turbulence kinetic energy dissipation rate')
    Turb.epsilon.attrs['standard_name'] = (
        'Ocean_turbulence_kinetic_energy_dissipation_rate')
    Turb.epsilon.attrs['units'] = 'W kg^-1'

    Turb['chi-t'].attrs['long_name'] = (
        'Ocean dissipation rate of thermal variance from microtemperature')
    Turb['chi-t'].attrs['standard_name'] = (
        'Ocean_dissipation_rate_of_thermal_variance_from_microtemperature')
    Turb['chi-t'].attrs['units'] = 'C^2 s^-1'

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

    fileprefix = '../estimates/' + fileprefix
    print('Writing to file.')
    Turb.to_netcdf(fileprefix + '_10min.nc')
    (Turb.resample(time='1H', keep_attrs=True).mean(dim='time')
     .to_netcdf(fileprefix + '_hourly.nc'))
    (Turb.resample(time='6H', keep_attrs=True).mean(dim='time')
     .to_netcdf(fileprefix + '_6hourly.nc'))


def nc_to_binned_df(dataset='../estimates/bay_merged_sorted_hourly.nc',
                    bins=default_density_bins,
                    moor=None):
    ''' reads KT from merged .nc file and returns DataFrame version
        suitable for processing.'''

    if isinstance(dataset, str):
        turb = xr.open_dataset(dataset).load()
    elif isinstance(dataset, xr.Dataset):
        turb = dataset.load()

    turb['season'] = turb.time.monsoon.labels

    df = (turb.to_dataframe()
          .dropna(subset=['KT'])
          .reset_index())

    latlon = (df['lat'].astype('float32').astype('str') + 'N, '
              + df['lon'].astype('float32').astype('str')
              + 'E').astype('category')
    df['season'] = df['season'].astype('category')

    moornames = {'RAMA12': '12.0N, 90.0E',
                 'RAMA15': '15.0N, 90.0E',
                 'NRL1': '5.0N, 85.5E',
                 'NRL2': '6.5N, 85.5E',
                 'NRL3': '8.0N, 85.5E',
                 'NRL4': '8.0N, 87.0E',
                 'NRL5': '8.0N, 88.5E'}

    df['moor'] = (latlon
                  .map(dict(zip(moornames.values(),
                                moornames.keys())))
                  .astype('category'))

    if moor is not None:
        if isinstance(moor, str):
            moor = list(str)

        df = df.loc[df.moor.isin(moor)]

    df = bin_ml_bl_rho(df, bins)

    return df


def bin_ml_bl_rho(df, bins):

    error_depth = 5

    depth_minus_mld = df.z - df.mld
    depth_minus_ild = df.z - df.ild
    mask_ml = depth_minus_mld <= error_depth
    mask_bl = np.logical_and(np.logical_not(mask_ml),
                             depth_minus_ild <= error_depth)

    mask_ml_plus = np.logical_and(
        np.logical_not(np.logical_or(mask_ml, mask_bl)),
        depth_minus_mld <= 15)
    mask_ml_plus = np.zeros_like(mask_ml).astype('bool')
    mask_deep = np.logical_not(np.logical_or(
        np.logical_or(mask_ml, mask_bl),
        mask_ml_plus))

    nrl3_ml = np.logical_and(df['moor'] == 'NRL3', df['N2'] < 3e-6)
    nrl3_bl = np.logical_and(np.logical_and(df['moor'] == 'NRL3',
                                            df['Tz'] < 2e-3),
                             df['N2'] > 3e-6)

    mask_ml[nrl3_ml] = True
    mask_bl[nrl3_bl] = True

    df['bin'] = ''
    df.loc[mask_ml, 'bin'] = 'ML'
    df.loc[mask_bl, 'bin'] = 'BL'
    # df.bin[mask_ml_plus] = 'ML+'
    # bins = get_kmeans_bins(7, df['ρ'][mask_deep])
    # df.bin = pd.qcut(df.ρ, 10, precision=1)
    df.loc[mask_deep, 'bin'] = pd.cut(df.ρ.loc[mask_deep],
                                      bins,
                                      precision=1)
    df['bin'] = df['bin'].astype('category')

    assert(np.sum(df.bin == '') == 0)

    return df


def remake_summaries(moorings=None):
    ''' Remake all summary images '''

    from .read_data import read_all_moorings

    if moorings is None:
        moorings = read_all_moorings()

    print('making all summaries')
    for m in moorings:
        m.Summarize(savefig=True)


def calc_wind_input(kind='merra2', dask=False):

    if kind == 'ncep':
        trange = dict(lat=slice(24, 2, None),
                      lon=region['lon'],
                      time=slice('2013-12-01', '2014-11-30'))

        u = xr.open_mfdataset('/home/deepak/datasets/ncep/uwnd*.nc').load()
        v = xr.open_mfdataset('/home/deepak/datasets/ncep/vwnd*.nc').load()
        taux = u.uwnd.sel(trange).copy(
            data=airsea.windstress.stress(u.uwnd.sel(trange)))
        tauy = v.vwnd.sel(trange).copy(
            data=airsea.windstress.stress(v.vwnd.sel(trange)))

        taux = taux * np.sign(u.uwnd)
        tauy = tauy * np.sign(v.vwnd)

        tau = (xr.merge([taux, tauy]).sortby('lat')
               .rename({'uwnd': 'taux', 'vwnd': 'tauy'})).compute()

        windstr = 'NCEP 6-hourly winds'
        windshortstr = 'ncep'

    elif kind == 'merra2':
        trange = dict(time=slice('2013-12-01', '2014-11-30'))
        merra = (xr.open_mfdataset('/home/deepak/bay/datasets/merra2/*.nc',
                                   parallel=dask)
                 .sel(trange))
        taux = merra.U10M.copy(data=airsea.windstress.stress(merra.U10M))
        tauy = merra.V10M.copy(data=airsea.windstress.stress(merra.V10M))

        taux = taux * np.sign(merra.U10M)
        tauy = tauy * np.sign(merra.V10M)

        windstr = 'MERRA-2 hourly winds'
        windshortstr = 'merra2'

        tau = (xr.merge([taux, tauy]).sortby('lat')
               .rename({'U10M': 'taux', 'V10M': 'tauy'})
               .compute())  # needed because we take FFT along time

    mimoc = dcpy.oceans.read_mimoc()
    mld = (mimoc.DEPTH_MIXED_LAYER
           .sel(**region).load())

    # append so that I can interpolate properly
    append = xr.DataArray(mld.isel(time=slice(-2, None)))
    append['time'] = pd.to_datetime(['2013-11-15', '2013-12-15'])
    mld = xr.concat([append, mld], 'time')

    wind_input, ZI = dcpy.oceans.calc_wind_power_input(
        tau.taux + 1j * tau.tauy,
        mld=mld.interp_like(tau.taux),
        f0=dcpy.oceans.coriolis(tau.lat))
    wind_input.name = 'wind_input'
    wind_input = wind_input.to_dataset()

    dt = (wind_input.time.diff('time').values.mean()
          .astype('timedelta64[s]').astype('float32'))
    wind_input['cumulative'] = np.cumsum(wind_input.wind_input) * dt
    wind_input['cumulative'].attrs['long_name'] = 'Energy'
    wind_input['cumulative'].attrs['units'] = 'J/m²'

    wind_input.attrs['description'] = (
        'Near-inertial power input calculated using ' + windstr +
        'and the MIMOC mixed layer climatology using the '
        'method of Alford (2003) to solve the '
        'Pollard & Millard slab model.')

    wind_input['taux'] = tau.taux
    wind_input['tauy'] = tau.tauy
    wind_input['mld'] = mld

    wind_input['ui'] = np.real(ZI)
    wind_input.ui.attrs['long_name'] = '$u_i$'
    wind_input.ui.attrs['description'] = 'Inertial zonal velocity'
    wind_input.ui.attrs['units'] = 'm/s'

    wind_input['vi'] = np.imag(ZI)
    wind_input.vi.attrs['long_name'] = '$v_i$'
    wind_input.vi.attrs['description'] = 'Inertial meridional velocity'
    wind_input.vi.attrs['units'] = 'm/s'

    wind_input.taux.attrs['long_name'] = '$τ_x$'
    wind_input.taux.attrs['units'] = 'N/m²'
    wind_input.taux.attrs['description'] = windshortstr
    wind_input.tauy.attrs['long_name'] = '$τ_y$'
    wind_input.tauy.attrs['units'] = 'N/m²'
    wind_input.tauy.attrs['description'] = windshortstr

    wind_input.mld.attrs['long_name'] = 'MLD'
    wind_input.mld.attrs['units'] = 'm'
    wind_input.mld.attrs['description'] = 'MIMOC mixed layer depth'

    wind_input.to_netcdf('~/bay/estimates/' + windshortstr
                         + '-wind-power-input.nc')


def bin_and_to_dataframe(KTm, ρbins=None, Sbins=None):
    ''' OLD DEPRECATED VERSION. DO NOT USE. '''

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


def seasonal_rms(data, split=False):
    def rms(data):
        calc = np.sqrt((np.abs(data)**2).mean('time'))
        count = data.count('time')/len(data.time)

        return calc.where(count > 0.8)

    if split:
        labels = data.time.monsoon.splitlabels
    else:
        labels = data.time.monsoon.labels

    return data.groupby(labels).apply(rms)


def seasonal_mean(data, split=False):
    def mean(data):
        calc = data.mean('time')
        count = data.count('time')/len(data.time)

        return calc.where(count > 0.8)

    if split:
        labels = data.time.monsoon.splitlabels
    else:
        labels = data.time.monsoon.labels

    return (data.groupby(labels).apply(mean))


def calc_isohaline_depth(S0=34.75, data=None, region=None, split=False):

    if region is None:
        region = dict(lon=slice(80, 94), lat=slice(4, 24))

    if data is None:
        data = (dcpy.oceans.read_argo_clim()
                ['S']
                .sel(**region, pres=slice(0, 800))
                .load()
                .rename({'pres': 'depth'}))

    # NIO atlas doesn't need this
    if 'monsoon' not in data.coords:
        if split:
            groupby = data.time.monsoon.splitlabels
        else:
            groupby = data.time.monsoon.labels

        data = (data.groupby(groupby).mean('time')
                .transpose('lon', 'lat', 'depth', 'monsoon'))

    isodepth = dcpy.interpolate.pchip_roots(data, "depth", S0).squeeze()
    isodepth.attrs['long_name'] = 'Depth of ' + str(S0) + ' isohaline'
    isodepth.attrs['units'] = 'm'

    # isodepth = data.S.isel(depth=1).drop('depth').copy()
    # zmat = xr.broadcast(data.S, data.depth)[1]
    # for tt in range(data.S.shape[-1]):
    #     isodepth.values[:, :, tt] = dcpy.util.calc_iso_surface(
    #         data.S.values[:, :, :, tt], S0, zmat.values[:, :, :, tt],
    #         interp_order=3)

    # isodepth = (zmat.where(np.logical_and(nio.S > S0-0.1,
    #                                       nio.S < S0+0.1))
    #             .mean(dim='depth')
    #             .sel(**bay.region))

    return isodepth


def generate_mean_median_dataframe(dataset, out):

    print('Generating table of means & medians')

    import scikits.bootstrap as bs

    turb = nc_to_binned_df(dataset)

    dflist = []
    for (season, name), group in tqdm.tqdm(
            turb[['KT', 'z']].groupby([turb.season, turb.bin])):

        ci_mean = bs.ci(group.KT, np.mean)
        ci_median = bs.ci(group.KT, np.median)
        dflist.append(pd.DataFrame({'season': season,
                                    'bin': name,
                                    'KT_mean': group.KT.mean(),
                                    'KT_median': group.KT.median(),
                                    'ci_mean': [ci_mean],
                                    'ci_median': [ci_median],
                                    'z': sp.stats.trim_mean(group.z, 0.1)},
                                   index=[0]))

    df = pd.concat(dflist)
    df['season'] = df.season.astype('category')
    df['bin'] = df['bin'].astype('category')

    df.to_csv(out)

    return df
