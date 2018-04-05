import numpy as np
import xarray as xr
import tqdm


def make_merged_nc(moorings):
    ''' Makes merged netCDF files with turbulence info.

    Inputs
    ------
    A list of mooring objects (moor)

    Outputs
    -------
    None
    '''

    vards = []
    for var in tqdm.tqdm(['KT', 'Jq', 'χ', 'ε']):
        varlist = []
        print('Processing ' + var)
        for m in moorings:
            subset = (m.__dict__[var].copy()
                      .reset_coords(['ρ', 'S', 'T', 'z', 'mld', 'ild']))

            # ktsubset['depth'].values = np.round(ktsubset.z.mean(dim='time'),
            #                                     decimals=1)

            subset = subset.expand_dims(['lat', 'lon'])
            subset['season'] = subset.time.monsoon.labels

            depth_season = np.round(subset.z.groupby(subset['season'])
                                    .median(dim='time')).astype('int64')

            # depth_week = np.round(subset.z.groupby(subset.time.dt.weekofyear)
            #                       .mean(dim='time')).astype('int64')

            # depth_bin = depth_week

            # bin depths
            # depth_season.values[np.logical_and(
            #     depth_season.values > 55,
            #     depth_season.values < 63)] = 60

            # depth_season.values[np.logical_and(
            #     depth_season.values >= 64,
            #     depth_season.values < 68)] = 65

            # depth_season.values[np.logical_and(
            #     depth_season.values > 68,
            #     depth_season.values < 82)] = 75

            # depth_season.values[np.logical_and(
            #     depth_season.values > 84.5,
            #     depth_season.values < 87.2)] = 85

            # depth_season.values[depth_season.values > 87.2] = 100

            _, seas = xr.broadcast(subset[var], subset['season'])

            mean_depth = xr.zeros_like(subset[var])
            for ss in ['NE', 'NESW', 'SW', 'SWNE']:
                mask = seas.values == ss
                if np.all(mask == False):
                    continue

                _, zz = xr.broadcast(seas == ss, depth_season.sel(season=ss))
                mean_depth.values[mask] = zz.values[mask]

            # get a reasonable depth for the subset
            subset['mean_depth'] = mean_depth

            varlist.append(subset.reset_coords())

        print('\t\t merging ' + var)
        vards.append(xr.merge(varlist))

        # make sure the merging worked
        for m in moorings:
            merged = (vards[-1]
                      .sel(lon=m.lon, lat=m.lat)
                      .reset_coords()[var]
                      .dropna(dim='depth', how='all')
                      .dropna(dim='time', how='all'))
            orig = (m.__dict__[var]
                    .reset_coords()[var]
                    .dropna(dim='time', how='all'))
            xr.testing.assert_equal(merged, orig)

    Turb = xr.merge(vards)
    Turb = Turb.rename({'ε': 'epsilon', 'χ': 'chi'})
    print('Writing to file.')
    # Turb.to_netcdf('merged_KT_test.nc', format='netCDF4')
    Turb.resample(time='1H').mean(dim='time').to_netcdf('bay_merged_hourly.nc')
    Turb.resample(time='6H').mean(dim='time').to_netcdf('bay_merged_6hourly.nc')


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
    KTdf['moor'] = KTdf['latlon'].map(dict(zip(moornames.values(),
                                               moornames.keys()))).astype('category')
    return KTdf
