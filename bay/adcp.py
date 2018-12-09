import dcpy
import numpy as np
import seawater as sw
import tqdm
import xarray as xr
import xfilter
import xrscipy as xrsp


# smooth shear
def smooth_shear(var, n):
    weights = xr.DataArray(np.hanning(n), dims=['window_dim'])
    var = (var.rolling(depth=n, center=True)
           .construct('window_dim', stride=1)
           .dot(weights))

    var.attrs['description'] = 'Smoothed over ' + str(n) + ' points.'

    return var


def to_isothermal_space(data, varnames, isotherms):
    isos = isotherms.values

    isoT = xr.Dataset()
    isoT['mean_depth'] = xr.DataArray(
        isotherms.depth.values, dims=['T'],
        attrs={'long_name': 'mean isothermal depth',
               'units': 'm'})
    isoT = isoT.set_coords('mean_depth')

    isoT['depth'] = (xr.DataArray(np.zeros((len(isos), len(data.time))),
                                  dims=['T', 'time'],
                                  coords={'T': isos,
                                          'time': data.time})
                     * np.nan)

    for var in varnames:
        isoT[var] = xr.zeros_like(isoT.depth) * np.nan
        isoT[var].attrs = data[var].attrs

    for tt, t in tqdm.tqdm(enumerate(data.T.time.values)):
        for var in varnames:
            mask = (~np.isnan(data.T[:, tt].values)
                    & ~np.isnan(data[var][:, tt].values))

            if not np.any(mask):
                continue

            isoT[var][:, tt] = np.flip(np.interp(
                np.flip(isos), np.flip(data.T.values[mask, tt]),
                np.flip(data[var].values[mask, tt]),
                left=np.nan, right=np.nan))

    return isoT


def backrotate(ds, f0):
    ''' assumes that f0 is in cpd. '''

    tvec = xr.DataArray((ds.time - ds.time[0]).values
                        .astype('timedelta64[D]').astype('float32'),
                        dims=['time'], coords={'time': ds.time})

    br = (ds.uz + 1j * ds.vz) * np.exp(1j * 2 * np.pi * f0 * tvec)
    br.attrs['long_name'] = 'Inertial backrotated shear'
    br.attrs['units'] = '1/s'

    return br


def process_adcp(mooring, argo_superset=None, nsmooth_shear=4):

    # estimate climatological N
    if not argo_superset:
        argo_superset = dcpy.oceans.read_argo_clim()[['Tmean', 'Smean']]

    argo = (argo_superset
            .sel(lat=mooring.lat, lon=mooring.lon, method='nearest'))
    argo['ρmean'] = xr.DataArray(sw.pden(argo.Smean, argo.Tmean, argo.pres),
                                 dims=['pres'], coords={'pres': argo.pres})
    argo['N2'] = argo.ρmean.differentiate('pres') * 9.81/1025
    argo['N'] = np.sqrt(argo.N2)

    vel = (mooring.vel[['u', 'v']].copy()
           .interpolate_na('depth')
           .interpolate_na('time')
           .dropna('depth', how='any')
           .dropna('time', how='any'))

    vel['good_data'] = ~np.isnan(mooring.vel.u)

    vel['uz'] = smooth_shear(vel.u.differentiate('depth'), nsmooth_shear)
    vel['vz'] = smooth_shear(vel.v.differentiate('depth'), nsmooth_shear)

    # truncate to valid range
    if mooring.name == 'NRL4':
        vel = vel.sel(time=slice(None, '2015-03-01'))

    # map temperature to ADCP grid
    vel['T'] = (mooring.ctd.T.rename({'depth2': 'depth'})
                .interp(time=vel.time, depth=vel.depth)
                .interpolate_na('depth')
                .interpolate_na('time', limit=24))
    vel['T'].attrs['description'] = 'CTD T interpolated to ADCP grid'
    vel['T'].attrs['units'] = '°C'

    # WKB scale velocities
    vel['Nmean'] = (argo.N.interp(pres=vel.depth.values)
                    .rename({'pres': 'depth'}))
    vel['wkb_factor'] = vel.Nmean/vel.Nmean.mean('depth')
    vel['wkb_factor'].attrs['long_name'] = 'N(z) / Nmean'
    wkbz = xrsp.integrate.cumtrapz(vel.wkb_factor, 'depth').values
    vel['wkbz'] = xr.DataArray(wkbz, dims=['wkbz'], coords={'wkbz': wkbz},
                               attrs={'long_name': 'WKB scaled depth',
                                      'units': 'm'})

    vel['wkbu'] = xr.DataArray((vel.u / np.sqrt(vel.wkb_factor)).values,
                               dims=['wkbz', 'time'],
                               attrs={'long_name': 'WKB u', 'units': 'm/s'})
    vel['wkbv'] = xr.DataArray((vel.v / np.sqrt(vel.wkb_factor)).values,
                               dims=['wkbz', 'time'],
                               attrs={'long_name': 'WKB v', 'units': 'm/s'})
    vel.wkbz.attrs['long_name'] = 'WKB scaled depth'
    vel.wkbz.attrs['units'] = 'm'
    vel['wkbKE'] = 0.5 * np.hypot(vel.wkbu, vel.wkbv)
    vel.wkbKE.attrs['long_name'] = 'WKB scaled KE'
    vel.wkbKE.attrs['units'] = 'm²/s²'

    vel['KE'] = 0.5 * np.hypot(vel.u, vel.v)
    vel.KE.attrs['long_name'] = 'KE'
    vel.KE.attrs['units'] = 'm²/s²'

    # backrotate
    br = backrotate(vel, mooring.inertial)
    vel['uz_back_real'] = np.real(br)
    vel['uz_back_imag'] = np.imag(br)

    if mooring.name == 'NRL5':
        z0 = 60
    else:
        z0 = 0

    # isothermal frame
    isotherms = (mooring.ctd.T.mean('time')
                 .rename({'depth2': 'depth'})
                 .sel(depth=slice(z0, None, 1))
                 .dropna('depth', how='all'))
    isotherms.values = np.flip(np.sort(isotherms.values))

    isoT = to_isothermal_space(vel, ['wkbu', 'wkbv', 'uz', 'vz',
                                     'uz_back_real', 'uz_back_imag'],
                               isotherms)

    # near-inertial bandpass filtering
    filtered = vel[['u', 'v', 'wkbu', 'wkbv', 'uz', 'vz']].apply(
        xfilter.bandpass, coord='time',
        freq=[0.8, 1.2]*mooring.inertial.values,
        cycles_per='D')

    iso_filtered = (isoT[['wkbu', 'wkbv']]
                    .interpolate_na('time', limit=12)
                    .apply(xfilter.bandpass, coord='time',
                           freq=[0.8, 1.2] * mooring.inertial.values,
                           cycles_per='D'))

    filtered = (filtered.dropna('depth', how='all')
                .sel(time=slice('2013-12', '2014-11')))
    iso_filtered = (iso_filtered.dropna('T', how='all')
                    .sel(time=slice('2013-12', '2014-11')))

    filtered['wkbKE'] = 0.5 * np.hypot(filtered.wkbu, filtered.wkbv)
    filtered.wkbKE.attrs['long_name'] = 'Near-inertial WKB scaled KE'
    filtered.wkbKE.attrs['units'] = 'm²/s²'
    filtered['KE'] = 0.5 * np.hypot(filtered.u, filtered.v)
    filtered.KE.attrs['long_name'] = 'Near-inertial KE'
    filtered.KE.attrs['units'] = 'm²/s²'

    # don't think this is a good idea, though not sure why.
    # filtered['uz_back'] = backrotate(filtered, mooring.inertial)

    vel['inertial'] = mooring.inertial
    filtered['inertial'] = mooring.inertial
    isoT['inertial'] = mooring.inertial

    vel.to_netcdf('../estimates/ebob-adcp/'
                  + mooring.name.lower() + '-vel.nc')
    filtered.to_netcdf('../estimates/ebob-adcp/'
                       + mooring.name.lower() + '-vel-niw-filtered.nc')
    isoT.to_netcdf('../estimates/ebob-adcp/'
                   + mooring.name.lower() + '-vel-isoT.nc')


def read_adcp(name):

    vel = dict()

    vel['vel'] = xr.open_dataset('../estimates/ebob-adcp/'
                                 + name.lower() + '-vel.nc')
    vel['niw'] = xr.open_dataset('../estimates/ebob-adcp/'
                                 + name.lower() + '-vel-niw-filtered.nc')
    vel['iso'] = xr.open_dataset('../estimates/ebob-adcp/'
                                 + name.lower() + '-vel-isoT.nc')

    # backrotation
    for var in ['vel', 'iso']:
        vel[var]['uz_back'] = (vel[var].uz_back_real
                               + 1j * vel[var].uz_back_imag)
        vel[var] = vel[var].drop(['uz_back_real', 'uz_back_imag'])

    for vv in vel:
        vel[vv].attrs['name'] = name.upper()

    return vel
