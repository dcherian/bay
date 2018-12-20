import dcpy
import numpy as np
import seawater as sw
import tqdm
import xarray as xr
import xfilter
import xrft
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


def process_adcp(mooring, argo_superset=None, nsmooth_shear=4,
                 isothermal=False):

    freq_factor = [0.6, 2.1]
    filter_kwargs = dict(
        coord='time', cycles_per='D', order=3,
        freq=freq_factor * mooring.inertial.values
    )

    # estimate climatological N
    if not argo_superset:
        argo_superset = dcpy.oceans.read_argo_clim()[['Tmean', 'Smean']]

    argo = (argo_superset
            .sel(lat=mooring.lat, lon=mooring.lon, method='nearest'))
    argo['ρmean'] = xr.DataArray(sw.pden(argo.Smean, argo.Tmean, argo.pres),
                                 dims=['pres'], coords={'pres': argo.pres})
    argo['N2'] = argo.ρmean.differentiate('pres') * 9.81/1025
    argo['N'] = np.sqrt(argo.N2)

    # truncate to valid range
    if mooring.name == 'NRL4':
        vel = (mooring.vel[['u', 'v']]
               .sel(time=slice(None, '2015-03-01'))
               .copy())
    else:
        vel = mooring.vel[['u', 'v']].copy()

    vel = (vel.interpolate_na('depth')
           .interpolate_na('time')
           .dropna('depth', how='any')
           .dropna('time', how='any'))

    vel['good_data'] = ~np.isnan(mooring.vel.u)

    vel['uz'] = smooth_shear(vel.u.differentiate('depth'), nsmooth_shear)
    vel['vz'] = smooth_shear(vel.v.differentiate('depth'), nsmooth_shear)

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

    # near-inertial bandpass filtering
    filtered = vel[['u', 'v', 'wkbu', 'wkbv', 'uz', 'vz']].apply(
        xfilter.bandpass, **filter_kwargs)
    filtered = filtered.dropna('depth', how='all')
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

    vel['zχpod'] = mooring.zχpod.interp(time=vel.time)
    filtered['zχpod'] = vel['zχpod']

    filtered.attrs['niw_lo'] = freq_factor[0]
    filtered.attrs['niw_hi'] = freq_factor[1]
    filtered.attrs['description'] = ('bandpass filtered in ' +
                                     str(freq_factor) + 'f_0 = ' +
                                     str(filter_kwargs['freq']) + ' cpd.')

    vel.to_netcdf('../estimates/ebob-adcp/'
                  + mooring.name.lower() + '-vel.nc')
    filtered.to_netcdf('../estimates/ebob-adcp/'
                       + mooring.name.lower() + '-vel-niw-filtered.nc')

    # isothermal frame
    if isothermal:
        isotherms = (mooring.ctd.T.mean('time')
                     .rename({'depth2': 'depth'})
                     .sel(depth=slice(z0, None, 1))
                     .dropna('depth', how='all'))
        isotherms.values = np.flip(np.sort(isotherms.values))

        isoT = to_isothermal_space(vel, ['wkbu', 'wkbv', 'uz', 'vz',
                                         'uz_back_real', 'uz_back_imag'],
                                   isotherms)

        iso_filtered = (isoT[['wkbu', 'wkbv']]
                        .interpolate_na('time', limit=12)
                        .apply(xfilter.bandpass, **filter_kwargs))

        iso_filtered = iso_filtered.dropna('T', how='all')

        isoT['inertial'] = mooring.inertial

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

    vel['low'] = (vel['vel'][['u', 'v', 'uz', 'vz']]
                  .apply(xfilter.lowpass, coord='time', freq=0.1,
                         cycles_per='D'))

    for var in vel:
        vel[var]['shear'] = vel[var].uz + 1j * vel[var].vz
        if var is not 'vel' and var is not 'iso':
            vel[var]['good_data'] = vel['vel'].good_data
        if 'u' in vel[var]:
            vel[var]['w'] = vel[var].u + 1j * vel[var].v
        if 'wkbu' in vel[var]:
            vel[var]['wkbw'] = vel[var].wkbu + 1j * vel[var].wkbv

    vel['niw']['shear'].dc.set_name_units('NIW shear', '1/s')
    vel['vel']['shear'].dc.set_name_units('shear', '1/s')
    vel['iso']['shear'].dc.set_name_units('isothermal shear', '1/s')

    vel['niw']['w'].dc.set_name_units('NIW velocity', 'm/s')
    vel['vel']['w'].dc.set_name_units('velocity', 'm/s')
    # vel['iso']['w'].dc.set_name_units('isothermal velocity', 'm/s')

    for vv in vel:
        vel[vv].attrs['name'] = name.upper()
        vel[vv].attrs['mooring'] = name.upper()
        for dd in vel[vv]:
            vel[vv][dd].attrs['mooring'] = name.upper()

    return vel


def partition_niw_up_down(data):
    '''
    Runs fft along depth, zeros out positive or negative wavenumber
    and then runs ifft to filter out up and down near-inertial motions.
    '''

    dim = list(data.dims)
    dim.remove('time')
    fft = xrft.dft(data, dim=dim, shift=False, window=False,
                   detrend='constant')
    axis = fft.get_axis_num('freq_' + dim[0])
    # ifft = np.fft.ifft(np.fft.ifftshift(fft, axes=axis), axis=axis)

    upkernel = (1 + np.tanh(fft.freq_depth/0.005))/2
    downkernel = (1 + np.tanh(-fft.freq_depth/0.005))/2

    up = xr.DataArray(np.fft.ifft(fft * upkernel, axis=axis),
                      dims=data.dims, coords=data.coords,
                      attrs={'long_name': 'up ' + data.attrs['long_name']},
                      name='up')

    down = xr.DataArray(np.fft.ifft(fft * downkernel, axis=axis),
                        dims=data.dims, coords=data.coords,
                        attrs={'long_name': 'down ' + data.attrs['long_name']},
                        name='down')

    # remove edge effects
    down[0, :] = np.nan
    down[-1, :] = np.nan
    up[0, :] = np.nan
    up[-1, :] = np.nan

    updown = xr.merge([up, down])

    updown.attrs = data.attrs

    return updown
