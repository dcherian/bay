def StitchRamaFluxes():
    '''
    Stich together RAMA flux observations
    '''

    import xarray as xr
    import numpy as np
    import matplotlib.pyplot as plt
    import dcpy.plots

    ramadir = '/home/deepak/TaoTritonPirataRama/high_resolution/'

    # nc = (no current?) absolute wind speed
    # nclw = absolute wind speed + long wave climatology
    # nlw = (no longwave?) relative wind speed + long wave climatology
    # otherwise relative wind speed = needs ocean current

    #------------------------- 15N90E
    lhf = xr.open_dataset(ramadir + '/hr/qlatnc15n90e_hr.cdf').QL_137
    lhf.values[lhf > 1e6] = np.nan
    lhf.name = 'lhf'

    shf = xr.open_dataset(ramadir + '/hr/qsennc15n90e_hr.cdf').QS_138
    shf.values[shf > 1e6] = np.nan
    shf.name = 'shf'

    swr = xr.open_dataset(ramadir + '/hr/swnetnc15n90e_hr.cdf').SWN_1495
    swr.values[swr > 1e6] = np.nan
    swr.name = 'swr'

    lwr = xr.open_dataset(ramadir + '/hr/lwnetnc15n90e_hr.cdf').LWN_1136
    lwr.values[lwr > 1e6] = np.nan
    lwr.name = 'lwr'

    qnet = xr.open_dataset(ramadir + '/hr/qnetnc15n90e_hr.cdf').QT_210
    qnet.values[qnet > 1e6] = np.nan
    qnet.name = 'Jq0'

    # lwr152m = xr.open_dataset(ramadir + '/2m/lw15n90e_2m.cdf').Ql_136
    # lwr152m.values[lwr152m > 1e6] = np.nan

    r15 = xr.merge([lhf, shf, swr, lwr, qnet])

    PlotStitched(r15)
    plt.savefig('/home/deepak/bay/images/stitched-fluxes-ra15.png', bbox_inches='tight')

    def interpolate_time(out_time, invar, lat, lon, name):
        return xr.DataArray(np.interp(out_time.astype('float32'),
                                   invar.time.astype('float32'),
                                   invar.values.squeeze(),
                                 left=np.nan, right=np.nan)
                       [:, np.newaxis, np.newaxis, np.newaxis],
                       dims=['time', 'depth', 'lat', 'lon'],
                       coords=[out_time, [0.], [lat], [lon]],
                       name=name)

    # ---------------------------  12N90E
    lhf = xr.open_dataset(ramadir + '/hr/qlat_nlw12n90e_hr.cdf').QL_137
    lhf.values[lhf > 1e6] = np.nan
    lhf.name = 'lhf'

    shf = xr.open_dataset(ramadir + '/hr/qsen_nlw12n90e_hr.cdf').QS_138
    shf.values[shf > 1e6] = np.nan
    shf.name = 'shf'

    swr = xr.open_dataset(ramadir + '/hr/swnet_nclw12n90e_hr.cdf').SWN_1495
    swr.values[swr > 1e6] = np.nan
    swr.name = 'swr'

    # interpolate 15n swr to 12n grid : fills in gap starting 2014-aug
    swr15i = interpolate_time(swr.time, r15.swr, 12, 90, 'swr')
    shf15i = interpolate_time(shf.time, r15.shf, 12, 90, 'lhf')
    lhf15i = interpolate_time(lhf.time, r15.lhf, 12, 90, 'lhf')

    swr.values[np.isnan(swr)] = swr15i.values[np.isnan(swr)]
    lhf.values[np.isnan(shf)] = lhf15i.values[np.isnan(lhf)]
    shf.values[np.isnan(shf)] = shf15i.values[np.isnan(shf)]

    # compare latent and sensible
    f, ax = plt.subplots(1, 3)
    ax[0].plot(lhf15i.squeeze(), lhf.squeeze(), '.', ms=2)
    dcpy.plots.line45(ax=ax[0])
    ax[0].set_xlabel('15N latent')
    ax[0].set_ylabel('12N latent')

    ax[1].plot(shf15i.squeeze(), shf.squeeze(), '.', ms=2)
    dcpy.plots.line45(ax=ax[1])
    ax[1].set_xlabel('15N sensible')
    ax[1].set_ylabel('12N sensible')

    ax[2].plot(swr15i.squeeze(), swr.squeeze(), '.', ms=2)
    dcpy.plots.line45(ax=ax[2])
    ax[2].set_xlabel('15N SW')
    ax[2].set_ylabel('12N SW')

    plt.savefig('/home/deepak/bay/images/stitched-fluxes-rama-12n-vs-15ng.png',
                bbox_inches='tight')

    # no longwave at 12N, take 15n and interpolate to 12N time
    lwr15 = xr.open_dataset(ramadir + '/hr/lwnetnc15n90e_hr.cdf').LWN_1136
    lwr15.values[lwr15 > 1e6] = np.nan
    lwr = interpolate_time(swr.time, lwr15, 12, 90, 'lwr')

    qnet = swr - lwr - lhf - shf
    qnet.name = 'Jq0'

    r12 = xr.merge([lhf, shf, swr, lwr, qnet])

    r12.squeeze().to_netcdf('./rama-12n-fluxes.nc')
    r15.squeeze().to_netcdf('./rama-15n-fluxes.nc')

    PlotStitched(r12)
    plt.savefig('/home/deepak/bay/images/stitched-fluxes-ra12.png', bbox_inches='tight')
    # return r12, r15


def PlotStitched(data):

    import matplotlib.pyplot as plt

    f, ax = plt.subplots(5, 1, sharex=True)
    f.set_size_inches((11.5, 8.5))
    data.swr.squeeze().plot(ax=ax[0], lw=0.5)
    data.lwr.squeeze().plot(ax=ax[1], lw=0.5)
    data.shf.squeeze().plot(ax=ax[2], lw=0.5)
    data.lhf.squeeze().plot(ax=ax[3], lw=0.5)
    data.Jq0.squeeze().plot(ax=ax[4], lw=0.5)

    ax[0].set_xlim(['2013-12-01', '2015-12-31'])


StitchRamaFluxes()
