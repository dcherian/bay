def StitchRamaFluxes():
    '''
    Stich together RAMA flux observations
    '''

    import xarray as xr
    import scipy as sp
    import numpy as np

    ramadir = '~/TaoTritonPirataRama/high_resolution/'

    ########### RAMA 15N90E
    lhf = xr.open_dataset(ramadir + '/hr/qlat15n90e_hr.cdf').QL_137
    lhf.values[lhf > 1e6] = np.nan
    lhf.name = 'lhf'

    shf = xr.open_dataset(ramadir + '/hr/qsen15n90e_hr.cdf').QS_138
    shf.values[shf > 1e6] = np.nan
    shf.name = 'shf'

    swr = xr.open_dataset(ramadir + '/hr/swnet15n90e_hr.cdf').SWN_1495
    swr.values[swr > 1e6] = np.nan
    swr.name = 'swr'

    lwr = xr.open_dataset(ramadir + '/hr/lwnet15n90e_hr.cdf').LWN_1136
    lwr.values[lwr > 1e6] = np.nan
    lwr.name = 'lwr'

    qnet = xr.open_dataset(ramadir + '/hr/qnet15n90e_hr.cdf').QT_210
    qnet.values[qnet > 1e6] = np.nan
    qnet.name = 'Jq0'

    r15 = xr.merge([lhf, shf, swr, lwr, qnet])

    ############# RAMA 12N90E
    lhf = xr.open_dataset(ramadir + '/hr/qlat_nclw12n90e_hr.cdf').QL_137
    lhf.values[lhf > 1e6] = np.nan
    lhf.name = 'lhf'

    shf = xr.open_dataset(ramadir + '/hr/qsen_nclw12n90e_hr.cdf').QS_138
    shf.values[shf > 1e6] = np.nan
    shf.name = 'shf'

    swr = xr.open_dataset(ramadir + '/hr/swnet_nclw12n90e_hr.cdf').SWN_1495
    swr.values[swr > 1e6] = np.nan
    swr.name = 'swr'

    lwr15 = xr.open_dataset(ramadir + '/hr/lwnet15n90e_hr.cdf').LWN_1136
    lwr15.values[lwr15 > 1e6] = np.nan
    lwr = xr.DataArray(np.interp(swr.time.astype('float32'),
                                 lwr15.time.astype('float32'),
                                 lwr15.values.squeeze())[:, np.newaxis,
                                                         np.newaxis,
                                                         np.newaxis],
                       dims=['time', 'depth', 'lat', 'lon'],
                       coords=[swr.time, [0.], [12.], [90.]],
                       name='lwr')

    qnet = swr - lwr - lhf - shf
    qnet.name = 'Jq0'

    r12 = xr.merge([lhf, shf, swr, lwr, qnet])

    r12.to_netcdf('./rama-12n-fluxes.nc')
    r15.to_netcdf('./rama-15n-fluxes.nc')
    return r12, r15
