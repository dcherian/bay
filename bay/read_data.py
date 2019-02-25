import importlib

import moor
import chipy
import numpy as np
import tqdm

from .adcp import process_adcp

moor = importlib.reload(moor)
chipy = importlib.reload(chipy)


def read_all_moorings(minimal=False):
    print('Reading all moorings...')
    ra12 = read_ra12(minimal)
    ra15 = read_ra15(minimal)
    nrl1 = read_nrl1(minimal)
    nrl2 = read_nrl2(minimal)
    nrl3 = read_nrl3(minimal)
    nrl4 = read_nrl4(minimal)
    nrl5 = read_nrl5(minimal)

    return [ra12, ra15, nrl1, nrl2, nrl3, nrl4, nrl5]


def read_ra12(minimal=False):

    ra12 = moor.moor(90, 12, 'RAMA 12N', 'ra12', 'rama', '../rama/RAMA13/')
    ra12.AddChipod(526, 15, 'mm1w', 'Turb.mat')
    ra12.AddChipod(527, 30, 'mm1w', 'Turb.mat')
    ra12.AddChipod(810, 15, 'mmw', 'Turb.mat', dir='../rama/RAMA14/')
    ra12.AddChipod(811, 30, 'mm1w', 'Turb.mat', dir='../rama/RAMA14/')
    ra12.AddChipod(812, 45, 'mm2w', 'Turb.mat', dir='../rama/RAMA14/')
    ra12.ReadMet('../rama/data/met12n90e_10m.cdf', WindType='pmel')
    ra12.ReadMet(FluxType='pmel')
    ra12.ReadVel('../rama/data/cur12n90e_30m.cdf', FileType='pmel')

    if not minimal:
        # ra12.ReadMet('../rama/data/jq0_12n90e_hr.mat', FluxType='merged')
        ra12.ReadMet(FluxType='precip')
        ra12.ReadTropflux('../rama/tropflux/')

    # ra12.ReadMet('../rama/data/qnet12n90e_hr.cdf', FluxType='pmel')
    ra12.ReadCTD('../rama/RamaPrelimProcessed/RAMA13-corrected.mat',
                 'ramaprelim')
    ra12.ReadCTD('../rama/RamaPrelimProcessed/RAMA14-12N-corrected.mat',
                 'ramaprelim')
    # ra12.AddDeployment('RAMA13', '2013-Nov-29', '2014-Dec-')
    # ra12.AddDeployment('RAMA14', 't0', 't1')

    ra12.AddEvents('FW1', '2014-Jan-15', '2014-Jan-19')
    ra12.AddEvents('Hudhud', '2014-Oct-08', '2014-Oct-11', 526)
    ra12.AddEvents('FW2', '2015-Apr-05', '2015-Apr-10')
    ra12.AddEvents('FW3', '2015-Oct-28', '2015-Nov-05')

    ra12.χpod[526].mixing_seasons = dict(
        ne2014=slice('2013-11-29', '2014-03-22'),
        nesw2014=slice('2014-03-24', '2014-05-07'),
        sw2014=slice('2014-05-08', '2014-09-22'),
        swne2014=slice('2014-09-23', '2014-11-22'))

    ra12 = _filter_wda_rama(ra12)

    ra12 = __common(ra12, minimal)

    return ra12


def read_ra15(minimal=False):

    ra15 = moor.moor(90, 15, 'RAMA 15N', 'ra15', 'rama', '../rama/RAMA14/')
    ra15.AddChipod(813, 15, 'pmw', 'Turb.mat')
    # ra15.AddChipod(814, 30, 'mm1w', 'Turb.mat')
    ra15.ReadMet('../rama/data/met15n90e_10m.cdf', WindType='pmel')
    ra15.ReadMet(FluxType='pmel')
    ra15.ReadVel('../rama/data/cur15n90e_30m.cdf', FileType='pmel')

    if not minimal:
        ra15.χpod[813].load_pitot()
        # ra15.χpod[814].load_pitot()

    ra15.ReadCTD('../rama/RamaPrelimProcessed/RAMA14-15N.mat', 'ramaprelim')

    # ra15.AddSeason([813, 814], 'NE', '2014-Dec-01', '2015-Mar-01')
    # ra15.AddSeason([813, 814], 'NE→SW', '2015-Mar-01', '2015-May-15')
    # ra15.AddSeason([813, 814], 'SW', '2015-May-15', '2015-Oct-14')

    ra15.AddEvents('FW4', '2015-Aug-10', '2015-Aug-20', 813)

    ra15.χpod[813].mixing_seasons = {
        'NE': slice('2014-12-01', '2015-03-22'),
        'NESW': slice('2015-03-22', '2015-05-15'),
        'SW': slice('2015-05-15', '2015-09-30'),
        'SWNE': slice('2015-10-01', '2015-11-30')}

    # ra15.χpod[814].mixing_seasons = {
    #     'NE': slice('2014-12-01', '2015-03-15'),
    #     'NESW': slice('2015-03-15', '2015-05-15'),
    #     'SW': slice('2015-05-15', '2015-09-30')}

    ra15 = _filter_wda_rama(ra15)

    ra15 = __common(ra15, minimal)

    return ra15


def read_nrl1(minimal=False):
    nrl1 = moor.moor(85.5, 5.0, 'NRL1', 'nrl1', 'ebob', '../ebob/')
    nrl1.ReadCTD('NRL1', FileType='ebob')
    nrl1.AddChipod(500, depth=56, best='mm', fname='Turb.mat')
    nrl1.AddChipod(501, depth=76, best='mm1', fname='Turb.mat')

    nrl1.ReadVel('NRL1', FileType='ebob')

    if not minimal:
        nrl1.ReadNIW()

    nrl1.AddEvents("SW1", '2014-01-23', '2014-02-02')
    nrl1.AddEvents("FW1", '2014-07-18', '2014-07-30')
    nrl1.AddEvents("Shear", '2014-08-07', '2014-09-30')

    nrl1 = __common(nrl1, minimal)

    return nrl1


def read_nrl2(minimal=False):

    nrl2 = moor.moor(85.5, 6.5, 'NRL2', 'nrl2', 'ebob', '../ebob/')
    nrl2.ReadCTD('NRL2', FileType='ebob')
    nrl2.AddChipod(504, 69, 'mm', 'Turb.mat')
    nrl2.ReadVel('NRL2', FileType='ebob')

    nrl2 = __common(nrl2, minimal)

    return nrl2


def read_nrl3(minimal=False):

    nrl3 = moor.moor(85.5, 8, 'NRL3', 'nrl3', 'ebob', '../ebob/')
    nrl3.ReadCTD('NRL3', FileType='ebob')
    nrl3.AddChipod(505, 28, 'mm', 'Turb.mat')
    nrl3.AddChipod(511, 48, 'mm', 'Turb.mat')
    nrl3.AddEvents('SLD', '2014-06-01', '2014-09-01')
    # this period has salinity gap that I filled in using TS fits.
    nrl3.AddEvents('Sal gap', '2014-03-14', '2014-09-29')
    nrl3.ReadVel('NRL3', FileType='ebob')

    if not minimal:
        nrl3.ReadNIW()

    nrl3 = __common(nrl3, minimal)

    return nrl3


def read_nrl4(minimal=False):

    nrl4 = moor.moor(87, 8, 'NRL4', 'nrl4', 'ebob', '../ebob/')
    nrl4.ReadCTD('NRL4', FileType='ebob')
    nrl4.AddChipod(514, 55, 'mm1', 'Turb.mat')
    nrl4.AddChipod(516, 75, 'mm2', 'Turb.mat')
    nrl4.AddEvents('SLD', '2014-06-01', '2014-09-01')
    nrl4.ReadVel('NRL4', FileType='ebob')

    if not minimal:
        nrl4.ReadNIW()

    nrl4 = __common(nrl4, minimal)

    nrl4.χpod[516].mixing_seasons = dict(
        himix0=slice('2014-05-08', '2014-06-22'),
        himix=slice('2014-07-23', '2014-10-08'),
        himix1=slice('2014-03-01', '2014-05-01'))

    return nrl4


def read_nrl5(minimal=False):

    nrl5 = moor.moor(88.5, 8, 'NRL5', 'nrl5', 'ebob', '../ebob/')
    nrl5.ReadCTD('NRL5', FileType='ebob')
    nrl5.AddChipod(518, depth=84, best='mm', fname='Turb.mat')
    nrl5.AddChipod(519, depth=104, best='mm1', fname='Turb.mat')
    # 519 mm2 reads low frequently
    nrl5.AddEvents('Storm+IW', '2014-07-17', '2014-08-07')
    nrl5.AddEvents('Storm-zoomin', '2014-07-25', '2014-08-07 11:59')
    nrl5.AddEvents("nomix", '2014-03-17', '2014-05-06')
    nrl5.ReadVel('NRL5', FileType='ebob')

    if not minimal:
        nrl5.ReadNIW()

    nrl5.χpod[519].mixing_seasons = dict(
        lowmix=slice('2014-03-17', '2014-05-05'),
        himix=slice('2014-05-06', '2014-11-01'),
        himix0=slice('2014-02-11', '2014-03-16'))

    nrl5 = __common(nrl5, minimal)

    return nrl5


def __common(mooring, minimal=False):
    mooring.calc_mld_ild_bld()
    mooring.ReadTropflux('../tropflux/')
    mooring.calc_niw_input()
    mooring.CombineTurb()
    if not minimal:
        mooring.ReadSSH()

    return mooring


def _filter_wda_rama(mooring):
    for unit in mooring.χpod:
        pod = mooring.χpod[unit]
        best = pod.chi[pod.best]
        if 'w' in pod.best:  # e.g. mm1w
            tzm = pod.chi[pod.best[:-1]].dTdz
            tzm = tzm.interp(time=best.time)
            jq0 = mooring.flux.Jq0.interp(time=tzm.time)

            mooring.χpod[unit].chi[pod.best] = best.where(
                ~((np.abs(tzm) < 1e-3) & (jq0 < 0)))

    return mooring


def process_adcps():
    for ff in tqdm.tqdm([read_nrl1, read_nrl2, read_nrl3, read_nrl4, read_nrl5]):
        mooring = ff()
        process_adcp(mooring)
