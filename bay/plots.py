import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import xarray as xr


def make_map(pods, DX=0.6, DY=0.55, add_year=True, highlight=[]):

    # markers = {'RAMA': 'o', 'NRL': '^', 'OMM/WHOI': 'o'}
    # colors = {'RAMA': '#0074D9', 'NRL': '#3D9970', 'OMM/WHOI': '#FF4136'}
    # colors = {'RAMA': '#1696A3', 'NRL': '#F89B1F', 'OMM/WHOI': '#EA4D5B'}
    colors = {'RAMA': '#1696A3', 'NRL': '#F89B1F', 'OMM/WHOI': '#EA4D5B'}

    import cartopy.crs as ccrs
    from cartopy import feature
    # from mpl_toolkits.basemap import cm

    labelargs = {'fontsize': 11.5,
                 'bbox': {
                     'alpha': 0.25,
                     'edgecolor': 'none',
                     'boxstyle': 'round'}}

    # ProjParams = {'min_latitude' : 0.0, # same as lat_0 in proj4 string
    #               'max_latitude' : 30.0, # same as lat_0 in proj4 string
    #               'central_longitude' : 88.0, # same as lon_0
    # }
    proj = ccrs.PlateCarree()

    coastline = feature.NaturalEarthFeature(name='coastline',
                                            category='physical',
                                            scale='50m',
                                            edgecolor='black',
                                            facecolor='#b2ccba')

    etopo = xr.open_dataset('~/datasets/ETOPO2v2g_f4.nc4', autoclose=True)
    extract = {'x': slice(60, 96), 'y': slice(-5, 25)}

    def get_color(name, highlight):
        if name in highlight or len(highlight) == 0:
            return colors[name]
        else:
            return '#888888'

    with plt.rc_context({'font.size': 14}):
        ax = plt.axes(projection=proj)
        ax.set_extent([80, 96, 2, 24])
        ax.add_feature(coastline)
        # ax.coastlines('10m', color='slategray', facecolor='slategray', lw=1)

        etopo.z.sel(**extract).plot.contour(
            transform=ccrs.PlateCarree(),
            add_colorbar=False,
            levels=np.sort([-100, -500, -1000,
                            -2500, -4000, -5000]),
            colors=['gray'], linewidths=0.5)
        ax.set_xlabel('')
        ax.set_ylabel('')

        for name in pods:
            pod = pods[name]
            ax.plot(pod['lon'], pod['lat'], transform=ccrs.PlateCarree(),
                    linestyle='',
                    marker='o',
                    color=get_color(pod['label'], highlight),
                    label=pod['label'], zorder=10)

            text = ''
            if add_year:
                if pod['label'] == 'RAMA' and pod['lat'] == 15:
                    text += '2015\n\n'
                if pod['label'] == 'OMM/WHOI':
                    text += '2014-15\n\n'

            for z in pod['depths']:
                text += z
                if pod['label'] == 'RAMA' and pod['lat'] == 12 and add_year:
                    text += ' / ' + pod['depths'][z]

                text += '\n'

            text = text[:-1]

            dx = 0
            dy = 0
            if pod['ha'] is 'left':
                dx = DX
            elif pod['ha'] is 'right':
                dx = -1 * DX
            if pod['va'] is 'bottom':
                dy = DY
            elif pod['va'] is 'top':
                dy = - DY

            if name == 'NRL3':
                dx += 0.6

            if pod['label'] == 'OMM/WHOI':
                dy += 1.25

            labelargs['bbox']['facecolor'] = get_color(pod['label'], highlight)

            ax.text(pod['lon']+dx, pod['lat']+dy, text,
                    transform=ccrs.PlateCarree(),
                    ha=pod['ha'], va=pod['va'],
                    multialignment='center', **labelargs)

        ax.set_facecolor(None)

    return ax, colors


def bin_ktdf(KTdf, bins):

    error_depth = 5

    depth_minus_mld = (KTdf.z - KTdf.mld)
    depth_minus_ild = (KTdf.z - KTdf.ild)
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

    KTdf['bin'] = ''
    KTdf.bin[mask_ml] = 'ML'
    KTdf.bin[mask_bl] = 'BL'
    # KTdf.bin[mask_ml_plus] = 'ML+'
    # bins = get_kmeans_bins(7, KTdf['ρ'][mask_deep])
    # KTdf.bin = pd.qcut(KTdf.ρ, 10, precision=1)
    KTdf.bin[mask_deep] = pd.cut(KTdf.ρ[mask_deep],
                                 bins,
                                 precision=1)
    KTdf['bin'] = KTdf['bin'].astype('category')
    assert(np.sum(KTdf.bin == '') == 0)

    return KTdf


def get_kmeans_bins(k, data):
    centroids, _ = sp.cluster.vq.kmeans(data, k)
    cent = np.sort(centroids)
    bins = np.hstack([data.min(), (cent[:-1]+cent[1:])/2])
    return bins


def trim_horiz_violin(hdl):
    # halve the voilin
    for b in hdl['bodies']:
        m = np.mean(b.get_paths()[0].vertices[:, 1])
        b.get_paths()[0].vertices[:, 1] = np.clip((b.get_paths()[0]
                                                   .vertices[:, 1]),
                                                  -np.inf, m)


def format_moor_names(names):
    rama = []
    nrl = []

    for nn in names:
        if nn[0:4] == 'RAMA':
            rama.append(nn[4:])
        elif nn[0:3] == 'NRL':
            nrl.append(nn[3])

    string = ''
    if rama != []:
        string += 'R' + ','.join(rama)

    if nrl != []:
        if string is not '':
            string += '\n'

        string += 'N' + ','.join(nrl)

    return string


def _get_color_from_hdl(hdl):
    try:
        facecolor = hdl['bodies'][0].get_facecolor()[0]
    except TypeError:
        facecolor = hdl[2][0].get_facecolor()

    return facecolor[:-1]


def plot_distrib(ax, plotkind, var, zloc, zstd, width=12, percentile=False):
    ''' Actually plot the distribution '''

    base_marker_size = 5

    if np.all(var < 1):
        # if in log-space transform out and back
        median = np.log10(np.median(10**var))
        mean = np.log10(np.mean(10**var))
        if percentile:
            prc = np.log10(np.percentile(10**var, 99.5))
    else:
        median = np.median(var)
        mean = np.mean(var)
        if percentile:
            prc = np.percentile(var, 99.5)

    if plotkind is 'violin':
        if percentile:
            var[var > prc] = np.nan

        hdl = ax.violinplot(var[~np.isnan(var)], positions=[zloc],
                            widths=[width],
                            vert=False, showmedians=False)

        for pc in hdl['bodies']:
            pc.set_alpha(0.55)

        trim_horiz_violin(hdl)
        hdl['cmaxes'].set_visible(False)
        hdl['cmins'].get_paths()[0].vertices[:, 1] = \
            (zloc + np.array([-1, 1])*zstd)
        if percentile:
            # use percentile instead of max
            hdl['cbars'].get_paths()[0].vertices[1, 0] = prc

    elif plotkind is 'hist':
        hdl = ax.hist(var, bottom=zloc, density=True)

    color = _get_color_from_hdl(hdl)

    ax.plot(median, zloc, 'o',
            color=color, zorder=12, ms=base_marker_size-1)
    ax.plot(median, zloc, 'o',
            color='w', zorder=11, ms=base_marker_size+1)
    ax.plot(mean, zloc, '^',
            color=color, zorder=12, ms=base_marker_size)
    ax.plot(mean, zloc, '^',
            color='w', zorder=12, ms=base_marker_size+2,
            fillstyle='none')

    return hdl, median, mean


def vert_distrib(KTdf, bins, varname='KT', pal=None, f=None, ax=None,
                 label_moorings=True, label_bins=True, adjust_fig=True,
                 width=12, percentile=False, add_offset=True, **kwargs):

    import cycler

    plotkind = 'violin'
    nadd = 3  # number of extra colors to generate
    pal_dist = sns.color_palette("GnBu_d", n_colors=len(bins.unique())+nadd)
    pal_dist.reverse()
    pal_dist = pal_dist[0:-(nadd)]
    if 'ML' in bins.cat.categories:
        pal_dist = np.roll(pal_dist, -1, axis=0)
    # if 'ML+' in bins.cat.categories:
    #     pal_dist = np.roll(pal_dist, -1, axis=0)
    if 'BL' in bins.cat.categories:
        pal_dist = np.roll(pal_dist, -1, axis=0)
        if 'ML' in bins.cat.categories:
            temp = pal_dist[-2, :].copy()
            pal_dist[-2, :] = pal_dist[-1, :]
            pal_dist[-1, :] = temp

    # map_kind = {'NRL1': 'NRL',
    #             'NRL2': 'NRL',
    #             'NRL3': 'NRL',
    #             'NRL4': 'NRL',
    #             'NRL5': 'NRL',
    #             'RAMA12': 'RAMA',
    #             'RAMA15': 'RAMA'}

    if pal is None:
        pal = pal_dist

    if f is None:
        f, axx = plt.subplots(1, 4, sharex=True, sharey=True)
        ax = {'NE': axx[0], 'NESW': axx[1], 'SW': axx[2], 'SWNE': axx[3]}

    if varname is 'KT':
        title = '$\log_{10}$ hourly averaged $K_T$ (m²/s)'
        xlim = kwargs.pop('xlim', [-7.5, 2])
        xlines = kwargs.pop('xlines', [-5, -4, -1])
    else:
        title = varname
        xlim = kwargs.pop('xlim', None)
        xlines = kwargs.pop('xlines', [])

    months = {'NE': 'Dec-Feb', 'NESW': 'Mar-May',
              'SW': 'Jun-Sep', 'SWNE': 'Oct-Nov'}

    for seas in ax:
        aa = ax[seas]
        aa.set_prop_cycle(cycler.cycler('color', pal))
        aa.set_title(seas + '\n(' + months[seas] + ')')
        if xlim is not None:
            aa.set_xlim(xlim)
            aa.set_xticks(range(-7, 0))

    # for plotting mean, median profile
    zvec = dict()
    mdnvec = dict()
    meanvec = dict()
    for seas in months.keys():
        zvec[seas] = []
        mdnvec[seas] = []
        meanvec[seas] = []

    # seaborn treats things as categoricals booo...
    # do things the verbose matplotlib way
    for index, (label, df) in enumerate(KTdf.groupby([bins, 'season'])):
        interval = label[0]
        season = label[1]
        zloc = sp.stats.trim_mean(df.z, 0.1)

        if add_offset:
            if type(interval) != str and interval.mid < 1020:
                if season == 'SWNE':
                    zloc += 5
            elif interval == 'BL':
                if season == 'NESW':
                    zloc -= 5
                else:
                    zloc -= 10
            elif interval == 'ML':
                zloc -= 5

        bin_name = (np.round(interval.mid-1000, 1).astype('str')
                    if isinstance(interval, pd.Interval)
                    else interval)

        var = df[varname]
        if varname == 'ρ':
            var -= 1000

        for mm in df['moor'].unique():
            if len(df['moor'].ix[df['moor'] == mm])/len(df) <= 0.10:
                df = df[df['moor'] != mm]

        hdl, median, mean = plot_distrib(ax[season], plotkind, var,
                                         zloc, df.z.std(), width, percentile)
        color = _get_color_from_hdl(hdl)

        zvec[season].append(zloc)
        mdnvec[season].append(median)
        meanvec[season].append(mean)

        try:
            xtxt = hdl['cbars'].get_paths()[0].vertices[1, 0]
            ytxt = zloc

            if label_bins:
                ax[season].text(xtxt, ytxt, '     '+bin_name,
                                color=color, ha='left', va='center',
                                fontsize=7)
            if label_moorings:
                ax[season].text(2, ytxt,
                                format_moor_names(df['moor'].unique()),
                                color=color, ha='left',
                                va='center', fontsize=6)
        except TypeError:
            pass

    ax['NE'].set_ylabel('depth (m)')
    ax['NE'].set_ylim([120, 0])
    # ax['NE'].set_yticks(np.arange(120,0,-20))
    # ax['NE'].set_yticklabels(np.arange(120,0,-10).astype('str'))

    for season in ax:
        # args = {'linestyle': '--', 'color': 'dimgray'}
        # idx = np.argsort(zvec)
        # ax[season].plot(np.array(mdnvec[season])[idx],
        #                 np.array(zvec[season])[idx], **args)
        # ax[season].plot(np.array(meanvec[season])[idx],
        #                 np.array(zvec[season])[idx], **args)
        aa = ax[season]
        for xx in xlines:
            aa.axvline(xx, lw=0.5, ls='dotted', zorder=-50, color='gray')

    if adjust_fig:
        sns.despine(fig=f, left=False, bottom=False, trim=True)
        f.set_size_inches((8.5, 5.5))
        f.suptitle(title, y=0.03, va='baseline')
        plt.subplots_adjust(wspace=0.25)

    return f, ax


def make_vert_distrib_plot(varname, bins):

    turb = xr.open_dataset('bay_merged_hourly.nc', autoclose=True).load()

    turb[varname].values = np.log10(turb[varname].values)

    turb['season'] = turb.time.monsoon.labels

    df = (turb[[varname, 'T', 'S', 'z', 'ρ', 'mld', 'ild', 'season']]
          .to_dataframe()
          .dropna(axis=0, subset=[varname])
          .reset_index())

    df['latlon'] = (df['lat'].astype('float32').astype('str') + 'N, '
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

    df['moor'] = (df['latlon']
                  .map(dict(zip(moornames.values(),
                              moornames.keys())))
                  .astype('category'))

    df = bin_ktdf(df, bins)

    f, ax = vert_distrib(df, df.bin, label_moorings=False, percentile=True)

    plt.subplots_adjust(wspace=-0.06)
