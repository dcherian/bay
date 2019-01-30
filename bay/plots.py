import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns

import cartopy.crs as ccrs
import dcpy
import xarray as xr
from cartopy import feature

from .bay import default_density_bins, nc_to_binned_df

# markers = {'RAMA': 'o', 'NRL': '^', 'OMM/WHOI': 'o'}
# colors = {'RAMA': '#0074D9', 'NRL': '#3D9970', 'OMM/WHOI': '#FF4136'}
# colors = {'RAMA': '#1696A3', 'NRL': '#F89B1F', 'OMM/WHOI': '#EA4D5B'}
colors = {'RAMA': '#1696A3', 'NRL': '#F89B1F', 'OMM/WHOI': '#EA4D5B'}
# colors = {'RAMA': '#1c666e', 'NRL': '#F89B1F', 'OMM/WHOI': '#EA4D5B'}
# colors = {'RAMA': '#1696A3', 'NRL': '#F89B1F', 'OMM/WHOI': 'gray'}


def plot_coastline(ax=None, facecolor="#FEF9E4"):

    if ax is None:
        ax = plt.gca()

    # facecolor = '#b2ccba'
    coastline = feature.NaturalEarthFeature(name='coastline',
                                            category='physical',
                                            scale='50m',
                                            edgecolor='black',
                                            facecolor=facecolor)

    ax.set_extent([80, 96, 2, 24])
    ax.add_feature(coastline)

    # ax.coastlines('10m', color='slategray', facecolor='slategray', lw=1)


def make_map(pods, DX=0.6, DY=0.7, add_year=True, highlight=[],
             ax=None, topo=True):

    labelargs = {'fontsize': 'small',
                 'bbox': {
                     'alpha': 0.85,
                     'edgecolor': 'none',
                     'boxstyle': 'round'}}

    proj = ccrs.PlateCarree()

    def get_color(name, highlight):
        if name in highlight or len(highlight) == 0:
            return colors[name]
        else:
            return '#888888'

    if ax is None:
        ax = plt.axes(projection=proj)

    plot_coastline(ax)

    if topo:
        etopo = xr.open_dataset('~/datasets/ETOPO2v2g_f4.nc4')
        extract = {'x': slice(60, 96), 'y': slice(-5, 25)}

        etopo.z.sel(**extract).plot.contour(
            transform=ccrs.PlateCarree(),
            add_colorbar=False,
            levels=np.sort([-100, -500, -1000,
                            -2500, -4000, -5000]),
            colors=['gray'], linewidths=0.5, ax=ax)

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
                text += 'RAMA 15 / 2015\n'
            if pod['label'] == 'RAMA' and pod['lat'] == 12:
                text += 'RAMA 12\n'
            if pod['label'] == 'OMM/WHOI':
                text += '2014-15\n'
            if 'NRL' in name:
                text += name + '\n'

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
        if 'RAMA' in name:
            dy += 0.75 * DY

        if name == 'NRL3':
            dx += 0.1
        if name == 'NRL5':
            dx -= 0.1
        if name in ['NRL3', 'NRL5']:
            dy += 0.3

        if pod['label'] == 'OMM/WHOI':
            dy += 2

        labelargs['bbox']['facecolor'] = get_color(pod['label'], highlight)

        ax.text(pod['lon']+dx, pod['lat']+dy, text,
                transform=ccrs.PlateCarree(),
                ha=pod['ha'], va=pod['va'], color='w',
                multialignment='center', **labelargs)

        ax.set_facecolor(None)

    return ax, colors


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
            var = var.where(var < prc)

        hdl = ax.violinplot(var[~np.isnan(var)], positions=[zloc],
                            widths=[width],
                            vert=False, showmedians=False)

        for pc in hdl['bodies']:
            pc.set_alpha(0.7)

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


def vert_distrib(df, bins, varname='KT', kind='distribution',
                 pal=None, f=None, ax=None,
                 label_moorings=True, label_bins=True, adjust_fig=True,
                 width=12, percentile=False, add_offset=True, **kwargs):
    '''
        Function to make vertical distribution plot when provided with
        appropriately formatted DataFrame.
    '''
    import cycler

    plotkind = 'violin'
    nadd = 2  # number of extra colors to generate
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

    if pal is None:
        pal = pal_dist

    if f is None:
        f, axx = plt.subplots(1, 4, sharex=True, sharey=True)
    else:
        axx = ax

    ax = {'NE': axx[0], 'NESW': axx[1], 'SW': axx[2], 'SWNE': axx[3]}

    if varname is 'KT':
        title = '$\\log_{10}$ hourly averaged $K_T$ [m²/s]'
        xlim = kwargs.pop('xlim', [-7.5, 2])
        xlines = kwargs.pop('xlines', [-5, -4])
    else:
        title = varname
        xlim = kwargs.pop('xlim', None)
        xlines = kwargs.pop('xlines', [])

    months = {'NE': 'Dec-Feb', 'NESW': 'Mar-Apr',
              'SW': 'May-Sep', 'SWNE': 'Oct-Nov'}

    for seas in ax:
        aa = ax[seas]
        aa.set_prop_cycle(cycler.cycler('color', pal))
        aa.set_title(seas + '\n(' + months[seas] + ')')
        if xlim is not None:
            aa.set_xlim(xlim)
            aa.set_xticks(range(-7, max(xlim)))

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
    for index, (label, df) in enumerate(df.groupby([bins, 'season'])):
        interval = label[0]
        season = label[1]
        zloc = sp.stats.trim_mean(df.z, 0.1)

        if add_offset:
            if type(interval) != str and interval.mid < 1020:
                if season == 'SWNE' or season == 'NE':
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

        if kind == 'distribution':
            for mm in df['moor'].unique():
                if (len(df['moor'].where(df.moor == mm).dropna())/len(df)
                    <= 0.10):
                    df = df.loc[df['moor'] != mm]

            hdl, median, mean = plot_distrib(ax[season], plotkind, var,
                                             zloc, df.z.std(), width,
                                             percentile)

            zvec[season].append(zloc)
            mdnvec[season].append(median)
            meanvec[season].append(mean)

            xtxt = hdl['cbars'].get_paths()[0].vertices[1, 0]
            color = _get_color_from_hdl(hdl)

        elif kind == 'mean_ci_profile':
            mean = np.log10(df[varname].values)
            err = np.abs(mean - np.log10(df.ci.values[0]))[:, np.newaxis]
            hdl = ax[season].errorbar(mean, zloc,
                                      xerr=err, fmt='^')

            xtxt = hdl.lines[-1][0].get_segments()[0][1, 0]
            color = hdl.lines[0].get_color()

        ytxt = zloc

        if label_bins:
            ax[season].text(xtxt, ytxt, '     '+bin_name,
                            color=color, ha='left', va='center',
                            fontsize=7)
        if label_moorings:
            ax[season].text(-0.75, ytxt,
                            format_moor_names(df['moor'].unique()),
                            color=color, ha='left',
                            va='center', fontsize=6)

    ax['NE'].set_ylabel('depth [m]')
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

    if label_moorings:
        plt.subplots_adjust(wspace=-0.1)
    else:
        plt.subplots_adjust(wspace=-0.2)

    return f, ax


def mark_moors(color='w',
               labels=True,
               colortext='k',
               markersize=14,
               fontsize=10,
               ax=None, **kwargs):
    lons = [85.5, 85.5, 87, 88.5, 90, 90]
    lats = [5, 8, 8, 8, 12, 15]
    names = ['1', '3', '4', '5', '12', '15']

    if ax is None:
        ax = plt.gca()

    ax.plot(lons, lats, 'o', color=color, ms=markersize, **kwargs)

    if labels:
        for lon, lat, name in zip(lons, lats, names):
            if len(name) == 2:
                ds = -1.5
            else:
                ds = 0

            ax.text(lon, lat, name,
                    ha='center', va='center',
                    fontdict=dict(color=colortext),
                    fontsize=fontsize+ds,
                    transform=ax.transData)


def mark_moors_clean(ax):
    mark_moors(ax=ax, markersize=4.5, fontsize=6, color='k',
               colortext='w', labels=False)
    mark_moors(ax=ax, markersize=3, fontsize=6, color='w',
               colortext='w', labels=False)


def make_vert_distrib_plot(varname,
                           bins=default_density_bins,
                           moor=None,
                           label_moorings=False, **kwargs):

    ''' user-friendly wrapper function to make vertical distribution plot. '''

    df = nc_to_binned_df(bins=bins, moor=moor)

    if varname == 'KT':
        df['KT'] = np.log10(df['KT'])

    vert_distrib(df, df.bin,
                 label_moorings=label_moorings,
                 percentile=True, **kwargs)


def make_labeled_map(ax=None):

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
                 '30 m': '2015'}},
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

    ax, colors = make_map(pods, add_year=True,
                          highlight=['RAMA', 'NRL', 'OMM/WHOI'], ax=ax)

    ax.text(87, 6.8, 'EBoB\n(2014)',
            color=colors['NRL'], ha='center', va='center')
    ax.text(90-0.35, 18, 'OMM/WHOI',
            color=colors['OMM/WHOI'], ha='right', va='center')
    ax.text(90, 13.5, 'RAMA',
            color=colors['RAMA'], ha='left', va='center')
    ax.set_xticks([80, 82, 84, 85.5, 87, 88.5, 90, 92, 94, 96])
    ax.set_yticks([2, 4, 5, 8, 12, 15, 18, 20, 22, 24])


def mark_χpod_depths_on_clim(ax=[], orientation='horizontal'):

    argoT = xr.open_dataset('~/datasets/argoclim/RG_ArgoClim_Temperature_2016.nc',
                            decode_times=False)
    argoS = xr.open_dataset('~/datasets/argoclim/RG_ArgoClim_Salinity_2016.nc',
                            decode_times=False)

    def mark_range(ax, top, middle, bottom, color=colors['NRL']):
        xlim = ax.get_xlim()
        ax.fill_between([xlim[0], xlim[1]], bottom, top,
                        facecolor=color, alpha=0.1, zorder=-5)
        dcpy.plots.liney(middle, ax=ax,
                         color=colors['NRL'], linestyle='-')

    def plot_profiles(ax, lon, lat, color):

        region = {'LATITUDE': lat, 'LONGITUDE': lon, 'method': 'nearest'}

        ax.plot(argoT.ARGO_TEMPERATURE_MEAN.sel(**region).squeeze(),
                argoT.PRESSURE, color=color, lw=2)
        ax2 = ax.twiny()
        ax2.plot(argoS.ARGO_SALINITY_MEAN.sel(**region).squeeze(),
                 argoS.PRESSURE, '--', color=color, lw=2)

        ax2.spines['top'].set_visible(True)
        ax.set_xlim([23, 30])
        ax.set_ylabel('Pressure')
        ax.text(29, 10, 'T', color=color)
        ax.text(23.75, 10, 'S', color=color)
        # ax.text(25.5, 114, 'Argo clim.', color='black',
        #         ha='center', fontsize=12)

        return ax2

    if len(ax) == 0:
        if orientation == 'horizontal':
            _, ax = plt.subplots(1, 2, sharey=True, constrained_layout=True)

        elif orientation == 'vertical':
            _, ax = plt.subplots(2, 1, sharey=True, constrained_layout=True)

    plot_profiles(ax[0], 90, 12, color=colors['RAMA'])
    plot_profiles(ax[1], 85.5, 5, color=colors['NRL'])
    mark_range(ax[1], 55, 60, 100)
    mark_range(ax[1], 75, 80, 115)
    mark_range(ax[1], 28, 32, 78)
    mark_range(ax[1], 48, 52, 100)
    mark_range(ax[1], 60, 63, 85)
    mark_range(ax[1], 80, 83, 105)
    dcpy.plots.liney([85, 105], ax=ax[1],
                     color=colors['NRL'], linestyle='-')
    dcpy.plots.liney([15, 30, 45], ax=ax[0],
                     color=colors['RAMA'], linestyle='-')

    ax[0].set_ylim([120, 0])
    ax[1].set_ylim([120, 0])


def KT_TS(turb, ctd, which_moorings='all', varname='KT', axes=None,
          cbar_kwargs={}):

    from dcpy.oceans import TSplot

    rama = ['ra12', 'ra15']
    ebob = ['nrl1', 'nrl2', 'nrl3', 'nrl4', 'nrl5']

    extra_filter = None

    if which_moorings == 'rama':
        moorings = rama
        extra_filter = turb.lat > 10
    elif which_moorings == 'ebob':
        moorings = ebob
        extra_filter = turb.lat < 10
    elif which_moorings == 'all':
        moorings = rama + ebob
    else:
        raise ValueError('moorings must be one of rama, ebob, all.')

    seasons = turb.time.monsoon.labels

    if axes is None:
        f, ax = plt.subplots(2, 2, sharex=True, sharey=True,
                             constrained_layout=True)
        f.set_constrained_layout_pads(wpad=1/72, hpad=1/72,
                                      wspace=0.0, hspace=0.0)
        axes = dict(zip(np.unique(seasons), ax.ravel()))
        f.set_size_inches(6, 6.5)

    else:
        assert all((season in axes) for season in ['NE', 'NESW', 'SW', 'SWNE'])
        f = axes['NE'].get_figure()

    extent = [31, 36, 16, 32]
    axes['NE'].set_xlim(extent[:2])
    axes['NE'].set_ylim(extent[2:])

    if varname == 'KT':
        levels = [1e-5, 1e-4, 1e-3]
        cmap = sns.light_palette('purple', n_colors=len(levels)+2,
                                 as_cmap=True)
        label = '$K_T$ [m²/s]'

    elif varname == 'Jq':
        levels = [-50, -25, -10, 0, 10, 25, 50]
        cmap = sns.diverging_palette(240, 10, n=levels, as_cmap=True)
        label = '$J_q^t$ [W/m²]'

    elif varname == 'Js':
        levels = [1e-4, 1e-3, 0.01]
        cmap = sns.light_palette('seagreen', n_colors=len(levels)+2,
                                 as_cmap=True)
        label = '$J_s^t$ [g/m²/s]'

    cmap_params = xr.plot.utils._determine_cmap_params(
        turb[varname].values.ravel(), levels=levels, cmap=cmap)

    if extra_filter is not None:
        turb = turb.where(extra_filter)

    for season in np.unique(seasons):
        subset = turb.where(seasons == season)

        for mooring in moorings:
            ctdsubset = ctd[mooring].where(
                ctd[mooring].time.monsoon.labels == season)
            TSplot(ctdsubset.S,
                   ctdsubset['T_S'] if 'T_S' in ctdsubset else ctdsubset['T'],
                   rho_levels=None, hexbin=False, plot_distrib=False,
                   ax=axes[season], size=1, color='lightgray',
                   plot_kwargs={'alpha': 0.1})

        handles, _ = TSplot(subset.S, subset['T'],  color=subset[varname],
                            ax=axes[season], hexbin=True,
                            plot_distrib=False, labels=True,
                            rho_levels=default_density_bins,
                            plot_kwargs=dict(cmap=cmap_params['cmap'],
                                             mincnt=10,
                                             norm=cmap_params['norm'],
                                             gridsize=50,
                                             edgecolors='#333333',
                                             linewidths=0.25,
                                             extent=extent))

        axes[season].text(0.05, 0.9, season, color='k',
                          transform=axes[season].transAxes)

    [aa.set_ylabel('') for aa in [axes['NESW'], axes['SWNE']]]
    [aa.set_xlabel('') for aa in [axes['NE'], axes['SWNE']]]

    cbar_defaults = dict(orientation='horizontal',
                         shrink=0.8, pad=0.0,
                         label=label)
    cbar_defaults.update(cbar_kwargs)

    f.colorbar(handles['ts'], ax=list(axes.values()),
               extend=cmap_params['extend'], **cbar_defaults)

    if extra_filter is not None:
        axes['NE'].text(0.05, 0.1, 'only ' + which_moorings,
                        transform=axes['NE'].transAxes)


def plot_moor(moor, idepth, axx, time_range='2014', events=None):

    axes = dict(zip(['met', 'KT', 'jq', 'N2'], axx[0:4]))
    axes['js'] = axes['jq'].twinx()
    axes['Tz'] = axes['N2'].twinx()
    axes['coverage'] = axes['KT'].twinx()

    if len(moor.met) != 0:
        hmet = (moor.met.τ.resample(time='D').mean()
                .sel(time=time_range).plot(ax=axes['met'], color='k', lw=1.2))
    else:
        hmet = (moor.tropflux.tau
                .sel(time=time_range).plot(ax=axes['met'], color='k', lw=1.2))

    hmet[0].set_clip_on(False)
    hmet[0].set_in_layout(False)

    hkt = (moor.KT.isel(depth=idepth).sel(time=time_range)
           .resample(time='D').mean('time')
           .plot(ax=axes['KT'], _labels=False, lw=1.2, color='k'))

    hjq = (moor.Jq.isel(depth=idepth).sel(time=time_range)
           .resample(time='D').mean('time')
           .plot(ax=axes['jq'], _labels=False, lw=1.2, color='k'))

    hjs = ((moor.Js/1e-2).isel(depth=idepth).sel(time=time_range)
           .resample(time='D').mean('time')
           .plot(ax=axes['js'], _labels=False, lw=1.2, color='C0'))

    htz = ((moor.Tz).isel(depth=idepth).sel(time=time_range)
           .resample(time='D').mean('time')
           .plot(ax=axes['Tz'], _labels=False, lw=1.2, color='C0'))
    # axes['Tz'].set_yscale('symlog', linthreshy=5e-3, linscaley=0.5)
    axes['Tz'].axhline(0, color='C0', zorder=-1, lw=0.5)
    hn2 = ((moor.N2/1e-4).isel(depth=idepth).sel(time=time_range)
           .resample(time='D').mean('time')
           .plot(ax=axes['N2'], _labels=False, lw=1.2, color='k'))
    axes['N2'].set_ylim([0, None])

    fraction = (moor.KT.sel(time=time_range).isel(depth=idepth)
                .groupby(moor.KT.sel(time=time_range).time.dt.floor('D'))
                .count()/144)
    fraction.where(fraction > 0).plot(ax=axes['coverage'], color='C0')

    for hh in [hjq, hjs]:
        hh[0].set_clip_on(False)
        hh[0].set_in_layout(False)

    # axes['js'].set_zorder(-1)
    axes['js'].spines['right'].set_visible(True)

    # labels
    htxt = [axes['met'].text(time, 0.35, season, va='bottom', zorder=-1)
            for time, season
            in zip(['2014-02-01', '2014-03-21', '2014-07-16', '2014-10-20',
                    '2014-12-15'],
                   ['NE', 'NESW', 'SW', 'SWNE', 'NE'])]

    axes['met'].set_xlim(('2014-01', '2015-01'))
    axes['met'].set_ylim([0, 0.35])
    axes['met'].set_ylabel('$τ$ [N/m²]')

    # customize axes
    axes['KT'].set_ylim([1e-7, 1e-1])
    axes['KT'].set_yticks([1e-6, 1e-5, 1e-4, 1e-3])
    axes['KT'].set_yscale('log')
    axes['KT'].set_ylabel('Daily avg. $K_T$ [m²/s]')
    axes['KT'].grid(False, axis='x')
    axes['KT'].grid(True, which='both', axis='y')

    axes['jq'].set_ylabel('$J_q^t$ [W/m²]')
    axes['js'].set_ylabel('$J_s^t$ \n [$10^{-2}$ g/m²/s]')
    [dcpy.plots.set_axes_color(axes[aa], 'C0', 'right')
     for aa in ['js', 'Tz', 'coverage']]

    axes['Tz'].set_ylabel('$T_z$ [°C/m]')
    axes['N2'].set_ylabel('$N²$ \n [$10^{-4}$ $s^{-2}$]')

    axes['coverage'].set_ylabel('Fraction\ndaily coverage')
    axes['coverage'].set_yticks([0, 0.5, 1])
    axes['coverage'].spines['right'].set_visible(True)

    [aa.set_xlabel('') for aa in axx]
    [aa.set_title('') for aa in axes.values()]
    [moor.MarkSeasonsAndEvents(ax=axes[aa], events=events)
     for aa in ['jq', 'KT', 'met', 'N2']]

    # axx[-1].set_xlabel('2014')
    axx[-1].xaxis.set_tick_params(rotation=0)
    [tt.set_ha('center') for tt in axx[-1].get_xticklabels()]
    # axx[-1].xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y-%b'))

    return axes


def mark_seasons(ax=None, zorder=-20):

    dates = dict()

    dates[2013] = {
        'NE': ('2012-Dec-01', '2013-Mar-01'),
        'NESW': ('2013-Mar-01', '2013-May-01'),
        'SW': ('2013-May-01', '2013-Oct-01'),
        'SWNE': ('2013-Oct-01', '2013-Dec-01')
    }

    dates[2014] = {
        'NE': ('2013-Dec-01', '2014-Mar-01'),
        'NESW': ('2014-Mar-01', '2014-May-01'),
        'SW': ('2014-May-01', '2014-Oct-01'),
        'SWNE': ('2014-Oct-01', '2014-Dec-01')
    }

    dates[2015] = {
        'NE': ('2014-Dec-01', '2015-Mar-01'),
        'NESW': ('2015-Mar-01', '2015-Apr-30'),
        'SW': ('2015-Apr-30', '2015-Sep-30'),
        'SWNE': ('2015-Oct-1', '2015-Nov-30')
    }

    if ax is None:
        ax = plt.gca()

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    seasonColor = {
        'NE': 'beige',  # 'beige',
        'NESW': 'lemonchiffon',  # lemonchiffon
        'SW': 'wheat',  # wheat
        'SWNE': 'honeydew'  # honeydew
    }

    for pp in dates:
        for ss in dates[pp]:
            clr = seasonColor[ss]
            xx = pd.to_datetime(dates[pp][ss])
            ax.fill_between(xx, 0, 1,
                            transform=ax.get_xaxis_transform('grid'),
                            facecolor=clr, alpha=0.9,
                            zorder=zorder, edgecolor=None)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
