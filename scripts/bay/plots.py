import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns

import cartopy.crs as ccrs
import xarray as xr
import dcpy
from cartopy import feature

import xfilter

from .bay import default_density_bins, nc_to_binned_df, pods, season_months

# markers = {'RAMA': 'o', 'NRL': '^', 'OMM/WHOI': 'o'}
# colors = {'RAMA': '#0074D9', 'NRL': '#3D9970', 'OMM/WHOI': '#FF4136'}
# colors = {'RAMA': '#1696A3', 'NRL': '#F89B1F', 'OMM/WHOI': '#EA4D5B'}
colors = {'RAMA': '#1696A3', 'NRL': 'C3', 'OMM/WHOI': 'C7'}
# colors = {'RAMA': '#1c666e', 'NRL': '#F89B1F', 'OMM/WHOI': '#EA4D5B'}
# colors = {'RAMA': '#1696A3', 'NRL': '#F89B1F', 'OMM/WHOI': 'gray'}


def plot_coastline(ax=None, facecolor="#FEF9E4", rivers=True, **kwargs):

    if ax is None:
        ax = plt.gca()

    # facecolor = '#b2ccba'
    coastline = feature.NaturalEarthFeature(name='coastline',
                                            category='physical',
                                            scale='50m',
                                            edgecolor='black',
                                            facecolor=facecolor,
                                            **kwargs)

    ax.set_extent([77, 96, 2, 24])
    ax.add_feature(coastline)
    if rivers:
        rivers = feature.NaturalEarthFeature(
            category='physical', name='rivers_lake_centerlines',
            scale='10m', facecolor='none', edgecolor='C0', linewidths=0.5,
            **kwargs)
        ax.add_feature(rivers)


def make_map(pods, DX=0.6, DY=0.7, add_year=True, highlight=[],
             ax=None, topo=True, add_depths=True):

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

        if add_depths:
            text = ''
            if add_year:
                if pod['label'] == 'RAMA' and pod['lat'] == 15:
                    text += 'RAMA 15\n'
                if pod['label'] == 'RAMA' and pod['lat'] == 12:
                    text += 'RAMA 12\n'
                if pod['label'] == 'OMM/WHOI':
                    text += '2014-15\n'
                if 'NRL' in name:
                    text += name + '\n'

            for z in pod['depths']:
                text += z
                if pod['label'] == 'RAMA' and add_year:
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


def trim_horiz_violin(hdl, keep='top'):

    # halve the voilin
    for b in hdl['bodies']:
        m = np.mean(b.get_paths()[0].vertices[:, 1])

        if keep == 'top':
            lower = -np.inf
            upper = m
        elif keep == 'bottom':
            lower = m
            upper = np.inf

        b.get_paths()[0].vertices[:, 1] = np.clip(
            (b.get_paths()[0].vertices[:, 1]), lower, upper)


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


def plot_distrib(ax, plotkind, var, zloc, zstd, width=12, percentile=False,
                 color=None, trim_keep='top', markers=None, overlay=False):
    ''' Actually plot the distribution '''

    base_marker_size = 5

    if markers is None:
        markers = {'median': '^', 'mean': 'o'}
    else:
        assert(('median' in markers) and ('mean' in markers))

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

        hdl = ax.violinplot(var[~np.isnan(var)],
                            positions=[zloc],
                            widths=[width],
                            vert=False,
                            showmedians=False)

        for pc in hdl['bodies']:
            pc.set_alpha(1)

        trim_horiz_violin(hdl, keep=trim_keep)
        hdl['cmaxes'].set_visible(False)

        if color is not None:
            for pc in hdl['bodies']:
                pc.set_color(color)
                pc.set_edgecolor('w')

            hdl['cbars'].set_color(color)
            hdl['cmins'].set_color(color)
            hdl['cmins'].set_zorder(20)

        if trim_keep == 'bottom':
            hdl['cbars'].set_visible(False)
            hdl['cmins'].set_visible(False)
        else:
            hdl['cbars'].set_zorder(hdl['bodies'][0].get_zorder())
            hdl['cmins'].get_paths()[0].vertices[:, 1] = (
                zloc + np.array([-1, 1]) * zstd)

        if percentile:
            # use percentile instead of max
            hdl['cbars'].get_paths()[0].vertices[1, 0] = prc

    elif plotkind is 'hist':
        hdl = ax.hist(var, bottom=zloc, density=True)

    # color = _get_color_from_hdl(hdl)

    # if trim_keep == 'bottom':
    #     zloc += 2.5
    # else:
    #     zloc -= 2.5

    dz = 2.1 if overlay else 0

    if not overlay:
        whitez = 11
        colorz = 12
        white_size = base_marker_size + 1
        color_size = base_marker_size - 1
    else:
        whitez = 12
        colorz = 12
        white_size = base_marker_size
        color_size = base_marker_size + 2

    ax.plot(median, zloc+dz, markers['median'],
            color=color, zorder=colorz, ms=color_size)

    ax.plot(median, zloc+dz, markers['median'],
            color='w', zorder=whitez, ms=white_size,
            fillstyle='full' if overlay else None)

    ax.plot(mean, zloc+dz, markers['mean'],
            color=color, zorder=colorz, ms=color_size)

    ax.plot(mean, zloc+dz, markers['mean'],
            color='w', zorder=whitez, ms=white_size,
            fillstyle='full' if overlay else 'none')

    return hdl, median, mean


def get_pal_dist(bins):

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

    return pal_dist


def vert_distrib(df, bins, varname='KT', kind='distribution',
                 pal=None, f=None, ax=None,
                 label_moorings=True, label_bins=True, adjust_fig=True,
                 width=12, percentile=False, add_offset=True, trim_keep='top',
                 **kwargs):
    '''
        Function to make vertical distribution plot when provided with
        appropriately formatted DataFrame.
    '''

    plotkind = 'violin'

    if pal is None:
        pal = get_pal_dist(bins)

    if f is None:
        f, axx = plt.subplots(1, 4, sharex=True, sharey=True)
    else:
        axx = ax

    ax = {'NE': axx[0], 'NESW': axx[1], 'SW': axx[2], 'SWNE': axx[3]}

    if 'ML' in df.bin.cat.categories:
        df.bin.cat.reorder_categories(
            ['ML', 'BL'] + list(df.bin.cat.categories[:-2]),
            inplace=True)

    if varname == 'KT':
        title = '$\\log_{10}$ hourly averaged $K_T$ [m²/s]'
        xlim = kwargs.pop('xlim', [-7, -1])
        xlines = kwargs.pop('xlines', [-6, -5, -4, -3])
    else:
        title = varname
        xlim = kwargs.pop('xlim', None)
        xlines = kwargs.pop('xlines', [])

    markers = kwargs.pop('markers', None)
    overlay = kwargs.pop('overlay', False)
    months = {'NE': 'Dec-Feb', 'NESW': 'Mar-Apr',
              'SW': 'May-Sep', 'SWNE': 'Oct-Nov'}

    coverage_levels = np.array([0.3, 0.75, 1.5, 2])
    coverage_pal = sns.cubehelix_palette(len(coverage_levels), as_cmap=True)

    # I'm adding sorted gradient estimates
    if trim_keep == 'bottom':
        df = df.where((df.bin == 'ML') | (df.bin == 'BL') |
                      (df.bin == df.bin.cat.categories[2]))

    for seas in ax:
        aa = ax[seas]
        # aa.set_prop_cycle(cycler.cycler('color', pal))
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
    for index, (label, ddff) in enumerate(df.groupby([bins, 'season'])):
        interval = label[0]
        season = label[1]
        zloc = sp.stats.trim_mean(ddff.z, 0.1)

        bin_name = (np.round(interval.mid-1000, 1).astype('str')
                    if isinstance(interval, pd.Interval)
                    else interval)

        if add_offset:
            if type(interval) != str:
                #if interval.mid < 1020:
                #    if season == 'SWNE' or season == 'NE':
                #         zloc += 5
                if bin_name == "23.2" and season == "SWNE":
                    zloc += 7.5

                if bin_name == "23.2" and season == "SW":
                    zloc -= 2.5

                if bin_name == "22.8" and season == "SWNE":
                    zloc += 2.5

            elif interval == 'BL':
                zloc -= 6
            elif interval == 'ML':
                zloc -= 10


        var = ddff[varname]
        if varname == 'ρ':
            var -= 1000

        if kind == 'distribution':
            for mm in ddff['moor'].unique():
                if (len(ddff['moor'].where(ddff.moor == mm).dropna())/len(ddff)
                    <= 0.10):
                    ddff = ddff.loc[ddff['moor'] != mm]

            fraction_coverage = var.count() / (30 * 24 * season_months[season])

            # at least 1/3 season covered by 1 χpod
            if fraction_coverage < 0.3:
                continue

            color = coverage_pal(fraction_coverage / coverage_levels.max())

            hdl, median, mean = plot_distrib(ax[season], plotkind, var,
                                             zloc, ddff.z.std(), width,
                                             percentile, color,
                                             trim_keep=trim_keep,
                                             markers=markers,
                                             overlay=overlay)

            zvec[season].append(zloc)
            mdnvec[season].append(median)
            meanvec[season].append(mean)

            xtxt = hdl['cbars'].get_paths()[0].vertices[1, 0]
            #if xtxt > -1:
            #    xtxt = -2
            color = _get_color_from_hdl(hdl)

        elif kind == 'mean_ci_profile':
            mean = np.log10(ddff[varname].values)
            err = np.abs(mean - np.log10(ddff.ci.values[0]))[:, np.newaxis]
            hdl = ax[season].errorbar(mean, zloc,
                                      xerr=err, fmt='^')

            xtxt = hdl.lines[-1][0].get_segments()[0][1, 0]
            color = hdl.lines[0].get_color()

        ytxt = zloc

        if label_bins:
            ax[season].text(xtxt, ytxt, ' '+bin_name,
                            color=color, ha='left', va='center',
                            fontsize=7)
        if label_moorings:
            ax[season].text(-0.75, ytxt,
                            format_moor_names(ddff['moor'].unique()),
                            color=color, ha='left',
                            va='center', fontsize=6)

    ax['NE'].set_ylabel('depth [m]')
    ax['NE'].set_ylim([120, 0])

    axins = ax['NE'].inset_axes([-6, 110, 5, 5],
                                zorder=100,
                                transform=ax['NE'].transData)
    # axins = f.add_axes([0.1, 0.1, 0.7, 0.1], facecolor='lightgray', zorder=5)
    axins.pcolormesh(
        np.array(coverage_levels).reshape(1, len(coverage_levels)),
        cmap=coverage_pal, edgecolors='w', linewidths=2)
    axins.set_clip_on(False)
    axins.set_in_layout(False)
    axins.set_yticks([])
    axins.set_yticklabels([])
    axins.spines['left'].set_visible(False)
    axins.spines['bottom'].set_visible(False)
    axins.set_xticklabels(coverage_levels.astype('str'))
    axins.set_xticks(np.arange(len(coverage_levels)))
    axins.tick_params(length=0)
    # axins.set_xlabel('Coverage\n[instrument-seasons]')

    # ax['NE'].set_yticks(np.arange(120,0,-20))
    # ax['NE'].set_yticklabels(np.arange(120,0,-10).astype('str'))

    for season in ax:
        # args = {'linestyle': '--', 'color': 'dimgray'}
        # idx = np.argsort(zvec)
        # ax[season].plot(np.array(mdnvec[season])[idx],
        #                 np.array(zvec[season])[idx], **args)
        # ax[season].plot(np.array(meanvec[season])[idx],
        #                 np.array(zvec[season])[idx], **args)
        for xx in xlines:
            ax[season].axvline(xx, lw=1, ls='-', zorder=2, color='w')

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
    # lons = [85.5, 85.5, 87, 88.5, 90, 90]
    # lats = [5, 8, 8, 8, 12, 15]
    # names = ['1', '3', '4', '5', '12', '15']

    lons = [85.5, 87, 88.5]
    lats = [8, 8, 8]
    names = ['3', '4', '5']

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


def mark_moors_clean(ax, markersize=4.5):
    mark_moors(ax=ax, markersize=markersize, fontsize=6, color='k',
               colortext='w', labels=False)
    mark_moors(ax=ax, markersize=markersize-1.5, fontsize=6, color='w',
               colortext='w', labels=False)


def make_vert_distrib_plot(varname,
                           dataset='../estimates/bay_merged_sorted_hourly.nc',
                           bins=default_density_bins,
                           moor=None,
                           label_moorings=False, **kwargs):

    ''' user-friendly wrapper function to make vertical distribution plot. '''

    df = nc_to_binned_df(dataset=dataset, bins=bins, moor=moor)

    if varname == 'KT':
        df['KT'] = np.log10(df['KT'])

    vert_distrib(df, df.bin,
                 label_moorings=label_moorings,
                 percentile=True, **kwargs)


def make_labeled_map(ax=None, add_depths=True, pods=pods):

    ax, colors = make_map(pods, add_year=True, add_depths=add_depths,
                          highlight=['RAMA', 'NRL', 'OMM/WHOI'], ax=ax)

    # ax.text(87, 6.8, 'EBoB\n[2014]',
    #         color=colors['NRL'], ha='center', va='center')
    # ax.text(90-0.35, 18, 'OMM/WHOI',
    #         color=colors['OMM/WHOI'], ha='right', va='center')
    # ax.text(90, 13.5, 'RAMA',
    #         color=colors['RAMA'], ha='left', va='center')
    ax.set_xticks([78, 80, 82, 84, 85.5, 87, 88.5, 90, 92, 94, 96])
    ax.set_yticks([2, 4, 5, 8, 12, 15, 18, 20, 22, 24])


def mark_range(ax, top, middle, bottom, color=colors['NRL']):
    ax.fill_between([0, 1], bottom, top,
                    facecolor=color, alpha=0.1, zorder=-5,
                    transform=ax.get_yaxis_transform('grid'))
    dcpy.plots.liney(middle, ax=ax,
                     color=colors['NRL'], linestyle='-')

def mark_χpod_depths_on_clim(ax=[], orientation='horizontal'):

    argo = dcpy.oceans.read_argo_clim()

    def plot_profiles(ax, lon, lat, color):

        region = {'lat': lat, 'lon': lon, 'method': 'nearest'}

        Tmean = argo.Tmean.sel(**region).sel(pres=slice(0, 150)).squeeze()
        Smean = argo.Smean.sel(**region).sel(pres=slice(0, 150)).squeeze()

        ax2 = ax.twiny()

        Tmean.plot(y='pres', ax=ax, color=color, lw=2, _labels=False)
        Smean.plot(y='pres', ax=ax2, ls='--', color=color, lw=2, _labels=False)

        ax2.spines['top'].set_visible(True)
        Slim = np.array([32, 35])
        ax2.set_xlim(Slim)
        ax2.spines['right'].set_visible(True)
        ax.set_xlim(18 + np.array([0, np.diff(Slim) * 7.6e-4/1.7e-4]))
        ax.set_ylabel('Pressure')
        # ax.text(25.5, 114, 'Argo clim.', color='black',
        #         ha='center', fontsize=12)

        return ax2

    if len(ax) == 0:
        if orientation == 'horizontal':
            _, ax = plt.subplots(1, 2, sharey=True, constrained_layout=True)

        elif orientation == 'vertical':
            _, ax = plt.subplots(2, 1, sharey=True, constrained_layout=True)

    plot_profiles(ax[0], 90, 12, color=colors['RAMA'])
    ax[0].text(29.5, 10, 'T', color=colors['RAMA'])
    ax[0].text(20, 10, 'S', color=colors['RAMA'])

    plot_profiles(ax[1], 85.5, 5, color=colors['NRL'])
    ax[1].text(29.5, 10, 'T', color=colors['NRL'])
    ax[1].text(26.5, 10, 'S', color=colors['NRL'])

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


def plot_rama(moor, idepth, axx, time_range='2014', events=None):

    axes = dict(zip(['met', 'KT', 'jq', 'js', 'coverage', 'N2'], axx[0:6]))
    # axes['js'] = axes['jq'].twinx()
    axes['Tz'] = axes['N2']
    # axes['coverage'] = axes['KT'].twinx()

    if len(moor.met) != 0:
        hmet = (moor.met.τ.resample(time='D').mean()
                .sel(time=time_range).plot(ax=axes['met'], color='k', lw=1.2))
    else:
        hmet = (moor.tropflux.tau
                .sel(time=time_range).plot(ax=axes['met'], color='k', lw=1.2))

    hkt = (moor.KT.isel(depth=idepth).sel(time=time_range)
           .resample(time='D').mean('time')
           .plot(ax=axes['KT'], _labels=False, lw=1.2, color='k'))
    dcpy.plots.annotate_end(hkt[0], 'sorted')

    jq = moor.Jq.isel(depth=idepth).sel(time=time_range)
    t = jq.time.resample(time='D').mean('time').values
    hjq = axes['jq'].bar(t,
                         jq.where(jq > 0).resample(time='D').mean('time'),
                         color='r')
    hjq = axes['jq'].bar(t,
                         jq.where(jq < 0).resample(time='D').mean('time'),
                         color='C0')

    hjs = ((moor.Js/1e-2).isel(depth=idepth).sel(time=time_range)
           .resample(time='D').mean('time')
           .plot(ax=axes['js'], _labels=False, lw=1.2, color='k'))

    htz = ((9.81 * 1.7e-4 * moor.Tz/1e-4)
           .isel(depth=idepth).sel(time=time_range)
           .resample(time='D').mean('time')
           .plot(ax=axes['N2'], _labels=False, lw=1.2, color='k'))
    # axes['Tz'].set_yscale('symlog', linthreshy=5e-3, linscaley=0.5)

    hn2 = ((moor.N2/1e-4)
           .isel(depth=idepth).sel(time=time_range)
           .resample(time='D').mean('time')
           .plot(ax=axes['N2'], _labels=False, lw=1.2, color='gray'))
    # axes['N2'].set_ylim([-5, 5])

    fraction = (moor.KT.sel(time=time_range).isel(depth=idepth)
                .groupby(moor.KT.sel(time=time_range).time.dt.floor('D'))
                .count()/144)
    fraction.where(fraction > 0).plot(ax=axes['coverage'], color='k')

    for hh in [hmet, htz, hn2, hjq, hjs]:
        hh[0].set_clip_on(False)
        hh[0].set_in_layout(False)

    # dcpy.plots.annotate_end(htz[0], '$gαT_z$', va='top')
    # dcpy.plots.annotate_end(hn2[0], '$N^2$', va='bottom')

    # axes['js'].set_zorder(-1)
    # axes['js'].spines['right'].set_visible(True)

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
    # [dcpy.plots.set_axes_color(axes[aa], 'C0', 'right')
     # for aa in ['js', 'coverage']]

    # axes['Tz'].set_ylabel('$gαT_z$ [s^{-2}]')
    axes['N2'].set_ylabel('[$10^{-4}$ $s^{-2}$]')

    axes['coverage'].set_ylabel('Fraction\ndaily coverage')
    axes['coverage'].set_yticks([0, 0.5, 1])
    axes['coverage'].spines['right'].set_visible(True)

    [aa.set_xlabel('') for aa in axx]
    [aa.set_title('') for aa in axes.values()]
    [moor.MarkSeasonsAndEvents(ax=axes[aa], events=events)
     for aa in ['jq', 'KT', 'met', 'N2', 'js', 'coverage']]

    axx[-1].set_xlabel('2014')
    axx[-1].xaxis.set_tick_params(rotation=0)
    [tt.set_ha('center') for tt in axx[-1].get_xticklabels()]
    axx[-1].xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y-%b'))

    return axes


def plot_moor_old(moor, idepth, axx, time_range='2014', events=None):

    axes = dict(zip(['met', 'KT', 'jq', 'N2'], axx[0:4]))
    axes['js'] = axes['jq'].twinx()
    axes['Tz'] = axes['N2']
    axes['coverage'] = axes['KT'].twinx()

    if len(moor.met) != 0:
        hmet = (moor.met.τ.resample(time='D').mean()
                .sel(time=time_range).plot(ax=axes['met'], color='k', lw=1.2))
    else:
        hmet = (moor.tropflux.tau
                .sel(time=time_range).plot(ax=axes['met'], color='k', lw=1.2))

    hkt = (moor.KT.isel(depth=idepth).sel(time=time_range)
           .resample(time='D').mean('time')
           .plot(ax=axes['KT'], _labels=False, lw=1.2, color='k'))

    hjq = (moor.Jq.isel(depth=idepth).sel(time=time_range)
           .resample(time='D').mean('time')
           .plot(ax=axes['jq'], _labels=False, lw=1.2, color='k'))

    hjs = ((moor.Js/1e-2).isel(depth=idepth).sel(time=time_range)
           .resample(time='D').mean('time')
           .plot(ax=axes['js'], _labels=False, lw=1.2, color='C0'))

    htz = ((9.81 * 1.7e-4 * moor.Tz/1e-4)
           .isel(depth=idepth).sel(time=time_range)
           .resample(time='D').mean('time')
           .plot(ax=axes['N2'], _labels=False, lw=1.2, color='C0'))
    # axes['Tz'].set_yscale('symlog', linthreshy=5e-3, linscaley=0.5)
    axes['Tz'].axhline(0, color='C0', zorder=-1, lw=0.5)

    hn2 = ((moor.N2/1e-4)
           .isel(depth=idepth).sel(time=time_range)
           .resample(time='D').mean('time')
           .plot(ax=axes['N2'], _labels=False, lw=1.2, color='k'))
    # axes['N2'].set_ylim([-5, 5])

    fraction = (moor.KT.sel(time=time_range).isel(depth=idepth)
                .groupby(moor.KT.sel(time=time_range).time.dt.floor('D'))
                .count()/144)
    fraction.where(fraction > 0).plot(ax=axes['coverage'], color='C0')

    for hh in [hmet, htz, hn2, hjq, hjs]:
        hh[0].set_clip_on(False)
        hh[0].set_in_layout(False)

    dcpy.plots.annotate_end(htz[0], '$gαT_z$', va='top')
    dcpy.plots.annotate_end(hn2[0], '$N^2$', va='bottom')

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
    axes['js'].set_ylabel('$J_s^t$ \n [$10^{-1}$ [psu] kg/m²/s]')
    [dcpy.plots.set_axes_color(axes[aa], 'C0', 'right')
     for aa in ['js', 'coverage']]

    # axes['Tz'].set_ylabel('$gαT_z$ [s^{-2}]')
    axes['N2'].set_ylabel('[$10^{-4}$ $s^{-2}$]')

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


def plot_nrl(mooring):

    from dcpy.plots import annotate_end, set_axes_color

    shear, low_shear, _, niw_shear, _, fm24 = mooring.filter_interp_shear(
        'filter_then_sample', wkb_scale=False, maxgap_time=12)
    niw_shear += fm24.interp(time=niw_shear.time, depth=niw_shear.depth)
    residual = shear.shear - niw_shear - low_shear

    N2 = (xfilter.lowpass(mooring.N2.isel(depth=1),
                           coord='time',
                           freq=1/30,
                           cycles_per='D',
                           num_discard=0)
                 .interp(time=niw_shear.time).interpolate_na("time"))

    f5, axx5 = plt.subplots(6, 1, sharex=True, constrained_layout=True)
    f5.set_constrained_layout_pads(hspace=0.001, h_pad=0)
    f5.set_size_inches((8, 8))

    hniw = ((niw_shear.rolling(time=7*24, center=True).reduce(dcpy.util.ms) / N2)
            .sel(time='2014')
            .plot(ax=axx5[-2], _labels=False, color='g', lw=1.5))
    hlow = ((low_shear.rolling(time=7*24, center=True).reduce(dcpy.util.ms) / N2)
            .sel(time='2014')
            .plot(ax=axx5[-2], _labels=False, color='k', lw=1.5))
    hres = ((residual.rolling(time=7*24, center=True).reduce(dcpy.util.ms) / N2)
            .sel(time='2014')
            .plot(ax=axx5[-2], _labels=False, color='C1', lw=1.5))
    annotate_end(hlow[0], '$S²_{low}$', va='top')
    annotate_end(hniw[0], '$S²_{niw+}$', va='bottom')
    annotate_end(hres[0], '$S²_{res}$', va='center')

    # trgrid = axx5[-2].get_xaxis_transform('grid')
    # axx5[-2].text('2014-08-12', 0.8, 'near-inertial',
    #               color=hniw[0].get_color(), transform=trgrid)
    # axx5[-2].text('2014-10-06', 0.8, 'low-frequency',
    #               color=hlow[0].get_color(), transform=trgrid)
    # axx5[-2].text('2014-12-02', 0.8, 'residual',
    #               color=hres[0].get_color(), transform=trgrid)
    hniw[0].set_clip_on(False)
    hniw[0].set_in_layout(False)
    hlow[0].set_clip_on(False)
    hlow[0].set_in_layout(False)
    mooring.MarkSeasonsAndEvents(events='Storm-zoomin', ax=axx5[-2])
    axx5[-2].set_ylabel('$S²/N²$')

    axmooring = plot_moor_old(mooring, idepth=1,
                              axx=axx5, events='Storm-zoomin')

    # fill in the 20m gap with linear interpolation
    # then interpolate velocity to CTD depths
    # then difference to get shear
    # zinterp = mooring.ctd.depth.isel(z=slice(1, 3))
    # vel_interp = (mooring.vel[['u', 'v']].interpolate_na('depth')
    #               .interp(time=zinterp.time, depth=zinterp.drop('depth')))
    # shear_interp = (np.hypot(vel_interp.u.diff('z')/15,
    #                          vel_interp.v.diff('z')/15)
    #                 .squeeze())

    # N2 = ((9.81/1025 * mooring.ctd.ρ.diff('z')/mooring.ctd.depth.diff('z'))
    #       .isel(z=1))
    # # Ri = (N2.where(N2 > 0)/shear_interp**2).sel(time='2014')

    # axmooring['ri'] = axx5[-1]
    # ((Ri.where(Ri < 5).resample(time='D').count()/144)
    #  .plot(ax=axmooring['ri'], label='< 10', _labels=False, color='k'))
    # axmooring['ri'].set_ylabel('Fraction of day\nwith Ri < 5')

    axmooring['depth'] = axx5[-1] # axmooring['ri'].twinx()
    (mooring.zχpod.sel(num=1).resample(time='D').mean('time')
     .plot(ax=axmooring['depth'], _labels=False, color='k', lw=1.2, yincrease=False))
    axmooring['depth'].set_ylabel('$χ$pod depth [m]')
    mooring.MarkSeasonsAndEvents(events='Storm-zoomin', ax=axx5[-1])
    # set_axes_color(axmooring['depth'], 'C0', spine='right')

    axmooring['input'] = axmooring['met'].twinx()
    axmooring['input'].plot(mooring.niw.time, mooring.niw.true_flux*1000,
                            color='C0')
    axmooring['input'].set_ylabel('Near-inertial input\n$\Pi$[mW/m²]')
    set_axes_color(axmooring['input'], 'C0', spine='right')

    dcpy.plots.label_subplots(axx5, x=0.025, y=0.83)

    [tt.set_rotation(0) for tt in axx5[-1].get_xticklabels()]
    [tt.set_ha('center') for tt in axx5[-1].get_xticklabels()]

    if mooring.name == 'NRL3':
        axx5[-2].set_ylim([0, 40])

    if mooring.name == 'NRL4':
        axx5[-2].set_ylim([0, 25])

    if mooring.name == 'NRL5':
        axmooring['depth'].set_ylim([60, 140])
        #axx5[-2].set_ylim([0, 10])
        axmooring['jq'].set_ylim([-20, 0])
        axmooring['js'].set_ylim([0, 0.5])

    return f5, axmooring
