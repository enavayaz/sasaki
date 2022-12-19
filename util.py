import matplotlib
from matplotlib.lines import Line2D
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import matplotlib.pylab as plt
from matplotlib import ticker
import matplotlib.pyplot as pylt
import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.geometry.discrete_curves import SRVMetric, Euclidean, R2, DiscreteCurves
import numpy as np
import pandas as pd
from math import *
from morphomatics.geom import BezierSpline
from morphomatics.manifold import Sphere, CubicBezierfold
from PIL import Image
# import pillow
try:
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.basemap import Basemap
except: print('mpl_toolkits not available')
from matplotlib import cm as cm
from scipy.interpolate import CubicSpline
import matplotlib.patches as mpatches
colors = {0: "b", 1: "orange", 2: "r"}
group_0 = mpatches.Patch(color=colors[0], label="Cat 0")
group_1 = mpatches.Patch(color=colors[1], label="Cat 1-3")
group_2 = mpatches.Patch(color=colors[2], label="Cat 4-5")
legend_handle = [group_0, group_1, group_2]
cmap_cat = cm.get_cmap('jet')
cnorm_cat = cm.colors.Normalize(vmin=20, vmax=137)
lbl_type = {'TD': 0, 'TS': 1, 'HU': 2, 'EX': 3, 'SD': 4, 'SS': 5, 'LO': 6, 'WV': 7, 'DB': 8}
subj_dict = {'Cyclone Nr': 0, 'Name': 1, 'n_sample': 2}
seq_dict = {'Date': 0, 'Time': 1, 'StatusType': 2, 'coord_2D': [3, 4], 'MaxWind': 5, 'MinPressure': 6, 'coord_3D': [7, 8, 9]}
N_SUBJ, N_SAMPLES = 218, 32  #N_SAMPLES:average = 32, Max = 96
STR_MAXWIND = 'Maximum sustained wind in knots'
# %matplotlib inline
# import imageio
# matplotlib.use("Agg")  # NOQA


def initial_mean(pu, metric):
    """
    Initialize mean geodesic
    """
    # compute mean of base points
    mean_p = FrechetMean(metric)
    mean_p.fit(pu[:, 0])
    # compute mean of tangent vectors
    PT = lambda p, u: metric.parallel_transport(u, p, end_point=mean_p.estimate_)
    mean_v = gs.mean([PT(*pu_i) for pu_i in pu], 0)
    return gs.array([mean_p.estimate_, mean_v])


def visSphere(points_list, color_list, size=15):
    """
    Visualize groups of points on the 2D-sphere
    """
    # label_list = ['Random geodesic', 'Mean geodesic']
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(111, projection="3d")
    sphere = visualization.Sphere()
    sphere.draw(ax, marker=".")
    for i in range(len(points_list)):
        for points in points_list[i]:
            points = gs.to_ndarray(points, to_ndim=2)
            sphere.draw_points(ax, points=points, color=color_list[i], marker=".")
    # ax.set_title("")
    plt.show(block=False)


def visKen(points_list, color_list, marker_list='.', size=11):
    """
    Visualize landmarks
    """
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(111)
    points = gs.to_ndarray(points_list, to_ndim=2)
    points = points.reshape(-1, 2)
    plt_plot, = plt.plot(points[:, 0], points[:, 1], 'o', markersize=5, markerfacecolor=color_list[0],
                         markeredgewidth=.5, markeredgecolor='k')

    plt.text(0.38, 0.25, 'Brg')
    plt.text(-0.02, 0.25, 'Lara')
    plt.text(-0.22, 0.2, 'IPP')
    plt.text(-0.3, 0.0, 'Opi')
    plt.text(-0.25, -0.15, 'Bas')
    plt.text(-0.05, -0.15, 'SOS')
    plt.text(0.2, -0.11, 'ISS')
    plt.text(0.45, -0.07, 'SES')

    ax.set_title(f'Rat skull measurements: 8 landmarks at 8 time points per subject (Subject=0 plots all data)')

    def upd(Subject):
        subj = Subject - 1
        if subj >= 0:
            points = gs.to_ndarray(points_list[int(subj * 8):int((subj + 1) * 8)], to_ndim=2)
        else:
            points = gs.to_ndarray(points_list, to_ndim=2)
        points = points.reshape(-1, 2)
        plt_plot.set_xdata(points[:, 0])
        plt_plot.set_ydata(points[:, 1])
        fig.canvas.draw_idle()

    #

    interact(upd, Subject=widgets.IntSlider(min=0, max=18, step=1, value=0))


def visKenGeo(geo_list, mean, size=10):
    """
    Visualize landmark-components of geodesics and their mean geodesic
    """
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(111)
    k_landmarks = geo_list.shape[2]
    for geo in geo_list:
        # points = gs.to_ndarray(points, to_ndim=2)
        p, q = geo[0], geo[0] + geo[1]
        for j in range(k_landmarks):
            plt.plot([p[j, 0], q[j, 0]], [p[j, 1], q[j, 1]], color='#008c04', linewidth=1.5, alpha=0.8)

    p, q = mean[0], mean[0] + mean[1]
    for j in range(k_landmarks):
        plt.plot([p[j, 0], q[j, 0]], [p[j, 1], q[j, 1]], color='k', linewidth=2.5)

    plt.text(0.38, 0.25, 'Brg')
    plt.text(-0.02, 0.25, 'Lara')
    plt.text(-0.22, 0.2, 'IPP')
    plt.text(-0.3, 0.0, 'Opi')
    plt.text(-0.25, -0.15, 'Bas')
    plt.text(-0.05, -0.15, 'SOS')
    plt.text(0.2, -0.11, 'ISS')
    plt.text(0.45, -0.07, 'SES')

    ax.set_title("Mean geodesic and individual geodesics")
    indivGeo = Line2D([0], [0], label='Individual geodesics', color='#008c04')
    meanLGeo = Line2D([0], [0], label='Mean geodesic', color='k')
    plt.legend(handles=[indivGeo, meanLGeo], loc='center')

    plt.show(block=False)


def visTPCA(explained_variance, scores, lbl=None, dim3=False, size=10, legend_handles=None, colors=None, txt=True, centroids=None):
    """
    Visualize explained variance and distribution of tPCA scores of input shapes
    """
    rel_cum_var = np.cumsum(explained_variance)
    tot_var = rel_cum_var[-1]
    rel_cum_var = 100.0 * rel_cum_var / tot_var

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2 * size, size))
    plt.rcParams['font.size'] = 14
    fig.suptitle('Tangent principal component analysis')
    ax1.set(ylabel='Relative cumulative variance [%]')
    ax1.set_title('tPCA relative cumulative variance')
    ax1.bar([f'{i + 1}' for i in range(len(rel_cum_var))], rel_cum_var, color='grey')

    ax2.set(xlabel='1st axis of variation', ylabel='2nd axis of variation')
    ax2.set_title('tPCA scores')
    if lbl is None:
        ax2.plot(scores[:, 0], scores[:, 1], 'o', markersize=10, markerfacecolor='#008c04', markeredgewidth=.5,
             markeredgecolor='k')
    else:
        #colors = ['blue', 'red']
        if dim3:
            ax2 = fig.add_subplot(111, projection='3d')
            if colors is None:
                ax2.scatter(scores[:, 0], scores[:, 1], scores[:, 2], c=lbl, cmap=cmap_cat)
            else:
                ax2.scatter(scores[:, 0], scores[:, 1], scores[:, 2], c=lbl, s=20, alpha=0.8)
        else:
            ax2.scatter(scores[:, 0], scores[:, 1], c=lbl, cmap=cmap_cat)
            if centroids is not None:
                lc = np.array([colors[i] for i in range(len(centroids))])
                ax2.scatter(centroids[:, 0], centroids[:, 1], c=lc, edgecolor='k', s=160, linewidths=3, cmap=cmap_cat)
    if txt is True:
        for i, txt in enumerate(range(1, len(scores) + 1)):
            ax2.annotate(txt, (scores[i, 0] + 0.00075, scores[i, 1] + 0.00075))
    if legend_handles is not None:
        plt.legend(handles=legend_handles)
    #ax2.legened(handles=legend_handles)
    plt.show(block=False)


def load_data():
    return np.load('datasets/rat_skulls.npy')


def load_data_hur():
    # path = 'hur.csv'
    return pd.read_csv('datasets/hur.csv', header=None)

def load_splines():
    S2 = Sphere()
    cBfS2 = CubicBezierfold(S2, 1)
    CP_file = np.load('datasets/splines.npz')

    splines_S2 = []
    max_wind_spline_coefficients = []
    for P, c in zip(CP_file['cubic_CP_S2'], CP_file['coeff_wind']):
        splines_S2.append(cBfS2.from_velocity_representation(P))
        max_wind_spline_coefficients.append(c)

    return splines_S2, max_wind_spline_coefficients

# Earth Science
def coord_2D3D(lat, lon, h=0.0):
    """
    this function converts latitude,longitude and height above sea level
    to earthcentered xyx coordinates in wgs84, lat and lon in decimal degrees
    e.g. 52.724156(West and South are negative), heigth in meters
    for algoritm see https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_geodetic_to_ECEF_coordinates
    for values of a and b see https://en.wikipedia.org/wiki/Earth_radius#Radius_of_curvature
    """
    #a = 1  # 6378137.0             #radius a of earth in meters cfr WGS84
    #b = 1  # 6356752.3             #radius b of earth in meters cfr WGS84
    #e2 = 1 - (b ** 2 / a ** 2)
    latr = np.pi*lat/180  # latitude in radians
    lonr = np.pi*lon/180  # longituede in radians
    #Nphi = a / sqrt(1 - e2 * sin(latr) ** 2)
    x = np.cos(latr) * np.cos(lonr)  # (Nphi + h) * cos(latr) * cos(lonr)
    y = np.cos(latr) * np.sin(lonr)  # (Nphi + h) * cos(latr) * sin(lonr)
    z = np.sin(latr)  # (b ** 2 / a ** 2 * Nphi + h) * sin(latr)
    return x, y, z


def coord_3D2D(xyz):
    x, y, z = xyz[0], xyz[1], xyz[2]
    lat = np.sign(z)*180*np.arctan(z/sqrt(x**2 + y**2))/np.pi
    lon = 180*np.arctan2(y, x)/np.pi # West is negative
    return lat, lon


def visEarth(seq_list, cat_list, title=None):
    fig = plt.figure(figsize=(12, 12))
    plt.rcParams['font.size'] = 14
    # set perspective angle
    #lat_viewing_angle,  lon_viewing_angle = 10, -45

    # define color maps for water and land
    ocean_map = (plt.get_cmap('ocean'))(210)
    cmap = plt.get_cmap('gist_earth')

    # call the basemap and use orthographic projection at viewing angle
    m = Basemap(projection='ortho', lat_0=10, lon_0=-45)
    #m = Basemap(projection='lcc', lon_0=-60, lat_0=20, lat_1=45, lat_2=55, width=1.2E7, height=1.0E7)  # conic

    # coastlines, map boundary, fill continents/water, fill ocean, draw countries
    m.drawcoastlines()
    m.drawmapboundary(fill_color=ocean_map)
    m.fillcontinents(color=cmap(200), lake_color=ocean_map)
    m.drawcountries()

    # latitude/longitude line vectors
    lat_line_range, lat_lines = [-90, 90], 8
    lat_line_count = (lat_line_range[1] - lat_line_range[0]) / lat_lines
    merid_range, merid_lines = [-180, 180], 8
    merid_count = (merid_range[1] - merid_range[0]) / merid_lines
    m.drawparallels(np.arange(lat_line_range[0], lat_line_range[1], lat_line_count))
    m.drawmeridians(np.arange(merid_range[0], merid_range[1], merid_count))

    # add points
    # x0, y0 = m(2, 41)
    # m.plot(x0, y0, marker='D', color='r')
    # plt.annotate('Barcelona', xy=(x0, y0), xycoords='data', xytext=(-90, 10), textcoords='offset points',
    #             color='r', arrowprops=dict(arrowstyle="->", color='g'))
    #x0, y0 = m(0, 0)  # origin (lat = lon = 0)
    #m.plot(x0, y0, marker='.', color='b')
    #plt.annotate('origin', xy=(x0, y0))
    if title is not None:
        plt.title(title)
    latlons = []
    if seq_list[0].shape[-1] == 3:
        for s in seq_list:
            latlon = np.zeros((s.shape[0], 2))
            for j in range(s.shape[0]):
                latlon[j, 0], latlon[j, 1] = coord_3D2D(s[j])
            latlons.append(latlon)
    else:
        latlons = seq_list
    for i in range(len(seq_list)):
        lats, lons, cat = latlons[i][:, 0], latlons[i][:, 1], cat_list[i]
        #lats, lons = seq[:, 0], seq[:, 1]  # [40, 30, 10, 0, 0, -5]  # [-10, -20, -25, -10, 0, 10]
        color = cat #max(cat)*cat/137  # np.array([0, 10, 20, 30, 40, 45, 50, 55, 60, 70, 80, 90, 100])
        norm = cnorm_cat  # gleich: plt.Normalize(vmin=0, vmax=137)
        x, y = m(lons, lats)
        # m.plot(x, y, marker=None, color='m')
        sc = m.scatter(x, y, marker='.', linewidth=.5, c=color, cmap=cmap_cat, norm=norm)
        #divider = make_axes_locatable(ax)
        #cax = divider.append_axes("right", size="5%", pad=0.1)
        #plt.colorbar(sc, fraction=.05, shrink=.5, label='Maximum sustained wind (in knots)')
        #mappable = cm.ScalarMappable(cnorm,cmap)
    cbar = m.colorbar(mappable=None, location='right', size='3%')
    cbar.set_label('Maximum sustained wind in knots', size=12)
    #plt.colorbar(sc, ax=None, fraction=.05, shrink=.5, label='Maximum sustained wind (in knots)')  # TODO
    plt.clim(0, 137)
    #plt.savefig('orthographic_map_example_python.png', dpi=150, transparent=True)
    plt.savefig('figures/tracks.png')
    plt.show(block=None)


def init_hur():
    db_origin = load_data_hur()
    print('Latitude first entry', db_origin.at[1, 4])
    print('Longitude first entry: ', db_origin.at[1, 5])
    db_origin.drop(columns=[8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], inplace=True)
    df_subjects = db_origin.loc[(db_origin[3].isnull()) & (db_origin[4].isnull())]
    df_subjects.drop(columns=[3, 4, 5, 6, 7], inplace=True)
    db_origin.drop(columns=[2], inplace=True)
    #ids = df_subjects.index
    db_origin[[8, 9, 10]] = 0.0
    db_seq = db_origin.loc[(db_origin[4].notnull()) | (db_origin[5].notnull())].values
    #seq = seq.to_numpy()
    seq =np.zeros(db_seq.shape)
    for i in range(len(seq)):
        z = db_seq[i, :]
        seq[i, 0], seq[i, 1] = np.float64(z[0]), np.float64(z[1])
        seq[i, 2] = lbl_type[np.str.strip(z[2])]
        if np.char.endswith(z[3], 'N'):
            z[3] = np.float64(np.char.rstrip(z[3], 'N'))
        elif np.char.endswith(z[3], 'S'):
            z[3] = -np.float64(np.char.rstrip(z[3], 'S'))
        if np.char.endswith(z[4], 'W'):
            z[4] = -np.float64(np.char.rstrip(z[4], 'W'))
        elif np.char.endswith(z[4], 'E'):
            z[4] = np.float64(np.char.rstrip(z[4], 'E'))

        seq[i, 3], seq[i, 4] = z[3], z[4]
        seq[i, 5], seq[i, 6] = z[5], z[6]
        seq[i, 7], seq[i, 8], seq[i, 9] = coord_2D3D(z[3], z[4])
    subj = df_subjects.to_numpy()
    subj[:, 2] = np.int64(subj[:, 2])
    info = [['Cyclone Nr', 'Name', 'Number of Entries'],
            ['Date', 'Time', 'Status Type', 'Latitude', 'Longitude',
             'Max Wind', 'Min Pressure', 'x', 'y', 'z']]
    ids = np.repeat(0, len(subj))
    for i in range(len(subj)):
        ids[i] = np.sum(subj[0:i, 2])
    np.savez('hur', subj=subj, seq=seq, ids=ids, info=info)


def get_CatName(MaxWind):
    f = gs.array([0, 34, 64, 83, 96, 113, 137])
    x = gs.array(range(-1, 6))
    return gs.floor(np.interp(MaxWind, f, x))

def get_label(CatName):
    y = 0 if CatName <= 0 else (1 if CatName <= 3 else 2)
    return y

def interpolate1(metric: RiemannianMetric, curve, n_points):
    """Interpolate a discrete curve with nb_points from a discrete curve.
    TODO: Generalize to manifold
    Returns
    -------
    interpolation : discrete curve with nb_points points
    """
    old_length = curve.shape[0]
    interpolation = gs.zeros((n_points, curve.shape[1]))
    incr = old_length / n_points
    pos = 0
    for i in range(n_points):
        index = int(gs.floor(pos))
        #interpolation[i] = curve[index] + (pos - index) * (curve[(index + 1) % old_length] - curve[index])
        interpolation[i] = metric.exp(base_point=curve[index], tangent_vec=(pos - index) * metric.log(
            point=curve[(index + 1) % old_length], base_point=curve[index]))
        pos += incr
    return interpolation

def interpolate(metric: RiemannianMetric, curve, n_points):
    """Interpolate a discrete curve with nb_points from a discrete curve
    Returns
    -------
    interpolation : discrete curve with n_points points
    """
    old_length = curve.shape[0]
    interpolation = gs.zeros((n_points, curve.shape[1]))
    no = (n_points-1) / (old_length-1)
    for j in range(old_length - 1):
        p = curve[j]
        v = metric.log(point=curve[j + 1], base_point=curve[j])
        i = no*j
        while i < no*(j + 1):
            tij = i/no - j
            interpolation[int(i)] = metric.exp(base_point=p, tangent_vec=tij*v)
            i += 1
        interpolation[int(i)] = curve[j+1]
    return interpolation

def resample(subj, seq, ids, n_samples):
    from geomstats.geometry.hypersphere import Hypersphere
    S2 = Hypersphere(dim=2)
    S2_metric = S2.metric
    if S2.default_coords_type == 'extrinsic':
        S2_metric.shape = (S2.dim + 1,)
    subj_old, seq_old, ids_old = subj, seq, ids
    ids = np.zeros(N_SUBJ, int)
    f = np.zeros((N_SUBJ, n_samples))
    x = np.zeros((N_SUBJ, n_samples, 3))
    seq = np.zeros((n_samples * N_SUBJ, 10))
    # upsample maxwinds via linear interpolation
    for i in range(N_SUBJ):
        a = seq_old[ids_old[i]:ids_old[i] + subj_old[i, 2], 5]
        b = seq_old[ids_old[i]:ids_old[i] + subj_old[i, 2], 7:10]
        f[i] = interpolate(Euclidean(1).metric, np.reshape(a, (a.shape[0], 1)), n_samples).flatten()
        x[i] = interpolate(S2_metric, np.reshape(b, (b.shape[0], 3)), n_samples)
        rg = range(i * n_samples, (i + 1) * n_samples)
        seq[rg, 7:10] = x[i]
        seq[rg, 5] = f[i]
        for j in range(n_samples):
            seq[j + i * n_samples: j + (i + 1) * n_samples, 3:5] = coord_3D2D(x[i, j])
        subj[i, 2],  ids[i] = n_samples, n_samples * i
    return subj, seq, ids


def sample_spline(B: BezierSpline, n: int = 50) -> np.array:
    """Sample a Bezier spline at n evenly spaced points"""

    return np.array([B.eval(t) for t in np.linspace(0, B.nsegments, n)])


def sample_cubic_polynomial(c: np.array, t_max: int = 1, n: int = 50) -> np.array:
    """Evaluate a real-valued cubic polynomial with given monomial basis factors given (see pythons polyfit for the
    construction of the factors)"""
    def p(t):
        return c[0] * t**3 + c[1] * t**2 + c[2] * t + c[3]

    return np.array([p(t) for t in np.linspace(0, t_max, n)])


def get_seq_date(seq, start, end):
    #group = np.zeros(N_SUBJ)
    a = seq[:, 0]
    x = [i for i in range(len(a)) if (a[i] >= start) & (a[i] <= end)]
    #x = np.where(subj[i,0][4:8]==start)
    return x

def get_subj_year(subj):
    year = 2010
    a = subj[:, 0]
    x = np.arange(0, 13, dtype=int)
    for j in range(11):
        x[j+1] = 1 + max([i for i in range(len(a)) if int(a[i][-4:]) == year])
        year += 1
    x[12] = N_SUBJ
    return x

def visWind(t, wind):
    plt.plot(t, wind)
    #plt.axis('equal')
    plt.xlabel('x')
    plt.xlabel('y')
    #ax.axis('off')
    plt.show()

def plot_3d(points, points_color, title, legend_handle, size=10):
    x, y, z = points.T
    fig, ax = plt.subplots(
        figsize=(size, size),
        facecolor="white",
        tight_layout=True,
        subplot_kw={"projection": "3d"},
    )
    fig.suptitle(title, size=16)
    col = ax.scatter(x, y, z, c=points_color, s=20, alpha=0.8)
    #ax.view_init(azim=-60, elev=9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.zaxis.set_major_locator(ticker.MultipleLocator(1))
    #fig.colorbar(col, ax=ax, orientation="horizontal", shrink=0.6, aspect=60, pad=0.01)
    #ax.legend(fontsize=12)
    ax.legend(handles=legend_handle)
    plt.show()

def plot_2d(points, points_color, title):
    fig, ax = plt.subplots(figsize=(3, 3), facecolor="white", constrained_layout=True)
    fig.suptitle(title, size=16)
    add_2d_scatter(ax, points, points_color)
    plt.show()


def add_2d_scatter(ax, points, points_color, title=None):
    x, y = points.T
    ax.scatter(x, y, c=points_color, s=50, alpha=0.8)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())

def mean_seq(trjs, winds=None, trj_metric=None, wind_metric=None):
    if trj_metric is None:
        from geomstats.geometry.discrete_curves import L2CurvesMetric
        from geomstats.geometry.hypersphere import Hypersphere
        trj_metric = L2CurvesMetric(Hypersphere(2))
    ini_mean = np.mean(np.array(trjs), axis=0)
    for i in range(ini_mean.shape[0]):
        ini_mean[i] = ini_mean[i] / np.linalg.norm(ini_mean[i])
    mean_gs = FrechetMean(trj_metric, point_type='matrix', init_point=ini_mean)
    mean_gs.fit(np.array(trjs))
    mean_trj = mean_gs.estimate_
    # TODO: Use elastic metric for winds
    mean_wind = None
    if winds is not None:
        mean_wind = np.mean(np.array(winds), axis=0)
    # TPCA
    return mean_trj, mean_wind
