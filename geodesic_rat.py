# ln 266 in pre_shape.py -> add full_matrices=False to svd call (otherwise no autodiff)
# from geomstats.geometry.pre_shape import PreShapeSpace, KendallShapeMetric
import os

os.environ['GEOMSTATS_BACKEND'] = 'autograd'

from sasaki_metric import SasakiMetric
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.learning.pca import TangentPCA
from geomstats.learning.geodesic_regression import GeodesicRegression
import geomstats.backend as gs
import numpy as np
from util import load_data_hur, visSphere, visEarth, initial_mean, visTPCA, coord_2D3D, preprocess_hur, get_CatName


"""
Application: Hurricane Tracks
"""
# Preprocessing, prepare data
#preprocess_hur()

# Import data
data = np.load('hur.npz', allow_pickle=True)
subj, seq, ids, info = data['subj'], data['seq'], data['ids'], data['info']
n_subj = 21  # only 2010
n_samples = ids[n_subj]
# subj, seq, ids = subj[0:n_subj], seq[0:n_samples], ids[0:n_subj+1]
# dataset contains 21 Hurricane tracks
print(f"Number of subjects: {n_subj}, Total number of observations: {n_samples}")

a=get_CatName(95)

# Vis 5 subjects
seq_list = []
for i in range(192, 193):
    seq_list.append(seq[ids[i]:ids[i + 1], 3:6])
visEarth(seq_list)

# geo_list = []
# for i in range(18, 21):
#     geo_list.append(seq[ids[i]:ids[i + 1], 7:10])
# visSphere(geo_list, ['r'] + ['b'] + ['r'] + ['b'] + ['g'])

# Prepare Riemannian structure
S2 = Hypersphere(dim=2)
S2_metric = S2.metric
if S2.default_coords_type == 'extrinsic':
    S2_metric.shape = (S2.dim + 1,)
sas = SasakiMetric(S2.metric)

"""
Geodesics (via regression) and Mean Geodesic
"""
# Regression
reg = GeodesicRegression(S2, S2_metric, center_X=False, method="riemannian", initialization="frechet")
# assume irregular time sampling to compensate for saturation in growth -> equidistant spread in shape space
points, cats = seq[:, 7:10], seq[:, 5]
geos, trjs = [], []
for i in range(191, 200):
    len_trj = subj[i, 2]
    x, trj = gs.linspace(0., 1., len_trj), points[ids[i]:ids[i] + len_trj]
    # set warm start
    reg.intercept_ = trj[0]
    reg.coef_ = S2_metric.log(point=trj[-1], base_point=trj[0])
    # compute best fitting geodesic
    reg.fit(x, trj, compute_training_score=True)
    print('R^2:', reg.training_score_)
    p, u = reg.intercept_, reg.coef_
    geos.append(gs.reshape([p, u], sas.shape))
    trjs.append(trj)
geos = gs.array(geos)
geo_list = []
t = gs.linspace(0., 1., 25)
for geo in geos:
    geo_list.append(S2_metric.geodesic(geo[0], initial_tangent_vec=geo[1])(t))
visSphere([geo_list] + [trjs], ['r'] + ['b'], size=15)
#visEarth([geo_list] + [trjs])

# group mean
print('Computing mean of geodesics')
initial = initial_mean(geos, sas.metric)
mean_gs = FrechetMean(sas, init_point=initial)
mean_gs.fit(geos)
mean = mean_gs.estimate_

data, meanvalue, geom = [], [], []
t = gs.linspace(0., 1., 20)
meanvalue.append(S2_metric.geodesic(mean[0], initial_tangent_vec=mean[1])(t))
geom.append(S2_metric.geodesic(mean[0], initial_tangent_vec=mean[1])(t))
for geo in geos:
    data.append(S2_metric.geodesic(geo[0], initial_tangent_vec=geo[1])(t))
visSphere([data] + [meanvalue] + [geom], ['r'] + ['k'] + ['b'], size=10)

# Tangent PCA
tpca = TangentPCA(metric=sas, n_components=2)
geos_proj = tpca.fit_transform(geos, base_point=mean_gs.estimate_)

# visualize tangent PCA
visTPCA(tpca.explained_variance_, geos_proj)
