#import numpy as gs
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.geometry.pre_shape import PreShapeSpace, KendallShapeMetric
from numpy.random import random
from sasaki_metric import SasakiMetric
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.grassmannian import Grassmannian
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.learning.pca import TangentPCA
from geomstats.learning.geodesic_regression import GeodesicRegression
from geomstats.geometry.euclidean import  Euclidean
#import geomstats.datasets.utils as data_utils
import geomstats.backend as gs
from util import visSphere, visKen, load_data, initial_mean
import geomstats.visualization as visualization
import logging
import matplotlib.pylab as plt

Nt = 25
t = gs.linspace(0, 1, Nt)
# """
# First Application: Discrete Geodesics on the 2-Sphere
# """
# S2 = Hypersphere(dim=2)
# S2_metric = S2.metric
# # bypass geomstats bug: shape property of HypersphereMetric not initialized consistently
# if S2.default_coords_type == 'extrinsic':
#     S2_metric.shape = (S2.dim+1,)
# sas = SasakiMetric(S2.metric)
# p0, u0 = gs.array([0, -1, 0]), gs.array([1, 0, 1])
# pu0 = gs.array([p0, u0])
# pL, uL = gs.array([1, 0, 0]), gs.array([0, 1, 1])
# puL = gs.array([pL, uL])
# #m = sm.mean([pu0] + [puL])
# # print('Computing shortest path of geodesics')
# z = sas.geodesic_discrete(pu0, puL)
# # vw0 = sm.log(puL, pu0)
# # xx = sm.exp(vw0, pu0)
# geo_list, color_list = [], []
# geods0L, geods = [], []
# for i in range(Nt):
#     geods0L.append(S2_metric.exp(t[i] * u0, p0))
#     geods0L.append(S2_metric.exp(t[i] * uL, pL))
# geo_list = [geods0L]
# color_list.append('r')
# for j in range(1, len(z) - 1):
#     p1, u1 = z[j][0], z[j][1]
#     for i in range(Nt):
#         geods.append(S2_metric.exp(t[i] * u1, p1))
# geo_list += [geods]
# color_list += 'b'
# visSphere(geo_list, color_list, 15)
# # # plot_and_save_video(geods)
# """
# Second Application: Clustering via Regression
# """
# m = gs.array([[0, -1, 0], [0, 0, 1]])
# n_samples, sigma = 10, gs.pi/12
# x = S2.random_riemannian_normal(m[0], n_samples=n_samples)
# y = S2.random_riemannian_normal(m[0], n_samples=n_samples)
# x = [S2_metric.exp(sigma*S2_metric.log(x[i], m[0]), m[0]) for i in range(n_samples)]
# u = [m[1] + sigma*S2_metric.log(y[i], m[0]) for i in range(n_samples)]
# samples = [gs.array([x[i], u[i]]) for i in range(n_samples)]
#
# print('Computing mean of geodesics')
# initial = initial_mean(samples, S2_metric)
# mean_gs = FrechetMean(sas, init_point=initial)
# mean_gs.fit(samples)
# mean = mean_gs.estimate_
# # mean = sas.mean(samples)
#
# meanvalue, data, geom = [], [], []
# for i in range(Nt):
#     ti = t[i]
#     meanvalue.append(S2_metric.exp(ti * m[1], m[0]))
#     geom.append(S2_metric.exp(ti * mean[1], mean[0]))
#     for j in range(len(samples)):
#         data.append(S2_metric.exp(ti * samples[j][1], samples[j][0]))
# geo_list = [meanvalue] + [data] + [geom]
# color_list = ['k'] + ['r'] + ['b']
# visSphere(geo_list, color_list, 15)
"""
Third application: Discrete Geodesics and Mean Geodesic in Kendall's Shape Space
"""
samples = load_data()
n_samples = len(samples)
print(f"Total number of rat skulls: {n_samples}")
n_subjects, n_trj, k_landmarks, dim = 18, 8, 8, 2
Ken = PreShapeSpace(k_landmarks, dim)
KenMetric = KendallShapeMetric(k_landmarks, dim)
KenMetric.shape = Ken.shape # fix: shape not set consistently
sas = SasakiMetric(KenMetric)
samples = [Ken.projection(samples[i]) for i in range(n_samples)]
sampels = gs.array(samples).reshape(n_subjects, n_trj, k_landmarks, dim)
visKen([samples], ['r'])
# Regression
# ln 266 in pre_shape.py -> add full_matrices=False to svd call (otherwise no autodiff)
reg = GeodesicRegression(Ken, KenMetric, method="riemannian", initialization="warm_start")
# TODO
geos = []
x = gs.linspace(0, 1, n_trj)
for trj in sampels:
    #Change backend to tensorflow, autodiff or pytorch and samples to array
    reg.intercept_ = trj[0]
    reg.coef_ = KenMetric.log(trj[-1], trj[0])
    reg.fit(x, trj, compute_training_score=True)
    print('R^2:', reg.training_score_)
    geos.append(gs.array([reg.intercept_, reg.coef_]))

# TPCA
mean_gs = FrechetMean(sas)
mean_gs.fit(geos)
mean = mean_gs.estimate_
tpca = TangentPCA(metric=KenMetric, n_components=2)
tpca = tpca.fit(geos, base_point=mean)
tangent_projected_data = tpca.transform(geos)
geodesic_0 = KenMetric.geodesic(
    initial_point=mean, initial_tangent_vec=tpca.components_[0]
)
geodesic_1 = KenMetric.geodesic(
    initial_point=mean, initial_tangent_vec=tpca.components_[1]
)
geodesic_points_0 = geodesic_0(t)
geodesic_points_1 = geodesic_1(t)
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)
xticks = gs.arange(1, 2 + 1, 1)
ax.xaxis.set_ticks(xticks)
ax.set_title("Explained variance")
ax.set_xlabel("Number of Principal Components")
ax.set_ylim((0, 1))
ax.bar(xticks, tpca.explained_variance_ratio_)
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
#TODO: plane statt S2
ax = visualization.plot(geodesic_points_0, ax, space="S2", linewidth=2, label="First component")
ax = visualization.plot(geodesic_points_1, ax, space="S2", linewidth=2, label="Second component")
ax = visualization.plot(samples, ax, space="S2", color="black", alpha=0.2, label="Data points")
ax = visualization.plot(mean, ax, space="S2", color="red", s=200, label="Fréchet mean")
ax.legend()
ax.set_box_aspect([1, 1, 1])
plt.show()
# Learnin