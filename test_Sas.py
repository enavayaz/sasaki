import numpy as np
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.geometry.pre_shape import PreShapeSpace, KendallShapeMetric
from numpy.random import random

from SasakiCls import Sasaki_metric
from geomstats.geometry.hypersphere import Hypersphere
import geomstats.backend as gs
from util import VisGeodesicsTM

import geomstats.visualization as visualization
import logging
"""
First Application: Discrete Geodesics on the 2-Sphere
"""
S2 = Hypersphere(dim=2)
S2_metric = S2.metric
sm = Sasaki_metric(S2_metric,5)
p0, u0 = np.array([0,-1,0]), np.array([1,0,1])
pu0=[p0,u0]
pL, uL = np.array([1,0,0]), np.array([0,1,1])
puL=[pL,uL]
#m=sm.mean([pu0]+[puL])
#plot_and_save_video(geods)
z = sm.geodesic([p0,u0],[pL,uL])
vw0 = sm.log(puL, pu0)
xx=sm.exp(vw0,pu0)
geo_list, color_list = [], []
geods0L, geods =[], []
Nt=25
t=np.linspace(0,1,Nt)
for i in range(Nt):
    geods0L.append(S2_metric.exp(t[i]*u0, p0))
    geods0L.append(S2_metric.exp(t[i]*uL, pL))
geo_list=[geods0L]
color_list.append('r')
for j in range(1,len(z)-1):
    p1, u1 =z[j][0], z[j][1]
    for i in range(Nt):
        geods.append(S2_metric.exp(t[i]*u1, p1))
geo_list += [geods]
color_list += 'b'
VisGeodesicsTM(geo_list,color_list,15)
"""
Second Application: Clustering via Regression
"""
m = np.array([[0, 1, 0], [1, 0, 1]])
x = S2.random_riemannian_normal(m[0], n_samples=10)
y = [S2.random_riemannian_normal(x[i]) for i in range(10)]
u = [S2_metric.log(y[i], x[i]) for i in range(10)]
mean = sm.mean(z)
mp, mu =mean[0], mean[1]
geom = []
for i in range(Nt):
    geom.append(S2_metric.exp(t[i]*mu, mp))
geo_list += [geom]
color_list += ['k']
VisGeodesicsTM(geo_list,color_list,15)
"""
Third application: Discrete Geodesics and Mean Geodesic in Kendall's Shape Space
"""
r = KendallShapeMetric(5,2)
# p = PreShapeSpace(5,2)
# s=Sasaki_metric(r)

