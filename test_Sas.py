import numpy as np
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.geometry.pre_shape import PreShapeSpace, KendallShapeMetric
from SasakiCls import Sasaki_metric
from geomstats.geometry.hypersphere import Hypersphere
import geomstats.backend as gs
from util import VisGeodesicsTM

import geomstats.visualization as visualization
import logging

px0=np.array([0,1])
ux0 =np.array([-1, 0])
pux0=[px0,ux0]
pxL=np.array([2,2])
uxL=np.array([-2,-2])
puxL=[pxL,uxL]
pux=[]
for j in range(3):
    pux.append([np.array([j,j]), np.array([j,3])])
b=[pux0]+pux+[puxL] #ok
"""
First Application: Discrete Geodesics on the 2-Sphere
"""
S2 = Hypersphere(dim=2)
S2_metric = S2.metric
sm = Sasaki_metric(S2_metric)
#pp=S2.random_uniform(2)
#p0=pp[0]
p0=np.array([0,-1,0])
u0=np.array([1,0,1])
pu0=[p0,u0]
#p00=pp[1]
#u0=S2_metric.log(p00,p0)
#qq=S2.random_uniform(2)
pL=np.array([1,0,0])
uL=np.array([0,1,1])
puL=[pL,uL]
geods=[]
geods0L=[]
Nt=25
t=np.linspace(0,1,Nt)
#plot_and_save_video(geods)
z = sm.geodesic([p0,u0],[pL,uL])
vw0 = sm.log(puL, pu0)
xx=sm.exp(vw0,pu0)
geo_list, color_list = [], []
for i in range(Nt):
    geods0L.append(S2.metric.exp(t[i]*u0, p0))
    geods0L.append(S2.metric.exp(t[i]*uL, pL))
geo_list=[geods0L]
color_list.append('r')
for j in range(1,len(z)-1):
    p1, u1 =z[j][0], z[j][1]
    for i in range(20):
        geods.append(S2.metric.exp(t[i]*u1, p1))
geo_list += [geods]
color_list += 'b'
VisGeodesicsTM(geo_list,color_list,15)
"""
Second Application: Clustering via Regression
"""
mean = sm.mean(z)
"""
Third application: Discrete Geodesics and Mean Geodesic in Kendall's Shape Space
"""
# r = KendallShapeMetric(5,2)
# p = PreShapeSpace(5,2)
# s=Sasaki_metric(r)

