import numpy as np
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.geometry.pre_shape import PreShapeSpace, KendallShapeMetric
from SasakiCls import Sasaki_metric
from geomstats.geometry.hypersphere import Hypersphere
from util import plt
from util import plot_and_save_video, plot_and_save_Image
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
S2 = Hypersphere(dim=2)
S2_metric = S2.metric
sm = Sasaki_metric(S2,S2_metric)
pp=S2.random_uniform(2)
p0=pp[0]
p0=np.array([0,0,1])
p00=pp[1]
u0=S2_metric.log(p00,p0)
qq=S2.random_uniform(2)
pL=qq[0]
pL=np.array([1,0,0])
q00=qq[1]
uL=S2_metric.log(q00,pL)
geods=[p0]
t=np.linspace(0,1,20)
for i in range(20):
    geods.append(S2.metric.exp(t[i]*S2.metric.log(pL,p0), p0))
#plot_and_save_video(geods)
plot_and_save_Image(geods)
[p, u] = sm.geodesic([p0,u0],[pL,uL])
r = KendallShapeMetric(5,2)
p = PreShapeSpace(5,2)
s=Sasaki_metric(p,r)
print("ja")
