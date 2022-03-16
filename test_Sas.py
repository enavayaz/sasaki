import numpy as np
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.geometry.pre_shape import PreShapeSpace, KendallShapeMetric
from SasakiCls import Sasaki_metric
from geomstats.geometry.hypersphere import Hypersphere


p0=np.array([0,1])
u0 =np.array([-1, 0])
pu0=[p0,u0]
pL=np.array([2,2])
uL=np.array([-2,-2])
puL=[pL,uL]
pu=[]
for j in range(3):
    pu.append([np.array([j,j]), np.array([j,3])])

b=[pu0]+pu+[puL] #ok
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
p, u = sm.geodesic([p0,u0],[pL,uL])
r = KendallShapeMetric(5,2)
p = PreShapeSpace(5,2)
s=Sasaki_metric(p,r)
print("ja")
