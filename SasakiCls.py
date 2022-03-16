import functools

import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import cumtrapz
import util

gradient_descent = util.gradient_descent

from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.geometry.manifold import Manifold
from geomstats.learning import geodesic_regression


class Sasaki_metric:
    """
    TODO
    """
    def __init__(self, mf: Manifold, metric: RiemannianMetric = None, Ns=4, Nt=20):
        # super(Sasaki_metric, self).__init__(metric)
        self.manifold = mf
        self.metric = metric
        self.Ns = Ns
        self.Nt = Nt
        # super().__init__(mf, metric, Ns, Nt)

    def exp(self, vw0, pu0):
        """
        Input: initial point pu= [p0,u0] in TM and initial
        velocity vw0=[v0,w0] tangent vectors at p0, where v0 horizontal and w0 vertical component
        Output: end point puL=[pL, uL] with pL footpoint and uL tangent vector
        """
        metric = self.metric
        par_trans = metric.parallel_transport
        itr = self.Nt + 1
        eps = 1 / itr
        v0, w0 = vw0[0], vw0[1]
        p0, u0 = pu0[0], pu0[1]
        for j in range(itr - 1):
            p = metric.exp(eps * v0, p0)
            u = par_trans(u0, p0, u0 + eps * w0, p)
            v = par_trans(u0, p0, v0 + eps * (metric.curvature(u0, w0, v0, p0)))
            w = par_trans(w0, p0, p, None, w0)
            p0, u0 = p, u
            v0, w0 = v, w
        return [p, u]

    def log(self, puL, pu0):
        """
        Input: pu0 and puL (with fields p and u) points in TM
        Output: structure vw with fields v and w; (v,w) is the initial vector of
        tangent bundle geodesic from (p0,u0) to (pL,uL)
        """
        metric = self.metric
        Ns = self.Ns
        p, u = self.geodesic(pu0, puL)
        eps = 1 / Ns
        p1, u1 = p[1], u[1]
        w = (metric.parallel_transport(u1, p1, None, pu0[0]) - pu0[1]) / eps
        v = metric.log(puL[0], pu0[0]) / eps
        return [v, w]

    def geodesic(self, pu0, puL):
        """
        Input: Points puo=(p0, u0) and puL=(pL, uL) of tangent bundle
        Ns - 1 is the number of intermidate points in the discretization
        of the geodesic from pu0 to puL
        Output: Discrete geodesic x(s)=(p(s), u(s)) in Sasaki metric
        In particular log(pu0, puL) is the tangent vector  from pu0 to puL
        """
        Ns, Nt = self.Ns, self.Nt
        metric = self.metric
        par_trans = metric.parallel_transport
        #mf = self.manifold
        #reg = geodesic_regression.GeodesicRegression
        p0, u0 = pu0[0], pu0[1]
        pL, uL = puL[0], puL[1]
        gp = [metric.log(p0, p0)]*(Ns-1)  # initial gradient with zero vectors
        gu = gp
        #p = [p0 for i in rg]
        #u = [u0 for i in rg]
        #p[Ns - 1], u[Ns - 1] = pL, uL
        eps = 1 / Ns

        def loss(pu):
            # h is the loss function
            pu = [pu0]+pu+[puL]
            #h = metric.dist(p[0], p[1])**2 + (np.linalg.norm(par_trans(u[1] - u[0], p[0], None, p[1])))**2
            h = 0
            for j in range(Ns):
                p1, u1, p2, u2 = pu[j][0], pu[j][1], pu[j+1][0], pu[j+1][1]
                v1, w1 = metric.log(p2, p1), par_trans(u2, p2, None, p1) - u1
                h += np.linalg.norm(v1)**2 + np.linalg.norm(w1)**2
            return .5 * h

        def grad(pu):
            pu = [pu0]+pu+[puL]
            delta = eps
            for j in range(Ns-1):
                p1, u1 = pu[j][0], pu[j][1]
                p2, u2 = pu[j+1][0], pu[j+1][1]
                p3, u3 = pu[j+2][0], pu[j+2][1]
                v, w = metric.log(p3, p2), par_trans(u3, p3, None, p2) - u2
                gp[j] = metric.log(p3, p2) + metric.log(p2, p1) - metric.curvature(u2, w, v, p2)
                gu[j] = par_trans(u2, p2, None, p1) - 2 * u1 + par_trans(u0, p0, None, p1)
                gp[j], gu[j] = - delta*gp[j], - delta*gu[j]
            return [gp, gu]
        # Initial values for gradient_descent
        v = metric.log(pL, p0)
        s = np.linspace(0, 1, Ns+1)
        pu_ini = []
        for i in range(1, Ns):
            p_ini = metric.exp(s[i] * v, p0)
            u_ini = (1 - s[i])*par_trans(u0, p0, None, p_ini) + s[i]*par_trans(uL, pL, None, p_ini)
            pu_ini.append([p_ini, u_ini])

        #pu_ini = np.array([p_ini, u_ini])
        # see, apply: examples.gradient_descent_s2
        x, _ = gradient_descent(pu_ini, loss, grad, metric)
        previous_x = pu_ini
        pu = []
        for x, _ in gradient_descent(pu_ini, loss, grad, metric):
            ini_tang_vec = [metric.log(point=x[0], base_point=previous_x[0]), x[1] - previous_x[1]]
            geodesic = [metric.geodesic(initial_point=previous_x[0], initial_tangent_vec=ini_tang_vec[0]), x[1]]
            pu.append(geodesic(s))
            previous_x = [x[0], x[1]]
        return [pu0, x, puL]

    def dist(self, pu0, puL):
        [v, w] = self.log(puL, pu0)
        return np.linalg.sqrt(np.linalg.norm(v) ** 2 + np.linalg.norm(w) ** 2)
