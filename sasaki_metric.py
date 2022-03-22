import functools
import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import cumtrapz
import util
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.geometry.manifold import Manifold
from geomstats.learning import geodesic_regression
from geomstats.learning.frechet_mean import FrechetMean
gradient_descent = util.gradient_descent


class SasakiMetric:
    """
    TODO
    """

    def __init__(self, metric: RiemannianMetric = None, Ns=4):
        self.metric = metric
        self.Ns = Ns
        # super().__init__(metric, Ns)

    def exp(self, vw0, pu0, t=1):
        """
        Input: initial point pu= [p0,u0] in TM and initial
        velocity vw0=[v0,w0] tangent vectors at p0, where v0 horizontal and w0 vertical component
        Output: end point puL=[pL, uL] with pL footpoint and uL tangent vector
        """
        metric = self.metric
        par_trans = metric.parallel_transport
        Ns = self.Ns
        eps = 1 / Ns
        v0, w0 = t * vw0[0], t * vw0[1]
        p0, u0 = pu0[0], pu0[1]
        p, u = p0, u0
        for j in range(Ns - 1):
            p = metric.exp(eps * v0, p0)
            u = par_trans(u0 + eps * w0, p0, None, p)
            v = par_trans(v0 - eps * (metric.curvature(u0, w0, v0, p0)), p0, None, p)
            w = par_trans(w0, p0, None, p)
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
        par_trans = metric.parallel_transport
        Ns = self.Ns
        eps = 1 / Ns
        pu = self.geodesic(pu0, puL)
        p1, u1 = pu[1][0], pu[1][1]
        p0, u0 = pu0[0], pu0[1]
        w = (par_trans(u1, p1, None, p0) - u0) / eps
        v = metric.log(point=p1, base_point=p0) / eps
        return np.array([v, w])

    def geodesic(self, pu0, puL):
        """
        Input: Points puo=(p0, u0) and puL=(pL, uL) of tangent bundle
        Ns - 1 is the number of intermidate points in the discretization
        of the geodesic from pu0 to puL
        Output: Discrete geodesic x(s)=(p(s), u(s)) in Sasaki metric
        In particular log(pu0, puL) is the tangent vector  from pu0 to puL
        """
        Ns = self.Ns
        metric = self.metric
        par_trans = metric.parallel_transport
        p0, u0 = pu0[0], pu0[1]
        pL, uL = puL[0], puL[1]
        eps = 1 / Ns

        def loss(pu):
            # h is the loss function
            pu = [pu0] + pu + [puL]
            # h = metric.dist(p[0], p[1])**2 + (np.linalg.norm(par_trans(u[1] - u[0], p[0], None, p[1])))**2
            h = 0
            for j in range(Ns):
                p1, u1, p2, u2 = pu[j][0], pu[j][1], pu[j + 1][0], pu[j + 1][1]
                v1, w1 = metric.log(p2, p1), par_trans(u2, p2, None, p1) - u1
                h += np.linalg.norm(v1) ** 2 + np.linalg.norm(w1) ** 2
            return .5 * h

        def grad(pu):
            gp = metric.log(p0, p0)  # initial gradient with zero vectors
            gu = gp
            g = np.array([np.vstack((gp, gu))] * (Ns - 1))
            pu = [pu0] + pu + [puL]  # add boundary points to the list of points
            for j in range(Ns - 1):
                p1, u1 = pu[j][0], pu[j][1]
                p2, u2 = pu[j + 1][0], pu[j + 1][1]
                p3, u3 = pu[j + 2][0], pu[j + 2][1]
                v, w = metric.log(p3, p2), par_trans(u3, p3, None, p2) - u2
                gp = metric.log(p3, p2) + metric.log(p1, p2) + metric.curvature(u2, w, v, p2)
                gu = par_trans(u3, p3, None, p2) - 2 * u2 + par_trans(u0, p0, None, p2)
                # g.append([gp, gu])
                g[j][0], g[j][1] = gp, gu
            return -Ns * g

        # Initial values for gradient_descent
        v = metric.log(pL, p0)
        s = np.linspace(0, 1, Ns + 1)
        pu_ini = []
        for i in range(1, Ns):
            p_ini = metric.exp(s[i] * v, p0)
            u_ini = (1 - s[i]) * par_trans(u0, p0, None, p_ini) + s[i] * par_trans(uL, pL, None, p_ini)
            pu_ini.append([p_ini, u_ini])

        x = gradient_descent(pu_ini, grad, self.exp)
        return [pu0] + x + [puL]

    def dist(self, pu0, puL):
        [v, w] = self.log(puL, pu0)
        return np.linalg.sqrt(np.linalg.norm(v) ** 2 + np.linalg.norm(w) ** 2)

    def mean(self, pu, mean_ini=None):
        lrate, max_iter = 0.5, 100

        def grad(x):
            # x = x[0]
            g = -np.sum(self.log(y, x) for y in pu) / len(pu)
            # return np.array([g])
            return g

        def grad1(x):
            # x = x[0]
            g = -np.sum(self.log(y, x[0]) for y in pu) / len(pu)
            return np.array([g])

        if mean_ini is None:
            mean_ini = pu[0]
        m1 = gradient_descent([mean_ini], grad1, self.exp, lrate=1, max_iter=100)
        m = mean_ini
        for _ in range(max_iter):
            g = grad(m)
            g_norm = np.linalg.norm(g)
            if g_norm < 1e-6: break
            m = self.exp(vw0=-lrate * g, pu0=m)
        return m
