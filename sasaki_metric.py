import geomstats.backend as gs
from geomstats.geometry.riemannian_metric import RiemannianMetric

import util
gradient_descent = util.gradient_descent

class SasakiMetric(RiemannianMetric):
    """
    TODO
    """

    def __init__(self, metric: RiemannianMetric = None, Ns=3):
        self.metric = metric  # Riemannian metric of underlying space
        self.Ns = Ns  # Number of discretization steps
        shape = (2, *metric.shape)

        super(SasakiMetric, self).__init__(2*metric.dim, shape = shape, default_point_type = 'matrix')

    def zerovector(self):
        return gs.zeros(self.shape)

    def exp(self, tangent_vec, base_point, n_steps=None, **kwargs):
        """
        Igsut: initial point pu= [p0,u0] in TM and initial
        velocity tangent_vec=[v0,w0] tangent vectors at base_point, where v0 horizontal and w0 vertical component
        Output: end point point=[pL, uL] with pL footpoint and uL tangent vector
        """
        bs_pts = [base_point] if base_point.shape == self.shape else base_point
        tngs = [tangent_vec] if tangent_vec.shape == self.shape else tangent_vec

        metric = self.metric
        par_trans = metric.parallel_transport
        Ns = self.Ns if n_steps is None else n_steps
        eps = 1 / Ns

        rslt = []
        for bs_pt, tng in zip(bs_pts, tngs):
            v0, w0 = tng[0], tng[1]
            p0, u0 = bs_pt[0], bs_pt[1]
            p, u = p0, u0
            for j in range(Ns - 1):
                p = metric.exp(eps * v0, p0)
                u = par_trans(u0 + eps * w0, p0, None, p)
                v = par_trans(v0 - eps * (metric.curvature(u0, w0, v0, p0)), p0, None, p)
                w = par_trans(w0, p0, None, p)
                p0, u0 = p, u
                v0, w0 = v, w
            rslt.append(gs.array([p, u]))

        return gs.array(rslt) if len(rslt) > 1 else rslt[0]

    def log(self, point, base_point, n_steps=None, **kwargs):
        """
        Igsut: base_point and point (with fields p and u) points in TM
        Output: structure vw with fields v and w; (v,w) is the initial vector of
        tangent bundle geodesic from (p0,u0) to (pL,uL)
        """
        pts = [point] if point.shape == self.shape else point
        bs_pts = [base_point] * len(pts) if base_point.shape == self.shape else base_point

        metric = self.metric
        par_trans = metric.parallel_transport
        Ns = self.Ns if n_steps is None else n_steps

        rslt = []
        for pt, bs_pt in zip(pts, bs_pts):
            pu = self.geodesic_discrete(bs_pt, pt, Ns)
            p1, u1 = pu[1][0], pu[1][1]
            p0, u0 = bs_pt[0], bs_pt[1]
            w = (par_trans(u1, p1, None, p0) - u0)
            v = metric.log(point=p1, base_point=p0)
            rslt.append(Ns * gs.array([v, w]))

        return gs.array(rslt) if len(rslt) > 1 else rslt[0]

    def geodesic_discrete(self, initial_point, end_point, n_steps=None, **kwargs):
        """
        Igsut: Points puo=(p0, u0) and point=(pL, uL) of tangent bundle
        Ns - 1 is the number of intermidate points in the discretization
        of the geodesic from base_point to point
        Output: Discrete geodesic x(s)=(p(s), u(s)) in Sasaki metric
        In particular log(base_point, point) is the tangent vector  from base_point to point
        """
        Ns = self.Ns if n_steps is None else n_steps
        metric = self.metric
        par_trans = metric.parallel_transport
        p0, u0 = initial_point[0], initial_point[1]
        pL, uL = end_point[0], end_point[1]
        eps = 1 / Ns

        def loss(pu):
            # h is the loss function
            pu = [initial_point] + pu + [end_point]
            # h = metric.dist(p[0], p[1])**2 + (gs.linalg.norm(par_trans(u[1] - u[0], p[0], None, p[1])))**2
            h = 0
            for j in range(Ns):
                p1, u1, p2, u2 = pu[j][0], pu[j][1], pu[j + 1][0], pu[j + 1][1]
                v1, w1 = metric.log(p2, p1), par_trans(u2, p2, None, p1) - u1
                h += gs.linalg.norm(v1) ** 2 + gs.linalg.norm(w1) ** 2
            return .5 * h

        def grad(pu):
            g = self.zerovector()  # initial gradient with zero vectors
            #gp = g[0]
            #gu = g[1]
            g = gs.array([g] * (Ns - 1))
            pu = [initial_point] + pu + [end_point]  # add boundary points to the list of points
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
        s = gs.linspace(0, 1, Ns + 1)
        pu_ini = []
        for i in range(1, Ns):
            p_ini = metric.exp(s[i] * v, p0)
            u_ini = (1 - s[i]) * par_trans(u0, p0, None, p_ini) + s[i] * par_trans(uL, pL, None, p_ini)
            pu_ini.append(gs.array([p_ini, u_ini]))
        x = gradient_descent(pu_ini, grad, self.exp)
        return [initial_point] + x + [end_point]

    def squared_dist(self, point_a, point_b, **kwargs):
        sqnorm = self.metric.squared_norm
        logs = self.log(point_a, point_b, **kwargs)
        b = point_b[0] if point_b.shape == self.shape else point_b[:, 0]
        v = logs[0] if logs.shape == self.shape else logs[:, 0]
        w = logs[1] if logs.shape == self.shape else logs[:, 1]
        return sqnorm(v, b) + sqnorm(w, b)

    def dist(self, point_a, point_b, **kwargs):
        return gs.linalg.sqrt(self.squared_dist(point_a, point_b, **kwargs))

    def squared_norm(self, vector, base_point=None):
        assert base_point is not None
        sqnorm = self.metric.squared_norm
        p = base_point[0] if base_point.shape == self.shape else base_point[:, 0]
        v = vector[0] if vector.shape == self.shape else vector[:, 0]
        w = vector[1] if vector.shape == self.shape else vector[:, 1]
        return sqnorm(v, p) + sqnorm(w, p)

    def norm(self, vector, base_point=None):
        assert base_point is not None
        return gs.linalg.sqrt(self.squared_norm(vector, base_point))