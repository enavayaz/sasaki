import geomstats.backend as gs
from geomstats.geometry.riemannian_metric import RiemannianMetric

import util
gradient_descent = util.gradient_descent

class SasakiMetric(RiemannianMetric):
    """
    A prominent natural metric on the tangent bundle TM of a Riemannian manifold M is the Sasaki metric.
    Its characterization: Canonical projection of TM becomes a Riemannian submersion, parallel vector fields
    along curves are orthogonal to their fibres, and restriction to any tangent space is Euclidean.
    For Details and computational aspects, see
    https://doi.org/10.1007/s10851-022-01079-x and https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4270017/
    """

    def __init__(self, metric: RiemannianMetric = None, Ns=3):
        self.metric = metric  # Riemannian metric of underlying space
        self.Ns = Ns  # Number of discretization steps
        shape = (2, gs.prod(metric.shape))

        super(SasakiMetric, self).__init__(2*metric.dim, shape=shape, default_point_type='matrix')

    def exp(self, tangent_vec, base_point, n_steps=None, **kwargs):
        """
        Input: initial point pu= [p0,u0] in TM and initial
        velocity tangent_vec=[v0,w0] tangent vectors at base_point, where v0 horizontal and w0 vertical component
        Output: end point point=[pL, uL] with pL footpoint and uL tangent vector
        """
        # unflatten
        bs_pts = gs.reshape(base_point, (-1, 2) + self.metric.shape)
        tngs = gs.reshape(tangent_vec, bs_pts.shape)

        metric = self.metric
        par_trans = metric.parallel_transport
        Ns = self.Ns if n_steps is None else n_steps
        eps = 1 / Ns

        v0, w0 = tngs[:, 0], tngs[:, 1]
        p0, u0 = bs_pts[:, 0], bs_pts[:, 1]
        p, u = p0, u0
        for j in range(Ns - 1):
            p = metric.exp(eps * v0, p0)
            u = par_trans(u0 + eps * w0, p0, None, p)
            v = par_trans(v0 - eps * (metric.curvature(u0, w0, v0, p0)), p0, None, p)
            w = par_trans(w0, p0, None, p)
            p0, u0 = p, u
            v0, w0 = v, w

        return gs.reshape(gs.array([p, u]), base_point.shape)

    def log(self, point, base_point, n_steps=None, **kwargs):
        """
        Input: base_point and point (with fields p and u) points in TM
        Output: structure vw with fields v and w; (v,w) is the initial vector of
        tangent bundle geodesic from (p0,u0) to (pL,uL)
        """
        # unflatten
        pts = gs.reshape(point, (-1, 2) + self.metric.shape)
        bs_pts = gs.reshape(base_point, (-1, 2) + self.metric.shape)

        metric = self.metric
        par_trans = metric.parallel_transport
        Ns = self.Ns if n_steps is None else n_steps

        rslt = []
        for i, pt in enumerate(pts):
            bs_pt = bs_pts[i % len(bs_pts)]
            pu = self.geodesic_discrete(bs_pt, pt, Ns)
            p1, u1 = pu[1][0], pu[1][1]
            p0, u0 = bs_pt[0], bs_pt[1]
            w = (par_trans(u1, p1, None, p0) - u0)
            v = metric.log(point=p1, base_point=p0)
            rslt.append(Ns * gs.array([v, w]))

        return gs.reshape(gs.array(rslt), point.shape)

    def geodesic_discrete(self, initial_point, end_point, n_steps=None, **kwargs):
        """
        This method employs a variational time discretization of geodesics.
        Input: Points puo=(p0, u0) and point=(pL, uL) of tangent bundle
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
            # initial gradient with zero vectors
            g = []
            pu = [initial_point] + pu + [end_point]  # add boundary points to the list of points
            for j in range(Ns - 1):
                p1, u1 = pu[j][0], pu[j][1]
                p2, u2 = pu[j + 1][0], pu[j + 1][1]
                p3, u3 = pu[j + 2][0], pu[j + 2][1]
                v, w = metric.log(p3, p2), par_trans(u3, p3, None, p2) - u2
                gp = metric.log(p3, p2) + metric.log(p1, p2) + metric.curvature(u2, w, v, p2)
                gu = par_trans(u3, p3, None, p2) - 2 * u2 + par_trans(u0, p0, None, p2)
                g.append([gp, gu])
            return -Ns * gs.array(g)

        # Initial values for gradient_descent
        v = metric.log(pL, p0)
        s = gs.linspace(0., 1., Ns + 1)
        pu_ini = []
        for i in range(1, Ns):
            p_ini = metric.exp(s[i] * v, p0)
            u_ini = (1 - s[i]) * par_trans(u0, p0, None, p_ini) + s[i] * par_trans(uL, pL, None, p_ini)
            pu_ini.append(gs.array([p_ini, u_ini]))
        x = gradient_descent(pu_ini, grad, self.exp)
        return [initial_point] + x + [end_point]

    def squared_dist(self, point_a, point_b, **kwargs):
        # compute dist as norm of logarithm of point_a w.r.t. point_b
        log = self.log(point_a, point_b, **kwargs)
        return self.squared_norm(log, point_b)

    def dist(self, point_a, point_b, **kwargs):
        return gs.sqrt(self.squared_dist(point_a, point_b, **kwargs))

    def squared_norm(self, vector, base_point=None):
        # unflatten
        vector = gs.reshape(vector, (-1, 2) + self.metric.shape)
        base_point = gs.reshape(base_point, (-1, 2) + self.metric.shape)
        # compute Sasaki inner product via metric of underlying manifold
        sqnorm = self.metric.squared_norm
        return sqnorm(vector[:, 0], base_point[:, 0]) + sqnorm(vector[:, 1], base_point[:, 0])

    def norm(self, vector, base_point=None):
        return gs.sqrt(self.squared_norm(vector, base_point))
