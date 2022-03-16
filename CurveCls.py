import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import cumtrapz
import util

class Curve:
    """T...
    """
    def __init__(self, point_shape=3, tspan=np.linspace(*(0, 2 * np.pi), 100), coords=None):
        self.point_shape = point_shape
        self.tspan = tspan
        self.coords = coords
        #super().__init__(point_shape, tspan, coords)

    @property
    def kappa(self, k):
        return k

    def kappa(s):
        return 1

    def tau(s):
        return 0

    def kappa_coords(self,t):
        # Curve.coords +1
        assert self.coords is not None
        return self.coords + t

    def tau_coords(self, t):
        # Curve.coords +1
        assert self.coords is not None
        return self.coords + t

    def length_coords(self, t):
        # Curve.coords +1
        assert self.coords is not None
        return self.coords + t


    def Frenet_Curve(self,T0, N0, B0):
        # T0, N0, B0 = [1, 0, 0], [0, 1, 0], [0, 0, 1]
        z0 = np.hstack([T0, N0, B0])
        # s_span = (0, 2*np.pi)  # start and final "time"
        s_span = (self.tspan[0], self.tspan[-1])
        # t_eval = np.linspace(*s_span, 100)  # define the number of point wanted in-between,
        t_eval = self.tspan

        # It is not necessary as the solver automatically
        # define the number of points.
        # It is used here to obtain a relatively correct
        # integration of the coordinates, see the graph

        def model(s, z):
            T = z[:3]  # z is a (9, ) shaped array, the concatenation of T, N and B
            N = z[3:6]
            B = z[6:]
            kappa = Curve.kappa
            tau = Curve.tau
            dTds = kappa(s) * N
            dNds = -kappa(s) * T + tau(s) * B
            dBds = -tau(s) * N

            return np.hstack([dTds, dNds, dBds])

        # Solve:
        sol = solve_ivp(model, s_span, z0, t_eval=t_eval, method='RK45')

        print(sol.message)
        # >> The solver successfully reached the end of the integration interval.

        # Unpack the solution:
        T, N, B = np.split(sol.y, 3)  # another way to unpack the z array
        s = sol.t

        # Bonus: integration of the normal vector in order to get the coordinates
        #        to plot the curve  (there is certainly better way to do this)
        return cumtrapz(T, x=s)  # coordinates