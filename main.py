import numpy as np
from scipy.integrate import solve_ivp, cumtrapz
from scipy import special
from CurveCls import Curve
import util
vis_curve = util.vis
vis_curve3D = util.vis3D

sigma = -2*np.pi
km = 6

# Define the parameters as regular Python functions:
def kappa_els(t, sigma, km):
    kap = km
    if km**2 + 2 - sigma == 0:
        return km
    if sigma < 2:
        q = km/np.sqrt(2 * (km**2 + 2 - sigma))
        kap = km * JacobiCN(.5 * t * km / q, q**2)  # Wavelike
    else:
        p = 2 * (km**2 + 2 - sigma) / km ** 2
        kap = km * JacobiDN(.5 * km * t, p)  # Orbitlike
    return kap

def kappa(t):
    return kappa_els(t, sigma, km)

def tau(t):
    return 0

def JacobiCN(u_, m_):
    return special.ellipj(u_, m_)[1]

def JacobiDN(u_, m_):
    return special.ellipj(u_, m_)[2]

c = Curve()
# c.kappa()=kappa_els(,1,2)
T0, N0, B0 = [1, 0, 0], [0, 1, 0], [0, 0, 1]
c.coords = c.Frenet_Curve(T0, N0, B0)
vis_curve(c.coords)

# The equations: dz/dt = model(s, z):
def model(s, z):
    T = z[:3]  # z is a (9, ) shaped array, the concatenation of T, N and B
    N = z[3:6]
    B = z[6:]

    dTds = kappa(s) * N
    dNds = -kappa(s) * T + tau(s) * B
    dBds = -tau(s) * N

    return np.hstack([dTds, dNds, dBds])

T0, N0, B0 = [1, 0, 0], [0, 1, 0], [0, 0, 1]
z0 = np.hstack([T0, N0, B0])
s_span = (0, 2*np.pi)  # start and final "time"
t_eval = np.linspace(*s_span, 100)  # define the number of point wanted in-between,
# It is not necessary as the solver automatically
# define the number of points.
# It is used here to obtain a relatively correct
# integration of the coordinates, see the graph

# Solve:
sol = solve_ivp(model, s_span, z0, t_eval=t_eval, method='RK45')
print(sol.message)
# >> The solver successfully reached the end of the integration interval.

# Unpack the solution:
T, N, B = np.split(sol.y, 3)  # another way to unpack the z array
s = sol.t

#  integration of the tangent vector in order to get the coordinates
#  to plot the curve  (there is certainly better way to do this)
coords = cumtrapz(T, x=s)
vis_curve(coords)
#vis_curve3D(coords)
# Press the green button in the gutter to run the script.
# if __name__ == '__main__':