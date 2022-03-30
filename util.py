import matplotlib
import matplotlib.pylab as plt
import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.learning.frechet_mean import FrechetMean
import logging
import numpy as np
matplotlib.use("Agg")  # NOQA

def initial_mean(pu, metric):
    mean_ini = pu[0]
    a = gs.array(pu)
    mean_p = FrechetMean(metric)
    mean_p.fit(a[:, 0])
    mean_ini[0] = mean_p.estimate_
    a[:, 1] = gs.array(
        [metric.parallel_transport(a[j, 1], a[j, 0], end_point=mean_ini[0]) for j in range(len(pu))])
    mean_ini[1] = gs.mean(a[:, 1], 0)
    return mean_ini

def gradient_descent(x_ini, grad, exp, loss=None, lrate=0.1, max_iter=100, tol=1e-6):
    """
    Apply a gradient descent until either max_iter or a given tolerance is reached.
    """
    L = len(x_ini)
    x = x_ini
    for i in range(max_iter):
        grad_x = grad(x)
        # loss_x = loss(x)
        grad_norm = gs.linalg.norm(grad_x)
        if grad_norm < tol:
            # logging.info("solution: %s", x)
            logging.info("reached tolerance %s", tol)
            logging.info("iterations: %d", i)
            logging.info("|grad|: %s", grad_norm)
            # logging.info("energy: %s", loss_x)
            break
        grad_x = -lrate * grad_x
        for j in range(L):
            x[j] = exp(grad_x[j], x[j])
    return list(x)

def visSphere(points_list, color_list, size=15):
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(111, projection="3d")
    sphere = visualization.Sphere()
    sphere.draw(ax, marker=".")
    for i in range(len(points_list)):
        for points in points_list[i]:
            points = gs.to_ndarray(points, to_ndim=2)
            sphere.draw_points(ax, points=points, color=color_list[i], marker=".")
    plt.show()

def visKen(points_list, color_list, size=10):  # TODO?
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(111)
    for i in range(len(points_list)):
        for points in points_list:
            #points = gs.to_ndarray(points, to_ndim=2)
            for point in points:
                ax.scatter(point[:, 0], point[:, 1], color=color_list[i], s=8, alpha=0.5)
    #ax.set_title("")
    #x.legend()
    plt.show()

def visKenPCA(geos, variance, samples, mean):  # TODO
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    xticks = gs.arange(1, 2 + 1, 1)
    ax.xaxis.set_ticks(xticks)
    ax.set_title("Explained variance")
    ax.set_xlabel("Number of Principal Components")
    ax.set_ylim((0, 1))
    ax.bar(xticks, variance)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    # TODO: plane statt S2
    for geo in geos:
        ax = visualization.plot(geo, ax, space="S2", linewidth=2, label="First component")
        ax = visualization.plot(geo, ax, space="S2", linewidth=2, label="Second component")
    ax = visualization.plot(samples, ax, space="S2", color="black", alpha=0.2, label="Data points")
    ax = visualization.plot(mean, ax, space="S2", color="red", s=200, label="Fréchet mean")
    ax.legend()
    ax.set_box_aspect([1, 1, 1])
    plt.show()

def load_data():
    return np.load('rat_skulls.npy')