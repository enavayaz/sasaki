import matplotlib
import matplotlib.pylab as plt
import matplotlib.animation as animation
import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.geometry.riemannian_metric import RiemannianMetric
import logging
import numpy as np
import csv
# matplotlib.use("Agg")  # NOQA

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

def vis(coords):
    plt.plot(coords[0, :], coords[1, :])
    plt.axis('equal')
    plt.xlabel('x')
    plt.xlabel('y')
    plt.show()

def vis3D(coords):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.plot(coords[0, :], coords[1, :], coords[2, :])
    # plt.axis('equal');#plt.xlabel('x');plt.ylabel('y');plt.zlabel('z')
    plt.show()

def plot_and_save_video(
        geodesics, loss=None, size=20, fps=10, dpi=100, out="out.mp4", color="red"):
    """Render a set of geodesics and save it to an mpeg 4 file."""
    FFMpegWriter = animation.writers["ffmpeg"]
    writer = FFMpegWriter(fps=fps)
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(111, projection="3d")
    sphere = visualization.Sphere()
    # sphere.plot_heatmap(ax, loss)
    points = gs.to_ndarray(geodesics[0], to_ndim=2)
    sphere.add_points(points)
    sphere.draw(ax, color=color, marker=".")
    with writer.saving(fig, out, dpi=dpi):
        for points in geodesics[1:]:
            points = gs.to_ndarray(points, to_ndim=2)
            sphere.draw_points(ax, points=points, color=color, marker=".")
            writer.grab_frame()

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

def visKen(points_list, color_list, size=10):
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(111)
    for i in range(len(points_list)):
        for points in points_list:
            #points = gs.to_ndarray(points, to_ndim=2)
            for j in range(len(points)):
                ax.scatter(points[j][:, 0], points[j][:, 1], color=color_list[i], alpha=0.5)
#    ax.plot(points[:, 0], points[:, 1], linestyle="dashed")
#    ax.scatter(gs.to_numpy(linear_mean[0]), gs.to_numpy(linear_mean[1]), label="Mean", s=80, alpha=0.5,)
    #ax.set_title("")
    #x.legend()
    plt.show()

def load_data():
    return np.load('rat_skulls.npy')