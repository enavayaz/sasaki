import matplotlib.pylab as plt
import matplotlib.animation as animation
import geomstats.visualization as visualization
from geomstats.geometry.riemannian_metric import RiemannianMetric
import geomstats.backend as gs
import logging
import numpy as np
#matplotlib.use("Agg")  # NOQA

def gradient_descent(pu_ini, grad, exp, loss=None, lrate=0.01, max_iter=256, tol=1e-6):
    """
    Apply a gradient descent until either max_iter or a given tolerance is reached.
    """
    L = len(pu_ini)
    pu = pu_ini
    for i in range(max_iter):
        grad_pu = grad(pu)
        #loss_pu = loss(pu)
        grad_norm = np.linalg.norm(grad_pu)
        if grad_norm < tol:
            #logging.info("[[point,vector]]: %s", pu)
            logging.info("reached tolerance %s", tol)
            logging.info("iterations: %d", i)
            logging.info("|grad|: %s", grad_norm)
            #logging.info("energy: %s", loss_pu)
            break
        grad_pu = -lrate * grad_pu
        for j in range(L):
            pu[j] = exp(grad_pu[j], pu[j])
    return pu

def vis(coords):
    plt.plot(coords[0, :], coords[1, :])
    plt.axis('equal'); plt.xlabel('x'); plt.xlabel('y')
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
    #sphere.plot_heatmap(ax, loss)
    points = gs.to_ndarray(geodesics[0], to_ndim=2)
    sphere.add_points(points)
    sphere.draw(ax, color=color, marker=".")
    with writer.saving(fig, out, dpi=dpi):
        for points in geodesics[1:]:
            points = gs.to_ndarray(points, to_ndim=2)
            sphere.draw_points(ax, points=points, color=color, marker=".")
            writer.grab_frame()

def VisGeodesicsTM(geo_list, color_list, size=15):
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(111, projection="3d")
    sphere = visualization.Sphere()
    sphere.draw(ax, marker=".")
    for i in range(len(color_list)):
        for points in geo_list[i]:
            points = gs.to_ndarray(points, to_ndim=2)
            sphere.draw_points(ax, points=points, color=color_list[i], marker=".")
    plt.show()