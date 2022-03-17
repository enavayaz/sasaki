import matplotlib.pylab as plt
import matplotlib.animation as animation
import geomstats.visualization as visualization
from geomstats.geometry.riemannian_metric import RiemannianMetric
import geomstats.backend as gs
import logging
#matplotlib.use("Agg")  # NOQA

def gradient_descent(start, loss, grad, metric: RiemannianMetric, lr=0.01, max_iter=256, tol=1e-6):
    """Operate a gradient descent on a given manifold.
    Until either max_iter or a given precision is reached.
    """
    x = start
    for i in range(max_iter):
        x_prev, grad = x, grad(x)
        for j in range(len(x)):
            pj, uj, gradpj, graduj = x[j][0], x[j][1], -lr*grad[0][j], -lr*grad[1][j]
            #euclidean_grad = gradpj
            #tang_vec = gradpj
            #if manifold is not None:
            #    tang_vec = manifold.to_tangent(vector=euclidean_grad, base_point=pj)
            #x[j][0] = manifold.metric.exp(base_point=pj, tang_vec=tang_vec)
            x[j][0] = metric.exp(base_point=pj, tangent_vec=gradpj)
            x[j][1] = metric.parallel_transport(tangent_vec=uj+graduj, base_point=pj, end_point=x[j][0])

        if gs.abs(loss(x) - loss(x_prev)) <= tol:
            logging.info("x: %s", x)
            logging.info("reached tolerance %s", tol)
            logging.info("iterations: %d", i)
            break
        #yield x, loss(x)
    return x, loss(x)

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
    geodesics, loss=None, size=20, fps=10, dpi=100, out="out.mp4", color="red"
):
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

def plot_and_save_Image(
        geodesics, loss=None, size=20, fps=10, dpi=100, out="out.png", color="red"
):
    """Render a set of geodesics and save it to an mpeg 4 file."""
    #writer = FFMpegWriter(fps=fps)
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(111, projection="3d")
    sphere = visualization.Sphere()
    #sphere.plot_heatmap(ax, loss)
    # points = gs.to_ndarray(geodesics[0], to_ndim=2)
    # sphere.add_points(points)
    sphere.draw(ax, color=color, marker=".")
    for points in geodesics[1:]:
        points = gs.to_ndarray(points, to_ndim=2)
        sphere.draw_points(ax, points=points, color=color, marker=".")
        #writer.grab_frame()

def VisGeodesicsTM(geo_in,geo_out,color_in, color_out, size=15):
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(111, projection="3d")
    sphere = visualization.Sphere()
    sphere.draw(ax, marker=".")
    for points in geo_in[1:]:
        points = gs.to_ndarray(points, to_ndim=2)
        sphere.draw_points(ax, points=points, color=color_in, marker=".")
    for points in geo_out[1:]:
        points = gs.to_ndarray(points, to_ndim=2)
        sphere.draw_points(ax, points=points, color=color_out, marker=".")
    plt.show()