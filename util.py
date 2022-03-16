import matplotlib.pylab as plt
import geomstats.backend as gs
import logging

def gradient_descent(start, loss, grad, manifold=None, lr=0.01, max_iter=256, precision=1e-5):
    """Operate a gradient descent on a given manifold.
    Until either max_iter or a given precision is reached.
    """
    x = start
    for i in range(max_iter):
        x_prev, grad = x, -lr*grad(x)
        for j in range(len(x)):
            pj, uj, gradpj, graduj = x[j][0], x[j][1], grad[j][0], grad[j][1]
            #euclidean_grad = gradpj
            tang_vec = gradpj
            #if manifold is not None:
            #    tangent_vec = manifold.to_tangent(vector=euclidean_grad, base_point=pj)
            x[j][0] = manifold.metric.exp(base_point=pj, tang_vec=tang_vec)

        if gs.abs(loss(x, use_gs=True) - loss(x_prev, use_gs=True)) <= precision:
            logging.info("x: %s", x)
            logging.info("reached precision %s", precision)
            logging.info("iterations: %d", i)
            break
        yield x, loss(x)

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