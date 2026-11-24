import warp as wp

from fluid_warp.grid import Grid


@wp.kernel
def init_q_fractal(q: Grid):
    i, j = wp.tid()
    # W = float(q.width - 1)
    # H = float(q.height - 1)
    # x = wp.cos(float(i) / W * wp.pi)
    # y = wp.sin(float(j) / H * wp.pi)
    # q.q[i, j] = x * y
    if (i - q.width / 2) & (j - q.height / 2) > 0:
        q.q[i, j] = 1.


@wp.kernel
def init_u_circular(u: Grid):
    i, j = wp.tid()
    W = float(u.width - 1)
    H = float(u.height)
    x = float(i) / W - 0.5
    y = (float(j) + 0.5) / H - 0.5
    u.q[i, j] = -y


@wp.kernel
def init_v_circular(v: Grid):
    i, j = wp.tid()
    W = float(v.width)
    H = float(v.height - 1)
    x = (float(i) + 0.5) / W - 0.5
    y = float(j) / H - 0.5
    v.q[i, j] = x


def init_circular_flow(u: Grid, v: Grid):
    wp.launch(kernel=init_u_circular,
              dim=u.q.shape,
              inputs=[u],
              device='cuda')
    wp.launch(kernel=init_v_circular,
              dim=v.q.shape,
              inputs=[v],
              device='cuda')



@wp.kernel
def init_u_taylor(u: Grid):
    i, j = wp.tid()
    W = float(u.width - 1)
    H = float(u.height)
    x = float(i) / W
    y = (float(j) + 0.5) / H
    u.q[i, j] = wp.sin(wp.pi * x) * wp.cos(wp.pi * y)


@wp.kernel
def init_v_taylor(v: Grid):
    i, j = wp.tid()
    W = float(v.width)
    H = float(v.height - 1)
    x = (float(i) + 0.5) / W
    y = float(j) / H
    v.q[i, j] = -wp.cos(wp.pi * x) * wp.sin(wp.pi * y)


def init_taylor_vortex(u: Grid, v: Grid):
    wp.launch(kernel=init_u_taylor,
              dim=u.q.shape,
              inputs=[u],
              device='cuda')
    wp.launch(kernel=init_v_taylor,
              dim=v.q.shape,
              inputs=[v],
              device='cuda')


@wp.kernel
def init_u_outward(u: Grid):
    i, j = wp.tid()
    W = float(u.width - 1)
    x = (float(i) - 0.5 * W) / W
    u.q[i, j] = x


@wp.kernel
def init_v_outward(v: Grid):
    i, j = wp.tid()
    H = float(v.height - 1)
    y = (float(j) - 0.5 * H) / H
    v.q[i, j] = y


def init_outward_flow(u: Grid, v: Grid):
    wp.launch(kernel=init_u_outward,
              dim=u.q.shape,
              inputs=[u],
              device='cuda')
    wp.launch(kernel=init_v_outward,
              dim=v.q.shape,
              inputs=[v],
              device='cuda')
