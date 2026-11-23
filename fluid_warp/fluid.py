import warp as wp

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation

from fluid_warp.grid import Grid, make_grid, copy_grid, advect_grid, grid_like, divergence


class Simulation:
    width: int
    height: int
    dx: float
    q: Grid  # some scalar quantity to be advected around

    # velocity staggered grid
    u: Grid  # x vel
    v: Grid  # y vel

    p: Grid  # scalar pressure field

    def __init__(self, width: int, height: int, dx: float):
        self.width = width
        self.height = height
        self.dx = dx
        self.q = make_grid(width + 1, height + 1, dx)
        self.u = make_grid(width + 1, height, dx)
        self.v = make_grid(width, height + 1, dx)
        self.p = make_grid(width, height, dx)

    def advect(self, h: float):
        new_q = copy_grid(self.q)
        new_u = copy_grid(self.u)
        new_v = copy_grid(self.v)
        wp.launch(kernel=advect_grid,
                  dim=(self.q.width, self.q.height),
                  inputs=[self.q, self.u, self.v, h, new_q],
                  device='cuda')
        wp.launch(kernel=advect_grid,
                  dim=(self.u.width, self.v.height),
                  inputs=[self.u, self.u, self.v, h, new_u],
                  device='cuda')
        wp.launch(kernel=advect_grid,
                  dim=(self.v.width, self.v.height),
                  inputs=[self.v, self.u, self.v, h, new_v],
                  device='cuda')
        self.q = new_q
        self.u = new_u
        self.v = new_v


@wp.kernel
def init_q(q: Grid):
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
    H = float(u.height - 1)
    y = (float(j) - 0.5 * H) / H
    u.q[i, j] = -y


@wp.kernel
def init_v_circular(v: Grid):
    i, j = wp.tid()
    W = float(v.width - 1)
    x = (float(i) - 0.5 * W) / W
    v.q[i, j] = x


if __name__ == '__main__':
    N = 1024
    s = Simulation(N, N, 1)
    dt = 0.5
    wp.launch(kernel=init_q,
              dim=(s.q.width, s.q.height),
              inputs=[s.q],
              device='cuda')
    wp.launch(kernel=init_u_circular,
              dim=(s.u.width, s.u.height),
              inputs=[s.u],
              device='cuda')
    wp.launch(kernel=init_v_circular,
              dim=(s.v.width, s.v.height),
              inputs=[s.v],
              device='cuda')
    matplotlib.use('TkAgg')
    fig = plt.figure()
    im = plt.imshow(s.q.q.numpy().transpose())
    def animate(i):
        print(f'frame: {i}')
        s.advect(dt)
        im.set_array(s.q.q.numpy().transpose())
        return im,
    anim = animation.FuncAnimation(fig, animate, interval=1, frames=200, repeat=False)
    anim.save('fluid.gif', fps=30)
    # plt.show()

