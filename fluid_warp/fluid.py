import warp as wp
import warp.sparse as sp
from warp.optim.linear import cg

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from fluid_warp.grid import Grid, make_grid, copy_grid, advect_grid, serialize_coord
from fluid_warp.init_fields import init_q_fractal, init_u_circular, init_v_circular, init_circular_flow, \
    init_taylor_vortex


@wp.kernel
def construct_div_mat(div: Grid, u: Grid, v: Grid,
                      rows_u: wp.array[int], cols_u: wp.array[int], vals_u: wp.array[float],
                      rows_v: wp.array[int], cols_v: wp.array[int], vals_v: wp.array[float]):
    i, j = wp.tid()
    div_index = serialize_coord(i, j, div)
    rows_u[div_index * 2] = div_index
    cols_u[div_index * 2] = serialize_coord(i, j, u)
    vals_u[div_index * 2] = -1. / u.dx
    rows_u[div_index * 2 + 1] = div_index
    cols_u[div_index * 2 + 1] = serialize_coord(i + 1, j, u)
    vals_u[div_index * 2 + 1] = 1. / u.dx
    rows_v[div_index * 2] = div_index
    cols_v[div_index * 2] = serialize_coord(i, j, v)
    vals_v[div_index * 2] = -1. / v.dx
    rows_v[div_index * 2 + 1] = div_index
    cols_v[div_index * 2 + 1] = serialize_coord(i, j + 1, v)
    vals_v[div_index * 2 + 1] = 1. / v.dx


class Simulation:
    width: int
    height: int
    dx: float
    q: Grid  # some scalar quantity to be advected around

    # velocity staggered grid
    u: Grid  # x vel
    v: Grid  # y vel

    p: Grid  # scalar pressure field

    def __init__(self, width: int, height: int, dx: float, density: float):
        self.width = width
        self.height = height
        self.dx = dx
        self.density = density
        self.q = make_grid(width + 1, height + 1, dx)
        self.u = make_grid(width + 1, height, dx)
        self.v = make_grid(width, height + 1, dx)
        self.p = make_grid(width, height, dx)

        mass_u = wp.ones_like(self.u.q)
        mass_u[0, :].fill_(0.)
        mass_u[self.width, :].fill_(0.)
        mass_v = wp.ones_like(self.v.q)
        mass_v[:, 0].fill_(0.)
        mass_v[:, self.height].fill_(0.)
        print(mass_u)
        print(mass_v)
        self.mass_u = sp.bsr_diag(mass_u.flatten(), self.u.q.size, self.u.q.size, float)
        self.mass_v = sp.bsr_diag(mass_v.flatten(), self.v.q.size, self.v.q.size, float)

        div_u_rows = wp.zeros(2 * self.p.q.size, dtype=int)
        div_u_cols = wp.zeros(2 * self.p.q.size, dtype=int)
        div_u_vals = wp.zeros(2 * self.p.q.size, dtype=float)
        div_v_rows = wp.zeros(2 * self.p.q.size, dtype=int)
        div_v_cols = wp.zeros(2 * self.p.q.size, dtype=int)
        div_v_vals = wp.zeros(2 * self.p.q.size, dtype=float)
        wp.launch(kernel=construct_div_mat,
                  dim=self.p.q.shape,
                  inputs=[self.p, self.u, self.v,
                          div_u_rows, div_u_cols, div_u_vals,
                          div_v_rows, div_v_cols, div_v_vals],
                  device='cuda')
        self.div_mat_u = sp.bsr_from_triplets(self.p.q.size, self.u.q.size, div_u_rows, div_u_cols, div_u_vals)
        self.div_mat_v = sp.bsr_from_triplets(self.p.q.size, self.v.q.size, div_v_rows, div_v_cols, div_v_vals)
        self.grad_mat_u = -self.div_mat_u.transpose()
        self.grad_mat_v = -self.div_mat_v.transpose()
        self.lapl_mat = sp.bsr_scale(self.div_mat_u @ self.grad_mat_u + self.div_mat_v @ self.grad_mat_v, -1.)

    def enforce_bcs(self):
        # should do the same thing as multiplying by precomputed mass matrices
        self.u.q[0, :].fill_(0.)
        self.u.q[self.width, :].fill_(0.)
        self.v.q[:, 0].fill_(0.)
        self.v.q[:, self.height].fill_(0.)

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

    def compute_pressure(self, h: float):
        # DOESN'T WORK RIGHT NOW
        p_vec = self.p.q.flatten()
        A = sp.bsr_scale(self.lapl_mat, h / (self.density * self.density))
        b = 1. / self.density * self.div_mat_u @ self.u.q + self.div_mat_v @ self.v.q
        cg(A, b, p_vec, atol=1e-12)

    def project(self, h: float):
        p_vec = self.p.q.flatten()
        u_vec = self.u.q.flatten()
        v_vec = self.v.q.flatten()
        u_vec -= h * self.density * self.grad_mat_u @ p_vec
        v_vec -= h * self.density * self.grad_mat_v @ p_vec

    def timestep(self, h: float):
        self.advect(h)
        self.enforce_bcs()
        self.compute_pressure(h)
        self.project(h)


if __name__ == '__main__':
    N = 2 ** 8
    max_frames = 10000
    s = Simulation(N, N, 1, 1)
    dt = 0.01
    wp.launch(kernel=init_q_fractal,
              dim=(s.q.width, s.q.height),
              inputs=[s.q],
              device='cuda')
    init_taylor_vortex(s.u, s.v)
    matplotlib.use('TkAgg')
    fig = plt.figure(figsize=(15, 15))
    im = plt.imshow(s.q.q.numpy().transpose()[:, ::-1], cmap='coolwarm')


    def animate(i):
        print(f'frame: {i}')
        s.timestep(dt)
        # s.advect(dt)
        im.set_array(s.q.q.numpy().transpose()[:, ::-1])
        return im,


    anim = animation.FuncAnimation(fig, animate, interval=100, frames=max_frames, repeat=False)
    # anim.save('fluid.gif', fps=30)
    plt.show()
