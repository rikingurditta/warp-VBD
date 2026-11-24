import random

import matplotlib.pyplot as plt
import warp as wp
import warp.sparse as sp
from warp.optim.linear import cg, cr, gmres, bicgstab, preconditioner

from fluid_warp.fluid import Simulation
from fluid_warp.grid import random_init, grid_like, divergence, copy_grid, visualize_vector_field, compute_divergence, \
    serialize_coord, make_grid, Grid
from fluid_warp.init_fields import init_u_circular, init_circular_flow, init_outward_flow, init_taylor_vortex
from util import sparse_to_dense


@wp.kernel
def init_array2d(arr: wp.array2d[float]):
    i, j = wp.tid()
    arr[i, j] = float(i * arr.shape[1] + j)


@wp.kernel
def flatten_using_serialized_index(g: Grid, flattened: wp.array[float]):
    i, j = wp.tid()
    index = serialize_coord(i, j, g)
    flattened[index] = g.q[i, j]


def test_flatten():
    g = make_grid(3, 3, 1)
    A = g.q
    wp.launch(kernel=init_array2d, dim=A.shape, inputs=[A], device='cuda')
    print(A)
    # shows that flatten() works in row-major order
    print('warp flatten:', A.flatten())
    a = wp.zeros(A.size, dtype=float)
    wp.launch(kernel=flatten_using_serialized_index, dim=A.shape, inputs=[g, a], device='cuda')
    print('kernel flatten:', a)


def test_div():
    N = 4
    s = Simulation(N, N, 1, 1)

    wp.launch(kernel=random_init,
              dim=s.u.q.shape,
              inputs=[random.randint(0, 1000000000), s.u],
              device='cuda')
    wp.launch(kernel=random_init,
              dim=s.v.q.shape,
              inputs=[random.randint(0, 1000000000), s.v],
              device='cuda')
    print(s.u.q)
    print(s.v.q)

    # div taken with sparse matrices
    d1 = s.div_mat_u @ s.u.q.flatten() + s.div_mat_v @ s.v.q.flatten()

    # div taken with divergence kernel
    d2_grid = grid_like(s.p)
    wp.launch(kernel=divergence,
              dim=d2_grid.q.shape,
              inputs=[s.u, s.v, d2_grid],
              device='cuda')
    d2 = d2_grid.q

    # diff between two div methods
    print(d1 - d2.flatten())


def test_projection():
    N = 128
    s = Simulation(N, N, 1 / N, 1)
    h = 1
    vis_scale = 10
    vis_samples = (30, 30)
    # init_circular_flow(s.u, s.v)
    init_outward_flow(s.u, s.v)
    # init_taylor_vortex(s.u, s.v)
    s.enforce_bcs()

    fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(30, 10))
    visualize_vector_field(s.u, s.v, ax1, vis_samples, scale=vis_scale)
    ax1.title.set_text('before projection')

    p = grid_like(s.p)
    p_vec = p.q.flatten()
    u_vec = s.u.q.flatten()
    v_vec = s.v.q.flatten()

    lapl = sp.bsr_scale(s.div_mat_u @ s.grad_mat_u + s.div_mat_v @ s.grad_mat_v, -1.)

    if N < 7:
        print('div u:', sparse_to_dense(s.div_mat_u))
        print('grad u:', sparse_to_dense(sp.bsr_scale(s.grad_mat_u, 1.)))
        print('div v:', sparse_to_dense(s.div_mat_v))
        print('grad v:', sparse_to_dense(sp.bsr_scale(s.grad_mat_v, 1.)))
        print('lapl:', sparse_to_dense(lapl))

    A = sp.bsr_scale(lapl, h / (s.density * s.density))
    b = 1. / s.density * (s.div_mat_u @ s.mass_u @ u_vec + s.div_mat_v @ s.mass_v @ v_vec)
    print(b)

    [n, rnorm, atol] = cg(A, b, p_vec, atol=1e-12, restart=1, maxiter=500)
    print(f'iterations: {n}\tresidual norm: {rnorm}\tatol: {atol}')
    r = b - A @ p_vec
    print('computed residual norm:', wp.sqrt(wp.utils.array_inner(r, r)))

    u_vec -= h * s.density * s.mass_u @ s.grad_mat_u @ p_vec
    v_vec -= h * s.density * s.mass_v @ s.grad_mat_v @ p_vec

    visualize_vector_field(s.u, s.v, ax2, vis_samples, scale=vis_scale)
    ax2.title.set_text('after projection')

    u_div = grid_like(s.u)
    v_div = grid_like(s.v)
    u_div_vec = h * s.density * s.grad_mat_u @ p_vec
    v_div_vec = h * s.density * s.grad_mat_v @ p_vec
    u_div.q = u_div_vec.reshape(u_div.q.shape)
    v_div.q = v_div_vec.reshape(v_div.q.shape)
    visualize_vector_field(u_div, v_div, ax3, vis_samples, scale=vis_scale)
    ax3.title.set_text('divergent component')

    div = compute_divergence(s.u, s.v)
    # print(div)

    plt.show()


if __name__ == '__main__':
    test_projection()
