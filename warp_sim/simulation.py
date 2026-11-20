from typing import Any

import warp as wp
import warp.optim.linear as linear
import warp.sparse as sparse
import warp.render as render

import numpy as np

from warp_sim.util import sparse_mat_array_to_numpy_blocks

np.set_printoptions(suppress=True)

from numpy_sim.potential_energy import LinearElasticity
from warp_sim.mesh import WarpMesh, mat32f
from numpy_sim.mesh import Mesh
from warp_sim.potential_energy import psi_linear, dpsi_dF_linear, d2psi_dF2_linear, MaterialParams

from numpy_sim.simulation import Simulation as SS

EPSILON = 1e-8


# noinspection PyTypeHints
@wp.kernel
def phi(E_dphi_dX: wp.array[mat32f],
        X: wp.array[wp.vec2],
        phi_vec: wp.array[wp.vec3]):
    element = wp.tid()
    out = E_dphi_dX[element] @ X[element]
    out[0] += 1.
    phi_vec[element] = out


# noinspection PyTypeHints
@wp.kernel
def deformation_gradient(V: wp.array[wp.vec2],
                         E: wp.array2d[int],
                         E_dphi_dX: wp.array[mat32f],
                         Fs: wp.array[wp.mat22f]):
    """
    compute deformation gradient for each element of mesh
    :param V: vertex positions
    :param E: elements (triangles)
    :param E_dphi_dX: derivative of shape function for each element
    :param Fs: output array
    :return:
    """
    element = wp.tid()
    V_el = wp.matrix(shape=(3, 2), dtype=float)  # vertex positions for element
    V_el[0, :] = V[E[element, 0]]
    V_el[1, :] = V[E[element, 1]]
    V_el[2, :] = V[E[element, 2]]
    F = wp.transpose(V_el) @ E_dphi_dX[element]
    Fs[element] = F


@wp.kernel
def E_pot(Fs: wp.array[wp.mat22f],
          E_vol0: wp.array[float],
          params: MaterialParams,
          energy: wp.array[float]):
    """
    compute potential energy for each element of mesh
    :param Fs: deformation gradient for each element
    :param E_vol0: volume of each element (area of triangle)
    :param params: lamé parameters
    :param energy: output array
    """
    element = wp.tid()
    energy[element] = E_vol0[element] * psi_linear(params, Fs[element])  # call helper based on elasticity model


# noinspection PyTypeHints
@wp.kernel
def dE_pot_dx(E: wp.array2d[int],
              Fs: wp.array[wp.mat22f],
              E_dphi_dX: wp.array[mat32f],
              E_vol0: wp.array[float],
              params: MaterialParams,
              denergy: wp.array[wp.vec2f]):
    """
    compute derivative of energy with respect to position
    :param E: elements
    :param Fs: deformation gradient for each element
    :param E_dphi_dX: shape function derivative for each element
    :param E_vol0: volume of each element
    :param params: lamé parameters
    :param denergy: output array
    """
    element = wp.tid()
    v0 = E[element, 0]
    v1 = E[element, 1]
    v2 = E[element, 2]

    dphi = E_dphi_dX[element]
    # B = dF/dx - based on lecture video
    B = wp.matrix(shape=(4, 6), dtype=float)
    B[0:2, 0] = dphi[0, :]
    B[2:4, 1] = dphi[0, :]
    B[0:2, 2] = dphi[1, :]
    B[2:4, 3] = dphi[1, :]
    B[0:2, 4] = dphi[2, :]
    B[2:4, 5] = dphi[2, :]

    F = Fs[element]
    dpsi = dpsi_dF_linear(params, F)  # (4,) - call helper based on elasticity model
    dE = wp.transpose(B) @ dpsi * E_vol0[element]  # (6,)
    # assemble gradient to vertices rather than just per element
    # this should use atomic_add by default so it should be thread-safe
    denergy[v0] += dE[0:2]
    denergy[v1] += dE[2:4]
    denergy[v2] += dE[4:6]


# noinspection PyTypeHints
@wp.kernel
def d2E_pot_dx2(E: wp.array2d[int],
                Fs: wp.array[wp.mat22f],
                E_dphi_dX: wp.array[mat32f],
                E_vol0: wp.array[float],
                params: MaterialParams,
                rows: wp.array[int],
                cols: wp.array[int],
                vals: wp.array[wp.mat22f]
                ):
    """
    populate rows, cols, vals that can be used to assemble sparse matrix
    :param E: elements
    :param Fs: deformation gradient for each element
    :param E_dphi_dX: derivative of shape function for each element
    :param E_vol0: volume of each element
    :param params: lamé parameters
    :param rows: row indices
    :param cols: col indices
    :param vals: vals[i] is matrix block to put at (rows[i], cols[i])
    """
    element = wp.tid()
    v0 = E[element, 0]
    v1 = E[element, 1]
    v2 = E[element, 2]

    dphi = E_dphi_dX[element]
    # B = dF/dx - based on lecture video
    B = wp.matrix(shape=(4, 6), dtype=float)
    B[0:2, 0] = dphi[0, :]
    B[2:4, 1] = dphi[0, :]
    B[0:2, 2] = dphi[1, :]
    B[2:4, 3] = dphi[1, :]
    B[0:2, 4] = dphi[2, :]
    B[2:4, 5] = dphi[2, :]

    F = Fs[element]
    d2psi = d2psi_dF2_linear(params, F)  # (4,)
    d2E = wp.transpose(B) @ d2psi @ B * E_vol0[element]  # (6,)

    rows[element * 9 + 0] = v0
    cols[element * 9 + 0] = v0
    vals[element * 9 + 0] = d2E[0:2, 0:2]
    rows[element * 9 + 1] = v0
    cols[element * 9 + 1] = v1
    vals[element * 9 + 1] = d2E[0:2, 2:4]
    rows[element * 9 + 2] = v0
    cols[element * 9 + 2] = v2
    vals[element * 9 + 2] = d2E[0:2, 4:6]
    rows[element * 9 + 3] = v1
    cols[element * 9 + 3] = v0
    vals[element * 9 + 3] = d2E[2:4, 0:2]
    rows[element * 9 + 4] = v1
    cols[element * 9 + 4] = v1
    vals[element * 9 + 4] = d2E[2:4, 2:4]
    rows[element * 9 + 5] = v1
    cols[element * 9 + 5] = v2
    vals[element * 9 + 5] = d2E[2:4, 4:6]
    rows[element * 9 + 6] = v2
    cols[element * 9 + 6] = v0
    vals[element * 9 + 6] = d2E[4:6, 0:2]
    rows[element * 9 + 7] = v2
    cols[element * 9 + 7] = v1
    vals[element * 9 + 7] = d2E[4:6, 2:4]
    rows[element * 9 + 8] = v2
    cols[element * 9 + 8] = v2
    vals[element * 9 + 8] = d2E[4:6, 4:6]


class Simulation:
    m0: WarpMesh  # rest position mesh
    params: MaterialParams  # lamé parameters
    Fs: wp.array[wp.mat22f]  # temp variable to store deformation gradients for each element

    def __init__(self, m_undeformed: WarpMesh, params: MaterialParams):
        self.m0 = m_undeformed
        self.params = params
        self.Fs = wp.empty(shape=self.m0.num_elements, dtype=wp.mat22f)

    def compute_Fs(self, wm: WarpMesh):
        wp.launch(kernel=deformation_gradient,
                  dim=self.m0.num_elements,
                  inputs=[wm.x, self.m0.E, self.m0.E_dphi_dX, self.Fs],
                  device='cuda')

    def E_step(self, m_curr: WarpMesh, vel: wp.array[wp.vec2f], h: float, wm: WarpMesh) -> float:
        """
        must call compute_Fs before this!!
        :param m_curr:
        :param vel:
        :param h:
        :param wm:
        :return:
        """
        potential_energy = wp.empty(shape=self.m0.num_elements, dtype=float)
        wp.launch(kernel=E_pot,
                  dim=self.m0.num_elements,
                  inputs=[self.Fs, self.m0.E_vol0, self.params, potential_energy],
                  device='cuda')
        V = wp.utils.array_sum(potential_energy)
        y = m_curr.x + h * vel
        K = 0.5 / h ** 2 * wp.utils.array_inner(wm.x - y, self.m0.M @ (wm.x - y))
        return V + K

    def dE_step_dx(self, m_curr: WarpMesh, vel: wp.array[wp.vec2f], h: float, wm: WarpMesh) -> wp.array[wp.vec2f]:
        """
        must call compute_Fs before this!!
        :param m_curr:
        :param vel:
        :param h:
        :param wm:
        :return:
        """
        dV_dx = wp.zeros(shape=self.m0.num_verts, dtype=wp.vec2f)
        wp.launch(kernel=dE_pot_dx,
                  dim=self.m0.num_elements,
                  inputs=[self.m0.E,
                          self.Fs,
                          self.m0.E_dphi_dX,
                          self.m0.E_vol0,
                          self.params,
                          dV_dx],
                  device='cuda')
        y = m_curr.x + h * vel
        dK_dx = self.m0.M @ (wm.x - y) / h ** 2
        return dV_dx + dK_dx

    def d2E_step_dx2(self, m_curr: WarpMesh, vel: wp.array[wp.vec2f], h: float, wm: WarpMesh) -> sparse.bsr_matrix_t[
        wp.mat22f]:
        """
        must call compute_Fs before this!!
        :param m_curr:
        :param vel:
        :param h:
        :param wm:
        :return:
        """
        rows = wp.zeros(shape=self.m0.num_elements * 9, dtype=int)
        cols = wp.zeros(shape=self.m0.num_elements * 9, dtype=int)
        vals = wp.zeros(shape=self.m0.num_elements * 9, dtype=wp.mat22f)
        wp.launch(kernel=d2E_pot_dx2,
                  dim=self.m0.num_elements,
                  inputs=[self.m0.E,
                          self.Fs,
                          self.m0.E_dphi_dX,
                          self.m0.E_vol0,
                          self.params,
                          rows, cols, vals],
                  device='cuda')
        d2V_dx2 = sparse.bsr_from_triplets(rows_of_blocks=self.m0.num_verts,
                                           cols_of_blocks=self.m0.num_verts,
                                           rows=rows,
                                           columns=cols,
                                           values=vals)
        d2K_dx2 = self.m0.M / h ** 2
        return d2V_dx2 + d2K_dx2

    def timestep(self, m: WarpMesh, m_prev: WarpMesh, h: float) -> WarpMesh:
        """
        Timestep using Newton's method with backtracking line search
        :param m: deformed mesh, m^k
        :param m_prev: deformed mesh at prev timestep, m^{k-1}
        :param h: timestep size
        :return: deformed mesh at next timestep, m^{k+1}
        """
        vel = (m.x - m_prev.x) / h
        m_new = m.copy()
        m_new.x += h * vel  # so this line also edits m_new.V
        # print('energy:', self.E_total(m, m_prev, h, m))
        self.compute_Fs(m_new)
        g = self.dE_step_dx(m, vel, h, m_new)
        steps = 0
        while wp.utils.array_inner(g, g) > EPSILON and steps < 100:
            self.compute_Fs(m_new)
            g = self.dE_step_dx(m, vel, h, m_new)
            H = self.d2E_step_dx2(m, vel, h, m_new)
            d = wp.empty_like(g)
            linear.gmres(H, -g, d)
            m_new.x += d
            # could add line search
            steps += 1
        return m_new


if __name__ == '__main__':
    h = 0.01
    max_frames = 1e10

    wp.init()
    renderer = render.OpenGLRenderer(draw_sky=False,
                                     draw_grid=False,
                                     draw_axis=False,
                                     camera_pos=(2, 0, 10),
                                     render_wireframe=True,
                                     background_color=(1, 1, 1),
                                     screen_width=1024,
                                     screen_height=1024,
                                     headless=False)
    m0 = Mesh(filename='../test_mesh.txt')
    m0.centre_at_zero()
    m_def = m0.copy()
    m_def.V[:, 0] += 0.001
    m_def.V[:, 1] *= 1.01

    p = MaterialParams()
    p.mu_lame = 1.
    p.lambda_lame = 1.

    wm0 = WarpMesh(m0)
    m_curr = WarpMesh(m_def)
    m_prev = wm0

    s = Simulation(wm0, p)

    frame = 0
    while renderer.is_running() and frame < max_frames:
        m_next = s.timestep(m_curr, m_prev, h)
        m_prev = m_curr.copy()
        m_curr = m_next.copy()

        renderer.begin_frame()
        renderer.render_mesh('undeformed', m0.V_3d(), m0.E, colors=(1., 0., 0.))
        renderer.render_mesh('deformed', m_curr.V_3d(), m0.E, colors=(0., 0., 1.))
        renderer.end_frame()
        print(f'frame: {frame}')
        frame += 1