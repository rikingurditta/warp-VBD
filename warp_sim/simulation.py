import warp as wp
from warp_sim.mesh import WarpMesh, mat32f
from numpy_sim.mesh import Mesh
from warp_sim.potential_energy import psi_linear, dpsi_dF_linear, d2psi_dF2_linear, MaterialParams

EPSILON = 1e-8


@wp.kernel
def phi(E_dphi_dX: wp.array[mat32f],
        X: wp.array[wp.vec2],
        phi_vec: wp.array[wp.vec3]):
    element = wp.tid()
    out = E_dphi_dX[element] @ X[element]
    out[0] += 1.
    phi_vec[element] = out


@wp.kernel
def deformation_gradient(V: wp.array[wp.vec2],
                         E: wp.array2d[int],
                         E_dphi_dX: wp.array[mat32f],
                         Fs: wp.array[wp.mat22f]):
    element = wp.tid()
    v0 = E[element, 0]
    v1 = E[element, 1]
    v2 = E[element, 2]
    V_el = wp.matrix(shape=(3, 2), dtype=float)
    V_el[0, :] = V[v0]
    V_el[1, :] = V[v1]
    V_el[2, :] = V[v2]
    F = wp.transpose(V_el) @ E_dphi_dX[element]
    Fs[element] = F


@wp.kernel
def E_pot(Fs: wp.array[wp.mat22f],
          E_vol0: wp.array[float],
          params: MaterialParams,
          energy: wp.array[float]):
    element = wp.tid()
    energy[element] = E_vol0[element] * psi_linear(params, Fs[element])


@wp.kernel
def dE_pot_dx(E: wp.array2d[int],
              Fs: wp.array[wp.mat22f],
              E_dphi_dX: wp.array[mat32f],
              E_vol0: wp.array[float],
              params: MaterialParams,
              denergy: wp.array[wp.vec2f]):
    element = wp.tid()

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
    dpsi = dpsi_dF_linear(params, F)  # (4,)
    dE = wp.transpose(B) @ dpsi * E_vol0[element]  # (6,)
    v0 = E[element, 0]
    v1 = E[element, 0]
    v2 = E[element, 0]
    # this should use atomic_add by default so it should be thread-safe
    denergy[v0] += dE[0:2]
    denergy[v1] += dE[2:4]
    denergy[v2] += dE[4:6]


# @wp.kernel
# def d2E_pot_dx2(E: wp.array2d[int],
#                 Fs: wp.array[wp.mat22f],
#                 E_dphi_dX: wp.array[mat32f],
#                 E_vol0: wp.array[float],
#                 params: MaterialParams,
#                 d2energy: wp.array2d[wp.mat22f]):
#     element = wp.tid()
#
#     dphi = E_dphi_dX[element]
#     # B = dF/dx - based on lecture video
#     B = wp.matrix(shape=(4, 6), dtype=float)
#     B[0:2, 0] = dphi[0, :]
#     B[2:4, 1] = dphi[0, :]
#     B[0:2, 2] = dphi[1, :]
#     B[2:4, 3] = dphi[1, :]
#     B[0:2, 4] = dphi[2, :]
#     B[2:4, 5] = dphi[2, :]
#
#     F = Fs[element]
#     d2psi = d2psi_dF2_linear(params, F)  # (4,)
#     d2E = wp.transpose(B) @ d2psi @ B * E_vol0[element]  # (6,6)
#     v0 = E[element, 0]
#     v1 = E[element, 0]
#     v2 = E[element, 0]
#     d2energy[v0, v0] += d2E[0:2, 0:2]
#     d2energy[v0, v1] += d2E[0:2, 2:4]
#     d2energy[v0, v2] += d2E[0:2, 4:6]
#     d2energy[v1, v0] += d2E[2:4, 0:2]
#     d2energy[v1, v1] += d2E[2:4, 2:4]
#     d2energy[v1, v2] += d2E[2:4, 4:6]
#     d2energy[v2, v0] += d2E[4:6, 0:2]
#     d2energy[v2, v1] += d2E[4:6, 2:4]
#     d2energy[v2, v2] += d2E[4:6, 4:6]


class Simulation:
    def __init__(self, m_undeformed: WarpMesh, params: MaterialParams):
        self.num_elements = m_undeformed.E.shape[0]
        self.m_undeformed = m_undeformed
        self.params = params
        self.Fs = wp.empty(shape=num_elements, dtype=wp.mat22f)

    def compute_Fs(self, wm: WarpMesh):
        wp.launch(kernel=deformation_gradient,
                  dim=num_elements,
                  inputs=[wm.V, wm0.E, wm0.E_dphi_dX, self.Fs],
                  device='cuda')

    def E_pot(self) -> float:
        energy = wp.empty(shape=num_elements, dtype=float)
        wp.launch(kernel=E_pot,
                  dim=num_elements,
                  inputs=[self.Fs, self.m_undeformed.E_vol0, self.params, energy],
                  device='cuda')
        return wp.utils.array_sum(energy)
#
#     def E_total(self, m: Mesh, vel: np.ndarray, h: float, m_temp: Mesh) -> float:
#         y = m.x() + h * vel
#         E0_kin = 0.5 / h ** 2 * (m_temp.x() - y).dot(m.M @ (m_temp.x() - y))
#         E0_pot = self.E_pot(m)
#         return E0_kin + E0_pot
#
#     def dE_total_dx(self, m: Mesh, vel: np.ndarray, h: float, m_temp: Mesh) -> npVector:
#         y = m.x() + h * vel
#         dE0_kin = m.M @ (m_temp.x() - y) / h ** 2
#         dE0_pot = self.dE_pot_dx(m)
#         return dE0_kin + dE0_pot
#
#     def d2E_total_dx2(self, m: Mesh, vel: np.ndarray, h: float, m_temp: Mesh) -> npMatrix:
#         d2E0_kin = m.M / h ** 2
#         d2E0_pot = self.d2E_pot_dx2(m)
#         return d2E0_kin + d2E0_pot
#
#     def timestep(self, m: Mesh, m_prev: Mesh, h: float) -> Mesh:
#         """
#         Timestep using Newton's method with backtracking line search
#         :param m: deformed mesh, m^k
#         :param m_prev: deformed mesh at prev timestep, m^{k-1}
#         :param h: timestep size
#         :return: deformed mesh at next timestep, m^{k+1}
#         """
#         vel = (m.x() - m_prev.x()) / h
#         m_new = m.copy()
#         x = m_new.x()  # .x() returns a view, so x refers to the same data as m_new.V
#         x[:] += h * (x - m_prev.x()) / h  # so this line also edits m_new.V
#         # print('energy:', self.E_total(m, m_prev, h, m))
#         g = self.dE_total_dx(m_new, vel, h, m_new)
#         steps = 0
#         while np.linalg.norm(g) > EPSILON and steps < 100:
#             g = self.dE_total_dx(m, vel, h, m_new)
#             H = self.d2E_total_dx2(m, vel, h, m_new)
#             d = np.linalg.solve(H, -g)
#             x0 = x.copy()
#             alpha = 1.0
#             # line search doesn't work, but simulation MOSTLY works without it (seems to be gaining energy)
#             # also, VBD doesn't need line search
#             # E = self.E_total(m, vel, h, m_new)
#             # while alpha > EPSILON:
#             #     x[:] = x0 + alpha * d
#             #     E_step = self.E_total(m, m_prev, h, m_new)
#             #
#             #     # coefficient in front of linear term is a parameter, seems like it changes the condition dramatically
#             #     lin_est = E + 0.1 * alpha * g.dot(d)
#             #
#             #     print(f'alpha={alpha:5.5}\t\tE_step={E_step:5.5}\t\tlin_est={lin_est:5.5}')
#             #
#             #     if E_step >= lin_est:
#             #         # seems like this condition is always being hit D: what's up with that?
#             #         alpha *= 0.5
#             #     else:
#             #         break
#             x[:] = x0 + alpha * d
#             steps += 1
#         return m_new


if __name__ == '__main__':
    m0 = Mesh(filename='../test_mesh.txt')
    m0.precompute()
    m_def = m0.copy()
    m_def.V[:, 1] *= 2
    num_elements = m0.E.shape[0]
    params = MaterialParams()
    params.mu_lame = 1.
    params.lambda_lame = 1.
    wm0 = WarpMesh(m0)
    wm0.precompute()
    wm_def = WarpMesh(m_def)

    s = Simulation(wm0, params)
    s.compute_Fs(wm_def)
    print(s.E_pot())
