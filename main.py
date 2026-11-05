from vtkmodules.vtkCommonMisc import VTK_PARSER_OR

from potential_energy import *
from mesh import Mesh

import numpy as np

import warp as wp
import warp.render

EPSILON = 1e-8

wp.init()
renderer = warp.render.OpenGLRenderer(draw_sky=False,
                                      draw_grid=False,
                                      draw_axis=False,
                                      camera_pos=(0, 0, 4),
                                      render_wireframe=True,
                                      background_color=(1, 1, 1))


def phi(V_undeformed: np.ndarray, X: np.ndarray):
    """
    :param V_undeformed: (3,2) array, rows are positions of vertices of undeformed element
    :param X: (2,) input position
    :return: barycentric coordinates of X
    """
    p0 = V_undeformed[0, :]
    p1 = V_undeformed[1, :]
    p2 = V_undeformed[2, :]
    T = np.zeros((2, 2))
    T[:, 0] = p1 - p0
    T[:, 1] = p2 - p0
    y = np.linalg.solve(T, X - p0)
    l1 = y[0]
    l2 = y[1]
    return np.array([1 - l1 - l2, l1, l2])


def dphi_dX(V_undeformed: np.ndarray) -> np.ndarray:
    """
    :param V_undeformed: (3,2) array, rows are positions of vertices of undeformed element
    :return: (3,2) matrix, derivative of barycentric coordinates wrt X
    (barycentric coordinates are linear so this is a constant matrix which is why there is no X input)
    """
    p0 = V_undeformed[0, :]
    p1 = V_undeformed[1, :]
    p2 = V_undeformed[2, :]
    T = np.zeros((2, 2))
    T[:, 0] = p1 - p0
    T[:, 1] = p2 - p0
    Tinv = np.linalg.inv(T)
    out = np.zeros((3, 2))
    out[0, :] = -np.ones((1, 2)) @ Tinv
    out[1:, :] = Tinv
    return out


def deformation_gradient(V_undeformed: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    :param V_undeformed: (3,2) array, rows are positions of vertices of undeformed element
    :param V: (3,2) array, rows are positions of vertices of deformed element
    :return: (2,2) deformation gradient F for element, aka dx/dX
    """
    return V.T @ dphi_dX(V_undeformed)


def E_pot(m_undeformed: Mesh, m: Mesh, V_pot: PotentialEnergy) -> float:
    """
    sum up potential energy for each element of mesh

    :param m_undeformed: object-space mesh
    :param m: world-space mesh - should have same E as m_undeformed
    :param V_pot: potential energy object
    :return: potential energy of entire mesh
    """
    E = 0
    for e in range(m.E.shape[0]):
        v0 = m.E[e, 0]
        v1 = m.E[e, 1]
        v2 = m.E[e, 2]
        tri_X = np.zeros((3, 2))
        tri_X[0, :] = m_undeformed.V[v0, :]
        tri_X[1, :] = m_undeformed.V[v1, :]
        tri_X[2, :] = m_undeformed.V[v2, :]
        tri_x = np.zeros((3, 2))
        tri_x[0, :] = m.V[v0, :]
        tri_x[1, :] = m.V[v1, :]
        tri_x[2, :] = m.V[v2, :]
        F = deformation_gradient(tri_X, tri_x)
        E += m.E_vol0[e] * V_pot.psi(F)
    return E


def dE_pot_dx(m_undeformed: Mesh, m: Mesh, V_pot: PotentialEnergy) -> np.ndarray:
    """
    :param m_undeformed: object-space mesh
    :param m: world-space mesh - should have same E as m_undeformed
    :param V_pot: potential energy object
    :return: gradient of potential energy wrt vertices of mesh
    """
    dE_dx = np.zeros_like(m.x())
    for e in range(m.E.shape[0]):
        # vertex indices for current element
        v0 = m.E[e, 0]
        v1 = m.E[e, 1]
        v2 = m.E[e, 2]
        # undeformed element positions
        tri_X = np.zeros((3, 2))
        tri_X[0, :] = m_undeformed.V[v0, :]
        tri_X[1, :] = m_undeformed.V[v1, :]
        tri_X[2, :] = m_undeformed.V[v2, :]
        # deformed element positions
        tri_x = np.zeros((3, 2))
        tri_x[0, :] = m.V[v0, :]
        tri_x[1, :] = m.V[v1, :]
        tri_x[2, :] = m.V[v2, :]

        dphi = dphi_dX(tri_X)  # (3,2)
        # B = dF/dx - based on lecture video
        B = np.zeros((4, 6))
        B[0:2, 0] = dphi[0, :]
        B[2:4, 1] = dphi[0, :]
        B[0:2, 2] = dphi[1, :]
        B[2:4, 3] = dphi[1, :]
        B[0:2, 4] = dphi[2, :]
        B[2:4, 5] = dphi[2, :]

        F = deformation_gradient(tri_X, tri_x)  # (2,2)
        dpsi = V_pot.dpsi_dF(F)  # (4,)
        dE = B.T @ dpsi * m.E_vol0[e]  # (6,)
        dE_dx[v0 * 2:v0 * 2 + 2] += dE[0:2]
        dE_dx[v1 * 2:v1 * 2 + 2] += dE[2:4]
        dE_dx[v2 * 2:v2 * 2 + 2] += dE[4:6]
    return dE_dx


def d2E_pot_dx2(m_undeformed: Mesh, m: Mesh, V_pot: PotentialEnergy) -> np.ndarray:
    """
    :param m_undeformed: object-space mesh
    :param m: world-space mesh - should have same E as m_undeformed
    :param V_pot: potential energy object
    :return: hessian of potential energy wrt vertices of mesh
    """
    d2E_dx2 = np.zeros((m.x().size, m.x().size))
    for e in range(m.E.shape[0]):
        # vertex indices for current element
        v0 = m.E[e, 0]
        v1 = m.E[e, 1]
        v2 = m.E[e, 2]
        # undeformed element positions
        tri_X = np.zeros((3, 2))
        tri_X[0, :] = m_undeformed.V[v0, :]
        tri_X[1, :] = m_undeformed.V[v1, :]
        tri_X[2, :] = m_undeformed.V[v2, :]
        # deformed element positions
        tri_x = np.zeros((3, 2))
        tri_x[0, :] = m.V[v0, :]
        tri_x[1, :] = m.V[v1, :]
        tri_x[2, :] = m.V[v2, :]

        dphi = dphi_dX(tri_X)  # (3,2)
        # B = dF/dx - based on lecture video
        B = np.zeros((4, 6))
        B[0:2, 0] = dphi[0, :]
        B[2:4, 1] = dphi[0, :]
        B[0:2, 2] = dphi[1, :]
        B[2:4, 3] = dphi[1, :]
        B[0:2, 4] = dphi[2, :]
        B[2:4, 5] = dphi[2, :]

        F = deformation_gradient(tri_X, tri_x)  # (2,2)
        d2psi = V_pot.d2psi_dF2(F)  # (4,)
        d2E = B.T @ d2psi @ B * m.E_vol0[e]  # (6,6)
        d2E_dx2[v0 * 2:v0 * 2 + 2, v0 * 2:v0 * 2 + 2] += d2E[0:2, 0:2]
        d2E_dx2[v0 * 2:v0 * 2 + 2, v1 * 2:v1 * 2 + 2] += d2E[0:2, 2:4]
        d2E_dx2[v0 * 2:v0 * 2 + 2, v2 * 2:v2 * 2 + 2] += d2E[0:2, 4:6]
        d2E_dx2[v1 * 2:v1 * 2 + 2, v0 * 2:v0 * 2 + 2] += d2E[2:4, 0:2]
        d2E_dx2[v1 * 2:v1 * 2 + 2, v1 * 2:v1 * 2 + 2] += d2E[2:4, 2:4]
        d2E_dx2[v1 * 2:v1 * 2 + 2, v2 * 2:v2 * 2 + 2] += d2E[2:4, 4:6]
        d2E_dx2[v2 * 2:v2 * 2 + 2, v0 * 2:v0 * 2 + 2] += d2E[4:6, 0:2]
        d2E_dx2[v2 * 2:v2 * 2 + 2, v1 * 2:v1 * 2 + 2] += d2E[4:6, 2:4]
        d2E_dx2[v2 * 2:v2 * 2 + 2, v2 * 2:v2 * 2 + 2] += d2E[4:6, 4:6]
    return d2E_dx2


def E_total(m_undeformed: Mesh, m: Mesh, m_prev: Mesh, V_pot: PotentialEnergy, h: float) -> float:
    vel = (m.V - m_prev.V) / h
    y = m.V + h * vel
    E_kin = 0.5 / h ** 2 * (m.V - y).flatten().dot(m.M @ (m.V - y).flatten())
    return E_kin + E_pot(m_undeformed, m, V_pot)


def dE_total_dx(m_undeformed: Mesh, m: Mesh, m_prev: Mesh, V_pot: PotentialEnergy, h: float) -> np.ndarray:
    vel = (m.V - m_prev.V) / h
    y = m.V + h * vel
    dE_kin = m.M @ (m.V - y).flatten() / h ** 2
    return dE_kin + dE_pot_dx(m_undeformed, m, V_pot)


def d2E_total_dx2(m_undeformed: Mesh, m: Mesh, m_prev: Mesh, V_pot: PotentialEnergy, h: float) -> np.ndarray:
    d2E_kin = m.M / h ** 2
    return d2E_kin + d2E_pot_dx2(m_undeformed, m, V_pot)


def timestep(m_undeformed: Mesh, m: Mesh, m_prev: Mesh, V_pot: PotentialEnergy, h: float) -> Mesh:
    """
    Timestep using Newton's method with backtracking line search
    :param m_undeformed: undeformed mesh
    :param m: deformed mesh, m^k
    :param m_prev: deformed mesh at prev timestep, m^{k-1}
    :param V_pot: potential energy function
    :param h: timestep size
    :return: deformed mesh at next timestep, m^{k+1}
    """
    m_new = m.copy()
    x = m_new.x()
    g = dE_total_dx(m_undeformed, m_new, m_prev, V_pot, h)
    steps = 0
    while np.linalg.norm(g) > EPSILON and steps < 2:
        # print(f'step {steps}')
        E = E_total(m_undeformed, m_new, m_prev, V_pot, h)
        g = dE_total_dx(m_undeformed, m_new, m_prev, V_pot, h)
        H = d2E_total_dx2(m_undeformed, m_new, m_prev, V_pot, h)
        d = np.linalg.solve(H, -g)
        x0 = x.copy()
        alpha = 1.0
        while alpha > EPSILON:
            x[:] = x0 + alpha * d
            E_step = E_total(m_undeformed, m_new, m_prev, V_pot, h)

            # 0.1 here is a parameter, seems like it changes the condition dramatically
            lin_est = E + alpha * 0.1 * g.dot(d)

            # print(f'alpha: {np.log10(alpha):>5.5}\t\tE step: {E_step:>5.8}\t\tE lin est: {lin_est:>5.8}\t\tdiff: {E_step - lin_est:>5.8}')

            if E_step > lin_est:
                # seems like this condition is always being hit D: what's up with that?
                alpha *= 0.5
            else:
                break
        x[:] = x0 + alpha * d
        steps += 1
    return m_new

if __name__ == "__main__":
    m0 = Mesh(filename='test_mesh.txt')
    m0.centre_at_zero()
    m0.precompute()

    # create copy of m0
    m_deformed = m0.copy()
    m_deformed.precompute()
    m_deformed.V[:, 1] *= 3

    # V_pot = Neohookean(1, 1)  # looks like g isn't 0 around rest ......... this is a problem for later
    V_pot = LinearElasticity(1, 1)

    print('V:\n', m0.V)
    print('E:\n', m0.E)
    print('V deformed:\n', m_deformed.V)
    print()

    EE = E_pot(m0, m_deformed, V_pot)
    gg = dE_pot_dx(m0, m_deformed, V_pot)
    HH = d2E_pot_dx2(m0, m_deformed, V_pot)

    print(f'E:\t\t{EE}',
          f'|g|:\t{np.linalg.norm(gg)}',
          f'H:\n{HH}',
          sep='\n')

    cols = np.random.rand(m_deformed.V.shape[0], 3)
    m_prev = m_deformed.copy()

    frame = 0
    t = renderer.time
    while renderer.is_running():
        # print('frame:', frame)
        renderer.begin_frame()
        t_new = renderer.time
        print(t_new)
        m_next = timestep(m0, m_deformed, m_prev, V_pot, 0.05)
        m_prev = m_deformed.copy()
        m_deformed = m_next
        renderer.render_mesh('undeformed', m0.V_3d(), m0.E, colors=(1, 1, 1))
        renderer.render_mesh('deformed', m_deformed.V_3d(), m_deformed.E)
        renderer.end_frame()
        t = t_new
        frame += 1
