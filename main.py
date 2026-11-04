from potential_energy import *
from mesh import Mesh

EPSILON = 1e-8

import numpy as np
# import warp as wp
# import warp.render


# wp.init()
# renderer = warp.render.OpenGLRenderer(draw_sky=False, draw_grid=False, draw_axis=False)


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
    dE_dx = np.zeros_like(m.x)
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
    d2E_dx2 = np.zeros((m.x.size, m.x.size))
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


if __name__ == "__main__":
    p0 = np.array([0, 0])
    p1 = np.array([1, 0])
    p2 = np.array([0.5, 1])
    V_undeformed = np.vstack([p0.reshape((1, 2)), p1.reshape((1, 2)), p2.reshape((1, 2))])
    E = np.array([[0, 1, 2]])
    m0 = Mesh(V=V_undeformed, E=E)
    m0.precompute()
    # check that volumes were precomputed properly
    print('vols:\n', m0.E_vol0)

    # create copy of m0 with all vertices rotated
    m_deformed = Mesh(V=V_undeformed, E=E)
    m_deformed.precompute()
    theta = 0
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    centre = np.array([0, 0])
    trans = np.array([0, 0])
    for v in range(m_deformed.V.shape[0]):
        m_deformed.V[v, :] = rot @ (m_deformed.V[v, :] - centre) + centre + trans

    # n = Neohookean(1, 1)
    n = LinearElasticity(1, 1)

    print('V:\n', m0.V)
    print('E:\n', m0.E)
    print('V deformed:\n', m_deformed.V)
    print()

    print('E:\n', E_pot(m0, m_deformed, n),
          '\ng:\n', dE_pot_dx(m0, m_deformed, n),
          '\nH:\n', d2E_pot_dx2(m0, m_deformed, n))

    # BAD!! looks like g is not 0 around rest position

    # each column is a colour
    # Colours = np.identity(3)
    # points_array = []
    # colours_array = []
    # for i in np.arange(0, 1, 0.01):
    #     for j in np.arange(0, 1, 0.01):
    #         bar = phi(V_undeformed, np.array([i, j]))
    #         if np.linalg.norm(bar - np.clip(bar, 0, 1)) < EPSILON:
    #             points_array.append(np.array([i, j, 0]))
    #             colours_array.append(Colours @ bar)
    # while renderer.is_running():
    #     renderer.begin_frame()
    #     renderer.render_points(name="barycentric coordinates",
    #                            radius=0.01,
    #                            points=np.array(points_array),
    #                            colors=np.clip(np.array(colours_array), 0, 1))
    #     renderer.end_frame()
