import numpy as np
from mesh import Mesh
from potential_energy import PotentialEnergy

EPSILON = 1e-8


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



class Simulation:
    def __init__(self, m_undeformed: Mesh, V_pot: PotentialEnergy):
        self.m_undeformed = m_undeformed
        self.V_pot = V_pot

    def E_pot(self, m: Mesh) -> float:
        """
        sum up potential energy for each element of mesh

        :param m: world-space mesh - should have same E as m_undeformed
        :return: potential energy of entire mesh
        """
        E = 0
        for e in range(m.E.shape[0]):
            v0 = m.E[e, 0]
            v1 = m.E[e, 1]
            v2 = m.E[e, 2]
            tri_X = np.zeros((3, 2))
            tri_X[0, :] = self.m_undeformed.V[v0, :]
            tri_X[1, :] = self.m_undeformed.V[v1, :]
            tri_X[2, :] = self.m_undeformed.V[v2, :]
            tri_x = np.zeros((3, 2))
            tri_x[0, :] = m.V[v0, :]
            tri_x[1, :] = m.V[v1, :]
            tri_x[2, :] = m.V[v2, :]
            F = deformation_gradient(tri_X, tri_x)
            E += m.E_vol0[e] * self.V_pot.psi(F)
        return E

    def dE_pot_dx(self, m: Mesh) -> np.ndarray:
        """
        :param m: world-space mesh - should have same E as m_undeformed
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
            tri_X[0, :] = self.m_undeformed.V[v0, :]
            tri_X[1, :] = self.m_undeformed.V[v1, :]
            tri_X[2, :] = self.m_undeformed.V[v2, :]
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
            dpsi = self.V_pot.dpsi_dF(F)  # (4,)
            dE = B.T @ dpsi * m.E_vol0[e]  # (6,)
            dE_dx[v0 * 2:v0 * 2 + 2] += dE[0:2]
            dE_dx[v1 * 2:v1 * 2 + 2] += dE[2:4]
            dE_dx[v2 * 2:v2 * 2 + 2] += dE[4:6]
        return dE_dx

    def d2E_pot_dx2(self, m: Mesh) -> np.ndarray:
        """
        :param m: world-space mesh - should have same E as m_undeformed
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
            tri_X[0, :] = self.m_undeformed.V[v0, :]
            tri_X[1, :] = self.m_undeformed.V[v1, :]
            tri_X[2, :] = self.m_undeformed.V[v2, :]
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
            d2psi = self.V_pot.d2psi_dF2(F)  # (4,)
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

    def E_total(self, m: Mesh, m_prev: Mesh, h: float, m_temp: Mesh) -> float:
        vel = (m.x() - m_prev.x()) / h
        y = m.x() + h * vel
        E0_kin = 0.5 / h ** 2 * (m_temp.x() - y).dot(m.M @ (m_temp.x() - y))
        E0_pot = self.E_pot(m)
        return E0_kin + E0_pot

    def dE_total_dx(self, m: Mesh, m_prev: Mesh, h: float, m_temp: Mesh) -> np.ndarray:
        vel = (m.x() - m_prev.x()) / h
        y = m.x() + h * vel
        dE0_kin = m.M @ (m_temp.x() - y) / h ** 2
        dE0_pot = self.dE_pot_dx(m)
        return dE0_kin + dE0_pot

    def d2E_total_dx2(self, m: Mesh, m_prev: Mesh, h: float, m_temp: Mesh) -> np.ndarray:
        d2E0_kin = m.M / h ** 2
        d2E0_pot = self.d2E_pot_dx2(m)
        return d2E0_kin + d2E0_pot

    def timestep(self, m: Mesh, m_prev: Mesh, h: float) -> Mesh:
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
        x += h * (x - m_prev.x()) / h
        g = self.dE_total_dx(m_new, m_prev, h, m_new)
        steps = 0
        while np.linalg.norm(g) > EPSILON and steps < 10:
            E = self.E_total(m, m_prev, h, m_new)
            g = self.dE_total_dx(m, m_prev, h, m_new)
            H = self.d2E_total_dx2(m, m_prev, h, m_new)
            d = np.linalg.solve(H, -g)
            x0 = x.copy()
            alpha = 1.0
            # while alpha > EPSILON:
            #     x[:] = x0 + alpha * d
            #     E_step = self.E_total(m, m_prev, h, m_new)
            #
            #     # 0.1 here is a parameter, seems like it changes the condition dramatically
            #     lin_est = E + alpha * 0.1 * g.dot(d)
            #
            #     if E_step > lin_est:
            #         # seems like this condition is always being hit D: what's up with that?
            #         alpha *= 0.5
            #     else:
            #         break
            x[:] = x0 + alpha * d
            steps += 1
        return m_new
