EPSILON = 1e-8

import numpy as np
import warp as wp
import warp.render
from neohookean import Neohookean

wp.init()
renderer = warp.render.OpenGLRenderer(draw_sky=False, draw_grid=False, draw_axis=False)


class Mesh:
    """
    right now just for 2d meshes
    reads an OBJ file and discards z coordinate
    """

    def __init__(self, filename: str = None, V: np.ndarray = None, E: np.ndarray = None):
        if filename is not None:
            vertices = []
            edges = []
            with open(filename, "r") as f:
                lines = f.readlines()
                for line in lines:
                    line_type = line[0]
                    rest = line[2:]
                    if line_type == "v":
                        vertices.append([np.float64(x) for x in rest.split(" ")])
                    if line_type == "e":
                        edges.append([np.int64(x) for x in rest.split(" ")])
            V = np.array(vertices, dtype=np.float64)
            self.V = V[:, :2]
            self.E = np.array(edges, dtype=np.int64)
        elif V is not None and E is not None:
            self.V = V
            self.E = E
        else:
            raise Exception()

        self.V_to_E = [[] for v in range(self.V.shape[0])]
        for i in range(self.E.shape[0]):
            for v in self.E[i, :]:
                self.V_to_E[v].append(i)
        print(self.V_to_E)


def phi(V_undeformed: np.ndarray, X: np.ndarray):
    """
    barycentric coordinates
    expects:
    - p0, p1, p2 are vertices of the triangle
    - X is input position

    returns barycentric coordinates of X
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


# derivative of barycentric coordinates wrt x
def dphi_dX(V_undeformed: np.ndarray) -> np.ndarray:
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
    expects:
    - V_undeformed is a 3x2 array, each row is coordinates of a vertex of the undeformed triangle
    - V            is a 3x2 array, each row is coordinates of a vertex of the deformed triangle
    """
    return V.T * dphi_dX(V_undeformed)


def E_neohookean_mesh(m_undeformed: Mesh, m: Mesh, mu_lamé: float, lambda_lamé: float) -> float:
    """
    sum up neohookean energy for each element of mesh

    :param m_undeformed: object-space mesh
    :param m: world-space mesh - should have same E as m_undeformed
    :param mu_lamé: Lamé parameter
    :param lambda_lamé: Lamé parameter

    :return: neohookean energy
    """
    neohookean = Neohookean(mu_lamé, lambda_lamé)

    E = 0
    for i in range(m.E.shape[0]):
        v0 = m.E[i, 0]
        v1 = m.E[i, 0]
        v2 = m.E[i, 0]
        tri_X = np.zeros((3, 2))
        tri_X[0, :] = m_undeformed.V[v0, :]
        tri_X[1, :] = m_undeformed.V[v1, :]
        tri_X[2, :] = m_undeformed.V[v2, :]
        tri_x = np.zeros((3, 2))
        tri_x[0, :] = m.V[v0, :]
        tri_x[1, :] = m.V[v1, :]
        tri_x[2, :] = m.V[v2, :]
        F = deformation_gradient(tri_X, tri_x)
        E += neohookean.E(F)
    return E


if __name__ == "__main__":
    p0 = np.array([0, 0])
    p1 = np.array([1, 0])
    p2 = np.array([0.5, 1])
    V_undeformed = np.vstack([p0.reshape((1, 2)), p1.reshape((1, 2)), p2.reshape((1, 2))])

    # each column is a colour
    Colours = np.identity(3)

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
    test = dphi_dX(V_undeformed)
    print(test)
