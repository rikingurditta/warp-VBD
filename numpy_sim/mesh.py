import numpy as np


def triangle_areas(m: Mesh):
    out = np.zeros(m.E.shape[0])
    for e in range(m.E.shape[0]):
        p0 = m.V[m.E[e, 0], :]
        p1 = m.V[m.E[e, 1], :]
        p2 = m.V[m.E[e, 2], :]
        a = p1 - p0
        b = p2 - p0
        out[e] = 0.5 * (a[0] * b[1] - a[1] * b[0])
    return out


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

        self.E_vol0 = np.zeros(self.E.shape[0])  # = Area in 2D
        self.M = np.identity(self.V.flatten().size)

    def precompute(self):
        self.E_vol0 = triangle_areas(self)
        for e in range(self.E.shape[0]):
            vol = self.E_vol0[e]
            v0 = self.E[e, 0]
            v1 = self.E[e, 1]
            v2 = self.E[e, 2]
            # TODO: do we need to multiply by a constant here? maybe *2?
            for v in [v0, v1, v2]:
                # diagonal entries are vol / 12
                self.M[v * 2, v * 2] += vol / 12
                self.M[v * 2 + 1, v * 2 + 1] += vol / 12
                # off-diagonal entries are vol / 24
                for w in [v0, v1, v2]:
                    if v != w:
                        self.M[v * 2, w * 2] += vol / 24
                        self.M[v * 2 + 1, w * 2 + 1] += vol / 24

    def x(self) -> np.ndarray:
        """
        :return: view of V flattened row-wise (can be an lvalue)
        """
        return self.V.ravel()

    def V_3d(self) -> np.ndarray:
        """
        :return: V embedded in 3d, aka a matrix whose xy columns are V and z column is 0
        """
        out = np.zeros((self.V.shape[0], 3))
        out[:, 0:2] = self.V
        return out

    def copy(self) -> Mesh:
        out = Mesh(V=self.V.copy(), E=self.E.copy())
        out.M = self.M.copy()
        out.V_to_E = self.V_to_E.copy()
        out.E_vol0 = self.E_vol0.copy()
        return out

    def centre_at_zero(self):
        avg = self.V.sum(axis=0) / self.V.shape[0]
        self.V -= avg
