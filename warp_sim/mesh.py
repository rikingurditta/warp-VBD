from typing import List
from numpy_sim.mesh import Mesh
import warp as wp
import warp.sparse as sparse
import numpy as np

mat32f = wp.types.matrix(shape=(3, 2), dtype=float)


class WarpMesh:
    num_verts: int
    num_elements: int
    x: wp.array[wp.vec2f]
    E: wp.array2d[int]
    M: sparse.bsr_matrix_t[wp.mat22f]
    V_to_E: List[List[int]]
    E_vol0: wp.array[float]
    E_dphi_dX: wp.array[mat32f]

    def __init__(self, m: Mesh | None):
        if m is None:
            return
        self.num_verts = m.V.shape[0]
        self.num_elements = m.E.shape[0]
        self.x = wp.array(m.V, dtype=float).view(dtype=wp.vec2f)
        self.E = wp.array(m.E, dtype=int)
        self.V_to_E = m.V_to_E
        self.E_vol0 = wp.array(m.E_vol0, dtype=float)

        # compute mass matrix
        rows = []
        cols = []
        vals = []

        for e in range(self.num_elements):
            vol = m.E_vol0[e]
            v0 = m.E[e, 0]
            v1 = m.E[e, 1]
            v2 = m.E[e, 2]
            # TODO: do we need to multiply by a constant here? maybe *2?
            for v in [v0, v1, v2]:
                # diagonal entries are vol / 12
                diag_block = np.array([[vol / 12, 0],
                                       [0, vol / 12]])
                rows.append(v)
                cols.append(v)
                vals.append(diag_block)

                # off-diagonal entries are vol / 24
                for w in [v0, v1, v2]:
                    if v != w:
                        block = np.array([[vol / 24, 0],
                                          [0, vol / 24]])
                        rows.append(v)
                        cols.append(v)
                        vals.append(block)

        rows = wp.array(rows, dtype=int)
        cols = wp.array(cols, dtype=int)
        vals = wp.array(vals, dtype=wp.mat22f)

        self.M = sparse.bsr_from_triplets(
            rows_of_blocks=self.num_verts,
            cols_of_blocks=self.num_verts,
            rows=rows,
            columns=cols,
            values=vals,
        )
        self.E_dphi_dX = wp.empty(self.E.shape[0], dtype=wp.types.matrix(shape=(3, 2), dtype=float))
        wp.launch(kernel=compute_dphi_dX,
                  dim=self.num_elements,
                  inputs=[self.x, self.E, self.E_dphi_dX],
                  device='cuda')

    def copy(self) -> WarpMesh:
        out = WarpMesh(None)
        out.num_verts = self.num_verts
        out.num_elements = self.num_elements
        out.x = wp.clone(self.x)
        out.E = wp.clone(self.E)
        out.M = sparse.bsr_copy(self.M)
        out.V_to_E = self.V_to_E.copy()
        out.E_vol0 = wp.clone(self.E_vol0)
        out.E_dphi_dX = wp.clone(self.E_dphi_dX)
        return out

    def V_3d(self) -> np.ndarray:
        V = self.x.numpy()
        out = np.zeros((self.num_verts, 3))
        out[:, :2] = V
        return out

# noinspection PyTypeHints
@wp.kernel
def compute_dphi_dX(V: wp.array[wp.vec2f],
                    E: wp.array2d[int],
                    E_dphi_dX: wp.array[mat32f]):
    element = wp.tid()
    v0 = E[element, 0]
    v1 = E[element, 1]
    v2 = E[element, 2]
    p0 = V[v0]
    p1 = V[v1]
    p2 = V[v2]
    T = wp.mat22f()
    T[:, 0] = p1 - p0
    T[:, 1] = p2 - p0
    Tinv = wp.inverse(T)
    out = wp.matrix(shape=(3, 2), dtype=float)
    out[0, :] = -(Tinv[0, :] + Tinv[1, :])
    out[1:, :] = Tinv
    E_dphi_dX[element] = out


if __name__ == "__main__":
    m = Mesh(filename='../test_mesh.txt')
    wm = WarpMesh(m)
    # print(wm.E_dphi_dX.numpy())
    print(wm.V_3d())
