from typing import Any

from numpy_sim.mesh import Mesh
import warp as wp

mat32f = wp.types.matrix(shape=(3, 2), dtype=float)


class WarpMesh:
    def __init__(self, m: Mesh | None):
        if m is not None:
            m.precompute()
            num_verts = m.V.shape[0]
            self.x = wp.array(m.V, dtype=float).view(dtype=wp.vec2f)
            self.E = wp.array(m.E, dtype=int)
            self.M = wp.empty((num_verts, num_verts), dtype=wp.mat22f)
            wp.launch(kernel=flat_M_to_mat22f,
                      dim=(num_verts, num_verts),
                      inputs=[wp.array(m.M, dtype=float), self.M],
                      device='cuda')
            self.V_to_E = m.V_to_E
            self.E_vol0 = wp.array(m.E_vol0, dtype=float)

    def copy(self) -> WarpMesh:
        out = WarpMesh(None)
        out.x = wp.clone(self.x)
        out.E = wp.clone(self.E)
        out.M = wp.clone(self.M)
        out.V_to_E = self.V_to_E.copy()
        out.E_vol0 = wp.clone(self.E_vol0)
        out.E_dphi_dX = wp.clone(self.E_dphi_dX)
        return out

    def precompute(self) -> None:
        self.E_dphi_dX = wp.empty(self.E.shape[0], dtype=wp.types.matrix(shape=(3, 2), dtype=float))
        wp.launch(kernel=compute_dphi_dX,
                  dim=self.E.shape[0],
                  inputs=[self.x, self.E, self.E_dphi_dX],
                  device='cuda')


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


@wp.kernel
def flat_M_to_mat22f(flat_M: wp.array2d[float], M: wp.array2d[wp.mat22f]):
    v0, v1 = wp.tid()
    block = wp.mat22f()
    block[0, 0] = flat_M[v0 * 2, v1 * 2]
    block[0, 1] = flat_M[v0 * 2, v1 * 2 + 1]
    block[1, 0] = flat_M[v0 * 2 + 1, v1 * 2]
    block[1, 1] = flat_M[v0 * 2 + 1, v1 * 2 + 1]
    M[v0, v1] = block


if __name__ == "__main__":
    m = Mesh(filename='../test_mesh.txt')
    wm = WarpMesh(m)
    wm.precompute()
    print(wm.E_dphi_dX.numpy())
