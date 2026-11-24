from typing import Any
import numpy as np
import warp as wp
import warp.sparse as sparse

@wp.kernel
def ke_sparse_to_dense(rows: wp.array[int], cols: wp.array[int], vals: wp.array[Any], de: wp.array2d[Any]):
    tid = wp.tid()
    de[rows[tid], cols[tid]] = vals[tid]

def sparse_to_dense(sp: sparse.bsr_matrix_t[Any]) -> wp.array2d[Any]:
    out = wp.zeros((sp.nrow, sp.ncol), dtype=sp.dtype)
    wp.launch(kernel=ke_sparse_to_dense,
              dim=sp.nnz,
              inputs=[sp.uncompress_rows(), sp.columns, sp.values, out],
              device='cuda')
    return out

def mat_array_to_numpy_blocks(A: wp.array2d[wp.mat22f]) -> np.ndarray:
    return np.block([[block for block in row] for row in A.numpy()])

def sparse_mat_array_to_numpy_blocks(A: sparse.bsr_matrix_t[wp.mat22f]) -> np.ndarray:
    return mat_array_to_numpy_blocks(sparse_to_dense(A))