# test if gradients and hessians are correct (i.e., match up with finite differences)

import numpy as np
from potential_energy import LinearElasticity, Neohookean

def d_dF_finite_diff(func, F, epsilon=1e-9):
    out = np.zeros(4)
    for i in range(4):
        pert = np.zeros_like(F)
        pert[i // 2, i % 2] += epsilon
        out[i] = (func(F + pert) - func(F)) / epsilon
    return out

def d2_dF2_finite_diff(func, F, epsilon=1e-9):
    out = np.zeros((4, 4))
    for i in range(4):
        pert_i = np.zeros_like(F)
        pert_i[i // 2, i % 2] += epsilon
        for j in range(4):
            pert_j = np.zeros_like(F)
            pert_j[j // 2, j % 2] += epsilon
            out[i, j] = (func(F + pert_i + pert_j) - func(F + pert_i) - func(F + pert_j) + func(F)) / (epsilon * epsilon)
    return out

if __name__ == "__main__":
    n = Neohookean(mu_lamé=1, lambda_lamé=1)
    l = LinearElasticity(mu_lamé=1, lambda_lamé=1)
    model = n
    F = np.eye(2)
    print('epsilon\t\tg diff\t\t\tH diff')
    for exp in range(7, 14):
        g_fd = d_dF_finite_diff(model.psi, F, epsilon=10 ** (-exp))
        g_an = model.dpsi_dF(F)
        g_diff = np.linalg.norm(g_fd - g_an)
        H_fd = d2_dF2_finite_diff(model.psi, F, epsilon=10 ** (-exp))
        H_an = model.d2psi_dF2(F)
        H_diff = np.linalg.norm(H_fd - H_an)

        print(f'1e-{exp}\t\t{g_diff:>.5}\t\t{H_diff:>.5}')
