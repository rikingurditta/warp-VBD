import numpy as np

class LinearElasticity:
    def __init__(self, mu_lamé: float, lambda_lamé: float):
        self.μ = mu_lamé
        self.λ = lambda_lamé

    def psi(self, F: np.ndarray) -> float:
        """
        psi = με:ε + λtr²(ε)
        :param F: (2,2) deformation gradient
        :return:
        """
        ε = 0.5 * (F + F.T) - np.eye(2)
        return self.μ * np.vdot(ε, ε) + 0.5 * self.λ * np.linalg.trace(ε) ** 2


    def dpsi_dF(self, F: np.ndarray) -> np.ndarray:
        """
        :param F: (2,2) deformation gradient
        :return: (4,) gradient of Neohookean energy psi wrt F (flattened into vector)
        """
        F00 = F[0, 0]
        F01 = F[0, 1]
        F10 = F[1, 0]
        F11 = F[1, 1]
        F_sqnorm = np.linalg.norm(F) ** 2

        # F was flattened row-wise, as in lecture
        # @formatter:off
        g = np.zeros(4)
        g[0] = F00*self.λ*(0.5*F_sqnorm-1)+self.μ*(2*F00*((1/2)*F00**2+(1/2)*F10**2-1/2)+2*F01*((1/2)*F00*F01+(1/2)*F10*F11))
        g[1] = F01*self.λ*(0.5*F_sqnorm-1)+self.μ*(2*F00*((1/2)*F00*F01+(1/2)*F10*F11)+2*F01*((1/2)*F01**2+(1/2)*F11**2-1/2))
        g[2] = F10*self.λ*(0.5*F_sqnorm-1)+self.μ*(2*F10*((1/2)*F00**2+(1/2)*F10**2-1/2)+2*F11*((1/2)*F00*F01+(1/2)*F10*F11))
        g[3] = F11*self.λ*(0.5*F_sqnorm-1)+self.μ*(2*F10*((1/2)*F00*F01+(1/2)*F10*F11)+2*F11*((1/2)*F01**2+(1/2)*F11**2-1/2))
        # @formatter:on

        return g


    def d2psi_dF2(self, F: np.ndarray) -> np.ndarray:
        """
        :param F: (2,2) deformation gradient
        :return: (4,4) Hessian of Neohookean energy psi wrt F (flattened into vector)
        """
        F00 = F[0, 0]
        F01 = F[0, 1]
        F10 = F[1, 0]
        F11 = F[1, 1]
        F_sqnorm = np.linalg.norm(F) ** 2

        # F was flattened row-wise, as in lecture
        # @formatter:off
        H = np.zeros((4, 4))
        H[0, 0] = F00**2*self.λ+self.λ*(0.5*F_sqnorm-1)+self.μ*(3*F00**2+F01**2+F10**2-1)
        H[0, 1] = F00*F01*self.λ+self.μ*(2*F00*F01+F10*F11)
        H[0, 2] = F00*F10*self.λ+self.μ*(2*F00*F10+F01*F11)
        H[0, 3] = F00*F11*self.λ+F01*F10*self.μ
        H[1, 0] = F00*F01*self.λ+self.μ*(2*F00*F01+F10*F11)
        H[1, 1] = F01**2*self.λ+self.λ*(0.5*F_sqnorm-1)+self.μ*(F00**2+3*F01**2+F11**2-1)
        H[1, 2] = F00*F11*self.μ+F01*F10*self.λ
        H[1, 3] = F01*F11*self.λ+self.μ*(F00*F10+2*F01*F11)
        H[2, 0] = F00*F10*self.λ+self.μ*(2*F00*F10+F01*F11)
        H[2, 1] = F00*F11*self.μ+F01*F10*self.λ
        H[2, 2] = F10**2*self.λ+self.λ*(0.5*F_sqnorm-1)+self.μ*(F00**2+3*F10**2+F11**2-1)
        H[2, 3] = F10*F11*self.λ+self.μ*(F00*F01+2*F10*F11)
        H[3, 0] = F00*F11*self.λ+F01*F10*self.μ
        H[3, 1] = F01*F11*self.λ+self.μ*(F00*F10+2*F01*F11)
        H[3, 2] = F10*F11*self.λ+self.μ*(F00*F01+2*F10*F11)
        H[3, 3] = F11**2*self.λ+self.λ*(0.5*F_sqnorm-1)+self.μ*(F01**2+F10**2+3*F11**2-1)
        # @formatter:on

        return H
