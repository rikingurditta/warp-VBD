import numpy as np


class Neohookean:
    def __init__(self, mu_lamé: float, lambda_lamé: float):
        self.μ = mu_lamé
        self.λ = lambda_lamé

    def psi(self, F: np.ndarray) -> float:
        """
        based on Psi_B from "Stable Neo-Hookean Flesh Simulation" [Smith et al. 2018]
        E = C (J^{-2/3} trace(F^T F) - 3) + D (J - 1)^2
        modified for 2d by changing -3 to -2
        """
        J = np.linalg.det(F)
        return (0.5 * self.μ * (J ** (-2 / 3) * np.linalg.trace(F.T @ F) - 2)
                + 0.5 * self.λ * (J - 1) ** 2)

    def dpsi_dF(self, F: np.ndarray) -> np.ndarray:
        """
        :param F: (2,2) deformation gradient
        :return: (4,) gradient of Neohookean energy psi wrt F (flattened into vector)
        """
        # used sympy
        F00 = F[0, 0]
        F01 = F[0, 1]
        F10 = F[1, 0]
        F11 = F[1, 1]

        # factor out common terms
        J = np.linalg.det(F)
        J_neg_2_3 = J ** (-2. / 3.)
        J_neg_5_3 = J ** (-5. / 3.)
        F_sqnorm = np.linalg.norm(F) ** 2

        # F was flattened row-wise, as in lecture
        # @formatter:off
        g = np.zeros(4)
        g[0] = F11*self.λ*(J-1)+(1/2)*self.μ*(2*F00/(J)**(2/3)-2/3*F11*(F_sqnorm)/(J)**(5/3))
        g[1] = -F10*self.λ*(J-1)+(1/2)*self.μ*(2*F01/(J)**(2/3)+(2/3)*F10*(F_sqnorm)/(J)**(5/3))
        g[2] = -F01*self.λ*(J-1)+(1/2)*self.μ*((2/3)*F01*(F_sqnorm)/(J)**(5/3)+2*F10/(J)**(2/3))
        g[3] = F00*self.λ*(J-1)+(1/2)*self.μ*(-2/3*F00*(F_sqnorm)/(J)**(5/3)+2*F11/(J)**(2/3))
        # @formatter:on

        return g

    def d2psi_dF2(self, F: np.ndarray) -> np.ndarray:
        """
        :param F: (2,2) deformation gradient
        :return: (4,4) Hessian of Neohookean energy psi wrt F (flattened into vector)
        """
        # used sympy
        F00 = F[0, 0]
        F01 = F[0, 1]
        F10 = F[1, 0]
        F11 = F[1, 1]

        # factor out common terms
        J = np.linalg.det(F)
        J_neg_2_3 = J ** (-2. / 3.)
        J_neg_5_3 = J ** (-5. / 3.)
        J_neg_8_3 = J ** (-8. / 3.)
        F_sqnorm = np.linalg.norm(F) ** 2

        # F was flattened row-wise, as in lecture
        # @formatter:off
        H = np.zeros((4, 4))
        H[0, 0] = F11**2*self.λ+(1/2)*self.μ*(-8/3*F00*F11/(J)**(5/3)+(10/9)*F11**2*(F_sqnorm)/(J)**(8/3)+2/(J)**(2/3))
        H[0, 1] = -F10*F11*self.λ+(1/2)*self.μ*((4/3)*F00*F10/(J)**(5/3)-4/3*F01*F11/(J)**(5/3)-10/9*F10*F11*(F_sqnorm)/(J)**(8/3))
        H[0, 2] = -F01*F11*self.λ+(1/2)*self.μ*((4/3)*F00*F01/(J)**(5/3)-10/9*F01*F11*(F_sqnorm)/(J)**(8/3)-4/3*F10*F11/(J)**(5/3))
        H[0, 3] = F00*F11*self.λ+self.λ*(J-1)+(1/2)*self.μ*(-4/3*F00**2/(J)**(5/3)+(10/9)*F00*F11*(F_sqnorm)/(J)**(8/3)-4/3*F11**2/(J)**(5/3)-2/3*(F_sqnorm)/(J)**(5/3))
        H[1, 0] = -F10*F11*self.λ+(1/2)*self.μ*((4/3)*F00*F10/(J)**(5/3)-4/3*F01*F11/(J)**(5/3)-10/9*F10*F11*(F_sqnorm)/(J)**(8/3))
        H[1, 1] = F10**2*self.λ+(1/2)*self.μ*((8/3)*F01*F10/(J)**(5/3)+(10/9)*F10**2*(F_sqnorm)/(J)**(8/3)+2/(J)**(2/3))
        H[1, 2] = F01*F10*self.λ-self.λ*(J-1)+(1/2)*self.μ*((4/3)*F01**2/(J)**(5/3)+(10/9)*F01*F10*(F_sqnorm)/(J)**(8/3)+(4/3)*F10**2/(J)**(5/3)+(2/3)*(F_sqnorm)/(J)**(5/3))
        H[1, 3] = -F00*F10*self.λ+(1/2)*self.μ*(-4/3*F00*F01/(J)**(5/3)-10/9*F00*F10*(F_sqnorm)/(J)**(8/3)+(4/3)*F10*F11/(J)**(5/3))
        H[2, 0] = -F01*F11*self.λ+(1/2)*self.μ*((4/3)*F00*F01/(J)**(5/3)-10/9*F01*F11*(F_sqnorm)/(J)**(8/3)-4/3*F10*F11/(J)**(5/3))
        H[2, 1] = F01*F10*self.λ-self.λ*(J-1)+(1/2)*self.μ*((4/3)*F01**2/(J)**(5/3)+(10/9)*F01*F10*(F_sqnorm)/(J)**(8/3)+(4/3)*F10**2/(J)**(5/3)+(2/3)*(F_sqnorm)/(J)**(5/3))
        H[2, 2] = F01**2*self.λ+(1/2)*self.μ*((10/9)*F01**2*(F_sqnorm)/(J)**(8/3)+(8/3)*F01*F10/(J)**(5/3)+2/(J)**(2/3))
        H[2, 3] = -F00*F01*self.λ+(1/2)*self.μ*(-10/9*F00*F01*(F_sqnorm)/(J)**(8/3)-4/3*F00*F10/(J)**(5/3)+(4/3)*F01*F11/(J)**(5/3))
        H[3, 0] = F00*F11*self.λ+self.λ*(J-1)+(1/2)*self.μ*(-4/3*F00**2/(J)**(5/3)+(10/9)*F00*F11*(F_sqnorm)/(J)**(8/3)-4/3*F11**2/(J)**(5/3)-2/3*(F_sqnorm)/(J)**(5/3))
        H[3, 1] = -F00*F10*self.λ+(1/2)*self.μ*(-4/3*F00*F01/(J)**(5/3)-10/9*F00*F10*(F_sqnorm)/(J)**(8/3)+(4/3)*F10*F11/(J)**(5/3))
        H[3, 2] = -F00*F01*self.λ+(1/2)*self.μ*(-10/9*F00*F01*(F_sqnorm)/(J)**(8/3)-4/3*F00*F10/(J)**(5/3)+(4/3)*F01*F11/(J)**(5/3))
        H[3, 3] = F00**2*self.λ+(1/2)*self.μ*((10/9)*F00**2*(F_sqnorm)/(J)**(8/3)-8/3*F00*F11/(J)**(5/3)+2/(J)**(2/3))
        # @formatter:on

        return H
