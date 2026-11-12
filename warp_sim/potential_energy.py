import warp as wp
from abc import ABC, abstractmethod


@wp.struct
class MaterialParams:
    """
    Lamé parameters
    """
    mu_lame: float
    lambda_lame: float


@wp.func
def psi_linear(params: MaterialParams, F: wp.mat22) -> float:
    """
    psi = με:ε + λtr²(ε)
    :param params: material parameters
    :param F: (2,2) deformation gradient
    :return: linear elasticity deformation energy
    """
    ε = 0.5 * (F + wp.transpose(F)) - wp.identity(2, dtype=float)
    return params.mu_lame * wp.ddot(ε, ε) + 0.5 * params.lambda_lame * wp.trace(ε) ** 2.


@wp.func
def dpsi_dF_linear(params: MaterialParams, F: wp.mat22) -> wp.vec4:
    """
    :param params: material parameters
    :param F: (2,2) deformation gradient
    :return: (4,) gradient of linear elasticity energy psi wrt F (flattened into vector)
    """
    F00 = F[0, 0]
    F01 = F[0, 1]
    F10 = F[1, 0]
    F11 = F[1, 1]
    F_sqnorm = wp.ddot(F, F)

    # F was flattened row-wise, as in lecture
    # @formatter:off
    g = wp.vec4f()
    g[0] = F00*params.lambda_lame*(0.5*F_sqnorm-1.)+params.mu_lame*(2.*F00*(0.5*F00**2.+0.5*F10**2.-0.5)+2.*F01*(0.5*F00*F01+0.5*F10*F11))
    g[1] = F01*params.lambda_lame*(0.5*F_sqnorm-1.)+params.mu_lame*(2.*F00*(0.5*F00*F01+0.5*F10*F11)+2.*F01*(0.5*F01**2.+0.5*F11**2.-0.5))
    g[2] = F10*params.lambda_lame*(0.5*F_sqnorm-1.)+params.mu_lame*(2.*F10*(0.5*F00**2.+0.5*F10**2.-0.5)+2.*F11*(0.5*F00*F01+0.5*F10*F11))
    g[3] = F11*params.lambda_lame*(0.5*F_sqnorm-1.)+params.mu_lame*(2.*F10*(0.5*F00*F01+0.5*F10*F11)+2.*F11*(0.5*F01**2.+0.5*F11**2.-0.5))
    # @formatter:on

    return g


@wp.func
def d2psi_dF2_linear(params: MaterialParams, F: wp.mat22) -> wp.mat44:
    """
    :param params: material parameters
    :param F: (2,2) deformation gradient
    :return: (4,4) Hessian of linear elasticity energy psi wrt F (flattened into vector)
    """
    F00 = F[0, 0]
    F01 = F[0, 1]
    F10 = F[1, 0]
    F11 = F[1, 1]
    F_sqnorm = wp.ddot(F, F)

    # F was flattened row-wise, as in lecture
    # @formatter:off
    H = wp.mat44f()
    H[0, 0] = F00**2.*params.lambda_lame+params.lambda_lame*(0.5*F_sqnorm-1.)+params.mu_lame*(3.*F00**2.+F01**2.+F10**2.-1.)
    H[0, 1] = F00*F01*params.lambda_lame+params.mu_lame*(2.*F00*F01+F10*F11)
    H[0, 2] = F00*F10*params.lambda_lame+params.mu_lame*(2.*F00*F10+F01*F11)
    H[0, 3] = F00*F11*params.lambda_lame+F01*F10*params.mu_lame
    H[1, 0] = F00*F01*params.lambda_lame+params.mu_lame*(2.*F00*F01+F10*F11)
    H[1, 1] = F01**2.*params.lambda_lame+params.lambda_lame*(0.5*F_sqnorm-1.)+params.mu_lame*(F00**2.+3.*F01**2.+F11**2.-1.)
    H[1, 2] = F00*F11*params.mu_lame+F01*F10*params.lambda_lame
    H[1, 3] = F01*F11*params.lambda_lame+params.mu_lame*(F00*F10+2.*F01*F11)
    H[2, 0] = F00*F10*params.lambda_lame+params.mu_lame*(2.*F00*F10+F01*F11)
    H[2, 1] = F00*F11*params.mu_lame+F01*F10*params.lambda_lame
    H[2, 2.] = F10**2.*params.lambda_lame+params.lambda_lame*(0.5*F_sqnorm-1.)+params.mu_lame*(F00**2.+3.*F10**2.+F11**2.-1.)
    H[2, 3] = F10*F11*params.lambda_lame+params.mu_lame*(F00*F01+2.*F10*F11)
    H[3, 0] = F00*F11*params.lambda_lame+F01*F10*params.mu_lame
    H[3, 1] = F01*F11*params.lambda_lame+params.mu_lame*(F00*F10+2.*F01*F11)
    H[3, 2] = F10*F11*params.lambda_lame+params.mu_lame*(F00*F01+2.*F10*F11)
    H[3, 3] = F11**2.*params.lambda_lame+params.lambda_lame*(0.5*F_sqnorm-1.)+params.mu_lame*(F01**2.+F10**2.+3.*F11**2.-1.)
    # @formatter:on

    return H


@wp.func
def psi_neohookean(params: MaterialParams, F: wp.mat22) -> float:
    """
    based on Psi_B from "Stable Neo-Hookean Flesh Simulation" [Smith et al. 2018]
    E = C (J^{-2/3} trace(F^T F) - 3) + D (J - 1)^2
    modified for 2d by changing -3 to -2
    """
    J = wp.determinant(F)
    return (0.5 * params.mu_lame * (J ** (-2 / 3) * wp.ddot(F, F) - 2)
            + 0.5 * params.lambda_lame * (J - 1) ** 2)


@wp.func
def dpsi_dF_neohookean(params: MaterialParams, F: wp.mat22) -> wp.vec4:
    """
    :param F: (2,2) deformation gradient
    :return: (4,) gradient of Neohookean energy psi wrt F (flattened into vector)
    """
    # TODO: not 0 around rest!!!!!
    # used sympy
    F00 = F[0, 0]
    F01 = F[0, 1]
    F10 = F[1, 0]
    F11 = F[1, 1]

    # factor out common terms
    J = wp.determinant(F)
    J_neg_2_3 = J ** (-2. / 3.)
    J_neg_5_3 = J ** (-5. / 3.)
    F_sqnorm = wp.ddot(F, F)

    # F was flattened row-wise, as in lecture
    # @formatter:off
    g = wp.vec4f()
    g[0] = F11*params.lambda_lame*(J-1)+(1/2)*params.mu_lame*(2*F00/(J)**(2/3)-2/3*F11*(F_sqnorm)/(J)**(5/3))
    g[1] = -F10*params.lambda_lame*(J-1)+(1/2)*params.mu_lame*(2*F01/(J)**(2/3)+(2/3)*F10*(F_sqnorm)/(J)**(5/3))
    g[2] = -F01*params.lambda_lame*(J-1)+(1/2)*params.mu_lame*((2/3)*F01*(F_sqnorm)/(J)**(5/3)+2*F10/(J)**(2/3))
    g[3] = F00*params.lambda_lame*(J-1)+(1/2)*params.mu_lame*(-2/3*F00*(F_sqnorm)/(J)**(5/3)+2*F11/(J)**(2/3))
    # @formatter:on

    return g


@wp.func
def d2psi_dF2_neohookean(params: MaterialParams, F: wp.mat22) -> wp.mat44:
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
    J = wp.determinant(F)
    J_neg_2_3 = J ** (-2. / 3.)
    J_neg_5_3 = J ** (-5. / 3.)
    J_neg_8_3 = J ** (-8. / 3.)
    F_sqnorm = wp.ddot(F, F)

    # F was flattened row-wise, as in lecture
    # @formatter:off
    H = wp.mat44f()
    H[0, 0] = F11**2*params.lambda_lame+(1/2)*params.mu_lame*(-8/3*F00*F11/(J)**(5/3)+(10/9)*F11**2*(F_sqnorm)/(J)**(8/3)+2/(J)**(2/3))
    H[0, 1] = -F10*F11*params.lambda_lame+(1/2)*params.mu_lame*((4/3)*F00*F10/(J)**(5/3)-4/3*F01*F11/(J)**(5/3)-10/9*F10*F11*(F_sqnorm)/(J)**(8/3))
    H[0, 2] = -F01*F11*params.lambda_lame+(1/2)*params.mu_lame*((4/3)*F00*F01/(J)**(5/3)-10/9*F01*F11*(F_sqnorm)/(J)**(8/3)-4/3*F10*F11/(J)**(5/3))
    H[0, 3] = F00*F11*params.lambda_lame+params.lambda_lame*(J-1)+(1/2)*params.mu_lame*(-4/3*F00**2/(J)**(5/3)+(10/9)*F00*F11*(F_sqnorm)/(J)**(8/3)-4/3*F11**2/(J)**(5/3)-2/3*(F_sqnorm)/(J)**(5/3))
    H[1, 0] = -F10*F11*params.lambda_lame+(1/2)*params.mu_lame*((4/3)*F00*F10/(J)**(5/3)-4/3*F01*F11/(J)**(5/3)-10/9*F10*F11*(F_sqnorm)/(J)**(8/3))
    H[1, 1] = F10**2*params.lambda_lame+(1/2)*params.mu_lame*((8/3)*F01*F10/(J)**(5/3)+(10/9)*F10**2*(F_sqnorm)/(J)**(8/3)+2/(J)**(2/3))
    H[1, 2] = F01*F10*params.lambda_lame-params.lambda_lame*(J-1)+(1/2)*params.mu_lame*((4/3)*F01**2/(J)**(5/3)+(10/9)*F01*F10*(F_sqnorm)/(J)**(8/3)+(4/3)*F10**2/(J)**(5/3)+(2/3)*(F_sqnorm)/(J)**(5/3))
    H[1, 3] = -F00*F10*params.lambda_lame+(1/2)*params.mu_lame*(-4/3*F00*F01/(J)**(5/3)-10/9*F00*F10*(F_sqnorm)/(J)**(8/3)+(4/3)*F10*F11/(J)**(5/3))
    H[2, 0] = -F01*F11*params.lambda_lame+(1/2)*params.mu_lame*((4/3)*F00*F01/(J)**(5/3)-10/9*F01*F11*(F_sqnorm)/(J)**(8/3)-4/3*F10*F11/(J)**(5/3))
    H[2, 1] = F01*F10*params.lambda_lame-params.lambda_lame*(J-1)+(1/2)*params.mu_lame*((4/3)*F01**2/(J)**(5/3)+(10/9)*F01*F10*(F_sqnorm)/(J)**(8/3)+(4/3)*F10**2/(J)**(5/3)+(2/3)*(F_sqnorm)/(J)**(5/3))
    H[2, 2] = F01**2*params.lambda_lame+(1/2)*params.mu_lame*((10/9)*F01**2*(F_sqnorm)/(J)**(8/3)+(8/3)*F01*F10/(J)**(5/3)+2/(J)**(2/3))
    H[2, 3] = -F00*F01*params.lambda_lame+(1/2)*params.mu_lame*(-10/9*F00*F01*(F_sqnorm)/(J)**(8/3)-4/3*F00*F10/(J)**(5/3)+(4/3)*F01*F11/(J)**(5/3))
    H[3, 0] = F00*F11*params.lambda_lame+params.lambda_lame*(J-1)+(1/2)*params.mu_lame*(-4/3*F00**2/(J)**(5/3)+(10/9)*F00*F11*(F_sqnorm)/(J)**(8/3)-4/3*F11**2/(J)**(5/3)-2/3*(F_sqnorm)/(J)**(5/3))
    H[3, 1] = -F00*F10*params.lambda_lame+(1/2)*params.mu_lame*(-4/3*F00*F01/(J)**(5/3)-10/9*F00*F10*(F_sqnorm)/(J)**(8/3)+(4/3)*F10*F11/(J)**(5/3))
    H[3, 2] = -F00*F01*params.lambda_lame+(1/2)*params.mu_lame*(-10/9*F00*F01*(F_sqnorm)/(J)**(8/3)-4/3*F00*F10/(J)**(5/3)+(4/3)*F01*F11/(J)**(5/3))
    H[3, 3] = F00**2*params.lambda_lame+(1/2)*params.mu_lame*((10/9)*F00**2*(F_sqnorm)/(J)**(8/3)-8/3*F00*F11/(J)**(5/3)+2/(J)**(2/3))
    # @formatter:on

    return H
