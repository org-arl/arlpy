import numpy as _np

def rnd(alpha=1.5, beta=0, scale=1, loc=0.0, size=1):
    """Generate independent stable random numbers.

    :param alpha: characteristic exponent (0.1 to 2.0)
    :param beta: skew (-1 to +1)
    :param scale: scale parameter
    :param loc: location parameter (mean for alpha > 1, median/mode when beta=0)
    :param size: size of the random array to generate

    This implementation is based on the method in J.M. Chambers, C.L. Mallows
    and B.W. Stuck, "A Method for Simulating Stable Random Variables," JASA 71 (1976): 340-4.
    McCulloch's MATLAB implementation (1996) served as a reference in developing this code.
    """
    if alpha < 0.1 or alpha > 2:
        raise ValueError('alpha must be in the range 0.1 to 2')
    if _np.abs(beta) > 1:
        raise ValueError('beta must be in the range -1 to 1')
    phi = (_np.random.uniform(size=size) - 0.5) * _np.pi
    if alpha == 1 and beta == 0:
        return loc + scale*_np.tan(phi)
    w = -_np.log(_np.random.uniform(size=size))
    if alpha == 2:
        return loc + 2*scale*_np.sqrt(w)*_np.sin(phi)
    if beta == 0:
        return loc + scale * ((_np.cos((1-alpha)*phi) / w) ** (1.0/alpha - 1) * _np.sin(alpha * phi) / _np.cos(phi) ** (1.0/alpha))
    cosphi = _np.cos(phi)
    if _np.abs(alpha-1) > 1e-8:
        zeta = beta * _np.tan(_np.pi*alpha/2)
        aphi = alpha * phi
        a1phi = (1-alpha) * phi
        return loc + scale * (( (_np.sin(aphi)+zeta*_np.cos(aphi))/cosphi * ((_np.cos(a1phi)+zeta*_np.sin(a1phi))) / ((w*cosphi)**((1-alpha)/alpha)) ))
    bphi = _np.pi/2 + beta*phi
    x = 2/_np.pi * (bphi*_np.tan(phi) - beta*_np.log(_np.pi/2*w*cosphi/bphi))
    if alpha != 1:
        x += beta * _np.tan(_np.pi*alpha/2)
    return loc + scale*x
