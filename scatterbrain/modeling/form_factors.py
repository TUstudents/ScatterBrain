# scatterbrain/modeling/form_factors.py
"""
Analytical form factors P(q) for common shapes.

Form factors describe the scattering from a single, isolated particle.
They are typically normalized such that P(q=0) = 1.
The total scattered intensity from N identical, non-interacting particles
is I(q) = N * (sld_particle - sld_solvent)^2 * V_particle^2 * P(q),
where sld is scattering length density and V is volume.
Often, the (sld_particle - sld_solvent)^2 * V_particle^2 part is incorporated
into a scale factor or I(0) when fitting I(q) = scale * P(q).
"""

import logging

import numpy as np
from scipy.special import j1 as bessel_j1

# Small constant to prevent division by zero or issues with q=0
# for functions where P(q=0) is handled by a limit.
_Q_EPSILON = 1e-9

logger = logging.getLogger(__name__)


def sphere_pq(q: np.ndarray, radius: float) -> np.ndarray:
    """
    Calculate the form factor P(q) for a monodisperse sphere.

    The form factor for a sphere is::

        P(q) = [ 3 * (sin(qR) - qR*cos(qR)) / (qR)^3 ]^2

    which gives P(0) = 1 via L'Hopital's rule.

    Parameters
    ----------
    q : np.ndarray
        Scattering vector magnitudes.
    radius : float
        Radius of the sphere (R). Must be positive.

    Returns
    -------
    np.ndarray
        The form factor P(q) values, normalized to P(0) = 1.

    Raises
    ------
    ValueError
        If radius is not positive.
    """
    if radius <= 0:
        raise ValueError("Sphere radius must be positive.")

    q_safe = np.asarray(q, dtype=float)
    # Handle q=0 case separately to avoid division by zero
    # P(q=0) = 1
    pq = np.ones_like(q_safe)

    # For qR > epsilon
    qr = q_safe * radius
    mask = qr > _Q_EPSILON

    # Using the (sin(x) - x*cos(x))/x^3 formulation for stability and directness
    # P(q) = [ (3 * (sin(qR) - qR*cos(qR))) / (qR)^3 ]^2
    # Let x = qR
    # Term in bracket: 3 * (sin(x) - x*cos(x)) / x^3
    # This is equivalent to 3 * j1(x) / x, where j1 is the spherical bessel func of order 1.
    # scipy.special.spherical_jn(1, x) is j1(x)

    # Using direct formula to avoid confusion with scipy.special.j1 (which is J_n not j_n)
    # sin_qr = np.sin(qr[mask])
    # cos_qr = np.cos(qr[mask])
    # pq_bracket_term = 3.0 * (sin_qr - qr[mask] * cos_qr) / (qr[mask]**3)
    # pq[mask] = pq_bracket_term**2

    # Simpler: P(q) = ( (3/(qR)^3) * (sin(qR) - qR cos(qR)) )^2
    # For x = qR:
    # P(qR) = ( (3/x^3) * (sin(x) - x cos(x)) )^2
    # The term (sin(x) - x cos(x)) / x^2 is the spherical Bessel function j_1(x).
    # So P(qR) = (3 * j_1(qR) / (qR))^2.
    # Scipy provides spherical_jn for j_n(x).
    # For n=1, spherical_jn(1, x) = (sin(x)/x - cos(x))/x = (sin(x) - x*cos(x))/x^2
    # So, P(q) = [3 * spherical_jn(1, qr) / qr]^2 is not quite right.
    # It should be: P(q) = [3 * (sin(qr)/(qr)^2 - cos(qr)/qr) / (1/qr)]^2 ? No.

    # Let's use the definition of j_1(x) = (sin(x) - x cos(x)) / x^2
    # Then P(q) = [3 * j_1(qR) / (qR)]^2 is indeed common.
    # If x = qR, then P(x) = (3 * j_1(x) / x)^2
    # Let's implement j_1(x) manually:
    x = qr[mask]
    j1_x = (np.sin(x) - x * np.cos(x)) / (x**2)
    pq[mask] = (
        3.0 * j1_x / x
    ) ** 2  # This expression does not look right for P(0)=1 without limit
    # (3 * j1(x) / x)^2 -> (3 * (x/3) / x)^2 = 1 as x->0

    # Let's use the most common form from literature:
    # P(qR) = [3 * (sin(qR) - qR*cos(qR)) / (qR)^3]^2
    # This form correctly gives P(0)=1 via L'Hopital's rule.
    # The term in brackets is F(qR) = 3 * (sin(qR) - qR*cos(qR)) / (qR)^3
    # F(x) = 3 * (sin(x) - x*cos(x)) / x^3
    # As x -> 0, F(x) -> 3 * (x - x^3/6 - x(1-x^2/2)) / x^3 + O(x^5)
    #            = 3 * (x - x^3/6 - x + x^3/2) / x^3
    #            = 3 * (-1/6 + 1/2) * x^3 / x^3 = 3 * (2/6) = 3 * (1/3) = 1
    # So P(qR) -> 1^2 = 1.

    x = qr[mask]  # x = qR
    func_val = 3.0 * (np.sin(x) - x * np.cos(x)) / (x**3)
    pq[mask] = func_val**2

    return pq


def cylinder_pq(q: np.ndarray, radius: float, length: float) -> np.ndarray:
    """
    Calculate the form factor P(q) for a monodisperse cylinder, randomly
    oriented in solution.

    The orientationally averaged form factor is:

        P(q, R, L) = integral_0^{pi/2}
            [2 * J1(q*R*sin(a)) / (q*R*sin(a))]^2
            * [sin(q*L/2*cos(a)) / (q*L/2*cos(a))]^2
            * sin(a) d_a

    where J1 is the cylindrical Bessel function of order 1.

    The integral is evaluated on a 64-point Gauss-Legendre quadrature grid
    over alpha in [0, pi/2], fully vectorized over q.

    Parameters
    ----------
    q : np.ndarray
        Scattering vector magnitudes.
    radius : float
        Cylinder radius R.  Must be positive.
    length : float
        Cylinder length (full length L).  Must be positive.

    Returns
    -------
    np.ndarray
        Form factor P(q) normalized so that P(0) = 1.

    Raises
    ------
    ValueError
        If *radius* or *length* is not positive.
    """
    if radius <= 0:
        raise ValueError("Cylinder radius must be positive.")
    if length <= 0:
        raise ValueError("Cylinder length must be positive.")

    q_arr = np.asarray(q, dtype=float).ravel()
    # 64-point Gauss-Legendre nodes and weights on [-1, 1]; map to [0, pi/2]
    nodes, weights = np.polynomial.legendre.leggauss(64)
    alpha = (nodes + 1.0) * (np.pi / 4.0)  # [0, pi/2]
    w = weights * (np.pi / 4.0)  # Jacobian

    sin_a = np.sin(alpha)  # shape (64,)
    cos_a = np.cos(alpha)  # shape (64,)

    # Broadcast: q_arr shape (N,), alpha shape (64,) -> (N, 64)
    q_col = q_arr[:, np.newaxis]

    # Radial term: 2*J1(u)/u, u = q*R*sin(a); handle u~0 via Taylor
    u = q_col * radius * sin_a  # (N, 64)
    safe_u = np.where(u < _Q_EPSILON, _Q_EPSILON, u)
    term_r = np.where(u < _Q_EPSILON, 1.0, 2.0 * bessel_j1(safe_u) / safe_u)

    # Axial term: sin(v)/v, v = q*L/2*cos(a)
    v = q_col * (length / 2.0) * cos_a  # (N, 64)
    safe_v = np.where(v < _Q_EPSILON, _Q_EPSILON, v)
    term_l = np.where(v < _Q_EPSILON, 1.0, np.sin(safe_v) / safe_v)

    integrand = term_r**2 * term_l**2 * sin_a  # (N, 64)
    pq = integrand @ w  # (N,)

    # Normalize so that P(0) = 1 (the integral at q=0 equals 1 by construction
    # since both terms -> 1 as q -> 0, and integral sin(a) da over [0,pi/2] = 1).
    # Divide by the q=0 value to be numerically safe.
    pq_0 = float(np.dot(sin_a, w))
    pq = pq / pq_0

    return pq.reshape(np.asarray(q, dtype=float).shape)  # type: ignore[no-any-return]


def core_shell_sphere_pq(
    q: np.ndarray,
    radius_core: float,
    shell_thickness: float,
    contrast_core: float = 1.0,
    contrast_shell: float = 0.5,
) -> np.ndarray:
    """
    Calculate the form factor P(q) for a spherically symmetric core-shell sphere.

    The amplitude is::

        F(q) = (4*pi/3) * [
            R_c^3 * (rho_c - rho_s) * f(q, R_c)
            + R_o^3 * rho_s * f(q, R_o)
        ]

    where f(q, R) = 3*(sin(qR) - qR*cos(qR)) / (qR)^3 is the sphere
    amplitude factor (= 1 at q=0), R_c = radius_core,
    R_o = radius_core + shell_thickness, and rho_c / rho_s are the
    scattering length density contrasts.

    P(q) = F(q)^2 / F(0)^2  (normalized to P(0) = 1).

    Parameters
    ----------
    q : np.ndarray
        Scattering vector magnitudes.
    radius_core : float
        Radius of the core.  Must be positive.
    shell_thickness : float
        Thickness of the shell (>= 0).  R_outer = radius_core + shell_thickness.
    contrast_core : float, optional
        Scattering length density contrast of the core.  Default 1.0.
    contrast_shell : float, optional
        Scattering length density contrast of the shell.  Default 0.5.

    Returns
    -------
    np.ndarray
        Form factor P(q) normalized to P(0) = 1.

    Raises
    ------
    ValueError
        If *radius_core* <= 0 or *shell_thickness* < 0.
    """
    if radius_core <= 0:
        raise ValueError("core_shell_sphere_pq: radius_core must be positive.")
    if shell_thickness < 0:
        raise ValueError("core_shell_sphere_pq: shell_thickness must be >= 0.")

    if contrast_core == contrast_shell:
        logger.debug(
            "core_shell_sphere_pq: contrast_core == contrast_shell; "
            "returning P(q) = 1 (degenerate uniform sphere)."
        )
        return np.ones_like(np.asarray(q, dtype=float))

    q_arr = np.asarray(q, dtype=float)
    r_core = float(radius_core)
    r_outer = r_core + float(shell_thickness)
    rho_c = float(contrast_core)
    rho_s = float(contrast_shell)

    def _sphere_amplitude_factor(qr: np.ndarray) -> np.ndarray:
        """f(qR) = 3*(sin(x) - x*cos(x)) / x^3, normalized to f(0) = 1."""
        out = np.ones_like(qr)
        mask = qr > _Q_EPSILON
        x = qr[mask]
        out[mask] = 3.0 * (np.sin(x) - x * np.cos(x)) / x**3
        return out

    qrc = q_arr * r_core
    qro = q_arr * r_outer

    f_core = _sphere_amplitude_factor(qrc)
    f_outer = _sphere_amplitude_factor(qro)

    # Amplitude (without the 4*pi/3 prefactor which cancels in the ratio)
    amp = r_core**3 * (rho_c - rho_s) * f_core + r_outer**3 * rho_s * f_outer

    # F(0): qR -> 0 means f(qR) -> 1
    amp_0 = r_core**3 * (rho_c - rho_s) + r_outer**3 * rho_s

    pq = (amp / amp_0) ** 2
    return pq


# Example usage:
if __name__ == "__main__":  # pragma: no cover
    import matplotlib.pyplot as plt

    q_values = np.geomspace(0.001, 1.0, 200)  # Use geomspace for log scale plotting
    radius_sphere = 10.0  # e.g., in nm, so q is in nm^-1

    pq_sphere = sphere_pq(q_values, radius_sphere)

    # Check P(0)
    print(
        f"P(q=0.001) for sphere (R={radius_sphere}): {pq_sphere[0]:.4f}"
    )  # Should be close to 1
    # For q=0 itself, it's handled to be 1.
    q_test_zero = np.array([0.0, 0.0001])
    pq_test_zero = sphere_pq(q_test_zero, radius_sphere)
    print(f"P(q=0) directly: {pq_test_zero[0]}")
    assert np.isclose(pq_test_zero[0], 1.0)

    plt.figure(figsize=(8, 6))
    plt.loglog(q_values, pq_sphere, label=f"Sphere P(q), R={radius_sphere} nm")
    plt.xlabel("q (nm$^{-1}$)")
    plt.ylabel("P(q)")
    plt.title("Form Factor for a Sphere")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.ylim(1e-5, 2)  # P(q) can go below 1e-4
    plt.show()

    # Test radius validation
    try:
        sphere_pq(q_values, -1.0)
    except ValueError as e:
        print(f"Caught expected error for negative radius: {e}")
