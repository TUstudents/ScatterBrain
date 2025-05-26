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

import numpy as np
from scipy.special import j1 # Bessel function of the first kind, order 1

# Small constant to prevent division by zero or issues with q=0
# for functions where P(q=0) is handled by a limit.
_Q_EPSILON = 1e-9


def sphere_pq(q: np.ndarray, radius: float) -> np.ndarray:
    """
    Calculate the form factor P(q) for a monodisperse sphere.

    The form factor for a sphere is given by:
    P(q) = [ (3 * (sin(qR) - qR*cos(qR))) / (qR)^3 ]^2
         = [ (9 * pi / 2) * (J_{3/2}(qR))^2 / (qR)^3 ]
         where J_{3/2}(x) = sqrt(2/(pi*x)) * (sin(x)/x - cos(x))
         This simplifies to the first expression.
         Also, J_1(x) is the Bessel function of first kind, order 1.
         sqrt(pi*x/2) * J_{3/2}(x) = sin(x)/x - cos(x)
         Another common representation using j1 (spherical Bessel function of order 1):
         P(q) = [3 * j1(qR) / (qR)]^2
         where j1(x) = (sin(x) - x*cos(x)) / x^2. Note that scipy.special.j1 is J_1, not j_1.

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
    pq[mask] = (3.0 * j1_x / x)**2 # This expression does not look right for P(0)=1 without limit
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

    x = qr[mask] # x = qR
    func_val = 3.0 * (np.sin(x) - x * np.cos(x)) / (x**3)
    pq[mask] = func_val**2

    return pq


# Example usage:
if __name__ == "__main__": # pragma: no cover
    import matplotlib.pyplot as plt

    q_values = np.geomspace(0.001, 1.0, 200) # Use geomspace for log scale plotting
    radius_sphere = 10.0 # e.g., in nm, so q is in nm^-1

    pq_sphere = sphere_pq(q_values, radius_sphere)

    # Check P(0)
    print(f"P(q=0.001) for sphere (R={radius_sphere}): {pq_sphere[0]:.4f}") # Should be close to 1
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
    plt.ylim(1e-5, 2) # P(q) can go below 1e-4
    plt.show()

    # Test radius validation
    try:
        sphere_pq(q_values, -1.0)
    except ValueError as e:
        print(f"Caught expected error for negative radius: {e}")