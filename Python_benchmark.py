"""
Benchmark for the "Bring-Jerrard quintic via Jacobi theta functions" write-up.

Part A  Step-by-step check of every number quoted in the proof (c = 1).
Part B  Does the final theta formula actually return the real root of y^5 + y + c?
        (sweep over many c, error measured against a 60-digit reference)
Part C  Speed and accuracy vs. standard root finders:
        Newton, Durand-Kerner, Aberth-Ehrlich, numpy.roots, and the theta pipeline.

Requires: numpy, scipy.   Run:  python3 benchmark_quintic.py
"""
import math
import time
import timeit
from decimal import Decimal, getcontext

import numpy as np
from scipy.optimize import brentq
from scipy.special import ellipk, ellipkm1  # ellipk(m) uses the PARAMETER m = k^2

getcontext().prec = 60


# --------------------------------------------------------------------------
# Reference solution (60-digit Newton on the single real root of y^5 + y + c)
# --------------------------------------------------------------------------
def reference_real_root(c):
    c = Decimal(repr(float(c)))
    y = Decimal(-1) if c > 0 else Decimal(1)
    for _ in range(300):
        step = (y**5 + y + c) / (5 * y**4 + 1)
        y -= step
        if abs(step) < Decimal(10) ** -55:
            break
    return y


# --------------------------------------------------------------------------
# Jacobi theta functions (nome q), plain-float series
# --------------------------------------------------------------------------
def theta2(q):
    s, n = 0.0, 0
    while n < 5000:
        t = q ** (n * (n + 1))
        s += t
        if t < 1e-18:
            break
        n += 1
    return 2.0 * q**0.25 * s


def theta3(q):
    s, n = 1.0, 1
    while n < 5000:
        t = 2.0 * q ** (n * n)
        s += t
        if t < 1e-18:
            break
        n += 1
    return s


def theta4(q):
    s, n = 1.0, 1
    while n < 5000:
        t = 2.0 * q ** (n * n)
        s += t if n % 2 == 0 else -t
        if t < 1e-18:
            break
        n += 1
    return s


# --------------------------------------------------------------------------
# The proof's pipeline
# --------------------------------------------------------------------------
def c_of_u(u):
    return (1 - u**2) / (2 * math.sqrt(u) * (1 + u**2) ** 0.75)


def solve_u(c):
    return brentq(lambda u: c_of_u(u) - c, 1e-14, 1 - 1e-14, xtol=1e-15, rtol=1e-14)


def nome_from_u(u):
    k = u**2                      # modulus, as in the proof
    m = k**2                      # scipy wants m = k^2
    K = ellipk(m)
    Kp = ellipkm1(m)              # K'(k) = K(sqrt(1-k^2)), accurate for tiny m
    return k, K, Kp, math.exp(-math.pi * Kp / K)


def theta_formula(q):
    """y0 = - th2(q^1/5) th3(q^1/5) / (th2(q) th3(q)) * th4(q^5)/th4(q)   (proof, section 2)"""
    q5r = q ** 0.2
    ratio = (theta2(q5r) * theta3(q5r)) / (theta2(q) * theta3(q))
    Phi = theta4(q**5) / theta4(q)
    return -ratio * Phi


def theta_root(c):
    u = solve_u(c)
    _, _, _, q = nome_from_u(u)
    return theta_formula(q)


# --------------------------------------------------------------------------
# Standard root finders for p(y) = y^5 + y + c
# --------------------------------------------------------------------------
def newton_real(c, y0=-1.0, tol=1e-15):
    y = y0
    for _ in range(100):
        step = (y**5 + y + c) / (5 * y**4 + 1)
        y -= step
        if abs(step) < tol:
            break
    return y


COEF = lambda c: np.array([1, 0, 0, 0, 1, c], dtype=complex)


def durand_kerner(c, tol=1e-14, maxit=500):
    a = COEF(c)
    n = 5
    z = (0.4 + 0.9j) ** np.arange(n)
    for _ in range(maxit):
        pz = np.polyval(a, z)
        diff = z[:, None] - z[None, :]
        np.fill_diagonal(diff, 1)
        dz = pz / diff.prod(axis=1)
        z = z - dz
        if np.max(np.abs(dz)) < tol:
            break
    return z


def aberth_ehrlich(c, tol=1e-14, maxit=200):
    a = COEF(c)
    da = np.polyder(a)
    n = 5
    z = 1.0 * np.exp(2j * np.pi * (np.arange(n) + 0.25) / n)
    for _ in range(maxit):
        pz, dpz = np.polyval(a, z), np.polyval(da, z)
        w = pz / dpz
        diff = z[:, None] - z[None, :]
        np.fill_diagonal(diff, np.inf)
        s = (1.0 / diff).sum(axis=1)
        dz = w / (1 - w * s)
        z = z - dz
        if np.max(np.abs(dz)) < tol:
            break
    return z


def numpy_roots(c):
    return np.roots([1, 0, 0, 0, 1, c])


def real_root_from(roots):
    r = roots[np.argmin(np.abs(roots.imag))]
    return r.real


# --------------------------------------------------------------------------
def hr(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def part_a():
    hr("PART A - step-by-step check of the numbers quoted in the proof (c = 1)")
    c = 1.0
    ref = float(reference_real_root(c))

    u = solve_u(c)
    k, K, Kp, q = nome_from_u(u)
    print(f"{'quantity':<28}{'proof claims':>16}{'computed':>20}")
    rows = [
        ("u  (from c = 1)", 0.1306, u),
        ("c(u) at the claimed u=0.1306", 1.0, c_of_u(0.1306)),
        ("k = u^2", 0.01706, k),
        ("q = exp(-pi K'/K)", 1.82e-5, q),
        ("q^(1/5)", 0.1132, q**0.2),
        ("theta2(q^(1/5))", 1.1788, theta2(q**0.2)),
        ("theta3(q^(1/5))", 1.2267, theta3(q**0.2)),
        ("theta2(q)", 0.1306, theta2(q)),
        ("theta3(q)", 1.0000, theta3(q)),
    ]
    for name, claim, val in rows:
        print(f"{name:<28}{claim:>16.6g}{val:>20.10g}")

    y_formula = theta_formula(q)
    y_claim = -0.754877669
    print()
    print(f"true real root (60-digit Newton) : {ref:.12f}")
    print(f"value claimed in the proof       : {y_claim:.12f}   |err| = {abs(y_claim-ref):.2e}")
    print(f"theta formula evaluated exactly  : {y_formula:.12f}   |err| = {abs(y_formula-ref):.2e}")
    print(f"residual |y^5+y+1| at claimed    : {abs(y_claim**5+y_claim+1):.2e}")
    print(f"residual |y^5+y+1| at formula    : {abs(y_formula**5+y_formula+1):.2e}")

    # factorisation sanity check: y^5+y+1 = (y^2+y+1)(y^3-y^2+1)
    r3 = np.roots([1, -1, 0, 1])
    print(f"\nsanity: y^5+y+1 = (y^2+y+1)(y^3-y^2+1); real root of the cubic = "
          f"{r3[np.argmin(abs(r3.imag))].real:.12f}")
    print("(note: for c = 1 the quintic is reducible, so it is a special, easy test case)")


def part_b():
    hr("PART B - does the theta formula reproduce the real root for general c?")
    cs = np.logspace(-0.7, 2, 25)
    print(f"{'c':>10}{'true root':>16}{'theta formula':>16}{'rel. error':>14}")
    worst = 0.0
    for c in cs:
        ref = float(reference_real_root(c))
        try:
            val = theta_root(c)
            err = abs(val - ref) / abs(ref)
        except Exception as e:  # pragma: no cover
            val, err = float("nan"), float("inf")
        worst = max(worst, err)
        print(f"{c:>10.4f}{ref:>16.9f}{val:>16.9f}{err:>14.2e}")
    print(f"\nworst relative error over sweep: {worst:.2e}")
    print("PASS" if worst < 1e-8 else "FAIL: the closed-form expression does not return the root")


def bench(fn, number=2000, repeat=5):
    t = min(timeit.repeat(fn, number=number, repeat=repeat)) / number
    return t * 1e6  # microseconds


def part_c():
    hr("PART C - speed and accuracy vs standard root finders (y^5 + y + 1)")
    c = 1.0
    ref = float(reference_real_root(c))
    ref_all = np.sort_complex(np.roots([1, 0, 0, 0, 1, c]))

    def max_err_all(z):
        return float(np.max(np.abs(np.sort_complex(z) - ref_all)))

    methods = [
        ("Newton (1 real root)", lambda: newton_real(c), lambda: abs(newton_real(c) - ref), 20000),
        ("Durand-Kerner (all 5)", lambda: durand_kerner(c), lambda: max_err_all(durand_kerner(c)), 1000),
        ("Aberth-Ehrlich (all 5)", lambda: aberth_ehrlich(c), lambda: max_err_all(aberth_ehrlich(c)), 1000),
        ("numpy.roots (all 5)", lambda: numpy_roots(c), lambda: max_err_all(numpy_roots(c)), 5000),
        ("theta pipeline (1 real root)", lambda: theta_root(c), lambda: abs(theta_root(c) - ref), 500),
    ]
    print(f"{'method':<32}{'time (us)':>12}{'abs error':>14}{'x slower than Newton':>24}")
    base = None
    for name, call, err, number in methods:
        call()  # warm-up
        t = bench(call, number=number)
        base = base or t
        print(f"{name:<32}{t:>12.2f}{err():>14.2e}{t/base:>24.1f}")

    # iteration counts
    def count_iters(step_fn, init, tol=1e-14):
        z, n = init, 0
        while n < 500:
            z, d = step_fn(z)
            n += 1
            if d < tol:
                break
        return n

    n_newton = count_iters(lambda y: (y - (y**5 + y + c) / (5 * y**4 + 1),
                                      abs((y**5 + y + c) / (5 * y**4 + 1))), -1.0)
    print(f"\nNewton iterations to 1e-14: {n_newton}")


if __name__ == "__main__":
    t0 = time.time()
    part_a()
    part_b()
    part_c()
    print(f"\n(total runtime {time.time()-t0:.1f}s)")