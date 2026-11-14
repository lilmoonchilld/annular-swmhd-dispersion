"""
Annular SWMHD radial eigenvalue problem -- dispersion relation with
mode continuation across the azimuthal wavenumber m.

Equation:   eta'' + (1/r) eta' + (K^2 + Lam/r - m^2/r^2) eta = 0
BC:         v_r(r1) = v_r(r2) = 0
            v_r = -i g [ omega* eta' - (m f / r) eta ] / (omega*^2 - f^2)

BRANCH TRACKING (downgraded on purpose, matches the Bessel script)
--------------------------------------------------------------------
Earlier versions of this script identified branches by radial node
count n (via count_radial_nodes / count_radial_nodes_hp, an
adaptive-precision mpmath eigenfunction evaluation), so that a branch
surviving a close crossing with another branch would not be silently
mislabeled.

That machinery has been removed. Branches are now tracked purely by
omega-continuity, exactly like the Bessel-function annular script:
  1. Seed roots at the first m via a full frequency scan.
  2. At each subsequent m, extrapolate every live branch's next omega
     via secant/Newton (continue_root) -- no check on what the
     eigenfunction looks like.
  3. Run a fresh full scan at that m. Any root not already matched to
     a continued branch is linked to the nearest branch that just
     died (within FRESH_LINK_TOL_FRAC*f), or starts a new branch.

KNOWN TRADEOFF (accepted, not fixed): two branches that pass close to
each other in (m, omega) can swap identities at the crossing, since
nothing here distinguishes them beyond proximity in omega. This is the
same blind spot the Bessel script has -- it is not a new failure mode
introduced by removing the node-count guard, just no longer patched.

Everything else (the fast mp.fp scan path, the exact mpmath disp()
used for root refinement, the double-precision vectorized evanescent
path) is unchanged from the previous version.
"""

import numpy as np
from scipy.special import hyp1f1, hyperu
from scipy.optimize import brentq, newton
import mpmath as mp
import matplotlib.pyplot as plt

mp.mp.dps = 25

# ---------------- physical parameters (user-supplied) ----------------
Omega = 5.0e-5
f     = 2 * Omega            # = 1.0e-4, matches the f = 2*Omega convention
D0    = 500.0                # H0
g     = 9.81
B0    = 8.5e-4
rho0  = 1000.0
r1v   = 5.0e5
r2v   = 1.0e6
r0    = 7.5e5
Ccg   = 9.375e8

mu0 = 4 * np.pi * 1e-7

betaEff = Omega**2 + 2 * Ccg / r0**3       # beta_eff = Om^2 + 2C/r0^3
omegaA2 = B0**2 / (mu0 * rho0 * D0**2)

SLOW_CUTOFF          = 0.05   # |omega/f| below this => "slow" branch
JUMP_TOL_FRAC        = 0.5    # reject a continuation step if |Delta omega| > this * f
FRESH_LINK_TOL_FRAC  = 1.50   # link a fresh-scan root to a dead branch within this * f
EXIST_MATCH_TOL_FRAC = 0.25   # a fresh-scan root this close to an already-placed root is a duplicate
OMEGA_ZERO_TOL       = 5e-4
RES_TOL              = 1e-8


# ==========================================================
# Singular-frequency lines: omega_* = +/- f
# ==========================================================
def resonance_frequencies():
    disc = f**2 - 4.0 * omegaA2
    if disc < 0:
        return []
    sqrt_disc = np.sqrt(disc)
    return [
        (r"$\omega_* = +f$ (upper)", (f + sqrt_disc) / 2.0, "red"),
        (r"$\omega_* = +f$ (lower)", (f - sqrt_disc) / 2.0, "orange"),
        (r"$\omega_* = -f$ (upper)", (-f + sqrt_disc) / 2.0, "blue"),
        (r"$\omega_* = -f$ (lower)", (-f - sqrt_disc) / 2.0, "purple"),
    ]


# ---------------- Whittaker functions: scalar (fast hybrid scipy/mpmath) --
def _is_complexish(x, tol=1e-15):
    return np.iscomplexobj(x) and abs(np.imag(x)) > tol


def frequency_is_safe(w):
    """False when omega is too close to omega=0 or omega_*^2 = f^2."""
    if not np.isfinite(w):
        return False
    if abs(w) < OMEGA_ZERO_TOL * f:
        return False
    ws = omega_star(w)
    res = ws**2 - f**2
    if abs(res) < RES_TOL * f**2:
        return False
    return True


# ---------------- combined value+derivative (mp context, no redundant calls) --
def whitM_and_dz(k, mu, z):
    """Returns (M_{k,mu}(z), d/dz M_{k,mu}(z)) with a single shared M0 call."""
    if _is_complexish(k) or _is_complexish(z):
        try:
            zz = complex(z); kk = complex(k)
            pref = mp.e**(-zz / 2) * mp.power(zz, mu + 0.5)
            M0 = mp.hyp1f1(mu - kk + 0.5, 2 * mu + 1, zz)
            M1 = mp.hyp1f1(mu - kk + 1.5, 2 * mu + 2, zz)
            val = pref * M0
            dval = ((mu + 0.5) / zz - 0.5) * val + pref * ((mu - kk + 0.5) / (2 * mu + 1.0)) * M1
            return complex(val), complex(dval)
        except Exception:
            return np.nan + 1j * np.nan, np.nan + 1j * np.nan
    try:
        zz = float(np.real(z)); kk = float(np.real(k))
        pref = np.exp(-zz / 2.0) * zz**(mu + 0.5)
        M0 = hyp1f1(mu - kk + 0.5, 2 * mu + 1, zz)
        M1 = hyp1f1(mu - kk + 1.5, 2 * mu + 2, zz)
        val = pref * M0
        dval = ((mu + 0.5) / zz - 0.5) * val + pref * ((mu - kk + 0.5) / (2 * mu + 1.0)) * M1
        if not (np.isfinite(val) and np.isfinite(dval)):
            return np.nan + 1j * np.nan, np.nan + 1j * np.nan
        return val, dval
    except Exception:
        return np.nan + 1j * np.nan, np.nan + 1j * np.nan


def whitW_and_dz(k, mu, z):
    """Returns (W_{k,mu}(z), d/dz W_{k,mu}(z)) with a single shared U0 call."""
    if _is_complexish(k) or _is_complexish(z):
        try:
            zz = complex(z); kk = complex(k)
            pref = mp.e**(-zz / 2) * mp.power(zz, mu + 0.5)
            U0 = mp.hyperu(mu - kk + 0.5, 2 * mu + 1, zz)
            U1 = mp.hyperu(mu - kk + 1.5, 2 * mu + 2, zz)
            val = pref * U0
            dval = ((mu + 0.5) / zz - 0.5) * val - pref * (mu - kk + 0.5) * U1
            return complex(val), complex(dval)
        except Exception:
            return np.nan + 1j * np.nan, np.nan + 1j * np.nan
    try:
        zz = float(np.real(z)); kk = float(np.real(k))
        pref = np.exp(-zz / 2.0) * zz**(mu + 0.5)
        U0 = hyperu(mu - kk + 0.5, 2 * mu + 1, zz)
        U1 = hyperu(mu - kk + 1.5, 2 * mu + 2, zz)
        val = pref * U0
        dval = ((mu + 0.5) / zz - 0.5) * val - pref * (mu - kk + 0.5) * U1
        if not (np.isfinite(val) and np.isfinite(dval)):
            return np.nan + 1j * np.nan, np.nan + 1j * np.nan
        return val, dval
    except Exception:
        return np.nan + 1j * np.nan, np.nan + 1j * np.nan


# ---------------- fast (double-precision) scan-only path, mpmath's fp context --
def whitM_and_dz_fp(k, mu, z):
    try:
        pref = mp.fp.e**(-z / 2) * z**(mu + 0.5)
        M0 = mp.fp.hyp1f1(mu - k + 0.5, 2 * mu + 1, z)
        M1 = mp.fp.hyp1f1(mu - k + 1.5, 2 * mu + 2, z)
        val = pref * M0
        dval = ((mu + 0.5) / z - 0.5) * val + pref * ((mu - k + 0.5) / (2 * mu + 1.0)) * M1
        return val, dval
    except Exception:
        return complex(np.nan, np.nan), complex(np.nan, np.nan)


def whitW_and_dz_fp(k, mu, z):
    try:
        pref = mp.fp.e**(-z / 2) * z**(mu + 0.5)
        U0 = mp.fp.hyperu(mu - k + 0.5, 2 * mu + 1, z)
        U1 = mp.fp.hyperu(mu - k + 1.5, 2 * mu + 2, z)
        val = pref * U0
        dval = ((mu + 0.5) / z - 0.5) * val - pref * (mu - k + 0.5) * U1
        return val, dval
    except Exception:
        return complex(np.nan, np.nan), complex(np.nan, np.nan)


def disp_fp(w, m):
    """Fast, approximate (double-precision mp.fp) dispersion function --
    SCAN ONLY. Do not use its output as a final root; use it only to find
    the sign of disp() cheaply so full_scan can bracket, then hand the
    bracket to the exact disp() for brentq refinement."""
    if not frequency_is_safe(w):
        return np.nan
    try:
        k2 = K2(w, m)
        rdv = complex(1.0 / (2.0 * mp.fp.sqrt(complex(-k2))))
        kap = Lam(w, m) * rdv
        mu = abs(m)
        wse = omega_star(w)

        z1 = r1v / rdv; z2 = r2v / rdv
        M1, M1p = whitM_and_dz_fp(kap, mu, z1)
        M2, M2p = whitM_and_dz_fp(kap, mu, z2)
        W1, W1p = whitW_and_dz_fp(kap, mu, z1)
        W2, W2p = whitW_and_dz_fp(kap, mu, z2)

        etaM1 = M1 / np.sqrt(r1v); etaM1p = M1p / (rdv * np.sqrt(r1v)) - M1 / (2.0 * r1v**1.5)
        etaM2 = M2 / np.sqrt(r2v); etaM2p = M2p / (rdv * np.sqrt(r2v)) - M2 / (2.0 * r2v**1.5)
        etaW1 = W1 / np.sqrt(r1v); etaW1p = W1p / (rdv * np.sqrt(r1v)) - W1 / (2.0 * r1v**1.5)
        etaW2 = W2 / np.sqrt(r2v); etaW2p = W2p / (rdv * np.sqrt(r2v)) - W2 / (2.0 * r2v**1.5)

        G1 = wse * etaM1p - (m * f / r1v) * etaM1
        G2 = wse * etaM2p - (m * f / r2v) * etaM2
        B1 = wse * etaW1p - (m * f / r1v) * etaW1
        B2 = wse * etaW2p - (m * f / r2v) * etaW2

        D = G1 * B2 - G2 * B1
        if not np.isfinite(D.real):
            return np.nan
        return D.real
    except Exception:
        return np.nan


# ---------------- Whittaker functions: vectorized real-domain path --------
def whitM_arr(k, mu, z):
    k = np.asarray(k, dtype=float); z = np.asarray(z, dtype=float)
    with np.errstate(all="ignore"):
        val = np.exp(-z / 2.0) * z**(mu + 0.5) * hyp1f1(mu - k + 0.5, 2 * mu + 1, z)
    return np.where(np.isfinite(val), val, np.nan)


def whitW_arr(k, mu, z):
    k = np.asarray(k, dtype=float); z = np.asarray(z, dtype=float)
    with np.errstate(all="ignore"):
        val = np.exp(-z / 2.0) * z**(mu + 0.5) * hyperu(mu - k + 0.5, 2 * mu + 1, z)
    return np.where(np.isfinite(val), val, np.nan)


def whitM_dz_arr(k, mu, z):
    k = np.asarray(k, dtype=float); z = np.asarray(z, dtype=float)
    with np.errstate(all="ignore"):
        pref = np.exp(-z / 2.0) * z**(mu + 0.5)
        M0 = hyp1f1(mu - k + 0.5, 2 * mu + 1, z)
        M1 = hyp1f1(mu - k + 1.5, 2 * mu + 2, z)
        val = pref * M0
        dval = ((mu + 0.5) / z - 0.5) * val + pref * ((mu - k + 0.5) / (2 * mu + 1.0)) * M1
    return np.where(np.isfinite(dval), dval, np.nan)


def whitW_dz_arr(k, mu, z):
    k = np.asarray(k, dtype=float); z = np.asarray(z, dtype=float)
    with np.errstate(all="ignore"):
        pref = np.exp(-z / 2.0) * z**(mu + 0.5)
        U0 = hyperu(mu - k + 0.5, 2 * mu + 1, z)
        U1 = hyperu(mu - k + 1.5, 2 * mu + 2, z)
        val = pref * U0
        dval = ((mu + 0.5) / z - 0.5) * val - pref * (mu - k + 0.5) * U1
    return np.where(np.isfinite(dval), dval, np.nan)


# ---------------- model building blocks (scalar; also work elementwise) --
def omega_star(w):
    return w + omegaA2 / w


def Gfac(w):
    ws = omega_star(w)
    return 1 + 4 * omegaA2 * ws / (w * (ws**2 - f**2))


def K2(w, m):
    ws = omega_star(w)
    return (w * (ws**2 - f**2)) / (g * D0 * ws) - (m * f * betaEff * Gfac(w)) / (g * D0 * ws)


def Lam(w, m):
    ws = omega_star(w)
    return (3 * m * f * Ccg * Gfac(w)) / (g * D0 * ws * r0**2)


def N_from_eta_deta(eta, deta, r, w, m):
    return omega_star(w) * deta - (m * f / r) * eta


def eta_and_deta_M(r, w, m):
    rdv = 1 / (2 * np.sqrt(complex(-K2(w, m))))
    kk = Lam(w, m) * rdv; mu = abs(m); z = r / rdv
    F, Fz = whitM_and_dz(kk, mu, z)
    eta = F / np.sqrt(r)
    deta = Fz / (rdv * np.sqrt(r)) - F / (2.0 * r**1.5)
    return eta, deta


def eta_and_deta_W(r, w, m):
    rdv = 1 / (2 * np.sqrt(complex(-K2(w, m))))
    kk = Lam(w, m) * rdv; mu = abs(m); z = r / rdv
    F, Fz = whitW_and_dz(kk, mu, z)
    eta = F / np.sqrt(r)
    deta = Fz / (rdv * np.sqrt(r)) - F / (2.0 * r**1.5)
    return eta, deta


def disp(w, m):
    """Scalar dispersion function -- exact (mp context). Used for brentq/
    Newton refinement of a bracket already found by the fast scan, and as
    the ground-truth check anywhere accuracy (not speed) matters."""
    if not frequency_is_safe(w):
        return np.nan
    try:
        etaM1, detaM1 = eta_and_deta_M(r1v, w, m)
        etaM2, detaM2 = eta_and_deta_M(r2v, w, m)
        etaW1, detaW1 = eta_and_deta_W(r1v, w, m)
        etaW2, detaW2 = eta_and_deta_W(r2v, w, m)

        G1 = N_from_eta_deta(etaM1, detaM1, r1v, w, m)
        G2 = N_from_eta_deta(etaM2, detaM2, r2v, w, m)
        B1 = N_from_eta_deta(etaW1, detaW1, r1v, w, m)
        B2 = N_from_eta_deta(etaW2, detaW2, r2v, w, m)

        D = G1 * B2 - G2 * B1
        if not np.isfinite(np.real(D)) or not np.isfinite(np.imag(D)):
            return np.nan
        return np.real(D)
    except Exception:
        return np.nan


def disp_batch(w_arr, m):
    """Vectorized dispersion function over an array of real frequencies,
    for fixed m -- SCAN USE ONLY (propagating-regime values are
    double-precision approximate, adequate for a sign change, not for a
    final root)."""
    w_arr = np.asarray(w_arr, dtype=float)
    out = np.full(w_arr.shape, np.nan)
    if w_arr.size == 0:
        return out

    k2 = K2(w_arr, m)
    evan_mask = k2 < 0
    prop_mask = ~evan_mask

    if np.any(evan_mask):
        we = w_arr[evan_mask]
        k2e = k2[evan_mask]
        rde = 1.0 / (2.0 * np.sqrt(-k2e))
        mu = abs(m)
        kappae = Lam(we, m) * rde
        z1 = r1v / rde
        z2 = r2v / rde

        M1 = whitM_arr(kappae, mu, z1); M1p = whitM_dz_arr(kappae, mu, z1)
        M2 = whitM_arr(kappae, mu, z2); M2p = whitM_dz_arr(kappae, mu, z2)
        W1 = whitW_arr(kappae, mu, z1); W1p = whitW_dz_arr(kappae, mu, z1)
        W2 = whitW_arr(kappae, mu, z2); W2p = whitW_dz_arr(kappae, mu, z2)

        etaM1 = M1 / np.sqrt(r1v); etaM1p = M1p / (rde * np.sqrt(r1v)) - M1 / (2.0 * r1v**1.5)
        etaM2 = M2 / np.sqrt(r2v); etaM2p = M2p / (rde * np.sqrt(r2v)) - M2 / (2.0 * r2v**1.5)
        etaW1 = W1 / np.sqrt(r1v); etaW1p = W1p / (rde * np.sqrt(r1v)) - W1 / (2.0 * r1v**1.5)
        etaW2 = W2 / np.sqrt(r2v); etaW2p = W2p / (rde * np.sqrt(r2v)) - W2 / (2.0 * r2v**1.5)

        wse = omega_star(we)
        G1 = wse * etaM1p - (m * f / r1v) * etaM1
        G2 = wse * etaM2p - (m * f / r2v) * etaM2
        B1 = wse * etaW1p - (m * f / r1v) * etaW1
        B2 = wse * etaW2p - (m * f / r2v) * etaW2

        D = G1 * B2 - G2 * B1
        out[evan_mask] = D

    if np.any(prop_mask):
        idxs = np.where(prop_mask)[0]
        for i in idxs:
            out[i] = disp_fp(float(w_arr[i]), m)

    return out


def make_frequency_grid():
    x1 = np.linspace(-15.0, -0.2, 250)
    x2 = np.linspace(0.2, 15.0, 250)
    x3 = np.linspace(-0.2, -1.0e-3, 1000)
    x4 = np.linspace(1.0e-3, 0.2, 1000)
    x = np.concatenate([x1, x3, x4, x2])
    x = np.unique(np.sort(x))
    return x * f


def valid_bracket(wa, wb):
    """Reject brackets that: cross omega=0, approach omega=0 too closely,
    or cross/approach omega_*^2 = f^2."""
    if wa * wb <= 0:
        return False
    if min(abs(wa), abs(wb)) < OMEGA_ZERO_TOL * f:
        return False
    res_a = omega_star(wa)**2 - f**2
    res_b = omega_star(wb)**2 - f**2
    if abs(res_a) < RES_TOL * f**2 or abs(res_b) < RES_TOL * f**2:
        return False
    if res_a * res_b < 0:
        return False
    return True


def full_scan(m, w_lo, w_hi, skip_near=None):
    """skip_near: optional array of omega values already placed at this m
    (from continuation) -- grid points within JUMP_TOL_FRAC*f of any of
    them are skipped, since continuation already covers that root."""
    ws = make_frequency_grid()
    ws = ws[(ws >= w_lo) & (ws <= w_hi)]

    if skip_near is not None and len(skip_near) > 0:
        skip_arr = np.asarray(skip_near, dtype=float)
        far_enough = ~np.any(
            np.abs(ws[:, None] - skip_arr[None, :]) < JUMP_TOL_FRAC * f, axis=1
        )
        ws = ws[far_enough]

    if len(ws) < 2:
        return []

    safe_mask = np.array([frequency_is_safe(w) for w in ws])
    vals = np.full(len(ws), np.nan)
    if np.any(safe_mask):
        vals[safe_mask] = disp_batch(ws[safe_mask], m)

    roots = []
    for i in range(len(ws) - 1):
        wa, wb = ws[i], ws[i + 1]
        a, b = vals[i], vals[i + 1]
        if not np.isfinite(a) or not np.isfinite(b):
            continue
        if not valid_bracket(wa, wb):
            continue
        if a * b < 0:
            try:
                root = brentq(lambda w: disp(w, m), wa, wb,
                               xtol=1e-14 * f, rtol=1e-12, maxiter=200)
                if np.isfinite(root) and frequency_is_safe(root):
                    roots.append(root)
            except Exception:
                pass

    cleaned = []
    for root in sorted(roots):
        if not cleaned or abs(root - cleaned[-1]) > 1e-4 * f:
            cleaned.append(root)
    return cleaned


def continue_root(m_new, prev_points):
    """prev_points: list of (m, omega) for this branch, most recent last.
    Pure omega-continuity -- no check on what the eigenfunction looks
    like. Same contract as the Bessel script's continue_root."""
    if len(prev_points) == 1:
        predicted = prev_points[-1][1]
    else:
        (m_a, w_a), (m_b, w_b) = prev_points[-2], prev_points[-1]
        slope = (w_b - w_a) / (m_b - m_a)
        predicted = w_b + slope * (m_new - m_b)

    if not frequency_is_safe(predicted):
        return None

    if abs(predicted) < 1e-3 * f:
        x1 = predicted + 1e-3 * f
    else:
        x1 = predicted * (1 + 1e-4)

    try:
        root = newton(lambda w: disp(w, m_new), x0=predicted, x1=x1,
                       tol=1e-13 * f, maxiter=60)
    except Exception:
        return None

    if not np.isfinite(root) or not frequency_is_safe(root):
        return None
    if abs(root - predicted) > JUMP_TOL_FRAC * f:
        return None

    return root


if __name__ == "__main__":
    m_values = list(range(1, 21))
    w_window = (-15 * f, 15 * f)

    branches = {}   # branch_id -> list of (m, omega)
    next_id = 0
    
    seed_roots = full_scan(m_values[0], *w_window)
    
    if not seed_roots:
        raise RuntimeError("No roots found at seed m.")
    
    # ------------------------------------------------------------
    # Identify the n=0 positive and negative frequency roots
    # ------------------------------------------------------------
    positive_seed_roots = [r for r in seed_roots if r > 0]
    negative_seed_roots = [r for r in seed_roots if r < 0]
    
    if not positive_seed_roots:
        raise RuntimeError("No positive-frequency roots found at seed m.")
    
    if not negative_seed_roots:
        raise RuntimeError("No negative-frequency roots found at seed m.")
    
    # n=0 positive branch = smallest positive-frequency root
    n0_positive_root = min(positive_seed_roots)
    
    # n=0 negative branch = negative root closest to zero
    n0_negative_root = max(negative_seed_roots)
    
    n0_positive_branch_id = None
    n0_negative_branch_id = None
    
    # ------------------------------------------------------------
    # Seed all branches and record the n=0 branch IDs
    # ------------------------------------------------------------
    for root in seed_roots:
    
        branches[next_id] = [(m_values[0], root)]
    
        if np.isclose(
            root,
            n0_positive_root,
            rtol=1e-10,
            atol=1e-14
        ):
            n0_positive_branch_id = next_id
    
        if np.isclose(
            root,
            n0_negative_root,
            rtol=1e-10,
            atol=1e-14
        ):
            n0_negative_branch_id = next_id
    
        next_id += 1
    
    
    print(
        f"m={m_values[0]}: seeded {len(seed_roots)} branch(es)"
    )
    
    print(
        f"Positive n=0 branch ID = {n0_positive_branch_id}, "
        f"omega/f = {n0_positive_root/f:.8f}"
    )
    
    print(
        f"Negative n=0 branch ID = {n0_negative_branch_id}, "
        f"omega/f = {n0_negative_root/f:.8f}"
    )
    for m in m_values[1:]:
        prev_active = [(bid, pts[-1][1]) for bid, pts in branches.items() if pts[-1][0] == m - 1]

        for bid, _ in prev_active:
            nr = continue_root(m, branches[bid])
            if nr is not None:
                branches[bid].append((m, nr))

        curr_vals = [pts[-1][1] for pts in branches.values() if pts[-1][0] == m]
        fresh_roots = full_scan(m, *w_window, skip_near=curr_vals)

        linked_bids = set()
        for root in fresh_roots:
            if any(abs(root - existing) <= EXIST_MATCH_TOL_FRAC * f for existing in curr_vals):
                continue
            candidates = [
                (abs(root - wp), bid) for bid, wp in prev_active
                if branches[bid][-1][0] == m - 1 and bid not in linked_bids
                and abs(root - wp) <= FRESH_LINK_TOL_FRAC * f
            ]
            if candidates:
                bid = min(candidates, key=lambda x: x[0])[1]
                branches[bid].append((m, root))
                linked_bids.add(bid)
            else:
                branches[next_id] = [(m, root)]
                next_id += 1
            curr_vals.append(root)

        n_alive = sum(1 for pts in branches.values() if pts[-1][0] == m)
        print(f"m={m}: {n_alive} branch(es) alive", flush=True)
    # ============================================================
    # PLOT SETTINGS
    # ============================================================
    
    POS_COLOR = "purple"
    NEG_COLOR = "orange"
    
    BRANCH_LW = 2.5
    BRANCH_SCATTER = 10
    
    N0_LW = 3.0
    N0_SCATTER = 18
    
    AXIS_LABEL_SIZE = 22
    TICK_SIZE = 18
    TITLE_SIZE = 22
    
    # Branch to skip
    SKIP_BRANCH_INDEX = 4
    
    resonances = resonance_frequencies()
    
    
    # ============================================================
    # HELPER FUNCTION
    # ============================================================
    
    def plot_branches_on_axis(ax, zoom=False):
    
        for idx, (bid, pts) in enumerate(branches.items()):
    
            # ----------------------------------------------------
            # Skip the fifth branch
            # ----------------------------------------------------
            if idx == SKIP_BRANCH_INDEX:
                continue
    
            if len(pts) < 2:
                continue
    
            ms_b = np.array([m for m, w in pts])
            ws_b = np.array([w / f for m, w in pts])
    
            # ----------------------------------------------------
            # n = 0 identification
            # ----------------------------------------------------
            is_n0_positive = (bid == n0_positive_branch_id)
            is_n0_negative = (bid == n0_negative_branch_id)
    
            # ====================================================
            # POSITIVE FREQUENCY --> PURPLE
            # ====================================================
    
            positive_mask = ws_b > 0.0
    
            if np.any(positive_mask):
    
                ms_pos = ms_b[positive_mask]
                ws_pos = ws_b[positive_mask]
    
                lw = N0_LW if is_n0_positive else BRANCH_LW
                s = N0_SCATTER if is_n0_positive else BRANCH_SCATTER
    
                ax.plot(
                    ms_pos,
                    ws_pos,
                    color=POS_COLOR,
                    linewidth=lw,
                    zorder=2
                )
    
                ax.scatter(
                    ms_pos,
                    ws_pos,
                    color=POS_COLOR,
                    s=s,
                    zorder=3
                )
    
            # ====================================================
            # NEGATIVE FREQUENCY --> ORANGE
            # ====================================================
    
            negative_mask = ws_b < 0.0
    
            if np.any(negative_mask):
    
                ms_neg = ms_b[negative_mask]
                ws_neg = ws_b[negative_mask]
    
                lw = N0_LW if is_n0_negative else BRANCH_LW
                s = N0_SCATTER if is_n0_negative else BRANCH_SCATTER
    
                ax.plot(
                    ms_neg,
                    ws_neg,
                    color=NEG_COLOR,
                    linewidth=lw,
                    zorder=2
                )
    
                ax.scatter(
                    ms_neg,
                    ws_neg,
                    color=NEG_COLOR,
                    s=s,
                    zorder=3
                )
    
    
    # ============================================================
    # 1. FULL DISPERSION DIAGRAM
    # ============================================================
    
    fig1, ax1 = plt.subplots(figsize=(9, 6))
    
    plot_branches_on_axis(ax1)
    
    
    # ------------------------------------------------------------
    # Resonance lines
    # ------------------------------------------------------------
    
    for label, wres, color in resonances:
    
        normalized = wres / f
    
        ax1.axhline(
            normalized,
            color=color,
            linestyle="--",
            linewidth=1.5,
            zorder=1
        )
    
    
    # ------------------------------------------------------------
    # Zero-frequency line
    # ------------------------------------------------------------
    
    ax1.axhline(
        0,
        color="black",
        linewidth=0.5,
        zorder=0
    )
    
    
    # ------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------
    
    ax1.set_xlabel(
        r"$m$ (azimuthal wavenumber)",
        fontsize=AXIS_LABEL_SIZE
    )
    
    ax1.set_ylabel(
        r"$\omega/f$",
        fontsize=AXIS_LABEL_SIZE
    )
    
    ax1.set_title(
        "Annular SWMHD radial modes",
        fontsize=TITLE_SIZE
    )
    
    ax1.set_xlim(1, 15)
    
    ax1.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_SIZE
    )
    
    ax1.grid(
        True,
        alpha=0.3
    )
    
    fig1.tight_layout()
    
    fig1.savefig(
        "annular_swmhd_dispersion_full.png",
        dpi=600,
        bbox_inches="tight"
    )
    
    plt.show()
    
    
    # ============================================================
    # 2. ZOOMED DISPERSION DIAGRAM
    #    -1 <= omega/f <= 1
    # ============================================================
    
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    
    plot_branches_on_axis(ax2, zoom=True)

    # --- Branch annotations ---

    ax.annotate(
        r'Magneto-Poincaré / inertial branches',
        xy=(4.8, 0.685),
        xytext=(6.8, 0.77),
        arrowprops=dict(
            arrowstyle='->',
            lw=1.2
        ),
        fontsize=14,
        ha='center'
    )
    
    ax.annotate(
        r'Magnetically modified slow branches',
        xy=(6.5, 0.325),
        xytext=(8.8, 0.43),
        arrowprops=dict(
            arrowstyle='->',
            lw=1.2
        ),
        fontsize=14,
        ha='center'
    )
    
    ax.annotate(
        r'Magnetostrophic branch',
        xy=(7.5, -0.075),
        xytext=(9.2, -0.18),
        arrowprops=dict(
            arrowstyle='->',
            lw=1.2
        ),
        fontsize=14,
        ha='center'
    )
        
    
    # ------------------------------------------------------------
    # Resonance lines
    # Only show those inside the zoomed frequency range
    # ------------------------------------------------------------
    
    for label, wres, color in resonances:
    
        normalized = wres / f
    
        if -1.0 <= normalized <= 1.0:
    
            ax2.axhline(
                normalized,
                color=color,
                linestyle="--",
                linewidth=1.5,
                zorder=1
            )
    
    
    # ------------------------------------------------------------
    # Zero-frequency line
    # ------------------------------------------------------------
    
    ax2.axhline(
        0,
        color="black",
        linewidth=0.5,
        zorder=0
    )
    
    
    # ------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------
    
    ax2.set_xlabel(
        r"$m$ (azimuthal wavenumber)",
        fontsize=AXIS_LABEL_SIZE
    )
    
    ax2.set_ylabel(
        r"$\omega/f$",
        fontsize=AXIS_LABEL_SIZE
    )
    
    # ax2.set_title(
    #     r"Zoomed view: $-1 \leq \omega/f \leq 1$",
    #     fontsize=TITLE_SIZE
    # )
    
    ax2.set_xlim(1, 15)
    
    ax2.set_ylim(
        -0.5,
        1.0
    )
    
    ax2.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_SIZE
    )
    
    ax2.grid(
        True,
        alpha=0.3
    )
    
    fig2.tight_layout()
    
    fig2.savefig(
        "annular_swmhd_dispersion_zoom.png",
        dpi=600,
        bbox_inches="tight"
    )
    
    plt.show()
    
    
    # ============================================================
    # 3. SEPARATE LEGEND FIGURE
    # ============================================================
    
    from matplotlib.lines import Line2D
    
    legend_handles = [
    
        # --------------------------------------------------------
        # Positive and negative branches
        # --------------------------------------------------------
        Line2D(
            [0], [0],
            color=POS_COLOR,
            linewidth=2.5,
            marker="o",
            markersize=4,
            label=r"Positive-frequency roots: $\omega>0$"
        ),
    
        Line2D(
            [0], [0],
            color=NEG_COLOR,
            linewidth=2.5,
            marker="o",
            markersize=4,
            label=r"Negative-frequency roots: $\omega<0$"
        )
    ]
    
    
    # ------------------------------------------------------------
    # Add resonance lines
    # ------------------------------------------------------------
    
    for label, wres, color in resonances:
    
        normalized = wres / f
    
        legend_handles.append(
            Line2D(
                [0], [0],
                color=color,
                linestyle="--",
                linewidth=1.5,
                label=rf"{label}: $\omega/f={normalized:.6f}$"
            )
        )
    
    
    # ------------------------------------------------------------
    # Create legend-only figure
    # ------------------------------------------------------------
    
    fig_legend = plt.figure(figsize=(9, 3.2))
    
    fig_legend.legend(
        handles=legend_handles,
        loc="center",
        fontsize=12,
        frameon=True,
        ncol=1
    )
    
    fig_legend.subplots_adjust(
        left=0.02,
        right=0.98,
        top=0.98,
        bottom=0.02
    )
    
    fig_legend.savefig(
        "annular_swmhd_dispersion_legend.png",
        dpi=600,
        bbox_inches="tight"
    )
    
    plt.show()
    
    
    # ============================================================
    # OUTPUT FILES
    # ============================================================
    
    print("\nSaved figures:")
    
    print(
        "  1. annular_swmhd_dispersion_full.png"
    )
    
    print(
        "  2. annular_swmhd_dispersion_zoom.png"
    )
    
    print(
        "  3. annular_swmhd_dispersion_legend.png"
    )
    # ============================================================
    # EXPORT ROOTS TO CSV
    # ============================================================
    csv_filename = "annular_swmhd_roots_B_0=0.csv"
    with open(csv_filename, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Branch_ID", "m", "omega", "omega_over_f", "sign"])
        
        for bid, pts in sorted(branches.items()):
            for m, w in pts:
                sign = "positive" if w > 0 else "negative"
                writer.writerow([bid, m, w, w / f, sign])

    print(f"  4. {csv_filename}")