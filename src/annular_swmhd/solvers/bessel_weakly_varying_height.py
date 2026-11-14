import numpy as np; import matplotlib.pyplot as plt
from scipy.special import jv, yv, iv, kv, jvp, yvp, ivp, kvp
from scipy.optimize import brentq, newton

Omega = 5.0e-5; f = 2.0 * Omega; D0 = 500.0; g = 9.81; Ccg = 1.8e9; B0 = 8.5e-4; rho0 = 1000.0
r1v = 5.0e5; r2v = 1.0e6; r0 = 7.5e5; mu0 = 4.0 * np.pi * 1.0e-7; betaEff = Omega**2
def H_eq_prime(r): return Omega**2 * r / g
omegaA2 = B0**2 / (mu0 * rho0 * D0**2)
SLOW_CUTOFF = 0.05;MIN_OMEGA_NORM = 1e-5; JUMP_TOL_FRAC = 0.5; K2_TOL = 1.0e-25; OMEGA_POLE_TOL = 1.0e-20;FRESH_LINK_TOL_FRAC = 1.50

def omega_star(w): return w - omegaA2 / w if abs(w) >= 1.0e-30 else np.inf
def Gfac(w): 
    ws = omega_star(w); denom = w - (ws**2 - f**2)
    return 1.0 + 4.0 * omegaA2 * ws / denom if abs(denom) >= 1.0e-30 else np.nan
def K2(w, m):
    ws, G = omega_star(w), Gfac(w)
    if not np.isfinite(ws) or not np.isfinite(G) or abs(ws) < 1.0e-30: return np.nan
    return (w * (ws**2 - f**2) - m * f * Omega**2 * G) / (g * D0 * ws)
def gradient_term_direct(w, m, r):
    ws, G = omega_star(w), Gfac(w)
    return (m * f * H_eq_prime(r) * G) / (r * D0 * ws) if np.isfinite(ws) and np.isfinite(G) else np.nan
def radial_basis(w, m):
    k2 = K2(w, m)
    if not np.isfinite(k2): return None
    if k2 > K2_TOL:
        k = np.sqrt(k2)
        return (lambda r: jv(m, k*r), lambda r: (m/r)*jv(m, k*r) - k*jv(m+1, k*r), lambda r: yv(m, k*r), lambda r: (m/r)*yv(m, k*r) - k*yv(m+1, k*r), "propagating")
    elif k2 < -K2_TOL:
        kap = np.sqrt(-k2)
        return (lambda r: iv(m, kap*r), lambda r: kap*iv(m-1, kap*r) - (m/r)*iv(m, kap*r), lambda r: kv(m, kap*r), lambda r: -kap*kv(m-1, kap*r) - (m/r)*kv(m, kap*r), "evanescent")
    return (lambda r: 1.0, lambda r: 0.0, lambda r: np.log(r), lambda r: 1.0/r, "K2_zero") if m == 0 else (lambda r: r**m, lambda r: m*r**(m-1), lambda r: r**(-m), lambda r: -m*r**(-m-1), "K2_zero")
def N_operator(eta_fun, deta_fun, r, w, m): return omega_star(w) * deta_fun(r) - (m * f / r) * eta_fun(r)
def disp(w, m):
    if abs(w) < 1.0e-12: return np.nan
    ws = omega_star(w)
    if not np.isfinite(ws) or abs(ws**2 - f**2) < 1.0e-12 * f**2: return np.nan
    basis = radial_basis(w, m)
    if basis is None: return np.nan
    e1, de1, e2, de2, _ = basis
    return float(np.real(N_operator(e1, de1, r1v, w, m) * N_operator(e2, de2, r2v, w, m) - N_operator(e1, de1, r2v, w, m) * N_operator(e2, de2, r1v, w, m)))
def make_frequency_grid(): return np.unique(np.sort(np.concatenate([np.linspace(-25.0, -0.2, 5000), np.linspace(-0.2, -1.0e-5, 60000), np.linspace(-1.0e-5, 0.2, 60000), np.linspace(0.2, 25.0, 5000)]))) * f
def classify(w, m):
    if abs(w / f) < SLOW_CUTOFF: return "slow"
    k2 = K2(w, m)
    return "invalid" if not np.isfinite(k2) else "evanescent" if k2 < 0 else "propagating"
def valid_bracket(wa, wb, m):
    if wa * wb <= 0 or min(abs(wa), abs(wb)) < OMEGA_POLE_TOL * f: return False
    resa, resb = omega_star(wa)**2 - f**2, omega_star(wb)**2 - f**2
    if not np.isfinite(resa) or not np.isfinite(resb) or resa * resb < 0: return False
    k2a, k2b = K2(wa, m), K2(wb, m)
    return False if not np.isfinite(k2a) or not np.isfinite(k2b) or k2a * k2b < 0 else True

def full_scan(m, w_lo, w_hi, n_scan=None):
    ws = make_frequency_grid(); ws = ws[(ws >= w_lo) & (ws <= w_hi)];# Exclude |omega/f| < 1e-4
    ws = ws[np.abs(ws / f) >= MIN_OMEGA_NORM]; vals = np.array([disp(w, m) for w in ws]); roots = []
    for i in range(len(ws) - 1):
        wa, wb = ws[i], ws[i + 1]; a, b = vals[i], vals[i + 1]
        if not np.isfinite(a) or not np.isfinite(b) or not valid_bracket(wa, wb, m) or a * b >= 0: continue
        try:
            root = brentq(lambda w: disp(w, m), wa, wb, xtol=1e-14*f, rtol=1e-12, maxiter=400)
            if np.isfinite(root) and abs(root / f) >= MIN_OMEGA_NORM: roots.append(root)
        except Exception: pass
    cleaned = []
    for r in sorted(roots):
        if not cleaned or abs(r - cleaned[-1]) > 1e-4 * f: cleaned.append(r)
    return cleaned

def continue_root(m_new, prev_points):
    if len(prev_points) == 1:
        pred = prev_points[-1][1]; x1 = pred * (1.0 + 1e-4) if pred != 0 else 1e-3 * f
    else:
        (ma, wa), (mb, wb) = prev_points[-2], prev_points[-1]; pred = wb + ((wb - wa) / (mb - ma)) * (m_new - mb); x1 = wb
    try: root = newton(lambda w: disp(w, m_new), x0=pred, x1=x1, tol=1e-13*f, maxiter=60)
    except Exception: return None
    return root if (np.isfinite(root) and abs(root / f) >= MIN_OMEGA_NORM and abs(root) >= OMEGA_POLE_TOL * f
                    and abs(root - pred) <= JUMP_TOL_FRAC * f) else None

def resonance_frequencies():
    disc = f**2 - 4.0 * omegaA2
    if disc < 0:
        return []
    sqrt_disc = np.sqrt(disc)
    return [(r"$\omega_* = +f$ (upper)",  (f + sqrt_disc) / 2.0, "red"),(r"$\omega_* = +f$ (lower)",  (f - sqrt_disc) / 2.0, "orange"),
        (r"$\omega_* = -f$ (upper)", (-f + sqrt_disc) / 2.0, "blue"),(r"$\omega_* = -f$ (lower)", (-f - sqrt_disc) / 2.0, "purple"),]

if __name__ == "__main__":
    m_values = list(range(1, 31)); w_window = (-15 * f, 15 * f); N_SCAN_SEED = 2000; N_SCAN = 1000; DETECT_EVERY = 1
    print(f"\n{'='*70}\nC = 0, H_eq'(r) != 0, B0 != 0\n{'='*70}\nOmega = {Omega:.6e}\nf = {f:.6e}\nD0 = {D0:.6e}\nB0 = {B0:.6e}\nrho0 = {rho0:.6e}\nC = {Ccg:.6e}\n\nbeta_eff = {betaEff:.6e}\nomega_A^2 = {omegaA2:.6e}\nomega_A = {np.sqrt(omegaA2):.6e}\n\nH_eq'(r) = Omega^2 r / g\nLambda = 0\n\n{'='*70}")
    print(f"\nGradient check at r0:\nH_eq'(r0) = {H_eq_prime(r0)}\nH_eq'(r0)/r0 = {H_eq_prime(r0)/r0}\nOmega^2/g = {Omega**2/g}")
    branches = {}; next_id = 0; seed_m = m_values[0]; seed_roots = full_scan(seed_m, *w_window, n_scan=N_SCAN_SEED)
    for r in seed_roots: branches[next_id] = [(seed_m, r)]; next_id += 1
    print(f"\nm={seed_m}: seeded {len(seed_roots)} branch(es)")

    FRESH_LINK_TOL_FRAC = 1.50

    for step, m in enumerate(m_values[1:], 1):
    
        # Branches that existed at the previous m
        prev_active = [
            (bid, pts[-1][1])
            for bid, pts in branches.items()
            if pts[-1][0] == m - 1
        ]
    
        # ------------------------------------------------------------
        # 1. Normal continuation
        # ------------------------------------------------------------
        for bid, _ in prev_active:
            pts = branches[bid]
            nr = continue_root(m, pts)
            if nr is not None:
                branches[bid].append((m, nr))
        # ------------------------------------------------------------
        # 2. Fresh scan
        # ------------------------------------------------------------
        if step % DETECT_EVERY == 0:
            fresh_roots = full_scan(
                m, *w_window, n_scan=N_SCAN
            )
            curr_vals = [
                pts[-1][1]
                for pts in branches.values()
                if pts[-1][0] == m
            ]
            # Keep track of branches already linked to a fresh root
            linked_bids = set()
            for r in fresh_roots:
                # ----------------------------------------------------
                # Already represented by an existing root at this m
                # ----------------------------------------------------
                if any(
                    abs(r - ex) <= 0.25 * f
                    for ex in curr_vals
                ):
                    continue
                # ----------------------------------------------------
                # Try to reconnect this fresh root to a branch
                # that existed at m-1 but whose Newton continuation
                # failed.
                # ----------------------------------------------------
                candidates = []
                for bid, w_prev in prev_active:
    
                    # Branch has not already been continued
                    if branches[bid][-1][0] != m - 1:
                        continue
                    if bid in linked_bids:
                        continue
                    distance = abs(r - w_prev)
                    if distance <= FRESH_LINK_TOL_FRAC * f:
                        candidates.append((distance, bid))
                # ----------------------------------------------------
                # Reconnect to nearest previous branch
                # ----------------------------------------------------
                if candidates:
                    _, bid = min(candidates, key=lambda x: x[0])   
                    branches[bid].append((m, r))
                    linked_bids.add(bid)
                    curr_vals.append(r)   
                # ----------------------------------------------------
                # Otherwise this is genuinely a new branch
                # ----------------------------------------------------
                else:
                    branches[next_id] = [(m, r)]
                    curr_vals.append(r)
                    next_id += 1
    
        print(f"m={m}: "f"{sum(1 for pts in branches.values() if pts[-1][0] == m)} "f"branch(es) alive")
        
    # for step, m in enumerate(m_values[1:], 1):
    #     for bid, pts in {k: v for k, v in branches.items() if v[-1][0] == m - 1}.items():
    #         nr = continue_root(m, pts)
    #         if nr is not None: branches[bid].append((m, nr))
    #     if step % DETECT_EVERY == 0:
    #         fresh_roots = full_scan(m, *w_window, n_scan=N_SCAN)
    #         curr_vals = [pts[-1][1] for pts in branches.values() if pts[-1][0] == m]
    #         for r in fresh_roots:
    #             if all(abs(r - ex) > JUMP_TOL_FRAC * f for ex in curr_vals): branches[next_id] = [(m, r)]; curr_vals.append(r); next_id += 1
    #     print(f"m={m}: {sum(1 for pts in branches.values() if pts[-1][0] == m)} branch(es) alive")
        
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 10), sharex=True); colors = {"evanescent": "green", "propagating": "blue", "slow": "red"}
    for bid, pts in branches.items():
        if len(pts) < 2: continue
        ms_prop, ws_prop = [], []
        for m, w in pts:
            if classify(w, m) in ("propagating", "slow"): ms_prop.append(m); ws_prop.append(w / f)
            else:
                if len(ms_prop) >= 2: ax1.plot(ms_prop, ws_prop, color="darkblue", lw=3, zorder=1); ax2.plot(ms_prop, ws_prop, color="darkblue", lw=3, zorder=1)
                ms_prop, ws_prop = [], []
        if len(ms_prop) >= 2: ax1.plot(ms_prop, ws_prop, color="darkblue", lw=3, zorder=1); ax2.plot(ms_prop, ws_prop, color="darkblue", lw=3, zorder=1)
        
    for lbl, col in colors.items():
        xs, ys = zip(*[(m, w/f) for pts in branches.values() for m, w in pts if classify(w, m) == lbl]) if any(classify(w, m) == lbl for pts in branches.values() for m, w in pts) else ([], [])
        if xs: ax1.scatter(xs, ys, color=col, s=8, label=lbl, zorder=2); ax2.scatter(xs, ys, color=col, s=8, label=lbl, zorder=2)
        
    # for label, wres, color in resonance_frequencies():
    #     normalized = wres / f
    #     ax1.axhline(normalized,color=color, linestyle="--",linewidth=1.5,zorder=0,label=rf"{label}: $\omega/f={normalized:.6f}$")
    #     ax2.axhline(normalized,color=color,linestyle="--",linewidth=1.5,zorder=0,label=rf"{label}: $\omega/f={normalized:.6f}$")    
    ax1.axhline(0, color="black", lw=0.5); ax2.axhline(0, color="black", lw=0.5)
    ax1.set(ylabel=r"$\omega/f$", title=r"Annular SWMHD dispersion: $C=0,\ H_{\rm eq}'\neq0,\ B_0\neq0$")
    ax2.set(xlabel=r"$m$ (azimuthal wavenumber)", ylabel=r"$\omega/f$", title=r"Zoomed view: $-1.5\leq\omega/f\leq1.5$")
    ax1.legend(); ax2.legend(); ax1.grid(True, alpha=0.2); ax2.grid(True, alpha=0.2);
    ax1.set_xlim(1, 30);ax2.set_xlim(1, 30);ax1.set_ylim(-20, 20);ax2.set_ylim(-1, 1); fig.tight_layout(); 
    fig.savefig("annular_swmhd_C0_gradient_B0_nonzero.png", dpi=150, bbox_inches="tight"); plt.show()
    print(f"\nTotal branches tracked (including length-1 fragments): {len(branches)}\nBranch lengths (top 10): {sorted((len(p) for p in branches.values()), reverse=True)[:10]}")