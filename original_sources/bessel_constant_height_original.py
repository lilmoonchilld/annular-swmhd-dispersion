#!/usr/bin/env python3
"""Constant-height annular SWMHD spectrum with branch-family colours and annotations.

Classification follows the requested ordering at the seed azimuthal mode m=1:
  * closest-to-zero positive/negative roots: Alfvén branches (black);
  * next positive/negative roots: magneto-Kelvin branches (red);
  * all remaining roots: magneto-Poincaré branches (blue).

The root solver itself is retained; annotations are placed in open parts of the
plot and their arrows target the numerically tracked branches.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.special import jv, yv
from scipy.optimize import brentq, newton

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.labelsize": 40,
    "axes.titlesize": 40,
    "xtick.labelsize": 35,
    "ytick.labelsize": 35,
    "axes.linewidth": 1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "figure.dpi": 150,
    "savefig.dpi": 600,
})

# Physical parameters
Omega = 5.0e-5
f = 2.0 * Omega
H0, g, B0, rho0, mu0 = 500.0, 9.81, 8.5e-4, 1000.0, 4.0 * np.pi * 1.0e-7
r1v, r2v = 5.0e5, 1.0e6
r0 = 0.5 * (r1v + r2v)

MIN_OMEGA_NORM = 1.0e-5
JUMP_TOL_FRAC = 0.5
FRESH_LINK_TOL_FRAC = 1.50
OMEGA_POLE_TOL = 1.0e-20
omegaA2 = B0**2 / (mu0 * rho0 * H0**2)


def omega_star(w):
    return np.inf if abs(w) < 1.0e-30 else w - omegaA2 / w


def K2(w, m):
    ws = omega_star(w)
    if not np.isfinite(ws) or abs(ws) < 1.0e-30:
        return np.nan
    denom = g * H0 * ws
    if abs(denom) < 1.0e-30:
        return np.nan
    return w * (ws**2 - f**2) / denom


def radial_k(w, m):
    k2 = K2(w, m)
    return np.nan + 0j if not np.isfinite(k2) else np.sqrt(complex(k2))


def boundary_J(w, m, r):
    ws, k = omega_star(w), radial_k(w, m)
    if not np.isfinite(ws) or not np.isfinite(k):
        return np.nan + 0j
    return ws * k * jv(m - 1, k * r) - (m * (ws + f) / r) * jv(m, k * r)


def boundary_Y(w, m, r):
    ws, k = omega_star(w), radial_k(w, m)
    if not np.isfinite(ws) or not np.isfinite(k):
        return np.nan + 0j
    return ws * k * yv(m - 1, k * r) - (m * (ws + f) / r) * yv(m, k * r)


def disp(w, m):
    if not np.isfinite(w) or abs(w) < 1.0e-12:
        return np.nan
    ws = omega_star(w)
    if not np.isfinite(ws) or abs(ws**2 - f**2) < 1.0e-12 * f**2:
        return np.nan
    k = radial_k(w, m)
    if not np.isfinite(k):
        return np.nan
    det = (
        boundary_J(w, m, r1v) * boundary_Y(w, m, r2v)
        - boundary_Y(w, m, r1v) * boundary_J(w, m, r2v)
    )
    return float(np.real(det)) if np.isfinite(det) else np.nan


def make_frequency_grid():
    # Dense near zero, where the Alfvén roots and omega=0 pole need resolution.
    grid = np.concatenate([
        np.linspace(-25.0, -0.2, 5000),
        np.linspace(-0.2, -1.0e-5, 60000),
        np.linspace(-1.0e-5, 0.2, 60000),
        np.linspace(0.2, 25.0, 5000),
    ])
    return np.unique(np.sort(grid)) * f


_FREQUENCY_GRID = make_frequency_grid()


def valid_bracket(wa, wb):
    if wa * wb <= 0 or min(abs(wa), abs(wb)) < OMEGA_POLE_TOL * f:
        return False
    resa, resb = omega_star(wa)**2 - f**2, omega_star(wb)**2 - f**2
    return not (
        not np.isfinite(resa) or not np.isfinite(resb) or resa * resb < 0
    )


def disp_array(w_array, m):
    """Vectorized determinant evaluation for rapid root bracketing."""
    w_array = np.asarray(w_array, dtype=float)
    result = np.full(w_array.shape, np.nan, dtype=float)
    mask = np.isfinite(w_array) & (np.abs(w_array) > 1.0e-12)
    if not np.any(mask):
        return result
    wv = w_array[mask]
    ws = wv - omegaA2 / wv
    ok = np.isfinite(ws) & (np.abs(ws**2 - f**2) >= 1.0e-12 * f**2) & (np.abs(ws) > 1.0e-30)
    if not np.any(ok):
        return result
    wv, ws = wv[ok], ws[ok]
    k2 = wv * (ws**2 - f**2) / (g * H0 * ws)
    good = np.isfinite(k2)
    if not np.any(good):
        return result
    wv, ws, k2 = wv[good], ws[good], k2[good]
    k = np.sqrt(k2.astype(complex))
    kr1, kr2 = k * r1v, k * r2v
    bj1 = ws * k * jv(m - 1, kr1) - (m * (ws + f) / r1v) * jv(m, kr1)
    by1 = ws * k * yv(m - 1, kr1) - (m * (ws + f) / r1v) * yv(m, kr1)
    bj2 = ws * k * jv(m - 1, kr2) - (m * (ws + f) / r2v) * jv(m, kr2)
    by2 = ws * k * yv(m - 1, kr2) - (m * (ws + f) / r2v) * yv(m, kr2)
    det = bj1 * by2 - by1 * bj2
    # Map only entries that passed every validity check back to the original array.
    original_indices = np.flatnonzero(mask)[np.flatnonzero(ok)[good]]
    finite = np.isfinite(det)
    result[original_indices[finite]] = np.real(det[finite])
    return result


def full_scan(m, w_lo, w_hi):
    grid = _FREQUENCY_GRID
    grid = grid[(grid >= w_lo) & (grid <= w_hi)]
    grid = grid[np.abs(grid / f) >= MIN_OMEGA_NORM]
    vals = disp_array(grid, m)
    left, right = grid[:-1], grid[1:]
    a, b = vals[:-1], vals[1:]
    candidate = np.isfinite(a) & np.isfinite(b) & (a * b < 0)
    candidate &= (left * right > 0)
    # Reject intervals crossing omega_*^2=f^2 poles/resonances.
    ws_left = left - omegaA2 / left
    ws_right = right - omegaA2 / right
    res_left, res_right = ws_left**2 - f**2, ws_right**2 - f**2
    candidate &= np.isfinite(res_left) & np.isfinite(res_right) & (res_left * res_right >= 0)
    candidate_indices = np.flatnonzero(candidate)
    roots = []
    for i in candidate_indices:
        wa, wb = left[i], right[i]
        try:
            root = brentq(
                lambda w: disp(w, m), wa, wb,
                xtol=1.0e-14 * f, rtol=1.0e-12, maxiter=400
            )
            if np.isfinite(root) and abs(root / f) >= MIN_OMEGA_NORM:
                roots.append(root)
        except (ValueError, RuntimeError, FloatingPointError, OverflowError):
            pass
    cleaned = []
    for root in sorted(roots):
        if not cleaned or abs(root - cleaned[-1]) > 1.0e-4 * f:
            cleaned.append(root)
    return cleaned


def continue_root(m_new, prev_points):
    if len(prev_points) == 1:
        pred = prev_points[-1][1]
        x1 = pred * 1.0001 if pred != 0 else 1.0e-3 * f
    else:
        (ma, wa), (mb, wb) = prev_points[-2], prev_points[-1]
        pred = wb + ((wb - wa) / (mb - ma)) * (m_new - mb)
        x1 = wb
    try:
        root = newton(
            lambda w: disp(w, m_new), x0=pred, x1=x1,
            tol=1.0e-13 * f, maxiter=60
        )
        if (
            np.isfinite(root)
            and abs(root / f) >= MIN_OMEGA_NORM
            and abs(root) >= OMEGA_POLE_TOL * f
            and abs(root - pred) <= JUMP_TOL_FRAC * f
        ):
            return root
    except (ValueError, RuntimeError, OverflowError, ZeroDivisionError):
        return None
    return None


def get_y_at(branch, m_target):
    """Return the tracked y-value (omega/f) nearest the requested m."""
    if not branch:
        return None
    m, w = min(branch, key=lambda item: abs(item[0] - m_target))
    return w / f


if __name__ == "__main__":
    m_values = list(range(1, 16))
    w_window = (-25.0 * f, 25.0 * f)
    DETECT_EVERY = 1

    print("\nConstant-height annular SWMHD: H_eq = H0, v_r(r1)=v_r(r2)=0")
    print(f"Omega = {Omega:.6e}; f = {f:.6e}; H0 = {H0:.6e}; B0 = {B0:.6e} T")
    print(f"omega_A/f = {np.sqrt(omegaA2) / f:.6f}")

    branches = {}
    next_id = 0
    seed_m = m_values[0]
    seed_roots = full_scan(seed_m, *w_window)
    positive_seed_roots = sorted(r for r in seed_roots if r > 0)
    negative_seed_roots = sorted((r for r in seed_roots if r < 0), reverse=True)

    if len(positive_seed_roots) < 2 or len(negative_seed_roots) < 2:
        raise RuntimeError(
            "At least two positive and two negative roots are needed to assign "
            "Alfvén and magneto-Kelvin pairs. Inspect the seed roots."
        )

    # The user-specified identification is made by root order at m=1.
    # The Alfvén branches are the two near-zero seed roots supplied by the
    # original branch identification. The second root in each sign is almost
    # horizontal too, so it is NOT automatically assumed to be magneto-Kelvin.
    alfven_pos = positive_seed_roots[0]
    alfven_neg = negative_seed_roots[0]  # negatives are sorted by proximity to zero
    alfven_ids = set()
    alfven_pos_id = alfven_neg_id = None

    for root in seed_roots:
        branches[next_id] = [(seed_m, root)]
        if np.isclose(root, alfven_pos, rtol=1e-10, atol=1e-14):
            alfven_ids.add(next_id); alfven_pos_id = next_id
        elif np.isclose(root, alfven_neg, rtol=1e-10, atol=1e-14):
            alfven_ids.add(next_id); alfven_neg_id = next_id
        next_id += 1

    print(f"m=1 roots: {len(seed_roots)}")
    print(f"Alfvén seed frequencies omega/f = {alfven_neg/f:.6f}, {alfven_pos/f:.6f}")

    for step, m in enumerate(m_values[1:], 1):
        prev_active = [
            (bid, pts[-1][1]) for bid, pts in branches.items()
            if pts[-1][0] == m - 1
        ]
        for bid, _ in prev_active:
            nr = continue_root(m, branches[bid])
            if nr is not None:
                branches[bid].append((m, nr))

        if step % DETECT_EVERY == 0:
            fresh_roots = full_scan(m, *w_window)
            curr_vals = [
                pts[-1][1] for pts in branches.values() if pts[-1][0] == m
            ]
            linked_bids = set()
            for root in fresh_roots:
                if any(abs(root - existing) <= 0.25 * f for existing in curr_vals):
                    continue
                candidates = [
                    (abs(root - wp), bid)
                    for bid, wp in prev_active
                    if branches[bid][-1][0] == m - 1
                    and bid not in linked_bids
                    and abs(root - wp) <= FRESH_LINK_TOL_FRAC * f
                ]
                if candidates:
                    bid = min(candidates, key=lambda item: item[0])[1]
                    branches[bid].append((m, root))
                    linked_bids.add(bid)
                else:
                    branches[next_id] = [(m, root)]
                    next_id += 1
                curr_vals.append(root)
        alive = sum(1 for pts in branches.values() if pts[-1][0] == m)
        print(f"m={m}: {alive} branch(es) alive")

    # Identify the first genuinely dispersive positive/negative branch after
    # the Alfvén roots. This skips the nearly horizontal near-resonant roots.
    def seed_frequency(bid):
        return branches[bid][0][1]

    def branch_change(bid):
        pts = sorted(branches[bid])
        return abs(pts[-1][1] - pts[0][1]) / f if len(pts) >= 2 else 0.0

    positive_candidates = [
        bid for bid, pts in branches.items()
        if pts[0][0] == 1 and pts[0][1] > 0 and bid not in alfven_ids
        and len(pts) >= 10 and branch_change(bid) > 0.5
    ]
    negative_candidates = [
        bid for bid, pts in branches.items()
        if pts[0][0] == 1 and pts[0][1] < 0 and bid not in alfven_ids
        and len(pts) >= 10 and branch_change(bid) > 0.5
    ]
    if not positive_candidates or not negative_candidates:
        raise RuntimeError("Could not identify the first dispersive magneto-Kelvin pair.")
    mk_pos_id = min(positive_candidates, key=seed_frequency)
    mk_neg_id = max(negative_candidates, key=seed_frequency)  # negative closest by frequency
    mk_ids = {mk_pos_id, mk_neg_id}
    print(f"Magneto-Kelvin seed frequencies omega/f = {seed_frequency(mk_neg_id)/f:.6f}, {seed_frequency(mk_pos_id)/f:.6f}")

    fig, ax = plt.subplots(figsize=(11.5, 7.5))
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.14, top=0.97)

    for bid, points in branches.items():
        if len(points) < 2:
            continue
        points = sorted(points)
        ms = [m for m, _ in points]
        ws_norm = [w / f for _, w in points]
        if bid in alfven_ids:
            colour, linewidth, marker_size, zorder = "black", 2.8, 24, 8
        elif bid in mk_ids:
            colour, linewidth, marker_size, zorder = "red", 2.8, 24, 7
        else:
            colour, linewidth, marker_size, zorder = "darkblue", 2.2, 18, 3
        ax.plot(ms, ws_norm, color=colour, lw=linewidth, zorder=zorder)
        ax.scatter(ms, ws_norm, color=colour, s=marker_size, zorder=zorder + 1)

    # Use actual tracked points for arrow targets. Place labels in the relatively
    # empty gaps between the central Alfvén pair and the rising wave families.
    alfven_y = get_y_at(branches.get(alfven_pos_id, []), 12)
    mk_y = get_y_at(branches.get(mk_pos_id, []), 8)
    poincare_candidates = [
        bid for bid, points in branches.items()
        if points[0][0] == 1 and points[0][1] > 0
        and bid not in alfven_ids and bid not in mk_ids
        and len(points) >= 10 and branch_change(bid) > 0.5
    ]
    poincare_candidates.sort(key=seed_frequency)
    poi_bid = poincare_candidates[0] if poincare_candidates else None

    if alfven_y is not None:
        ax.annotate(
            "Alfvén waves", xy=(12, alfven_y), xytext=(12.7, -7.8),
            fontsize=25, color="black", ha="center", va="center",
            arrowprops=dict(arrowstyle="->", lw=1.4, color="black",
                            shrinkA=4, shrinkB=4),
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.82),
        )
        alfven_neg_y = get_y_at(branches.get(alfven_neg_id, []), 12)
        if alfven_neg_y is not None:
            ax.annotate(
                "", xy=(12, alfven_neg_y), xytext=(12.7, -2.8),
                arrowprops=dict(arrowstyle="->", lw=1.2, color="black",
                                shrinkA=4, shrinkB=4),
            )

    if mk_y is not None:
        ax.annotate(
            "Magneto-Kelvin", xy=(8, mk_y), xytext=(7.4, 2.8),
            fontsize=25, color="red", ha="center", va="center",
            arrowprops=dict(arrowstyle="->", lw=1.5, color="red",
                            shrinkA=4, shrinkB=4),
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.82),
        )
        mk_neg_y = get_y_at(branches.get(mk_neg_id, []), 8)
        if mk_neg_y is not None:
            ax.annotate(
                "", xy=(8, mk_neg_y), xytext=(7.4, -2.8),
                arrowprops=dict(arrowstyle="->", lw=1.3, color="red",
                                shrinkA=4, shrinkB=4),
            )

    if poi_bid is not None:
        poi_branch = branches[poi_bid]
        poi_point = min(poi_branch, key=lambda item: abs(item[0] - 6))
        ax.annotate(
            "Magneto-Poincaré modes\n",
            xy=(poi_point[0], poi_point[1] / f), xytext=(11.15, 4.35),
            fontsize=25, color="darkblue", ha="center", va="center", rotation=4,
            arrowprops=dict(arrowstyle="->", lw=1.4, color="darkblue",
                            shrinkA=4, shrinkB=4),
            #bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.96),
        )

    # Radial-order cue moved to the upper-middle open region, as in the markup.
    # The arrow points toward a higher positive-frequency Poincare branch.
    ax.annotate(
        r"increasing $n$", xy=(7.8, 19.8), xytext=(8.65, 14.1),
        fontsize=22, color="darkblue", ha="center", va="center",
        arrowprops=dict(arrowstyle="->", lw=1.4, color="darkblue",
                        shrinkA=4, shrinkB=4, connectionstyle="arc3,rad=-0.12"),
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.92),
    )
    # A matching lower-side arrow shows increasing radial order away from zero
    # on the negative-frequency Poincare family as well.
    # ax.annotate(
    #     "", xy=(7.8, -19.2), xytext=(8.25, -11.5),
    #     arrowprops=dict(arrowstyle="->", lw=1.35, color="darkblue",
    #                     shrinkA=2, shrinkB=2, connectionstyle="arc3,rad=0.08"),
    # )

    ax.set_xlabel(r"Azimuthal wavenumber $m$", labelpad=12)
    ax.set_ylabel(r"$\omega/f$", labelpad=12)
    ax.set_xlim(1, 15)
    ax.set_xticks([1, 5, 10, 15])
    ax.grid(True, alpha=0.28)
    ax.set_axisbelow(True)

    # Data-driven vertical limits with breathing room.
    all_y = [w / f for points in branches.values() for _, w in points]
    if all_y:
        ymin, ymax = min(all_y), max(all_y)
        pad = 0.05 * (ymax - ymin)
        ax.set_ylim(ymin - pad, ymax + pad)

    fig.savefig("annular_swmhd_constant_H0_annotated.png", dpi=600, bbox_inches="tight")
    plt.show()
    print("Saved annular_swmhd_constant_H0_annotated.png")
    print(f"Tracked branch count: {len(branches)}")
