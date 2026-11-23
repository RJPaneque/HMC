"""
Minimal Dirac-Hartree-Fock-Slater atom (spherical average, Latter tail,
uniformly charged nucleus).  Used only to obtain bound-electron radial
amplitudes at the nuclear surface, from which the *relative* electron-capture
probabilities of sub-shells inside a major shell are derived (L1:L2:L3, M:N+).

Atomic units throughout (hbar = m = e = 1, c = 137.035999).
"""
import math
import numpy as np
from numba import njit

C = 137.035999084
BOHR_FM = 52917.721          # 1 a.u. of length in fm
HARTREE_KEV = 0.0272113862   # 1 a.u. of energy in keV


@njit(cache=True)
def _integrate(E, kappa, l, r, rh, V, Vh, h, V0):
    """RK4 outward integration in t = ln r.  Returns P, Q arrays and node count
    of P.  r/V on grid points, rh/Vh on half points."""
    n = r.shape[0]
    P = np.zeros(n)
    Q = np.zeros(n)
    a0 = (E - V0 + 2 * C * C) / C
    b0 = (E - V0) / C
    r0 = r[0]
    if kappa < 0:
        P[0] = r0 ** (l + 1)
        Q[0] = -b0 / (2 * l + 3) * r0 ** (l + 2)
    else:
        Q[0] = r0 ** l
        P[0] = a0 / (2 * l + 1) * r0 ** (l + 1)
    nodes = 0
    for i in range(n - 1):
        p = P[i]
        q = Q[i]
        # k1
        ra = r[i]
        va = V[i]
        k1p = -kappa * p + ra * (E - va + 2 * C * C) / C * q
        k1q = kappa * q - ra * (E - va) / C * p
        rb = rh[i]
        vb = Vh[i]
        pp = p + 0.5 * h * k1p
        qq = q + 0.5 * h * k1q
        k2p = -kappa * pp + rb * (E - vb + 2 * C * C) / C * qq
        k2q = kappa * qq - rb * (E - vb) / C * pp
        pp = p + 0.5 * h * k2p
        qq = q + 0.5 * h * k2q
        k3p = -kappa * pp + rb * (E - vb + 2 * C * C) / C * qq
        k3q = kappa * qq - rb * (E - vb) / C * pp
        rc = r[i + 1]
        vc = V[i + 1]
        pp = p + h * k3p
        qq = q + h * k3q
        k4p = -kappa * pp + rc * (E - vc + 2 * C * C) / C * qq
        k4q = kappa * qq - rc * (E - vc) / C * pp
        P[i + 1] = p + h / 6.0 * (k1p + 2 * k2p + 2 * k3p + k4p)
        Q[i + 1] = q + h / 6.0 * (k1q + 2 * k2q + 2 * k3q + k4q)
        if P[i + 1] * p < 0.0:
            nodes += 1
        if abs(P[i + 1]) > 1e200:
            # diverged; rescale impossible -> stop, fill rest with last value
            for j in range(i + 2, n):
                P[j] = P[i + 1]
                Q[j] = Q[i + 1]
            break
    return P, Q, nodes


def solve_orbital(nq, kappa, r, rh, V, Vh, h, V0, Elo, Ehi):
    """Bisection on the node count.  Returns E, normalised P, Q."""
    l = kappa if kappa > 0 else -kappa - 1
    want = nq - l - 1
    for _ in range(200):
        E = 0.5 * (Elo + Ehi)
        P, Q, nodes = _integrate(E, kappa, l, r, rh, V, Vh, h, V0)
        if nodes > want:
            Ehi = E
        else:
            Elo = E
        if Ehi - Elo < 1e-13 * max(1.0, abs(E)):
            break
    E = Elo
    P, Q, nodes = _integrate(E, kappa, l, r, rh, V, Vh, h, V0)
    # cut the diverging tail: beyond the outer classical turning point take the
    # point where |P| is smallest and zero everything after it
    Veff = V + l * (l + 1) / (2 * r * r)
    inside = np.where(Veff < E)[0]
    itp = inside[-1] if len(inside) else len(r) // 2
    tail = np.abs(P[itp:])
    icut = itp + int(np.argmin(tail))
    P = P.copy()
    Q = Q.copy()
    P[icut:] = 0.0
    Q[icut:] = 0.0
    norm = np.trapezoid((P * P + Q * Q) * r, dx=h)
    s = 1.0 / math.sqrt(norm)
    return E, P * s, Q * s


def configuration(Z):
    """(n, kappa, occupation) list, spherical average of the ground config."""
    order = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (3, 2), (4, 1),
             (5, 0), (4, 2), (5, 1), (6, 0), (4, 3), (5, 2), (6, 1), (7, 0)]
    special = {24: {(3, 2): 5, (4, 0): 1}, 29: {(3, 2): 10, (4, 0): 1},
               41: {(4, 2): 4, (5, 0): 1}, 42: {(4, 2): 5, (5, 0): 1}}
    occ = {}
    left = Z
    for (n, l) in order:
        if left <= 0:
            break
        cap = 2 * (2 * l + 1)
        k = min(cap, left)
        occ[(n, l)] = k
        left -= k
    if Z in special:
        occ.update(special[Z])
    shells = []
    for (n, l), k in sorted(occ.items()):
        if l == 0:
            shells.append((n, -1, float(k)))
        else:
            tot = 2 * (2 * l + 1)
            shells.append((n, l, k * (2 * l) / tot))           # j = l-1/2
            shells.append((n, -(l + 1), k * (2 * l + 2) / tot))  # j = l+1/2
    return shells


def dhfs(Z, A, xalpha=1.0, latter=True, h=0.006, rmax=60.0, verbose=False):
    Rn = 1.2 * A ** (1.0 / 3.0) / BOHR_FM
    r0 = Rn / 200.0
    n = int(math.log(rmax / r0) / h) + 1
    t = math.log(r0) + h * np.arange(n)
    r = np.exp(t)
    rh = np.exp(t + 0.5 * h)

    def vnuc(x):
        return np.where(x < Rn, -Z / (2 * Rn) * (3 - (x / Rn) ** 2), -Z / x)

    def moliere(x):
        a = 0.8853 * Z ** (-1.0 / 3.0)
        y = x / a
        return 0.35 * np.exp(-0.3 * y) + 0.55 * np.exp(-1.2 * y) + 0.10 * np.exp(-6 * y)

    shells = configuration(Z)
    Nel = sum(s[2] for s in shells)
    # screening (electronic) potential on the full + half grid
    Ve = (Z / r) * (1 - moliere(r))
    Veh = (Z / rh) * (1 - moliere(rh))
    energies = {(nq, k): None for (nq, k, _) in shells}
    for it in range(200):
        V = vnuc(r) + Ve
        Vh = vnuc(rh) + Veh
        if latter:
            tail = -(Z - Nel + 1) / r
            tailh = -(Z - Nel + 1) / rh
            V = np.minimum(V, tail)
            Vh = np.minimum(Vh, tailh)
        V0 = V[0]
        sig = np.zeros(n)
        orbs = {}
        for (nq, k, occ) in shells:
            e_old = energies[(nq, k)]
            if e_old is None:
                Elo, Ehi = -Z * Z * 1.2, -1e-6
            else:
                d = max(0.3 * abs(e_old), 0.05)
                Elo, Ehi = e_old - d, min(e_old + d, -1e-7)
            E, P, Q = solve_orbital(nq, k, r, rh, V, Vh, h, V0, Elo, Ehi)
            if e_old is not None and (abs(E - Elo) < 1e-9 or abs(E - Ehi) < 1e-9):
                E, P, Q = solve_orbital(nq, k, r, rh, V, Vh, h, V0, -Z * Z * 1.2, -1e-7)
            energies[(nq, k)] = E
            orbs[(nq, k)] = (E, P, Q)
            sig += occ * (P * P + Q * Q)
        # Hartree potential from radial density sig (int sig dr = N)
        inner = np.concatenate(([0.0], np.cumsum(0.5 * h * (sig[1:] * r[1:] + sig[:-1] * r[:-1]))))
        g = sig  # sig/r * r dt
        outer_rev = np.concatenate(([0.0], np.cumsum(0.5 * h * (g[::-1][1:] + g[::-1][:-1]))))
        outer = outer_rev[::-1]
        VH = inner / r + outer
        rho = sig / (4 * math.pi * r * r)
        Vx = -1.5 * xalpha * (3 * rho / math.pi) ** (1.0 / 3.0)
        Ve_new = VH + Vx
        diff = np.max(np.abs((Ve_new - Ve) * r))
        mix = 0.35
        Ve = Ve + mix * (Ve_new - Ve)
        Veh = np.exp(np.interp(t + 0.5 * h, t, np.log(np.maximum(Ve, 1e-300)))) if np.all(Ve > 0) else np.interp(t + 0.5 * h, t, Ve)
        if verbose and it % 10 == 0:
            print(it, diff)
        if diff < 2e-6:
            break
    iR = int(np.argmin(np.abs(r - Rn)))
    out = {}
    for (nq, k, occ) in shells:
        E, P, Q = orbs[(nq, k)]
        out[(nq, k)] = dict(E=E, occ=occ, g=P[iR] / r[iR], f=Q[iR] / r[iR])
    return out, Rn


def capture_weights(Z, A, **kw):
    """Relative capture weights per sub-shell.
    allowed  : beta^2 (g^2 for kappa=-1, f^2 for kappa=+1), filled shells
    unique 1 : coefficient of 1/q^2 for kappa=-2 (p3/2) relative to the s1/2
               weights, in keV^2 (see ensdf2nuc.py for the use).
    """
    orbs, Rn = dhfs(Z, A, **kw)
    res = {}
    for (nq, k), o in orbs.items():
        if k == -1:
            res[(nq, 's1/2')] = dict(E=o['E'], w=o['g'] ** 2 * o['occ'] / 2.0)
        elif k == 1:
            res[(nq, 'p1/2')] = dict(E=o['E'], w=o['f'] ** 2 * o['occ'] / 2.0)
        elif k == -2:
            # (g/R)^2 in a.u.^-2 -> (hbar c / a0)^2 = (alpha m c^2)^2 keV^2
            conv = (HARTREE_KEV * C) ** 2
            res[(nq, 'p3/2')] = dict(E=o['E'], w=(o['g'] / Rn) ** 2 * conv * o['occ'] / 4.0)
    return res


if __name__ == '__main__':
    import sys
    for Z, A in ((25, 52), (39, 86)):
        w = capture_weights(Z, A, verbose=False)
        print('Z =', Z)
        for key in sorted(w):
            print('  %d%s  E = %10.4f keV   w = %.6e' % (key[0], key[1], w[key]['E'] * HARTREE_KEV, w[key]['w']))
