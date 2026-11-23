"""
Sub-shell internal-conversion systematics for one element, built from the
BrIcc-based numbers stored in the Geant4 PhotonEvaporation level files
(z<Z>.a<A>): for every transition of pure multipolarity the file gives the
total ICC and the K, L1, L2, L3, M1-M5 and outer-shell shares.

They are used here ONLY to
  * split an ENSDF L-shell coefficient (LC) into L1 : L2 : L3, and
  * split a bare total coefficient (CC without KC/LC/MC records) into shells.
"""
import glob
import math
import os
import numpy as np

CODE = {2: 'E1', 3: 'M1', 4: 'E2', 5: 'M2', 6: 'E3', 7: 'M3'}
SHELLS = ('K', 'L1', 'L2', 'L3', 'M', 'N')
PAIR_THRESHOLD = 1000.0     # above ~1022 keV the Geant4 total includes pair formation


class Table:
    def __init__(self, Z, datadir):
        self.Z = Z
        self.raw = {m: [] for m in CODE.values()}
        for f in sorted(glob.glob(os.path.join(datadir, 'z%d.a*' % Z))):
            for line in open(f):
                p = line.split()
                if len(p) < 16:
                    continue
                try:
                    code = int(p[3])
                    delta = float(p[4])
                    E = float(p[1])
                    cc = float(p[5])
                    fr = [float(x) for x in p[6:16]]
                except ValueError:
                    continue
                if code not in CODE or delta != 0.0 or cc <= 0 or fr[1] <= 0:
                    continue
                shares = dict(K=fr[0], L1=fr[1], L2=fr[2], L3=fr[3], M=sum(fr[4:9]), N=fr[9])
                self.raw[CODE[code]].append((E, cc, shares))
        self.fit = {}
        for m, pts in self.raw.items():
            if len(pts) < 5:
                continue
            x = np.log([p[0] for p in pts])
            E = np.array([p[0] for p in pts])
            deg = 5 if len(pts) > 100 else (3 if len(pts) > 12 else 2)
            d = {'xmin': x.min(), 'xmax': x.max(), 'n': len(pts)}
            # ratios inside the L shell: valid at all energies
            for s in ('L2', 'L3'):
                y = np.log([max(p[2][s], 1e-30) / p[2]['L1'] for p in pts])
                d['r' + s] = self._robust(x, y, deg)
            # absolute sub-shell coefficients: only below the pair threshold
            low = E < PAIR_THRESHOLD
            if low.sum() >= 3:
                dl = min(deg, 5 if low.sum() > 100 else (3 if low.sum() > 12 else (2 if low.sum() > 5 else 1)))
                for s in SHELLS:
                    y = np.log([max(p[1] * p[2][s], 1e-300) for p in pts])
                    ok = low & (y > -600)
                    d['a' + s] = self._robust(x[ok], y[ok], dl)
                d['xlow'] = x[low].min()
                d['xhigh'] = min(x[low].max(), math.log(PAIR_THRESHOLD))
            self.fit[m] = d

    @staticmethod
    def _robust(x, y, deg):
        keep = np.ones(len(x), bool)
        c = None
        for _ in range(6):
            c = np.polyfit(x[keep], y[keep], deg)
            res = y - np.polyval(c, x)
            s = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep]))) + 1e-6
            new = np.abs(res) < 4 * s + 0.01
            if (new == keep).all():
                break
            keep = new
        return c

    def has(self, m):
        return m in self.fit

    def has_abs(self, m):
        return m in self.fit and 'aK' in self.fit[m]

    def lratio(self, m, E):
        """(L2/L1, L3/L1) for a pure multipole."""
        d = self.fit[m]
        x = min(max(math.log(E), d['xmin']), d['xmax'])
        return math.exp(np.polyval(d['rL2'], x)), math.exp(np.polyval(d['rL3'], x))

    def alpha(self, m, E):
        """dict of absolute K, L1, L2, L3, M, N(+) coefficients for a pure
        multipole.  Above the pair threshold the K/L1/M/N values are a log-log
        linear extrapolation (used only as mixing weights)."""
        d = self.fit[m]
        x = math.log(E)
        out = {}
        for s in SHELLS:
            c = d['a' + s]
            if x <= d['xhigh']:
                xx = max(x, d['xlow'])
                out[s] = math.exp(np.polyval(c, xx))
            else:
                x0 = d['xhigh']
                slope = np.polyval(np.polyder(c), x0)
                out[s] = math.exp(np.polyval(c, x0) + slope * (x - x0))
        if x > d['xhigh']:
            r2, r3 = self.lratio(m, E)
            out['L2'] = out['L1'] * r2
            out['L3'] = out['L1'] * r3
        return out

    def mixture(self, comps, E):
        """comps = [(multipole, weight)], weights sum to 1 (photon intensity
        shares).  Returns absolute sub-shell coefficients of the mixture."""
        tot = {s: 0.0 for s in SHELLS}
        for m, w in comps:
            a = self.alpha(m, E)
            for s in SHELLS:
                tot[s] += w * a[s]
        return tot
