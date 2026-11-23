#!/usr/bin/env python3
"""
ensdf2nuc.py - convert an ENSDF EC/beta+ decay data set into a PenNuc ".nuc"
file laid out like the DDEP/NUCLEIDE exports (PAR/AZP/NDA/DAU/DDE/Q, one block
of BEP/CK/CL1/CL2/CL3/CM/CN branches per level, one LED block of
GA/EK/EL1/EL2/EL3/EM/EN transitions per excited level).

What comes straight from ENSDF
    level energies, half-lives, gamma energies, absolute gamma intensities
    (%IG records, else RI*NR*BR), K/L/M/N conversion coefficients (KC, LC, MC,
    NC+OC), Q value, beta+ / EC shares and the CK, CL, CM+ capture fractions.

What ENSDF does not contain and is modelled here
    * L1:L2:L3 split of LC, and the shell split of a bare CC
          -> BrIcc systematics for the daughter element (icc.py)
    * L1:L2(:L3) split of CL and M:N+ split of CM+
          -> Dirac-Hartree-Fock-Slater amplitudes of the parent atom (dhfs.py)
             with exchange/overlap factors calibrated on the DDEP Mn-52 file

Usage:  python3 ensdf2nuc.py DATASET.ensdf -o Y-86.nuc --g4dir <PhotonEvaporation dir>
"""
import argparse
import collections
import math
import re
import sys
from decimal import Decimal, ROUND_HALF_UP

import ensdf
import icc as iccmod
import dhfs

ELEMENTS = ("H HE LI BE B C N O F NE NA MG AL SI P S CL AR K CA SC TI V CR MN FE CO NI CU ZN GA GE AS SE BR KR "
            "RB SR Y ZR NB MO TC RU RH PD AG CD IN SN SB TE I XE CS BA LA CE PR ND PM SM EU GD TB DY HO ER TM YB "
            "LU HF TA W RE OS IR PT AU HG TL PB BI PO AT RN FR RA AC TH PA U NP PU AM CM BK CF").split()

# electron binding energies of the DAUGHTER atom, keV (X-ray Data Booklet)
BINDING = {
    38: dict(K=16.105, L1=2.216, L2=2.007, L3=1.940,
             M=(0.3587 + 0.2803 + 0.2700 + 0.1360 + 0.1342) / 5,
             N=(0.0389 + 0.0213 + 0.0201) / 3),
    24: dict(K=5.989, L1=0.696, L2=0.5838, L3=0.5741, M=0.033, N=0.0),
}
AMU_KEV = 931494.10
ME2 = 1022.0          # 2 m_e c^2 as used in the DDEP files

# exchange/overlap calibration on the DDEP Mn-52 file (Z=25), see report
REF_XLK = 1.169       # (L/K)_DDEP / (L/K)_DHFS  at Z=25
REF_L2 = 0.887        # (L2/L1)_DDEP / (L2/L1)_DHFS at Z=25
REF_L3 = 0.842        # same for the unique-forbidden L3/L1 coefficient
REF_NM = 1.12         # (N/M)_DDEP / (N/M)_DHFS at Z=25


# ----------------------------------------------------------------- formatting
def plain(x, nd=None, sig=4):
    """plain-decimal string (no exponent), trailing zeros stripped."""
    if x is None:
        return ''
    if x == 0:
        return '0'
    d = Decimal(repr(float(x)))
    if nd is None:
        nd = sig - 1 - int(math.floor(math.log10(abs(x))))
    q = Decimal(1).scaleb(-nd)
    d = d.quantize(q, rounding=ROUND_HALF_UP) if nd >= 0 else Decimal(round(float(x), nd))
    s = format(d, 'f')
    if '.' in s:
        s = s.rstrip('0').rstrip('.')
    return s


def val_unc(x, u, sig=4):
    """value and uncertainty, value rounded where a 2-digit uncertainty ends."""
    if u is None or u <= 0 or not math.isfinite(u):
        return plain(x, sig=sig), ''
    nd = 1 - int(math.floor(math.log10(u)))
    if x != 0:
        nd = max(nd, -int(math.floor(math.log10(abs(x)))))   # keep >= 1 digit of the value
    return plain(x, nd=nd), plain(u, nd=nd)


def rec(key, fields):
    return '%-3s ' % key + ' ; '.join(fields)


def relu(v, u):
    return (u / v) if (u is not None and v) else None


def quad(*rels):
    rels = [r for r in rels if r is not None]
    return math.sqrt(sum(r * r for r in rels)) if rels else None


# -------------------------------------------------------- multipolarity model
def components(g):
    """[(multipole, photon share)] describing how the ICC of this gamma was
    built, or None if nothing is known."""
    s = g.M.replace('[', '').replace(']', '').strip()
    if not s:
        return None
    opt = re.search(r'\(\+\s*([EM]\d)\)', s)
    s2 = re.sub(r'\(\+\s*[EM]\d\)', '', s).replace('(', '').replace(')', '')
    alt = ',' in s2
    parts = [p.strip() for p in re.split('[+,]', s2) if p.strip() and p.strip() != 'E0']
    if opt and opt.group(1) != 'E0' and not alt:
        if g.MR is not None:
            parts.append(opt.group(1))
    if any(p in ('D', 'Q') for p in parts) or alt:
        ex = []
        for p in parts:
            ex += {'D': ['E1', 'M1'], 'Q': ['E2']}.get(p, [p])
        # BrIcc quotes the midpoint between the smallest and largest value:
        lo, hi = ('E1' if 'E1' in ex else ex[0]), ('E2' if 'E2' in ex else ex[-1])
        return [(lo, 0.5), (hi, 0.5)] if lo != hi else [(lo, 1.0)]
    if len(parts) == 1:
        return [(parts[0], 1.0)]
    if len(parts) == 2:
        if g.MR is not None:
            d = abs(g.MR) * (0.5 if g.MRflag in ('LT', 'LE') else 1.0)
        else:
            d = 1.0
        w2 = d * d / (1 + d * d)
        return [(parts[0], 1 - w2), (parts[1], w2)]
    return None


def shell_split(g, table):
    """absolute K, L1, L2, L3, M, N(+) conversion coefficients with relative
    uncertainties: {shell: (alpha, rel_unc)} or None when ENSDF has no ICC."""
    c = g.cont
    comps = components(g)
    if comps is None:
        comps = [('E1', 1 / 3), ('M1', 1 / 3), ('E2', 1 / 3)]      # unknown multipolarity
    comps = [(m, w) for m, w in comps if table.has(m)]
    usable = [(m, w) for m, w in comps if table.has_abs(m)]
    model = None
    if len(comps) == 1 or not usable:
        m = max(comps, key=lambda t: t[1])[0] if comps else 'E2'
        r2, r3 = table.lratio(m, g.E)
        lshare = dict(L1=1 / (1 + r2 + r3), L2=r2 / (1 + r2 + r3), L3=r3 / (1 + r2 + r3))
        if table.has_abs(m):
            model = table.alpha(m, g.E)
    else:
        tot = sum(w for _, w in usable)
        model = table.mixture([(m, w / tot) for m, w in usable], g.E)
        L = model['L1'] + model['L2'] + model['L3']
        lshare = {s: model[s] / L for s in ('L1', 'L2', 'L3')}
    out = {}
    if 'KC' in c or 'LC' in c:
        def get(k):
            v = c.get(k)
            return (v[0], relu(v[0], v[1])) if v and isinstance(v[0], float) else (0.0, None)
        K, L, M = get('KC'), get('LC'), get('MC')
        N, O = get('NC'), get('OC')
        out['K'] = K
        for s in ('L1', 'L2', 'L3'):
            out[s] = (L[0] * lshare[s], L[1])
        out['M'] = M
        npl = N[0] + O[0]
        out['N'] = (npl, quad(N[1] * N[0] / npl if N[1] else None, O[1] * O[0] / npl if O[1] else None) if npl else None)
        return {k: v for k, v in out.items() if v[0] > 0}
    if g.CC and g.CCflag == '' and model is not None:
        tot = sum(model.values())
        ru = relu(g.CC, g.dCC)
        return {s: (g.CC * model[s] / tot, ru) for s in iccmod.SHELLS}
    return None


def jpi(txt):
    """(J, parity) for an unambiguous assignment such as '4-', '(5+)', '(4)-'."""
    m = re.match(r'^\(?(\d+(?:/2)?)\)?([+-])\)?$', txt.strip())
    if not m:
        return None
    j = m.group(1)
    return (float(j[:-2]) / 2 if j.endswith('/2') else float(j)), m.group(2)


def jpi_flag(jp, jl):
    """last field of the branch records as used in the DDEP files: 0 for allowed
    and first-forbidden non-unique shapes, dJ-1 when |dJ| >= 2 (unique n-th
    forbidden, and the non-unique (n+1)-th forbidden treated with that shape)."""
    a, b = jpi(jp), jpi(jl)
    if not a or not b:
        return 0
    dj = abs(a[0] - b[0])
    return int(dj) - 1 if dj >= 2 else 0


# ------------------------------------------------------------------ main work
def convert(path, g4dir, feeding='balance', log=print):
    ds = ensdf.read(path)
    L = ds.levels
    par = ds.parent
    m = re.match(r'(\d+)([A-Z]+)', par.nucid)
    A, psym = int(m.group(1)), m.group(2)
    Zp = ELEMENTS.index(psym) + 1
    m = re.match(r'(\d+)([A-Z]+)', ds.nucid)
    dsym = m.group(2)
    Zd = ELEMENTS.index(dsym) + 1
    Q, dQ = par.Q, par.DQ
    norm = (ds.norm.NR or 1.0) * (ds.norm.BR or 1.0)
    mass = A * AMU_KEV
    bind = BINDING[Zd]
    table = iccmod.Table(Zd, g4dir)
    log('Data set : %s  [%s]' % (ds.title, ds.dsref))
    log('Parent   : %s (Z=%d)  Q = %g +- %g keV   daughter %s (Z=%d)   NR*BR = %g' % (par.nucid, Zp, Q, dQ, ds.nucid, Zd, norm))

    # ---- gammas: placement, absolute intensity, conversion
    skipped = []
    out = collections.defaultdict(float)
    inn = collections.defaultdict(float)
    for i, l in enumerate(L):
        keep = []
        for g in l.gammas:
            ig = g.cont.get('%IG')
            if (ig and ig[2] != '') or g.RIflag or (g.RI is None and not ig):
                skipped.append('level %s keV, gamma %s keV: intensity is only a limit (%s %s) -> not written'
                               % (l.Estr, g.Estr, g.RIstr, g.RIflag or (ig[2] if ig else '')))
                continue
            if ig:
                g.I, g.dI = ig[0] / 100.0, (ig[1] / 100.0 if ig[1] else None)
            else:
                g.I = g.RI * norm / 100.0
                g.dI = g.I * quad(relu(g.RI, g.dRI), relu(ds.norm.NR, ds.norm.DNR)) if g.dRI else None
            g.Etr = g.E + g.E ** 2 / (2 * mass)
            fl = g.cont.get('FL')
            target = fl[0] if fl and isinstance(fl[0], float) else l.E - g.Etr
            g.final = min(range(i), key=lambda k: abs(L[k].E - target))
            g.mis = L[g.final].E - target
            if abs(g.mis) > 1.0:
                log('  WARNING placement: level %s gamma %s -> level %s (mismatch %.2f keV)' % (l.Estr, g.Estr, L[g.final].Estr, g.mis))
            g.shells = shell_split(g, table)
            if g.shells is None and g.CC:
                log('  note: gamma %s has CC but could not be split' % g.Estr)
            atot = sum(v[0] for v in g.shells.values()) if g.shells else 0.0
            ipf = g.cont.get('IPC')
            g.tot = g.I * (1 + atot + (ipf[0] if ipf and isinstance(ipf[0], float) else 0.0))
            out[i] += g.tot
            inn[g.final] += g.tot
            keep.append(g)
        l.gammas = sorted(keep, key=lambda g: (-g.final, g.E))

    # ---- atomic model for the capture sub-shell ratios
    w = dhfs.capture_weights(Zp, A)
    W = lambda n, s: w.get((n, s), {'w': 0.0})['w']
    rawK, rawL1, rawL2 = W(1, 's1/2'), W(2, 's1/2'), W(2, 'p1/2')
    rawM = W(3, 's1/2') + W(3, 'p1/2')
    rawN = sum(W(n, s) for n in (4, 5, 6) for s in ('s1/2', 'p1/2'))
    BK, BL = -w[(1, 's1/2')]['E'] * dhfs.HARTREE_KEV, -w[(2, 's1/2')]['E'] * dhfs.HARTREE_KEV
    xs = []
    for l in L:
        e = l.feed
        if e and 'CK' in e.cont and 'CL' in e.cont and e.cont['CK'][0] > 0 and 'U' not in (e.UN or ''):
            q = Q - l.E
            xs.append(e.cont['CL'][0] / e.cont['CK'][0] / (((q - BL) / (q - BK)) ** 2) / ((rawL1 + rawL2) / rawK))
    xs.sort()
    xlk = xs[len(xs) // 2] if xs else 1.0
    s = min(max((xlk - 1) / (REF_XLK - 1), 0.0), 1.5)
    rL2 = rawL2 / rawL1 * (1 - (1 - REF_L2) * s)
    cL3 = 9.0 * W(2, 'p3/2') / rawL1 * (1 - (1 - REF_L3) * s)      # keV^2 ; L3/L1 = cL3 / q^2
    rNM = rawN / rawM * REF_NM
    log('Capture sub-shell model (Z=%d): L/K exchange factor seen in ENSDF = %.3f (scale %.2f)' % (Zp, xlk, s))
    log('   L2/L1 = %.5f    L3/L1 (1st unique) = %.0f keV^2 / q^2    N+/M = %.4f' % (rL2, cL3, rNM))

    # ---- feeding of every level
    feeds = {}
    for i, l in enumerate(L):
        e = l.feed
        bal = out[i] - inn[i] if i else 0.0
        if e is None:
            if i and bal > 1e-6:
                log('  note: level %s has intensity imbalance %.5f %% but no E record -> no branch written' % (l.Estr, 100 * bal))
            continue
        lim = bool(e.IBflag or e.IEflag or e.TIflag)
        ti = e.TI if e.TI is not None else ((e.IB or 0) + (e.IE or 0))
        dti = e.dTI if e.TI is not None else (e.dIE if e.IE else e.dIB)
        if feeding == 'ensdf':
            f = None if lim else ti / 100.0
        else:
            f = bal if bal > 0 else None
        if f is None or f <= 0:
            skipped.append('level %s keV: EC/b+ feeding given as %s%s, gamma balance %+.4f %% -> no branch written'
                           % (l.Estr, e.TIstr or e.IEstr, ' (limit)' if lim else '', 100 * bal))
            continue
        feeds[i] = dict(f=f, rel=None if lim else relu(ti, dti), e=e, ensdf=ti, lim=lim, bal=bal)
    total = sum(d['f'] for d in feeds.values())
    log('Sum of the feedings before normalisation: %.4f %%  -> all branches scaled by %.5f' % (100 * total, 1 / total))

    # ---- write
    lines = []
    A_ = lines.append
    A_(rec('PAR', ['%s%d' % (psym, A)]))
    A_(rec('AZP', ['%d' % A, '%d' % Zp]))
    A_(rec('NDA', ['1']))
    cut = re.search(r'CUT=([^$]+)', ds.history)
    cit = re.search(r'CIT=([^$]+)', ds.history)
    src = ('ENSDF %s cutoff %s' % (cit.group(1).strip() if cit else '', cut.group(1).strip() if cut else '')).strip()
    A_('COM ' + (' ' + src + ' ').center(62, '*'))
    A_('COM ===================== Daughter Separator =====================')
    A_(rec('DAU', ['%s%d' % (dsym, A)]))
    branch_lines = []
    sums = collections.Counter()
    for i in range(len(L) - 1, -1, -1):
        branch_lines.append('COM ---------------------- Branch Separator ----------------------')
        if i not in feeds:
            continue
        d = feeds[i]
        e = d['e']
        l = L[i]
        f = d['f'] / total
        c = e.cont
        ck, cl, cm = (c.get(k, (0.0, None, ''))[0] for k in ('CK', 'CL', 'CM+'))
        sc = ck + cl + cm
        if sc <= 0:                       # no capture fractions: use IB / IE
            fb = (e.IB or 0) / ((e.IB or 0) + (e.IE or 0))
            log('  WARNING: level %s has no CK/CL/CM+ record' % l.Estr)
        else:
            fb = 1 - sc if (e.IB is not None and sc < 1) else 0.0
            if fb == 0.0:
                ck, cl, cm = ck / sc, cl / sc, cm / sc
        flag = int(e.UN[0]) if (e.UN and e.UN[0].isdigit() and 'U' in e.UN) else jpi_flag(par.J, l.J)
        qec = Q - l.E
        qs, dqs = plain(qec, nd=3), plain(dQ)
        tail = [str(i), qs, dqs, str(flag)]
        if fb > 0 and qec - ME2 > 0:
            relb = (relu(e.IB, e.dIB) if (e.IB and e.dIB and not e.IBflag) else d['rel']) if not d['lim'] else None
            v, u = val_unc(f * fb, f * fb * relb if relb else None)
            branch_lines.append(rec('BEP', [v, u, str(i), plain(qec - ME2, nd=3), dqs, str(flag)]))
            sums['b+'] += f * fb
        rele = relu(e.IE, e.dIE) if (e.IE and e.dIE and not e.IEflag) else d['rel']
        r3 = cL3 / (qec - BL) ** 2 if flag >= 1 else 0.0
        den = 1 + rL2 + r3
        parts = [('CK', ck, c.get('CK'), None),
                 ('CL1', cl / den, c.get('CL'), None),
                 ('CL2', cl * rL2 / den, c.get('CL'), 0.05)]
        if flag >= 1:
            parts.append(('CL3', cl * r3 / den, c.get('CL'), 0.15))
        parts += [('CM', cm / (1 + rNM), c.get('CM+'), 0.03),
                  ('CN', cm * rNM / (1 + rNM), c.get('CM+'), 0.15)]
        for key, frac, src_rec, model_rel in parts:
            if frac <= 0:
                continue
            p = f * frac
            relc = relu(src_rec[0], src_rec[1]) if src_rec else None
            rr = quad(rele, relc, model_rel) if (rele is not None and not d['lim']) else None
            v, u = val_unc(p, p * rr if rr else None)
            branch_lines.append(rec(key, [v, u] + tail))
            sums[key[1:]] += p
            sums['ec'] += p
    nbr = sum(1 for b in branch_lines if not b.startswith('COM'))
    A_(rec('DDE', ['1', '', str(len(L) - 1), str(nbr)]))
    A_(rec('Q', [plain(Q, sig=8), plain(dQ)]))
    lines += branch_lines
    nga = nel = 0
    for i in range(len(L) - 1, 0, -1):
        l = L[i]
        A_('COM -------------- Level Separator - T1/2 in seconds -------------')
        tl = []
        for g in l.gammas:
            fin = str(g.final)
            v, u = val_unc(g.I, g.dI)
            tl.append(rec('GA', [v, u, g.Estr, plain(g.dE) if g.dE else '', fin]))
            nga += 1
            if not g.shells:
                continue
            dEe = math.sqrt((l.dE or 0) ** 2 + (L[g.final].dE or 0) ** 2) or (g.dE or 0)
            for sh, key in (('K', 'EK'), ('L1', 'EL1'), ('L2', 'EL2'), ('L3', 'EL3'), ('M', 'EM'), ('N', 'EN')):
                if sh not in g.shells:
                    continue
                a, ra = g.shells[sh]
                ee = g.Etr - bind[sh]
                if a <= 0 or ee <= 0:
                    continue
                p = g.I * a
                rr = quad(relu(g.I, g.dI), ra)
                v, u = val_unc(p, p * rr if rr else None)
                tl.append(rec(key, [v, u, plain(ee, nd=3), plain(dEe, sig=2), fin]))
                nel += 1
        if l.T is not None and l.Tflag == '':
            ts = ('%.4G' % l.T).replace('E-0', 'E-').replace('E+0', 'E+')
            us = ('%.2G' % l.dT).replace('E-0', 'E-').replace('E+0', 'E+') if l.dT else ''
        else:
            ts = us = ''
        A_(rec('LED', [l.Estr, plain(l.dE) if l.dE else '', str(len(tl)), ts, us, str(i)]))
        lines += tl

    # ---- report
    log('Written  : %d levels (+ g.s.), %d branch records, %d gamma records, %d conversion-electron records'
        % (len(L) - 1, nbr, nga, nel))
    log('Totals per decay: beta+ = %.4f   EC = %.4f  (K %.4f, L1 %.4f, L2 %.5f, L3 %.6f, M %.5f, N+ %.5f)   sum = %.6f'
        % (sums['b+'], sums['ec'], sums['K'], sums['L1'], sums['L2'], sums['L3'], sums['M'], sums['N'], sums['b+'] + sums['ec']))
    for sline in skipped:
        log('  skipped: ' + sline)
    # feedings versus the ENSDF E records
    log('Feeding used (after normalisation) versus ENSDF E record, levels where they differ by > 3 % of the value:')
    for i in sorted(feeds):
        d = feeds[i]
        used = 100 * d['f'] / total
        if d['ensdf'] and abs(used - d['ensdf']) > 0.03 * d['ensdf']:
            log('   level %-9s used %8.4f %%   ENSDF %s %s' % (L[i].Estr, used, d['e'].TIstr or d['e'].IEstr, '(limit)' if d['lim'] else ''))
    # cascade check: emission probabilities implied by the file vs ENSDF
    pop = collections.defaultdict(float)
    worst = []
    for i in range(len(L) - 1, 0, -1):
        p = pop[i] + (feeds[i]['f'] / total if i in feeds else 0.0)
        for g in L[i].gammas:
            t = p * g.tot / out[i]
            pop[g.final] += t
            worst.append((t / g.tot - 1, g.I, L[i].Estr, g.Estr))
    strong = [x for x in worst if x[1] >= 0.01]
    log('Cascade check (gamma emission probability implied by the file / ENSDF %IG - 1):')
    log('   lines >= 1 %%: max deviation %.2f %%;   all %d lines: %d deviate by > 5 %%'
        % (100 * max(abs(x[0]) for x in strong), len(worst), sum(abs(x[0]) > 0.05 for x in worst)))
    for x in sorted(worst, key=lambda x: -abs(x[0]))[:5]:
        log('   level %s gamma %s (I = %.4g %%): %+.1f %%' % (x[2], x[3], 100 * x[1], 100 * x[0]))
    log('   decays reaching the ground state: %.6f' % pop[0])
    return lines


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('ensdf')
    ap.add_argument('-o', '--out', required=True)
    ap.add_argument('--g4dir', required=True, help='Geant4 PhotonEvaporation directory (BrIcc sub-shell systematics)')
    ap.add_argument('--feeding', choices=('balance', 'ensdf'), default='balance',
                    help="'balance' (default): EC/b+ feeding of each level from the gamma intensity balance; "
                         "'ensdf': the TI/IE values printed on the E records")
    a = ap.parse_args()
    out_lines = convert(a.ensdf, a.g4dir, a.feeding)
    with open(a.out, 'w', newline='\n') as fh:
        fh.write('\n'.join(out_lines) + '\n')
    print('->', a.out)
