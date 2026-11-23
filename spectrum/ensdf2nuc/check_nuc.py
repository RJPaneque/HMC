#!/usr/bin/env python3
"""Independent consistency check of a PenNuc .nuc file (does not import the
converter).  Optionally cross-checks GA intensities against the %IG records
of an ENSDF file, read here with plain regular expressions."""
import re
import sys

BR = ('ALP', 'BEM', 'BEP', 'CK', 'CL', 'CL1', 'CL2', 'CL3', 'CM', 'CN', 'CO')
TR = ('GA', 'EK', 'EL', 'EL1', 'EL2', 'EL3', 'EM', 'EN')


def check(path, ensdf=None):
    errs = []
    E = errs.append
    recs = []
    for n, line in enumerate(open(path), 1):
        if line.endswith('\r\n'):
            E('line %d: CRLF line ending' % n)
        line = line.rstrip('\n')
        key, body = line[:3].strip(), line[4:]
        if line[3:4] != ' ':
            E('line %d: column 4 not blank' % n)
        recs.append((n, key, [f.strip() for f in body.split(';')] if key != 'COM' else body))
    keys = [r[1] for r in recs]
    for k in set(keys):
        if k not in BR + TR + ('PAR', 'AZP', 'NDA', 'COM', 'DAU', 'DDE', 'Q', 'LED'):
            E('unknown keyword %s' % k)
    get = lambda k: [r for r in recs if r[1] == k]
    dde = get('DDE')[0][2]
    Q = float(get('Q')[0][2][0])
    nlev, nbr = int(dde[2]), int(dde[3])
    branches = [r for r in recs if r[1] in BR]
    if len(branches) != nbr:
        E('DDE says %d branch records, file has %d' % (nbr, len(branches)))
    leds = get('LED')
    if len(leds) != nlev:
        E('DDE says %d levels, file has %d LED records' % (nlev, len(leds)))
    elev = {0: 0.0}
    for n, k, f in leds:
        elev[int(f[5])] = float(f[0])
    if sorted(elev) != list(range(nlev + 1)):
        E('level numbers are not 0..%d' % nlev)
    if [int(f[5]) for _, _, f in leds] != list(range(nlev, 0, -1)):
        E('LED blocks are not in descending level order')
    # numbers must be plain positive decimals
    for n, k, f in recs:
        if k in BR + TR:
            if len(f) != (6 if k in BR else 5):
                E('line %d: wrong number of fields' % n)
            if not re.fullmatch(r'\d+(\.\d+)?', f[0]) or float(f[0]) <= 0:
                E('line %d: bad probability %r' % (n, f[0]))
            if f[1] and not re.fullmatch(r'\d+(\.\d+)?', f[1]):
                E('line %d: bad uncertainty %r' % (n, f[1]))
    # branches
    tot = 0.0
    bplus = 0.0
    order = []
    for n, k, f in branches:
        p, lev, en, flag = float(f[0]), int(f[2]), float(f[3]), int(f[5])
        tot += p
        order.append(lev)
        if lev not in elev:
            E('line %d: branch to unknown level %d' % (n, lev))
            continue
        expect = Q - elev[lev] - (1022.0 if k == 'BEP' else 0.0)
        if abs(en - expect) > 0.011:
            E('line %d: %s energy %s, expected %.3f' % (n, k, f[3], expect))
        if k == 'BEP':
            bplus += p
        if k == 'CL3' and flag == 0:
            E('line %d: CL3 on an allowed branch' % n)
    if order != sorted(order, reverse=True):
        E('branches are not in descending level order')
    nsep = sum(1 for r in recs if r[1] == 'COM' and 'Branch Separator' in r[2])
    if nsep != nlev + 1:
        E('%d branch separators for %d levels (+g.s.)' % (nsep, nlev))
    # transitions
    cur = None
    count = 0
    want = {}
    nga = nel = 0
    kvac = sum(float(f[0]) for _, k, f in branches if k == 'CK')
    last_ga = None
    gammas = []
    for n, k, f in recs:
        if k == 'LED':
            if cur is not None and count != want[cur]:
                E('level %d: LED announces %d records, found %d' % (cur, want[cur], count))
            cur, count = int(f[5]), 0
            want[cur] = int(f[2])
            for x in (f[3], f[4]):
                if x:
                    float(x)
        elif k in TR:
            count += 1
            p, en, fin = float(f[0]), float(f[2]), int(f[4])
            if fin >= cur:
                E('line %d: final level %d not below level %d' % (n, fin, cur))
            if k == 'GA':
                nga += 1
                last_ga = (en, fin, p)
                gammas.append((elev[cur], en, p))
                if abs(elev[cur] - elev[fin] - en) > 1.0:
                    E('line %d: gamma %s does not connect levels %d -> %d (%.2f)' % (n, f[2], cur, fin, elev[cur] - elev[fin]))
            else:
                nel += 1
                if last_ga is None or fin != last_ga[1] or not (0 < en < last_ga[0] + 0.5):
                    E('line %d: electron line does not follow its gamma' % n)
                if p > last_ga[2] * 5:
                    E('line %d: implausible conversion coefficient' % n)
                if k == 'EK':
                    kvac += p
    if cur is not None and count != want[cur]:
        E('level %d: LED announces %d records, found %d' % (cur, want[cur], count))
    print('%s: %d levels, %d branch records, %d gammas, %d electron lines' % (path, nlev, len(branches), nga, nel))
    print('   sum of branch probabilities = %.6f   (beta+ %.4f, EC %.4f);  K vacancies/decay = %.4f' % (tot, bplus, tot - bplus, kvac))
    if ensdf:
        ref = {}
        lev = None
        last = None
        for line in open(ensdf, encoding='latin-1'):
            if line[5:9] == '  L ':
                lev = float(line[9:19])
            elif line[5:9] == '  G ' and lev is not None:
                last = (lev, float(line[9:19]))
            elif line[6:9] == ' G ' and line[5] != ' ' and last:
                m = re.search(r'%IG=([0-9.Ee+-]+)', line)
                if m:
                    ref[last] = float(m.group(1)) / 100
        bad = 0
        for key in gammas:
            r = ref.get((key[0], key[1]))
            if r is None or abs(r - key[2]) > 5e-4 * r + 1e-12:
                bad += 1
                if r is not None:
                    E('gamma %s from level %s: %g vs ENSDF %g' % (key[1], key[0], key[2], r))
        missing = sorted(set(ref) - {(g[0], g[1]) for g in gammas})
        print('   GA intensities identical to ENSDF %%IG for %d of %d gammas; %d without a %%IG value; ENSDF gammas not in file: %s'
              % (len(gammas) - bad, len(gammas), bad, missing or 'none'))
    print('   %s' % ('NO structural problems found' if not errs else '%d PROBLEMS:' % len(errs)))
    for e in errs[:30]:
        print('     -', e)
    return not errs


if __name__ == '__main__':
    ok = check(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    sys.exit(0 if ok else 1)
