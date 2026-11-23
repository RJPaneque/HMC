"""Small ENSDF decay-dataset reader (only what the PenNuc conversion needs)."""
import math
import re

UNITS = {'Y': 365.2422 * 86400, 'D': 86400.0, 'H': 3600.0, 'M': 60.0, 'S': 1.0,
         'MS': 1e-3, 'US': 1e-6, 'NS': 1e-9, 'PS': 1e-12, 'FS': 1e-15, 'AS': 1e-18}


def num(s):
    s = s.strip()
    if not s:
        return None
    s = s.replace('(', '').replace(')', '')
    try:
        return float(s)
    except ValueError:
        return None


def unc(val_str, unc_str):
    """ENSDF uncertainty (in units of the last digit) -> (absolute unc, flag).
    flag is '' for a normal number, or LT/GT/LE/GE/AP/CA/SY..."""
    u = unc_str.strip()
    v = val_str.strip().replace('(', '').replace(')', '')
    if not u or not v:
        return None, ''
    if u.isalpha():
        return None, u.upper()
    m = re.match(r'^\+(\d+)-(\d+)$', u)
    if m:
        u = str(max(int(m.group(1)), int(m.group(2))))
    if not u.isdigit():
        return None, u
    mant = re.split('[eE]', v)[0]
    expo = int(re.split('[eE]', v)[1]) if re.search('[eE]', v) else 0
    dec = len(mant.split('.')[1]) if '.' in mant else 0
    return int(u) * 10.0 ** (expo - dec), ''


def ndec(val_str):
    """number of decimals carried by a plain ENSDF number (None if exponent)."""
    v = val_str.strip()
    if re.search('[eE]', v):
        return None
    return len(v.split('.')[1]) if '.' in v else 0


class Rec(dict):
    __getattr__ = dict.get


def parse_cont(text):
    """'KC=0.000433 6$LC=4.70E-5 7' -> {'KC': (val, unc, flag)}"""
    out = {}
    for item in text.split('$'):
        item = item.strip()
        if not item:
            continue
        m = re.match(r'^(%?[A-Z+/0-9]+?)\s*(=|<|>| LT | GT | LE | GE | AP )\s*(\S+)(?:\s+(\S+))?', item)
        if not m:
            continue
        key, op, val, u = m.group(1), m.group(2).strip(), m.group(3), m.group(4) or ''
        if u.startswith('('):
            u = ''
        v = num(val)
        if v is None:
            out[key] = (val, None, 'TXT')
            continue
        du, flag = unc(val, u)
        if op in ('<', 'LT', 'LE'):
            flag = 'LT'
        elif op in ('>', 'GT', 'GE'):
            flag = 'GT'
        out[key] = (v, du, flag)
    return out


def read(path):
    ds = Rec(levels=[], unplaced=[], parent=None, norm=None, title='', history='')
    cur_level = None
    last = None
    for raw in open(path, encoding='latin-1'):
        line = raw.rstrip('\n').rstrip('\r').replace('\t', ' ').ljust(80)
        if not line.strip():
            continue
        nucid, cont, com, typ = line[0:5], line[5], line[6], line[7]
        if com not in ' ':
            continue                      # comment / documentation records
        if typ == ' ' and cont == ' ' and line[8] == ' ' and not ds.title:
            ds.title = line[9:39].strip()
            ds.nucid = nucid.strip()
            ds.dsref = line[39:80].strip()
            continue
        if typ == 'H':
            ds.history += line[9:80].strip()
            continue
        if cont != ' ':
            # continuation record
            if last is not None and typ in 'GLE' and typ == last.typ:
                last.cont.update(parse_cont(line[9:80]))
            continue
        if typ == 'N' and line[6] == ' ' and line[5] == ' ' and nucid.strip():
            f = lambda a, b: num(line[a:b])
            ds.norm = Rec(NR=f(9, 19), DNR=unc(line[9:19], line[19:21])[0],
                          NT=f(21, 29), BR=f(31, 39), DBR=unc(line[31:39], line[39:41])[0],
                          NB=f(41, 49))
            last = None
            continue
        if typ == 'P':
            ds.parent = Rec(nucid=nucid.strip(), E=num(line[9:19]) or 0.0, J=line[21:39].strip(),
                            T=line[39:49].strip(), DT=line[49:55].strip(),
                            Q=num(line[64:74]), DQ=unc(line[64:74], line[74:76])[0])
            last = None
            continue
        if typ == 'L':
            E = num(line[9:19])
            dE, _ = unc(line[9:19], line[19:21])
            tfield = line[39:49].split()
            T = dT = None
            Tflag = ''
            if len(tfield) == 2 and tfield[1] in UNITS and num(tfield[0]) is not None:
                T = num(tfield[0]) * UNITS[tfield[1]]
                du, Tflag = unc(tfield[0], line[49:55])
                dT = du * UNITS[tfield[1]] if du is not None else None
            cur_level = Rec(typ='L', E=E, dE=dE, Estr=line[9:19].strip(), J=line[21:39].strip(), T=T, dT=dT,
                            Tflag=Tflag, Ttxt=line[39:55].strip(), gammas=[], feed=None, cont={})
            ds.levels.append(cur_level)
            last = cur_level
            continue
        if typ == 'G':
            g = Rec(typ='G', E=num(line[9:19]), Estr=line[9:19].strip(), RIstr=line[21:29].strip(),
                    RI=num(line[21:29]), M=line[31:41].strip(), MR=num(line[41:49]), MRstr=line[41:49].strip(),
                    CC=num(line[55:62]), TI=num(line[64:74]), q=line[79], cont={})
            g.dE, _ = unc(line[9:19], line[19:21])
            g.dRI, g.RIflag = unc(line[21:29], line[29:31])
            g.dMR, g.MRflag = unc(line[41:49], line[49:55])
            g.dCC, g.CCflag = unc(line[55:62], line[62:64])
            (cur_level.gammas if cur_level is not None else ds.unplaced).append(g)
            last = g
            continue
        if typ in 'EB' and cur_level is not None:
            e = Rec(typ=typ, IB=num(line[21:29]), IE=num(line[31:39]), TI=num(line[64:74]),
                    UN=line[77:79].strip(), q=line[79], cont={},
                    IBstr=line[21:29].strip(), IEstr=line[31:39].strip(), TIstr=line[64:74].strip())
            e.dIB, e.IBflag = unc(line[21:29], line[29:31])
            e.dIE, e.IEflag = unc(line[31:39], line[39:41])
            e.dTI, e.TIflag = unc(line[64:74], line[74:76])
            cur_level.feed = e
            last = e
            continue
        last = None
    return ds
