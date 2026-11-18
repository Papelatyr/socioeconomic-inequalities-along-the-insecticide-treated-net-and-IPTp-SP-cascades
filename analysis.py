#!/usr/bin/env python3
r"""
analysis_all.py - one stand-alone script that reproduces every result of the article

  "Where malaria prevention is lost and for whom: socioeconomic inequalities along the
   insecticide-treated net and IPTp-SP cascades in Senegal, a cross-sectional analysis of the
   2023 Continuous Demographic and Health Survey" (International Health)

  From the DHS recode files it rebuilds the analytical populations, computes all estimates
  (design-based coverage with logit 95% CIs, Erreygers/Wagstaff/standard concentration indices
  with a design-based rescaling bootstrap, indirect standardisation, survey-weighted logistic
  regressions, design-adjusted Wald tests, VIFs) and writes:

    OUTPUT_FOLDER/figures/Figure1.tif ... Figure7.tif, FigureS1.tif
                      TIFF, LZW, RGB, 600 dpi, width <= 185 mm, Arial (Oxford University Press)
    OUTPUT_FOLDER/Tables_article.xlsx          Tables 1-3 of the article
    OUTPUT_FOLDER/Supplementary_tables.xlsx    Supplementary Tables S1-S13
    OUTPUT_FOLDER/Results_in_text.xlsx         other results quoted in the text and table notes

Data
    The DHS data cannot be redistributed. Register (free) at https://dhsprogram.com and download
    the 2023 Senegal Continuous DHS in Stata format. The script needs SNHR8RFL.DTA (households),
    SNPR8RFL.DTA (household members) and SNIR8RFL.DTA (women); give it the downloaded .zip file
    as is, or the folder where it was unzipped.

Usage
    python analysis_all.py  DHS_DATA  [OUTPUT_FOLDER]  [--reps 1000]  [--png]

    DHS_DATA       the DHS .zip file or the unzipped folder
    OUTPUT_FOLDER  default: ./outputs
    --reps N       bootstrap replicates for the concentration-index CIs (default 1000, as in the
                   article; seed 20261026). The whole run takes a few minutes.
    --png          also write small PNG previews of the figures

    Example (Windows):
        python analysis_all.py "C:\Users\me\Downloads\SN_2023_CONTINUOUSDHS_11252024_1850_180367.zip"

Requirements: Python 3.9+, pandas, numpy, scipy, matplotlib, pillow, pyreadstat, openpyxl
    pip install pandas numpy scipy matplotlib pillow pyreadstat openpyxl

Structure: Part 0 holds settings shared by all parts; Part 1 produces the figures, Part 2 the
tables of the article, Part 3 the supplementary tables and Part 4 the other results
quoted in the text. Functions that differ between parts
carry the prefix fig_, tab_ or sup_. Map boundaries: Natural Earth (public domain), simplified.
"""
import argparse, glob, io, json, os, sys, tempfile, warnings, zipfile

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch, PathPatch, FancyBboxPatch
from matplotlib.path import Path
from PIL import Image

warnings.simplefilter('ignore')


# ############################################################################################
# PART 0. Settings and helpers shared by all parts
# ############################################################################################

SEED = 20261026


K = range(1, 8)


REG = {1: 'Dakar', 2: 'Ziguinchor', 3: 'Diourbel', 4: 'Saint-Louis', 5: 'Tambacounda', 6: 'Kaolack',
       7: 'Thiès', 8: 'Louga', 9: 'Fatick', 10: 'Kolda', 11: 'Matam', 12: 'Kaffrine', 13: 'Kédougou',
       14: 'Sédhiou'}


ZONE = {'Dakar': 'West', 'Thiès': 'West', 'Diourbel': 'Centre', 'Kaolack': 'Centre', 'Fatick': 'Centre',
        'Kaffrine': 'Centre', 'Saint-Louis': 'North', 'Louga': 'North', 'Matam': 'North',
        'Ziguinchor': 'South', 'Kolda': 'South', 'Sédhiou': 'South',
        'Tambacounda': 'South-East', 'Kédougou': 'South-East'}


SEASON = {1: 'Jan-Mar', 2: 'Jan-Mar', 3: 'Jan-Mar', 4: 'Apr-May', 5: 'Apr-May', 6: 'Jun-Aug', 7: 'Jun-Aug', 8: 'Jun-Aug'}


def find(folder, name):
    hits = glob.glob(os.path.join(folder, '**', name), recursive=True)
    hits += glob.glob(os.path.join(folder, '**', name.lower()), recursive=True)
    if not hits:
        sys.exit(f'{name} not found in {folder}')
    return hits[0]


def hhid(df):
    return df.hv001.astype(int).astype(str) + '_' + df.hv002.astype(int).astype(str)


class Design:
    def __init__(self, w, strata, psu):
        self.w = np.asarray(w, float)
        key = pd.MultiIndex.from_arrays([np.asarray(strata), np.asarray(psu)])
        self.cl, uni = pd.factorize(key); self.ncl = len(uni)
        self.st_of_cl, st = pd.factorize(np.array([k[0] for k in uni])); self.nst = len(st)
        self.df = self.ncl - self.nst

    def var_total(self, z):
        z = np.atleast_2d(np.asarray(z, float).T).T
        Zc = np.zeros((self.ncl, z.shape[1])); np.add.at(Zc, self.cl, z)
        V = np.zeros((z.shape[1], z.shape[1]))
        for s in range(self.nst):
            zs = Zc[self.st_of_cl == s]; n = len(zs)
            if n > 1:
                d = zs - zs.mean(0); V += n / (n - 1) * d.T @ d
        return V


def frank(score, w):
    o = np.argsort(score, kind='mergesort'); s, ww = score[o], w[o]
    cw = np.cumsum(ww); r = (cw - ww / 2) / cw[-1]
    rr = pd.Series(r).groupby(s).transform('mean').values
    out = np.empty_like(rr); out[o] = rr
    return out


def multipliers(rng, strata, psu, B):
    key = pd.DataFrame({'h': strata, 'c': psu}).drop_duplicates().reset_index(drop=True)
    M = np.zeros((B, len(key)))
    for _, g in key.groupby('h'):
        idx = g.index.values; n = len(idx)
        draws = rng.integers(0, n, size=(B, n - 1)); cnt = np.zeros((B, n))
        for j in range(n - 1):
            np.add.at(cnt, (np.arange(B), draws[:, j]), 1)
        M[:, idx] = cnt * n / (n - 1)
    return M, pd.MultiIndex.from_frame(key)


def fn(n):
    return f'{n:,}'.replace(',', ' ') if n >= 10000 else str(n)


def fe(t):
    s = lambda x: ('+' if x >= 0 else '−') + f'{abs(x):.3f}'
    return f'{s(t[0])} ({s(t[1])} to {s(t[2])})'


def fo(x, limit=False):
    s = f'{x:.2f}'
    return f'{x:.3f}' if limit and s == '1.00' else s


def fpv(p):
    return '<0.001' if p < 0.001 else f'{p:.3f}'


# ############################################################################################
# PART 1. FIGURES (Figures 1-7 and Supplementary Figure S1, TIFF for OUP); names prefixed fig_
# ############################################################################################
# =====================================================================================
# 1. Output settings (OUP)
# =====================================================================================
DPI = 600            # dots per inch
MAX_WIDTH_MM = 185   # OUP two-column width


def pick_font():
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ['Arial', 'Liberation Sans', 'Helvetica', 'DejaVu Sans']:
        if name in have:
            return name
    return 'DejaVu Sans'


FONT = pick_font()
plt.rcParams.update({'font.family': FONT, 'font.size': 8, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
                     'ytick.major.width': 0.6, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5})
BLUE, ORANGE, TEAL, PURPLE = '#1f5c99', '#d55e00', '#2a8a8a', '#7b5ea7'
GREY, LIGHT = '#bdbdbd', '#e6e6e6'


def save(fig, outdir, name, png=False):
    """Render at 600 dpi, flatten to RGB on white, check width, write LZW-compressed TIFF."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=DPI, bbox_inches='tight', pad_inches=0.04, facecolor='white')
    plt.close(fig)
    im = Image.open(buf).convert('RGB')
    max_px = int(round(MAX_WIDTH_MM / 25.4 * DPI))
    if im.width > max_px:            # safety net only: figures are drawn to fit
        print(f'  note: {name} reduced from {im.width / DPI * 25.4:.0f} mm to {MAX_WIDTH_MM} mm')
        im = im.resize((max_px, round(im.height * max_px / im.width)), Image.LANCZOS)
    path = os.path.join(outdir, name + '.tif')
    im.save(path, format='TIFF', compression='tiff_lzw', dpi=(DPI, DPI))
    if png:
        prev = im.resize((im.width // 4, im.height // 4), Image.LANCZOS)
        prev.save(os.path.join(outdir, name + '_preview.png'))
    print(f'  {name}.tif  {im.width / DPI * 25.4:5.0f} x {im.height / DPI * 25.4:5.0f} mm, {DPI} dpi, RGB, LZW')


def fmt_n(n):
    return f'{n:,}'


# =====================================================================================
# 2. Data
# =====================================================================================


def fig_read_data(folder):
    import pyreadstat
    if folder.lower().endswith('.zip'):              # the DHS download can be given as is
        import tempfile, zipfile
        tmp = tempfile.mkdtemp(prefix='dhs_')
        with zipfile.ZipFile(folder) as z:
            for name in z.namelist():
                if name.upper().endswith(('SNHR8RFL.DTA', 'SNPR8RFL.DTA', 'SNIR8RFL.DTA')) or \
                        name.upper().endswith(('SNHR8RDT.ZIP', 'SNPR8RDT.ZIP', 'SNIR8RDT.ZIP')):
                    z.extract(name, tmp)
        for inner in glob.glob(os.path.join(tmp, '**', '*.[zZ][iI][pP]'), recursive=True):
            with zipfile.ZipFile(inner) as z:
                for name in z.namelist():
                    if name.upper().endswith('.DTA'):
                        z.extract(name, tmp)
        folder = tmp
    hr_cols = (['hv001', 'hv002', 'hv005', 'hv006', 'hv009', 'hv012', 'hv013', 'hv021', 'hv022', 'hv024',
                'hv025', 'hv270', 'hv271', 'hml1']
               + [f'{v}_{k}' for v in ['hml10', 'hml21', 'hml22', 'hml24', 'hml4'] for k in K])
    pr_cols = ['hml12', 'hml18', 'hv001', 'hv002', 'hv005', 'hv021', 'hv022', 'hv101', 'hv102', 'hv103',
               'hv104', 'hv105', 'hv106', 'hv218', 'hvidx']
    ir_cols = (['b19_01', 'v005', 'v006', 'v012', 'v021', 'v022', 'v024', 'v025', 'v106', 'v190', 'v191',
                'v201']
               + [f'{v}_{k}' for v in ['m80', 'm13', 'm14', 'm2a', 'm2b', 'm2c', 'm2n', 'm49a', 'ml1', 'ml2']
                  for k in range(1, 7)])
    hr, _ = pyreadstat.read_dta(find(folder, 'SNHR8RFL.DTA'), usecols=hr_cols)
    pr, _ = pyreadstat.read_dta(find(folder, 'SNPR8RFL.DTA'), usecols=pr_cols)
    ir, _ = pyreadstat.read_dta(find(folder, 'SNIR8RFL.DTA'), usecols=ir_cols)
    return hr, pr, ir


def fig_build(hr, pr, ir):
    """Household (h), de facto person (p) and women (ir) analytic files, as in the article."""
    itn = np.column_stack([(hr[f'hml10_{k}'] == 1).values for k in K])
    h = pd.DataFrame({'hhid': hr.hv001.astype(int).astype(str) + '_' + hr.hv002.astype(int).astype(str),
                      'psu': hr.hv021, 'strata': hr.hv022, 'w': hr.hv005 / 1e6, 'region': hr.hv024.map(REG),
                      'res': hr.hv025.map({1: 'Urban', 2: 'Rural'}), 'wscore': hr.hv271,
                      'defacto': hr.hv013, 'n_itn': itn.sum(1)})
    h['zone'] = h.region.map(ZONE)
    h['own'] = (h.n_itn > 0).astype(float)
    h['enough'] = np.where(h.defacto > 0, (2 * h.n_itn >= h.defacto).astype(float), np.nan)
    h['access_hh'] = np.where(h.defacto > 0, np.minimum(1, 2 * h.n_itn / h.defacto.replace(0, np.nan)), np.nan)

    p = pd.DataFrame({'hhid': pr.hv001.astype(int).astype(str) + '_' + pr.hv002.astype(int).astype(str),
                      'psu': pr.hv021, 'strata': pr.hv022, 'w': pr.hv005 / 1e6, 'defacto_flag': pr.hv103,
                      'sex': pr.hv104.map({1: 'Male', 2: 'Female'}), 'age': pr.hv105, 'preg': pr.hml18,
                      'hml12': pr.hml12})
    n_pr = len(p)
    p = p.merge(h.drop(columns=['psu', 'strata', 'w']), on='hhid', how='left', validate='m:1')
    p['use'] = p.hml12.isin([1, 2]).astype(float)
    p = p[p.defacto_flag == 1].copy()
    p['agegrp'] = pd.cut(p.age, [-1, 4, 14, 49, 97], labels=['0-4', '5-14', '15-49', '50+']).astype(object)
    p.loc[p.age >= 98, 'agegrp'] = np.nan

    blk = np.full(len(ir), np.nan)                     # maternity section of the most recent live birth
    for k in range(6, 0, -1):
        blk = np.where(ir[f'm80_{k}'] == 1, k, blk)
    for v in ['m13', 'm14', 'm2n', 'm49a', 'ml1']:
        vals = np.full(len(ir), np.nan)
        for k in range(1, 7):
            vals = np.where(blk == k, ir[f'{v}_{k}'], vals)
        ir[v] = vals
    ir['w'] = ir.v005 / 1e6; ir['psu'] = ir.v021; ir['strata'] = ir.v022
    ir['region'] = ir.v024.map(REG); ir['zone'] = ir.region.map(ZONE)
    ir['res'] = ir.v025.map({1: 'Urban', 2: 'Rural'}); ir['wscore'] = ir.v191
    ir['agegrp'] = pd.cut(ir.v012, [14, 19, 34, 49], labels=['15-19', '20-34', '35-49']).astype(object)
    ir['parity'] = pd.cut(ir.v201, [0, 1, 3, 20], labels=['1', '2-3', '4+']).astype(object)
    ir['inwin'] = ir.b19_01 < 24
    ir['eligible'] = ir.inwin & ir.m14.notna()
    ir['anc1'] = (ir.m2n == 0).astype(float)
    ir['anc4'] = ((ir.m14 >= 4) & (ir.m14 < 98)).astype(float)
    ir['ancvis'] = np.select([ir.m14 == 0, (ir.m14 >= 1) & (ir.m14 <= 3), (ir.m14 >= 4) & (ir.m14 < 98),
                              ir.m14 == 98], ['0', '1-3', '4+', 'DK'], None)
    ir.loc[ir.m2n == 1, 'ancvis'] = '0'
    ir['sp1'] = (ir.m49a == 1).astype(float)
    ir['doses'] = np.where(ir.m49a == 1, ir.ml1, 0)
    ir['sp3'] = ((ir.doses >= 3) & (ir.doses < 98)).astype(float)
    counts = dict(pr_rows=n_pr, defacto=len(p), hh=len(h), hh_defacto=int((h.defacto > 0).sum()),
                  women=len(ir), birth24=int(ir.eligible.sum()))
    return h, p, ir, counts


# =====================================================================================
# 3. Design-based estimators (stratified two-stage cluster design)
# =====================================================================================


def fig_mean(d, y, mask=None):
    """Weighted proportion with Taylor-linearised SE and logit-transformed 95% CI."""
    y = np.asarray(y, float)
    m = np.ones_like(y, bool) if mask is None else np.asarray(mask, bool)
    m = m & ~np.isnan(y)
    w = np.where(m, d.w, 0.0); yy = np.where(m, y, 0.0); W = w.sum(); est = (w * yy).sum() / W
    se = np.sqrt(d.var_total(w * (yy - est) / W)[0, 0]); t = stats.t.ppf(0.975, d.df)
    L, sl = np.log(est / (1 - est)), se / (est * (1 - est))
    return dict(est=est, lo=1 / (1 + np.exp(-(L - t * sl))), hi=1 / (1 + np.exp(-(L + t * sl))), n=int(m.sum()))


def logit_fit(d, X, y):
    w = d.w; b = np.zeros(X.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-(X @ b)))
        step = np.linalg.solve((X * (w * p * (1 - p))[:, None]).T @ X, X.T @ (w * (y - p)))
        b += step
        if np.abs(step).max() < 1e-10:
            break
    return b


# =====================================================================================
# 4. Estimates used in the figures
# =====================================================================================
def estimates(h, p, ir):
    R = {}
    dp = Design(p.w, p.strata, p.psu)
    R['pop'] = [fig_mean(dp, p.own), fig_mean(dp, p.access_hh), fig_mean(dp, p.use)]
    pw = ((p.preg == 1) & (p.sex == 'Female')).values
    R['preg'] = [fig_mean(dp, p.own, pw), fig_mean(dp, p.access_hh, pw), fig_mean(dp, p.use, pw)]
    R['n_pop'], R['n_preg'] = len(p), int(pw.sum())
    R['region'] = {}
    for rg in REG.values():
        m = (p.region == rg).values
        a, u = fig_mean(dp, p.access_hh, m), fig_mean(dp, p.use, m)
        R['region'][rg] = dict(access=a, use=u, ratio=u['est'] / a['est'])
    R['ratio_nat'] = R['pop'][2]['est'] / R['pop'][1]['est']
    E = ir.eligible.values
    di = Design(ir.w, ir.strata, ir.psu)
    R['ipt'] = [fig_mean(di, ir.anc1, E), fig_mean(di, ir.sp1, E & (ir.anc1 == 1)), fig_mean(di, ir.anc4, E & (ir.anc1 == 1)),
                fig_mean(di, ir.sp3, E & (ir.anc4 == 1))]
    e = ir[E]
    dcat = np.select([e.doses == 0, e.doses == 1, e.doses == 2, (e.doses >= 3) & (e.doses < 98)],
                     ['0', '1', '2', '3+'], 'DK')
    R['xtab'] = (pd.crosstab(e.ancvis, dcat, values=e.w, aggfunc='sum', normalize='index') * 100).to_dict('index')
    R['xtab_n'] = pd.crosstab(e.ancvis, dcat).sum(1).to_dict()
    R['n_anc0'] = int((e.anc1 == 0).sum()); R['n_dk'] = int((e.ancvis == 'DK').sum())
    return R


# ---------------- concentration indices (Erreygers), rescaling bootstrap


def fig_cindex(y, score, w):
    R_ = frank(score, w); W = w.sum(); mu = (w * y).sum() / W
    C = 2 * ((w * (y - mu) * (R_ - (w * R_).sum() / W)).sum() / W) / mu
    return 4 * mu * C                                     # Erreygers' corrected index


def fig_ci_E(df, y, mask, mult, ystd=None):
    keep = mask & df[y].notna().values
    d = df[keep]
    yy = d[y].values.astype(float) if ystd is None else ystd[keep]
    w, sc = d.w.values, d.wscore.values.astype(float)
    E = fig_cindex(yy, sc, w)
    M, key = mult
    pos = key.get_indexer(pd.MultiIndex.from_arrays([d.strata, d.psu]))
    bs = []
    for b in range(M.shape[0]):
        wb = w * M[b, pos]; k = wb > 0
        bs.append(fig_cindex(yy[k], sc[k], wb[k]))
    se = np.std(bs, ddof=1)
    return dict(E=E, lo=E - 1.96 * se, hi=E + 1.96 * se)


def fig_std_pred(df, mask, y, needs):
    """Indirect standardisation: observed - predicted (logit on need variables) + mean."""
    d = df[mask]
    X = pd.get_dummies(d[needs], drop_first=True, dtype=float); X.insert(0, 'const', 1.0)
    b = logit_fit(Design(d.w, d.strata, d.psu), X.values, d[y].values.astype(float))
    ph = 1 / (1 + np.exp(-(X.values @ b)))
    full = np.full(len(df), np.nan)
    full[np.where(mask)[0]] = d[y].values - ph + np.average(d[y], weights=d.w)
    return full


def fig_concentration(p, ir, B):
    rng = np.random.default_rng(SEED)
    out = {}
    pp = p.copy(); pp['w'] = pp.w.astype(float)
    mult = multipliers(rng, pp.strata.values, pp.psu.values, B)
    allp = np.ones(len(pp), bool)
    pw = ((pp.preg == 1) & (pp.sex == 'Female')).values
    out['own'] = fig_ci_E(pp, 'own', allp, mult)
    out['access'] = fig_ci_E(pp, 'access_hh', allp, mult)
    out['use'] = fig_ci_E(pp, 'use', allp, mult)
    out['use_enough'] = fig_ci_E(pp, 'use', (pp.enough == 1).values, mult)
    out['preg_own'] = fig_ci_E(pp, 'own', pw, mult)
    out['preg_access'] = fig_ci_E(pp, 'access_hh', pw, mult)
    out['preg_use'] = fig_ci_E(pp, 'use', pw, mult)
    m = pp.agegrp.notna().values
    out['use_std'] = fig_ci_E(pp, 'use', m, mult, fig_std_pred(pp, m, 'use', ['zone', 'res', 'agegrp', 'sex']))
    w = ir[ir.eligible].copy(); w['w'] = w.w.astype(float)
    mult = multipliers(rng, w.strata.values, w.psu.values, B)
    al = np.ones(len(w), bool)
    for v in ['anc1', 'anc4', 'sp1', 'sp3']:
        out[v] = fig_ci_E(w, v, al, mult)
    out['sp3_anc4'] = fig_ci_E(w, 'sp3', (w.anc4 == 1).values, mult)
    out['sp3_std'] = fig_ci_E(w, 'sp3', al, mult, fig_std_pred(w, al, 'sp3', ['zone', 'res', 'agegrp', 'parity']))
    return out


# =====================================================================================
# 5. Figures
# =====================================================================================
W_IN = 7.0      # drawing width in inches (178 mm); final width is checked against 185 mm


def figure1(outdir, png):
    """Study area: regions shaded by ecogeographical zone."""
    COL = {'West': '#c6dbef', 'Centre': '#fdd0a2', 'North': '#fff3b0', 'South': '#b8e0c8', 'South-East': '#d9c8e8'}
    OFF = {'Dakar': (-0.55, 0.15), 'Thiès': (0, 0.05), 'Fatick': (0.15, 0.05), 'Kaolack': (0, -0.05),
           'Kaffrine': (0.1, 0)}
    places = {'MAURITANIA': (-14.2, 16.85), 'MALI': (-11.6, 14.6), 'GUINEA': (-12.3, 12.2),
              'GUINEA-BISSAU': (-15.3, 12.05), 'THE GAMBIA': (-15.6, 13.2)}

    def patch(polys, **kw):
        verts, codes = [], []
        for poly in polys:
            for ring in poly:
                verts += ring; codes += [Path.MOVETO] + [Path.LINETO] * (len(ring) - 2) + [Path.CLOSEPOLY]
        return PathPatch(Path(verts, codes), **kw)

    fig, ax = plt.subplots(figsize=(W_IN * 0.76, 5.0))
    for polys in GEO['neighbours'].values():
        ax.add_patch(patch(polys, fc='#f2f2f2', ec='#9e9e9e', lw=0.5))
    for name, polys in GEO['regions'].items():
        ax.add_patch(patch(polys, fc=COL[ZONE[name]], ec='#4d4d4d', lw=0.6))
    for polys in GEO['zones'].values():
        ax.add_patch(patch(polys, fc='none', ec='black', lw=1.2))
    for name, (x, y) in GEO['label_points'].items():
        if name == 'Ziguinchor':
            ax.text(-16.3, 12.86, name, ha='center', va='center', fontsize=7.5, bbox=dict(fc=COL['South'], ec='none', pad=0.5))
            continue
        dx, dy = OFF.get(name, (0, 0))
        ax.annotate(name, (x, y), xytext=(x + dx, y + dy), ha='center', va='center', fontsize=7.5,
                    arrowprops=dict(arrowstyle='-', lw=0.5) if name == 'Dakar' else None)
    for t, (x, y) in places.items():
        ax.text(x, y, t, fontsize=6.5, color='#6e6e6e', ha='center')
    ax.text(-17.6, 14.0, 'Atlantic\nOcean', fontsize=6.5, color='#6e6e6e', ha='center', style='italic')
    ax.set_xlim(-17.9, -11.2); ax.set_ylim(12.0, 16.95); ax.set_aspect('equal'); ax.set_axis_off()
    ax.legend(handles=[Patch(facecolor=COL[z], edgecolor='#4d4d4d', label=z) for z in COL],
              title='Ecogeographical zone', loc='upper left', bbox_to_anchor=(1.0, 0.95), frameon=False,
              fontsize=7.5, title_fontsize=7.5)
    save(fig, outdir, 'Figure1', png)


def figure2(outdir, png, c):
    """Derivation of the analytical populations (flow diagram). Counts come from the data,
    except households selected and occupied, which are taken from the survey report."""
    EDGE, SHADE, G = '#5a6370', '#ecedef', '#5a6370'
    f = lambda n: fmt_n(n)
    fig = plt.figure(figsize=(W_IN, W_IN * 1500 / 2000))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 2000); ax.set_ylim(1500, 0); ax.axis('off')
    S = 2000 / (W_IN * 72)          # canvas units per point

    def box(x, y, w, h_, shade=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h_, boxstyle='round,pad=0,rounding_size=12',
                                    fc=SHADE if shade else 'white', ec=EDGE, lw=0.9))

    def txt(x, y, s, size=7, weight='normal', style='normal', color='#111', ha='center'):
        ax.text(x, y, s, fontsize=size, fontweight=weight, fontstyle=style, color=color, ha=ha, va='center',
                linespacing=1.3)

    def arrow(x0, y0, x1, y1):
        ax.annotate('', (x1, y1), (x0, y0), arrowprops=dict(arrowstyle='-|>,head_width=0.25,head_length=0.5',
                                                         color=EDGE, lw=0.9, shrinkA=0, shrinkB=0))

    def line(xs, ys):
        ax.plot(xs, ys, color=EDGE, lw=0.9, solid_capstyle='butt')

    def excl(x, y, s):
        txt(x, y, s, 6.5, style='italic', color=G, ha='left')

    line([50, 50], [62, 1500])
    for yb in [295, 570, 1175]:
        line([37, 70], [yb, yb])
    for s, y in zip(['SURVEY', 'DATA FILES', 'ANALYTICAL POPULATIONS', 'ANALYSIS SAMPLES'], [178, 432, 872, 1335]):
        ax.text(18, y, s, rotation=90, fontsize=6.5, fontweight='bold', color=G, ha='center', va='center')
    box(600, 80, 970, 160)
    txt(1085, 112, '2023 Senegal Continuous Demographic and Health Survey', 7.5)
    txt(1085, 145, '8,782 households selected  ·  8,591 occupied', 6.5, color=G)
    txt(1085, 180, f'{f(c["hh"])} households interviewed', 8.5, 'bold')
    txt(1085, 214, 'household response rate 98%', 6.5, style='italic', color=G)
    line([1085, 1085], [240, 288]); line([647, 1525], [288, 288])
    arrow(647, 288, 647, 350); arrow(1525, 288, 1525, 350)
    for x, w, cx, lines_ in [(212, 869, 647, ('Household Member Recode (PR)', 'linked to Household Recode (HR)',
                                              f'n = {f(c["pr_rows"])} members', f'listed in {f(c["hh"])} households')),
                             (1290, 470, 1525, ("Individual (women's) Recode (IR)", 'women aged 15–49 interviewed',
                                                f'n = {f(c["women"])}', 'one record per woman'))]:
        box(x, 350, w, 175, True)
        txt(cx, 390, lines_[0], 7.5); txt(cx, 420, lines_[1], 6.5, color=G)
        txt(cx, 453, lines_[2], 8.5, 'bold'); txt(cx, 487, lines_[3], 6.5, style='italic', color=G)
    arrow(647, 525, 647, 675)
    excl(670, 600, f'Excluded: {f(c["pr_rows"] - c["defacto"])} members who did not sleep\nin the household the previous night')
    arrow(1525, 525, 1525, 675)
    excl(1545, 600, f'Excluded: {f(c["women"] - c["birth24"])} women\nwithout a live birth in\nthe past 24 months')
    box(212, 675, 869, 165)
    txt(647, 720, 'ITN analytical population', 7.5)
    txt(647, 757, f'n = {f(c["defacto"])} de facto members', 8.5, 'bold')
    txt(647, 794, f'of {f(c["hh_defacto"])} households', 6.5, color=G)
    box(1290, 675, 470, 165)
    txt(1525, 705, 'Women with a live birth', 7.5); txt(1525, 738, 'in the past 24 months', 6.5, color=G)
    txt(1525, 769, f'n = {f(c["birth24"])}', 8.5, 'bold')
    txt(1525, 803, 'most recent live birth', 6.5, style='italic', color=G)
    line([647, 647], [840, 885]); line([271, 1022], [885, 885])
    for cx in [271, 647, 1022]:
        arrow(cx, 885, cx, 930)
    arrow(1525, 840, 1525, 950)
    excl(1545, 895, 'No exclusion')
    for x, cx, a, b in [(100, 271, 'Currently pregnant\nwomen aged 15–49', c['n_preg']),
                        (475, 647, 'Members of households\nowning ≥1 ITN', c['n_own']),
                        (850, 1022, 'Members of households\nwith ≥1 ITN per 2 people', c['n_enough'])]:
        box(x, 930, 344, 175)
        txt(cx, 995, a, 7); txt(cx, 1055, f'n = {f(b)}', 8.5, 'bold')
    box(1290, 950, 470, 155)
    txt(1525, 1005, 'IPTp-SP analytical population', 7.5); txt(1525, 1045, f'n = {f(c["birth24"])} women', 8.5, 'bold')
    arrow(647, 1105, 647, 1205); arrow(1022, 1105, 1022, 1205); arrow(1525, 1105, 1525, 1205)
    excl(1545, 1155, f'Excluded: {f(c["n_anc0"])} with no ANC visit')
    for x, cx, title, n, tot, rows in [(467, 647, 'Supplementary model A2', c['a2_n'], c['a2_ex'], c['a2_rows']),
                                       (842, 1022, 'Model A', c['a_n'], c['a_ex'], c['a_rows'])]:
        box(x, 1205, 360, 270, True)
        txt(cx, 1235, title, 7); txt(cx, 1270, f'n = {f(n)}', 8.5, 'bold')
        txt(cx, 1310, 'Excluded (missing data)', 6.5, 'bold', color=G)
        line([x + 20, x + 340], [1328, 1328])
        for i, (lab, v) in enumerate(zip(['Age', "Head's education", 'Net age'], rows)):
            txt(x + 22, 1352 + 29 * i, lab, 6.5, ha='left'); txt(x + 338, 1352 + 29 * i, f(v), 6.5, ha='right')
        line([x + 20, x + 340], [1430, 1430])
        txt(x + 22, 1452, 'Total', 6.5, 'bold', ha='left'); txt(x + 338, 1452, f(tot), 6.5, 'bold', ha='right')
    box(1290, 1205, 470, 270, True)
    txt(1525, 1235, 'Model B', 7.5); txt(1525, 1272, f'n = {f(c["b_n"])} women', 8.5, 'bold')
    txt(1525, 1305, 'with ≥1 ANC visit', 6.5, color=G)
    line([1370, 1680], [1330, 1330])
    txt(1525, 1403, f'No exclusion for missing\ncovariates; {f(c["n_dk"])} women with an\nunknown number of visits\n'
                    'kept as a separate category', 6.5, style='italic', color=G)
    save(fig, outdir, 'Figure2', png)


def regression_counts(p, ir):
    """Sequential exclusions for the regression samples (Figure 2)."""
    pp = p.copy()
    ok = pd.Series(True, index=pp.index)
    c = {}
    for key, mask in [('a2', pp.own == 1), ('a', pp.enough == 1)]:
        d = pp[mask]
        r1 = d.agegrp.isna(); r2 = ~r1 & d.head_ed.isna(); r3 = ~r1 & ~r2 & d.netage.isna()
        c[key + '_rows'] = (int(r1.sum()), int(r2.sum()), int(r3.sum()))
        c[key + '_ex'] = int(r1.sum() + r2.sum() + r3.sum()); c[key + '_n'] = len(d) - c[key + '_ex']
    c['n_own'] = int((pp.own == 1).sum()); c['n_enough'] = int((pp.enough == 1).sum())
    c['b_n'] = int((ir.eligible & (ir.anc1 == 1)).sum())
    return c


def figure3(outdir, png, R):
    fig, axs = plt.subplots(1, 2, figsize=(W_IN, 2.9))
    lab = ['Lives in household\nwith ≥1 ITN', 'Has access to an ITN\n(assuming ≤2 people\nper ITN)',
           'Slept under an ITN\nthe previous night']
    for ax, vals, title in [(axs[0], R['pop'], f'A. De facto household population (n = {fmt_n(R["n_pop"])})'),
                            (axs[1], R['preg'], f'B. Pregnant women aged 15–49 (n = {fmt_n(R["n_preg"])})')]:
        y = np.arange(3)[::-1]
        for i, (v, col) in enumerate(zip(vals, [BLUE, TEAL, ORANGE])):
            prev = vals[i - 1]['est'] * 100 if i else None
            if prev:
                ax.barh(y[i], prev, color=LIGHT, height=0.6)
                ax.text(prev + 1.5, y[i] + 0.42, f"−{prev - v['est'] * 100:.1f} pp", fontsize=6.5, color='#555')
            ax.barh(y[i], v['est'] * 100, color=col, height=0.6)
            ax.errorbar(v['est'] * 100, y[i], xerr=[[100 * (v['est'] - v['lo'])], [100 * (v['hi'] - v['est'])]],
                        color='black', capsize=2, lw=0.7)
            ax.text(v['hi'] * 100 + 1.5, y[i], f"{v['est'] * 100:.1f}%", va='center', fontsize=7.5, fontweight='bold')
        ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=7); ax.set_xlim(0, 100)
        ax.set_xlabel('Percentage (%)'); ax.set_title(title, fontsize=7.5, loc='left', fontweight='bold')
    axs[1].set_yticklabels([])
    plt.tight_layout(); save(fig, outdir, 'Figure3', png)


def figure4(outdir, png, R):
    reg = R['region']; order = sorted(reg, key=lambda k: reg[k]['access']['est'])
    pc = lambda r: 100 * r['est']
    fig, ax = plt.subplots(figsize=(W_IN * 0.95, 4.6))
    for i, rg in enumerate(order):
        a, u = reg[rg]['access'], reg[rg]['use']
        ax.plot([pc(u), pc(a)], [i, i], color=GREY, lw=2, zorder=1)
        ax.errorbar(pc(a), i + 0.13, xerr=[[pc(a) - 100 * a['lo']], [100 * a['hi'] - pc(a)]], fmt='o', color=BLUE,
                    ms=5, lw=0.8, zorder=3)
        ax.errorbar(pc(u), i - 0.13, xerr=[[pc(u) - 100 * u['lo']], [100 * u['hi'] - pc(u)]], fmt='o', color=ORANGE,
                    ms=5, lw=0.8, zorder=3)
        ax.text(86, i, f'{pc(a):.1f}', color=BLUE, va='center', ha='right', fontsize=7)
        ax.text(97, i, f'{pc(u):.1f}', color=ORANGE, va='center', ha='right', fontsize=7)
    na, nu = pc(R['pop'][1]), pc(R['pop'][2])
    ax.axvline(na, ls=':', color=BLUE, lw=0.9); ax.axvline(nu, ls=':', color=ORANGE, lw=0.9)
    top = len(order) - 0.35
    ax.text(na + 0.5, top, f'national access {na:.1f}%', color=BLUE, fontsize=6.5)
    ax.text(nu - 0.5, top, f'national use {nu:.1f}%', color=ORANGE, fontsize=6.5, ha='right')
    ax.text(86, top, 'Access, %', color=BLUE, fontsize=6.5, ha='right')
    ax.text(97, top, 'Use, %', color=ORANGE, fontsize=6.5, ha='right')
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order); ax.set_xlim(0, 100)
    ax.set_ylim(-0.7, len(order)); ax.set_xlabel('Percentage of de facto household population (%)')
    ax.legend(handles=[plt.Line2D([], [], marker='o', color=BLUE, ls='', label='Population access to an ITN'),
                       plt.Line2D([], [], marker='o', color=ORANGE, ls='', label='Slept under an ITN')],
              loc='lower right', bbox_to_anchor=(0.8, 0.0), frameon=False, fontsize=7)
    save(fig, outdir, 'Figure4', png)


def figure5(outdir, png, R):
    reg = R['region']; pc = lambda r: 100 * r['est']
    fig, ax = plt.subplots(figsize=(W_IN * 0.9, 4.4))
    xs = [pc(reg[r]['access']) for r in reg]; ys = [reg[r]['ratio'] for r in reg]
    ax.axvline(np.median(xs), ls=':', color='grey', lw=0.8); ax.axhline(np.median(ys), ls=':', color='grey', lw=0.8)
    ax.scatter(xs, ys, s=28, color=BLUE, zorder=3)
    na = pc(R['pop'][1])
    ax.scatter([na], [R['ratio_nat']], marker='D', s=32, color='black', zorder=4)
    ap = dict(arrowstyle='-', color='#888888', lw=0.6, shrinkA=0, shrinkB=3)
    ax.annotate(f"Senegal ({na:.1f}%, {R['ratio_nat']:.2f})", (na, R['ratio_nat']), xytext=(29.5, 0.755),
                fontsize=7, fontweight='bold', arrowprops=ap)
    pos = {'Dakar': (24.5, 0.595), 'Thiès': (33, 0.845), 'Tambacounda': (41.5, 0.655), 'Diourbel': (43.5, 0.84),
           'Kaolack': (44.5, 0.895), 'Sédhiou': (56, 0.80), 'Kédougou': (46.5, 0.752), 'Louga': (44.5, 0.700),
           'Kaffrine': (51.5, 0.655), 'Saint-Louis': (57.5, 0.645), 'Kolda': (63.5, 0.69), 'Fatick': (62.5, 0.895),
           'Ziguinchor': (64.5, 0.81), 'Matam': (58.5, 0.95)}
    for r in reg:
        ax.annotate(r, (pc(reg[r]['access']), reg[r]['ratio']), xytext=pos[r], fontsize=7, arrowprops=ap, va='center')
    ax.set_xlabel('Population access to an ITN (%)')
    ax.set_ylabel('Use-to-access ratio\n(population use / population access)')
    ax.set_xlim(15, 70); ax.set_ylim(0.55, 1.0)
    save(fig, outdir, 'Figure5', png)


def figure6(outdir, png, R):
    fig, axs = plt.subplots(1, 2, figsize=(W_IN, 3.3), gridspec_kw={'width_ratios': [1.3, 1]})
    ax = axs[0]
    steps = [('≥1 ANC visit\n(all women)', BLUE, 'a'), ('≥1 SP dose\n(among ANC1+)', TEAL, 'b'),
             ('≥4 ANC visits\n(among ANC1+)', PURPLE, 'b'), ('≥3 SP doses\n(among ANC4+)', ORANGE, 'c')]
    y = np.arange(4)[::-1]
    for i, ((lab, col, d), v) in enumerate(zip(steps, R['ipt'])):
        e = 100 * v['est']
        ax.barh(y[i], 100, color=LIGHT, height=0.6); ax.barh(y[i], e, color=col, height=0.6)
        ax.errorbar(e, y[i], xerr=[[e - 100 * v['lo']], [100 * v['hi'] - e]], color='black', capsize=2, lw=0.7)
        ax.text(100 * v['lo'] - 1.5, y[i], f'{e:.1f}%', va='center', ha='right', fontsize=7.5, fontweight='bold',
                color='white')
        ax.text(101, y[i], d, va='center', fontsize=6.5, color='#555')
    ax.set_yticks(y); ax.set_yticklabels([s[0] for s in steps], fontsize=7); ax.set_xlim(0, 104)
    ax.set_xlabel('Percentage of the step denominator (%)')
    ax.set_title('A. Nested IPTp-SP cascade', fontsize=7.5, loc='left', fontweight='bold')
    ax = axs[1]
    xt, nn = R['xtab'], R['xtab_n']
    rows, cats = ['1-3', '4+', 'DK'], ['0', '1', '2', '3+']
    cols = ['#d9d9d9', '#fdd0a2', '#fd8d3c', ORANGE]
    left = np.zeros(3)
    for cat, col in zip(cats, cols):
        vals = np.array([xt[r].get(cat, 0) for r in rows])
        ax.barh(np.arange(3)[::-1], vals, left=left, color=col, height=0.6,
                label=f'{cat} dose{"s" if cat != "1" else ""}')
        for j, v in enumerate(vals):
            if v > 6:
                ax.text(left[j] + v / 2, 2 - j, f'{v:.0f}', ha='center', va='center', fontsize=6.5)
        left += vals
    ylab = [f'1-3 visits\n(n = {fmt_n(nn["1-3"])})', f'4+ visits\n(n = {fmt_n(nn["4+"])})',
            f'Unknown no.\nof visits\n(n = {fmt_n(nn["DK"])})']
    ax.set_yticks(np.arange(3)[::-1]); ax.set_yticklabels(ylab, fontsize=7)
    ax.set_xlim(0, 100); ax.set_xlabel('Percentage of women (%)')
    ax.set_title('B. SP doses by number of ANC visits', fontsize=7.5, loc='left', fontweight='bold')
    ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.4, -0.27), frameon=False, fontsize=6.5, handlelength=1)
    plt.tight_layout(); save(fig, outdir, 'Figure6', png)


def figure7(outdir, png, C):
    rows = [('own', 'In household with ≥1 ITN', BLUE), ('access', 'Access to an ITN', BLUE),
            ('use', 'Slept under an ITN', BLUE), ('use_enough', 'ITN use | ≥1 ITN per 2 people', BLUE),
            ('use_std', 'ITN use, need-standardised', BLUE),
            ('preg_own', 'Pregnant: household with ≥1 ITN', TEAL), ('preg_access', 'Pregnant: access', TEAL),
            ('preg_use', 'Pregnant: slept under an ITN', TEAL),
            ('anc1', '≥1 ANC visit', ORANGE), ('anc4', '≥4 ANC visits', ORANGE), ('sp1', '≥1 SP dose', ORANGE),
            ('sp3', '≥3 SP doses (IPTp-SP3)', ORANGE), ('sp3_anc4', 'IPTp-SP3 | ≥4 ANC visits', ORANGE),
            ('sp3_std', 'IPTp-SP3, need-standardised', ORANGE)]
    fig, ax = plt.subplots(figsize=(W_IN * 0.76, 5.0))
    pos, yy = 0, []
    for i, (k, lab, col) in enumerate(rows):
        if i in (5, 8):
            pos += 0.7
        yy.append(pos); r = C[k]; sig = r['lo'] > 0 or r['hi'] < 0
        ax.errorbar(r['E'], -pos, xerr=[[r['E'] - r['lo']], [r['hi'] - r['E']]], fmt='o', color=col,
                    mfc=col if sig else 'white', ms=4.5, capsize=2, lw=0.8)
        ax.text(0.33, -pos, f"{r['E']:+.3f} ({r['lo']:+.3f} to {r['hi']:+.3f})".replace('-', '−'), va='center',
                fontsize=6.5)
        pos += 1
    ax.axvline(0, color='black', lw=0.7)
    ax.set_yticks([-p for p in yy]); ax.set_yticklabels([r[1] for r in rows], fontsize=7)
    ax.set_xlim(-0.35, 0.32)
    ax.set_xlabel('Erreygers-corrected concentration index (95% CI)\n'
                  '← concentrated among poorer      concentrated among wealthier →')
    ax.text(0.33, 0.9, 'E (95% CI)', fontsize=7, fontweight='bold')
    ax.text(-0.345, 0.9, 'ITN, household population', color=BLUE, fontsize=7, fontweight='bold')
    ax.text(-0.345, -5.2, 'ITN, pregnant women', color=TEAL, fontsize=7, fontweight='bold')
    ax.text(-0.345, -8.9, 'IPTp-SP, women with a live birth in past 24 months', color=ORANGE, fontsize=7,
            fontweight='bold')
    save(fig, outdir, 'Figure7', png)


def figureS1(outdir, png, p, ir):
    def curve(y, score, w):
        o = np.argsort(score, kind='mergesort'); y, w = y[o], w[o]
        return (np.concatenate([[0], np.cumsum(w) / w.sum()]), np.concatenate([[0], np.cumsum(w * y) / (w * y).sum()]))
    e = ir[ir.eligible]
    fig, axs = plt.subplots(1, 2, figsize=(W_IN, 3.6))
    panels = [(p, [('own', 'In household with ≥1 ITN', BLUE, '-'), ('access_hh', 'Access to an ITN', TEAL, '--'),
                   ('use', 'Slept under an ITN', ORANGE, '-.')], 'A. ITN cascade (household population)'),
              (e, [('anc4', '≥4 ANC visits', PURPLE, '--'), ('sp1', '≥1 SP dose', TEAL, ':'),
                   ('sp3', '≥3 SP doses', ORANGE, '-')], 'B. IPTp-SP cascade (women)')]
    for ax, (d, items, title) in zip(axs, panels):
        ax.plot([0, 1], [0, 1], color='black', lw=0.7, label='Line of equality')
        for v, lab, col, ls in items:
            x, y = curve(d[v].values.astype(float), d.wscore.values.astype(float), d.w.values)
            ax.plot(x, y, color=col, ls=ls, lw=1.2, label=lab)
        ax.set_xlabel('Cumulative share of population,\nranked from poorest to wealthiest')
        ax.set_ylabel('Cumulative share of outcome'); ax.set_title(title, fontsize=7.5, loc='left', fontweight='bold')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal'); ax.legend(frameon=False, fontsize=6.5, loc='upper left')
    plt.tight_layout(); save(fig, outdir, 'FigureS1', png)


# =====================================================================================
# 6. Main
# =====================================================================================


# Region, zone and neighbouring-country boundaries for Figure 1
# (Natural Earth 1:10m admin-0 and admin-1, public domain; simplified, tolerance 0.004 degrees)
GEO = json.loads(r"""{"regions":{"Kédougou":[[[[-11.9,13.382],[-11.898,13.371],[-11.855,13.351],[-11.84,13.317],[-11.823,13.307],[-11.775,13.343],[-11.767,13.383],[-11.757,13.4],[-11.741,13.408],[-11.727,13.406],[-11.704,13.38],[-11.65,13.384],[-11.644,13.356],[-11.613,13.361],[-11.606,13.32],[-11.554,13.257],[-11.572,13.217],[-11.569,13.199],[-11.539,13.163],[-11.533,13.112],[-11.45,13.075],[-11.459,13.055],[-11.436,13.017],[-11.423,13.009],[-11.433,12.997],[-11.439,12.959],[-11.412,12.961],[-11.402,12.989],[-11.393,12.981],[-11.384,12.933],[-11.403,12.916],[-11.427,12.919],[-11.415,12.897],[-11.429,12.866],[-11.425,12.85],[-11.433,12.836],[-11.408,12.804],[-11.405,12.782],[-11.415,12.76],[-11.402,12.732],[-11.41,12.719],[-11.437,12.712],[-11.448,12.686],[-11.462,12.673],[-11.461,12.658],[-11.442,12.648],[-11.437,12.637],[-11.442,12.596],[-11.437,12.571],[-11.44,12.559],[-11.468,12.544],[-11.449,12.535],[-11.43,12.508],[-11.425,12.526],[-11.417,12.529],[-11.401,12.48],[-11.39,12.498],[-11.378,12.48],[-11.388,12.404],[-11.416,12.405],[-11.482,12.428],[-11.515,12.432],[-11.643,12.418],[-11.72,12.389],[-11.758,12.383],[-11.84,12.387],[-11.924,12.419],[-11.998,12.386],[-12.079,12.408],[-12.104,12.407],[-12.192,12.349],[-12.361,12.306],[-12.377,12.314],[-12.406,12.356],[-12.423,12.369],[-12.467,12.384],[-12.505,12.39],[-12.571,12.364],[-12.594,12.362],[-12.603,12.382],[-12.648,12.426],[-12.688,12.436],[-12.773,12.435],[-12.783,12.453],[-12.839,12.466],[-12.849,12.476],[-12.849,12.495],[-12.874,12.516],[-12.913,12.536],[-12.949,12.536],[-12.97,12.477],[-12.981,12.467],[-13.052,12.472],[-13.067,12.48],[-13.078,12.496],[-13.083,12.516],[-13.064,12.566],[-13.059,12.603],[-13.064,12.622],[-13.076,12.636],[-13.127,12.635],[-13.145,12.644],[-13.155,12.633],[-13.169,12.639],[-13.202,12.633],[-13.222,12.648],[-13.255,12.642],[-13.284,12.653],[-13.339,12.639],[-13.36,12.65],[-13.357,12.665],[-13.365,12.685],[-13.378,12.691],[-13.407,12.682],[-13.443,12.684],[-13.466,12.695],[-13.481,12.709],[-13.473,12.771],[-13.489,12.812],[-13.477,12.835],[-13.485,12.852],[-13.471,12.852],[-13.465,12.861],[-13.473,12.892],[-13.504,12.923],[-13.514,12.951],[-13.507,12.965],[-13.512,12.996],[-13.533,13.01],[-13.529,13.029],[-13.57,13.066],[-13.572,13.09],[-13.6,13.115],[-13.607,13.13],[-13.615,13.17],[-13.613,13.226],[-13.623,13.24],[-13.623,13.249],[-13.6,13.26],[-13.599,13.313],[-13.582,13.325],[-13.586,13.308],[-13.568,13.298],[-13.558,13.299],[-13.547,13.317],[-13.513,13.33],[-13.5,13.326],[-13.507,13.304],[-13.494,13.309],[-13.485,13.333],[-13.476,13.338],[-13.397,13.331],[-13.365,13.341],[-13.356,13.325],[-13.378,13.292],[-13.377,13.276],[-13.366,13.249],[-13.353,13.249],[-13.328,13.229],[-13.331,13.214],[-13.321,13.191],[-13.328,13.164],[-13.315,13.154],[-13.356,13.139],[-13.356,13.133],[-13.335,13.124],[-13.328,13.044],[-13.309,13.022],[-13.29,13.017],[-13.203,13.04],[-13.173,13.037],[-13.147,13.028],[-13.129,13.014],[-13.112,13.014],[-13.094,13.037],[-13.073,13.037],[-13.032,13.058],[-12.977,13.058],[-12.956,13.049],[-12.938,13.031],[-12.924,13.031],[-12.874,12.99],[-12.853,12.984],[-12.845,13.011],[-12.836,13.019],[-12.803,13.025],[-12.801,13.043],[-12.792,13.052],[-12.757,13.043],[-12.742,13.046],[-12.742,13.063],[-12.733,13.081],[-12.695,13.113],[-12.701,13.16],[-12.686,13.172],[-12.645,13.175],[-12.616,13.193],[-12.586,13.175],[-12.563,13.172],[-12.531,13.181],[-12.513,13.193],[-12.501,13.213],[-12.498,13.234],[-12.487,13.248],[-12.466,13.251],[-12.46,13.266],[-12.445,13.278],[-12.419,13.284],[-12.396,13.278],[-12.369,13.251],[-12.357,13.254],[-12.337,13.307],[-12.325,13.313],[-12.266,13.307],[-12.252,13.325],[-12.258,13.348],[-12.287,13.383],[-12.293,13.401],[-12.278,13.416],[-12.278,13.439],[-12.255,13.448],[-12.14,13.378],[-12.079,13.36],[-12.067,13.316],[-12.046,13.316],[-12.02,13.333],[-11.993,13.363],[-11.9,13.382]]]],"Tambacounda":[[[[-12.264,14.775],[-12.257,14.746],[-12.237,14.728],[-12.223,14.703],[-12.18,14.693],[-12.192,14.667],[-12.19,14.656],[-12.165,14.642],[-12.166,14.625],[-12.198,14.588],[-12.204,14.554],[-12.23,14.537],[-12.236,14.519],[-12.22,14.435],[-12.222,14.391],[-12.116,14.355],[-12.109,14.342],[-12.113,14.33],[-12.128,14.324],[-12.12,14.311],[-12.098,14.288],[-12.044,14.265],[-11.998,14.166],[-12.003,14.103],[-12.025,14.027],[-12.025,13.985],[-12.015,13.959],[-11.963,13.909],[-11.957,13.89],[-11.963,13.829],[-11.979,13.785],[-12.012,13.751],[-12.098,13.704],[-12.07,13.673],[-12.037,13.598],[-12.015,13.564],[-11.997,13.544],[-11.939,13.503],[-11.918,13.475],[-11.89,13.465],[-11.883,13.454],[-11.9,13.382],[-11.993,13.363],[-12.02,13.333],[-12.046,13.316],[-12.067,13.316],[-12.079,13.36],[-12.14,13.378],[-12.255,13.448],[-12.278,13.439],[-12.278,13.416],[-12.293,13.401],[-12.287,13.383],[-12.258,13.348],[-12.252,13.325],[-12.266,13.307],[-12.325,13.313],[-12.337,13.307],[-12.357,13.254],[-12.369,13.251],[-12.396,13.278],[-12.419,13.284],[-12.445,13.278],[-12.46,13.266],[-12.466,13.251],[-12.487,13.248],[-12.498,13.234],[-12.501,13.213],[-12.513,13.193],[-12.531,13.181],[-12.563,13.172],[-12.586,13.175],[-12.616,13.193],[-12.645,13.175],[-12.686,13.172],[-12.701,13.16],[-12.695,13.113],[-12.733,13.081],[-12.742,13.063],[-12.742,13.046],[-12.757,13.043],[-12.792,13.052],[-12.801,13.043],[-12.803,13.025],[-12.836,13.019],[-12.845,13.011],[-12.853,12.984],[-12.874,12.99],[-12.924,13.031],[-12.938,13.031],[-12.956,13.049],[-12.977,13.058],[-13.032,13.058],[-13.073,13.037],[-13.094,13.037],[-13.112,13.014],[-13.129,13.014],[-13.147,13.028],[-13.173,13.037],[-13.203,13.04],[-13.29,13.017],[-13.309,13.022],[-13.328,13.044],[-13.335,13.124],[-13.356,13.133],[-13.356,13.139],[-13.315,13.154],[-13.328,13.164],[-13.321,13.191],[-13.331,13.214],[-13.328,13.229],[-13.353,13.249],[-13.366,13.249],[-13.377,13.276],[-13.378,13.292],[-13.356,13.325],[-13.365,13.341],[-13.397,13.331],[-13.476,13.338],[-13.485,13.333],[-13.494,13.309],[-13.507,13.304],[-13.5,13.326],[-13.513,13.33],[-13.547,13.317],[-13.558,13.299],[-13.568,13.298],[-13.586,13.308],[-13.582,13.325],[-13.599,13.313],[-13.6,13.26],[-13.623,13.249],[-13.623,13.24],[-13.641,13.241],[-13.648,13.246],[-13.658,13.274],[-13.69,13.294],[-13.688,13.325],[-13.705,13.341],[-13.692,13.362],[-13.695,13.372],[-13.751,13.368],[-13.77,13.375],[-13.771,13.384],[-13.76,13.4],[-13.757,13.428],[-13.729,13.464],[-13.764,13.478],[-13.795,13.468],[-13.808,13.474],[-13.81,13.486],[-13.801,13.501],[-13.798,13.534],[-13.83,13.536],[-13.844,13.515],[-13.883,13.504],[-13.901,13.542],[-13.918,13.555],[-13.988,13.579],[-14.034,13.56],[-14.093,13.555],[-14.139,13.532],[-14.197,13.519],[-14.271,13.477],[-14.33,13.456],[-14.366,13.455],[-14.47,13.522],[-14.487,13.547],[-14.51,13.609],[-14.523,13.626],[-14.586,13.66],[-14.627,13.664],[-14.667,13.653],[-14.741,13.615],[-14.796,13.645],[-14.831,13.736],[-14.86,13.766],[-14.838,13.809],[-14.813,13.84],[-14.716,13.863],[-14.629,13.939],[-14.619,13.999],[-14.595,14.01],[-14.586,14.024],[-14.577,14.082],[-14.589,14.146],[-14.593,14.222],[-14.634,14.323],[-14.633,14.511],[-14.509,14.54],[-14.461,14.573],[-14.431,14.608],[-14.398,14.631],[-14.357,14.644],[-14.313,14.604],[-14.304,14.585],[-14.269,14.563],[-13.834,14.402],[-13.785,14.394],[-13.511,14.399],[-13.238,14.587],[-13.219,14.595],[-12.949,14.546],[-12.755,14.566],[-12.731,14.585],[-12.649,14.706],[-12.573,14.894],[-12.57,14.91],[-12.582,14.949],[-12.69,15.092],[-12.632,15.105],[-12.602,15.079],[-12.53,15.034],[-12.485,15.017],[-12.472,14.998],[-12.448,14.907],[-12.382,14.846],[-12.32,14.817],[-12.264,14.775]]]],"Kolda":[[[[-13.36,12.65],[-13.405,12.662],[-13.728,12.673],[-15.146,12.679],[-15.154,12.741],[-15.131,12.802],[-15.131,12.884],[-15.169,12.937],[-15.169,12.999],[-15.236,13.043],[-15.242,13.099],[-15.278,13.175],[-15.325,13.184],[-15.36,13.207],[-15.366,13.243],[-15.345,13.319],[-15.342,13.363],[-15.277,13.38],[-15.248,13.399],[-15.228,13.425],[-15.218,13.515],[-15.16,13.581],[-15.138,13.59],[-15.015,13.496],[-14.95,13.463],[-14.863,13.447],[-14.829,13.427],[-14.79,13.417],[-14.732,13.359],[-14.698,13.345],[-14.662,13.341],[-14.592,13.353],[-14.53,13.307],[-14.47,13.298],[-14.431,13.261],[-14.369,13.236],[-14.285,13.239],[-14.231,13.229],[-14.202,13.23],[-14.117,13.282],[-14.099,13.282],[-14.057,13.297],[-14.016,13.298],[-13.976,13.308],[-13.901,13.315],[-13.851,13.336],[-13.823,13.378],[-13.819,13.429],[-13.843,13.477],[-13.883,13.504],[-13.844,13.515],[-13.83,13.536],[-13.803,13.536],[-13.796,13.53],[-13.801,13.501],[-13.81,13.486],[-13.808,13.474],[-13.795,13.468],[-13.764,13.478],[-13.729,13.464],[-13.757,13.428],[-13.76,13.4],[-13.771,13.384],[-13.77,13.375],[-13.751,13.368],[-13.701,13.374],[-13.692,13.369],[-13.705,13.341],[-13.688,13.325],[-13.69,13.294],[-13.658,13.274],[-13.648,13.246],[-13.615,13.233],[-13.614,13.164],[-13.607,13.13],[-13.6,13.115],[-13.572,13.09],[-13.57,13.066],[-13.529,13.029],[-13.531,13.006],[-13.512,12.996],[-13.507,12.965],[-13.512,12.946],[-13.5,12.915],[-13.473,12.892],[-13.465,12.861],[-13.471,12.852],[-13.485,12.852],[-13.477,12.841],[-13.488,12.806],[-13.473,12.771],[-13.481,12.709],[-13.466,12.695],[-13.443,12.684],[-13.407,12.682],[-13.378,12.691],[-13.365,12.685],[-13.357,12.665],[-13.36,12.65]]]],"Sédhiou":[[[[-15.146,12.679],[-15.224,12.675],[-15.25,12.664],[-15.308,12.623],[-15.369,12.595],[-15.425,12.555],[-15.677,12.439],[-15.712,12.432],[-15.884,12.442],[-15.947,12.54],[-15.962,12.57],[-15.964,12.602],[-15.94,12.581],[-15.864,12.578],[-15.84,12.557],[-15.806,12.565],[-15.769,12.593],[-15.748,12.584],[-15.73,12.591],[-15.716,12.582],[-15.699,12.584],[-15.68,12.55],[-15.657,12.538],[-15.633,12.54],[-15.588,12.558],[-15.539,12.615],[-15.51,12.636],[-15.507,12.65],[-15.53,12.666],[-15.521,12.674],[-15.54,12.727],[-15.543,12.758],[-15.535,12.783],[-15.523,12.79],[-15.48,12.79],[-15.466,12.796],[-15.446,12.825],[-15.426,12.805],[-15.407,12.8],[-15.394,12.81],[-15.392,12.838],[-15.407,12.822],[-15.453,12.831],[-15.493,12.807],[-15.549,12.794],[-15.555,12.78],[-15.555,12.722],[-15.544,12.684],[-15.542,12.649],[-15.641,12.559],[-15.653,12.559],[-15.672,12.571],[-15.666,12.601],[-15.697,12.614],[-15.768,12.625],[-15.816,12.595],[-15.833,12.591],[-15.848,12.609],[-15.86,12.612],[-15.902,12.598],[-15.953,12.625],[-15.956,12.646],[-15.981,12.653],[-16.001,12.687],[-16.013,12.696],[-16.024,12.738],[-16.009,12.822],[-15.986,12.857],[-15.982,12.888],[-15.987,12.927],[-15.942,12.997],[-15.902,13.079],[-15.907,13.094],[-15.928,13.103],[-15.924,13.123],[-15.897,13.157],[-15.833,13.157],[-15.827,13.162],[-15.819,13.334],[-15.807,13.34],[-15.737,13.346],[-15.693,13.36],[-15.596,13.356],[-15.52,13.387],[-15.379,13.363],[-15.342,13.363],[-15.345,13.319],[-15.366,13.243],[-15.36,13.207],[-15.325,13.184],[-15.278,13.175],[-15.242,13.099],[-15.236,13.043],[-15.169,12.999],[-15.169,12.937],[-15.131,12.884],[-15.131,12.802],[-15.154,12.741],[-15.146,12.679]]]],"Ziguinchor":[[[[-15.973,12.437],[-16.055,12.461],[-16.091,12.464],[-16.166,12.451],[-16.182,12.443],[-16.222,12.455],[-16.412,12.362],[-16.463,12.361],[-16.533,12.348],[-16.669,12.357],[-16.706,12.348],[-16.728,12.333],[-16.744,12.379],[-16.799,12.431],[-16.803,12.485],[-16.781,12.523],[-16.732,12.552],[-16.679,12.55],[-16.635,12.579],[-16.618,12.605],[-16.58,12.633],[-16.561,12.612],[-16.547,12.574],[-16.533,12.565],[-16.533,12.581],[-16.522,12.593],[-16.505,12.598],[-16.388,12.552],[-16.371,12.55],[-16.351,12.56],[-16.327,12.589],[-16.31,12.598],[-16.247,12.588],[-16.212,12.606],[-16.193,12.578],[-16.115,12.606],[-16.058,12.616],[-16.03,12.633],[-16.008,12.632],[-15.964,12.602],[-15.962,12.57],[-15.947,12.54],[-15.884,12.442],[-15.973,12.437]]],[[[-16.674,13.164],[-15.897,13.157],[-15.929,13.111],[-15.928,13.103],[-15.907,13.094],[-15.902,13.079],[-15.942,12.997],[-15.987,12.927],[-15.982,12.888],[-15.986,12.857],[-16.009,12.822],[-16.022,12.729],[-16.028,12.729],[-16.045,12.69],[-16.049,12.656],[-16.077,12.639],[-16.118,12.632],[-16.155,12.612],[-16.228,12.619],[-16.269,12.605],[-16.324,12.61],[-16.347,12.598],[-16.371,12.565],[-16.412,12.578],[-16.445,12.596],[-16.467,12.622],[-16.49,12.632],[-16.493,12.649],[-16.467,12.666],[-16.467,12.674],[-16.483,12.674],[-16.492,12.682],[-16.501,12.708],[-16.508,12.625],[-16.516,12.625],[-16.566,12.674],[-16.57,12.694],[-16.599,12.664],[-16.615,12.66],[-16.635,12.669],[-16.638,12.694],[-16.591,12.762],[-16.597,12.797],[-16.604,12.763],[-16.638,12.729],[-16.651,12.705],[-16.647,12.632],[-16.677,12.62],[-16.721,12.584],[-16.755,12.572],[-16.773,12.583],[-16.783,12.643],[-16.777,12.678],[-16.761,12.713],[-16.735,12.733],[-16.701,12.722],[-16.699,12.736],[-16.713,12.747],[-16.734,12.751],[-16.752,12.745],[-16.784,12.713],[-16.796,12.715],[-16.788,12.732],[-16.796,12.749],[-16.77,12.793],[-16.775,12.811],[-16.783,12.811],[-16.794,12.795],[-16.8,12.799],[-16.803,12.814],[-16.795,12.839],[-16.755,12.903],[-16.762,12.951],[-16.75,13.014],[-16.748,13.04],[-16.754,13.065],[-16.749,13.099],[-16.726,13.123],[-16.708,13.157],[-16.674,13.164]]]],"Saint-Louis":[[[[-14.949,16.671],[-14.931,16.643],[-14.912,16.641],[-14.882,16.648],[-14.711,16.636],[-14.656,16.646],[-14.608,16.629],[-14.555,16.641],[-14.493,16.632],[-14.472,16.636],[-14.426,16.66],[-14.407,16.661],[-14.343,16.637],[-14.331,16.624],[-14.335,16.586],[-14.278,16.581],[-14.268,16.548],[-14.214,16.535],[-14.073,16.401],[-14.04,16.382],[-14.016,16.356],[-13.978,16.338],[-13.971,16.324],[-13.979,16.3],[-13.963,16.272],[-13.968,16.246],[-13.964,16.23],[-13.908,16.21],[-13.884,16.194],[-13.859,16.142],[-13.845,16.126],[-13.827,16.119],[-13.811,16.122],[-13.779,16.143],[-13.747,16.182],[-13.729,16.19],[-13.711,16.189],[-13.723,16.146],[-13.72,16.136],[-13.761,16.051],[-13.863,16.012],[-13.877,15.975],[-13.99,15.877],[-14.298,15.638],[-14.338,15.667],[-14.498,15.747],[-14.592,15.877],[-14.705,15.954],[-14.757,15.95],[-14.854,15.997],[-15.427,15.999],[-15.434,15.932],[-15.591,15.921],[-15.808,16.155],[-15.841,16.182],[-15.891,16.147],[-15.922,16.103],[-16.116,15.997],[-16.151,15.969],[-16.168,15.938],[-16.226,15.905],[-16.257,15.845],[-16.304,15.839],[-16.34,15.818],[-16.378,15.809],[-16.541,15.806],[-16.52,15.846],[-16.51,15.902],[-16.508,16.01],[-16.47,16.106],[-16.466,16.174],[-16.459,16.194],[-16.444,16.206],[-16.407,16.215],[-16.385,16.235],[-16.335,16.456],[-16.314,16.493],[-16.274,16.523],[-16.251,16.528],[-16.195,16.523],[-16.156,16.548],[-16.132,16.555],[-16.11,16.547],[-16.074,16.505],[-16.054,16.495],[-16.007,16.496],[-15.986,16.506],[-15.958,16.498],[-15.897,16.518],[-15.862,16.51],[-15.813,16.514],[-15.768,16.499],[-15.735,16.498],[-15.688,16.481],[-15.672,16.481],[-15.644,16.495],[-15.61,16.521],[-15.551,16.515],[-15.525,16.522],[-15.491,16.565],[-15.464,16.581],[-15.449,16.58],[-15.427,16.553],[-15.415,16.548],[-15.386,16.552],[-15.328,16.572],[-15.232,16.566],[-15.125,16.583],[-15.093,16.6],[-15.096,16.614],[-15.118,16.643],[-15.108,16.665],[-15.079,16.665],[-15.044,16.632],[-15.024,16.64],[-15.013,16.667],[-14.997,16.682],[-14.974,16.691],[-14.949,16.671]]]],"Matam":[[[[-13.72,16.136],[-13.669,16.106],[-13.635,16.122],[-13.607,16.123],[-13.569,16.138],[-13.485,16.155],[-13.483,16.141],[-13.5,16.111],[-13.502,16.095],[-13.492,16.088],[-13.455,16.102],[-13.44,16.102],[-13.422,16.077],[-13.389,16.052],[-13.384,16.019],[-13.37,15.985],[-13.316,15.93],[-13.316,15.882],[-13.294,15.855],[-13.28,15.823],[-13.286,15.77],[-13.251,15.743],[-13.228,15.711],[-13.229,15.697],[-13.249,15.657],[-13.238,15.623],[-13.178,15.628],[-13.11,15.596],[-13.092,15.575],[-13.095,15.515],[-13.084,15.493],[-13.039,15.487],[-12.996,15.514],[-12.98,15.516],[-12.964,15.509],[-12.955,15.493],[-12.951,15.465],[-12.933,15.438],[-12.931,15.397],[-12.919,15.365],[-12.847,15.313],[-12.828,15.284],[-12.833,15.27],[-12.884,15.271],[-12.894,15.263],[-12.888,15.251],[-12.857,15.222],[-12.806,15.223],[-12.787,15.214],[-12.779,15.2],[-12.785,15.166],[-12.78,15.146],[-12.755,15.139],[-12.717,15.103],[-12.69,15.092],[-12.582,14.949],[-12.57,14.91],[-12.573,14.894],[-12.649,14.706],[-12.731,14.585],[-12.755,14.566],[-12.949,14.546],[-13.219,14.595],[-13.238,14.587],[-13.511,14.399],[-13.785,14.394],[-13.834,14.402],[-14.269,14.563],[-14.304,14.585],[-14.313,14.604],[-14.357,14.644],[-14.413,14.623],[-14.461,14.573],[-14.509,14.54],[-14.633,14.511],[-14.632,14.55],[-14.639,14.557],[-14.844,14.622],[-14.863,14.638],[-14.824,14.665],[-14.81,14.745],[-14.766,14.774],[-14.73,14.847],[-14.774,14.941],[-14.803,15.039],[-14.734,15.065],[-14.621,15.079],[-14.542,15.102],[-14.585,15.123],[-14.567,15.231],[-14.534,15.369],[-14.618,15.358],[-14.785,15.442],[-14.748,15.496],[-14.741,15.678],[-14.661,15.707],[-14.621,15.732],[-14.541,15.805],[-14.498,15.747],[-14.338,15.667],[-14.298,15.638],[-13.99,15.877],[-13.877,15.975],[-13.863,16.012],[-13.761,16.051],[-13.72,16.136]]]],"Kaffrine":[[[[-14.86,13.766],[-14.879,13.781],[-14.916,13.792],[-15.017,13.797],[-15.076,13.819],[-15.098,13.82],[-15.171,13.794],[-15.246,13.747],[-15.267,13.742],[-15.304,13.782],[-15.348,13.789],[-15.394,13.772],[-15.445,13.795],[-15.514,13.798],[-15.583,13.827],[-15.623,13.766],[-15.641,13.758],[-15.666,13.769],[-15.695,13.798],[-15.71,13.827],[-15.71,13.853],[-15.699,13.878],[-15.702,13.904],[-15.746,13.929],[-15.782,13.965],[-15.793,13.987],[-15.797,14.012],[-15.811,14.031],[-15.855,14.038],[-15.855,14.154],[-15.8,14.288],[-15.799,14.385],[-15.773,14.405],[-15.71,14.477],[-15.684,14.479],[-15.661,14.491],[-15.594,14.544],[-15.431,14.574],[-15.401,14.607],[-15.38,14.698],[-15.3,14.651],[-15.249,14.633],[-15.063,14.6],[-15.02,14.613],[-14.932,14.666],[-14.898,14.677],[-14.844,14.622],[-14.639,14.557],[-14.631,14.538],[-14.634,14.323],[-14.593,14.222],[-14.589,14.146],[-14.577,14.082],[-14.586,14.024],[-14.595,14.01],[-14.619,13.999],[-14.629,13.939],[-14.716,13.863],[-14.813,13.84],[-14.838,13.809],[-14.86,13.766]]]],"Kaolack":[[[[-15.498,13.642],[-15.502,13.588],[-15.518,13.583],[-16.085,13.585],[-16.094,13.612],[-16.154,13.678],[-16.166,13.738],[-16.227,13.786],[-16.241,13.805],[-16.238,13.842],[-16.206,13.875],[-16.197,13.898],[-16.209,13.974],[-16.248,14.021],[-16.232,14.059],[-16.235,14.104],[-16.249,14.123],[-16.302,14.16],[-16.372,14.163],[-16.365,14.168],[-16.38,14.168],[-16.366,14.187],[-16.356,14.221],[-16.306,14.294],[-16.278,14.326],[-16.257,14.34],[-16.113,14.36],[-16.104,14.359],[-16.099,14.349],[-16.014,14.335],[-15.993,14.342],[-15.975,14.361],[-15.954,14.37],[-15.831,14.365],[-15.799,14.385],[-15.8,14.288],[-15.855,14.154],[-15.855,14.038],[-15.811,14.031],[-15.797,14.012],[-15.793,13.987],[-15.782,13.965],[-15.746,13.929],[-15.702,13.904],[-15.699,13.878],[-15.71,13.853],[-15.71,13.827],[-15.695,13.798],[-15.666,13.769],[-15.641,13.758],[-15.623,13.766],[-15.583,13.827],[-15.514,13.798],[-15.445,13.795],[-15.394,13.772],[-15.436,13.741],[-15.468,13.708],[-15.498,13.642]]]],"Fatick":[[[[-16.085,13.585],[-16.561,13.587],[-16.576,13.612],[-16.576,13.64],[-16.508,13.688],[-16.529,13.701],[-16.507,13.728],[-16.509,13.74],[-16.529,13.743],[-16.531,13.707],[-16.542,13.694],[-16.557,13.695],[-16.564,13.674],[-16.579,13.66],[-16.599,13.654],[-16.618,13.66],[-16.636,13.678],[-16.64,13.696],[-16.63,13.76],[-16.576,13.839],[-16.549,13.825],[-16.557,13.846],[-16.522,13.852],[-16.513,13.813],[-16.502,13.802],[-16.487,13.805],[-16.501,13.873],[-16.516,13.88],[-16.487,14.003],[-16.518,13.952],[-16.526,13.91],[-16.549,13.869],[-16.57,13.873],[-16.58,13.86],[-16.638,13.846],[-16.649,13.837],[-16.659,13.811],[-16.683,13.794],[-16.7,13.769],[-16.71,13.774],[-16.715,13.806],[-16.737,13.82],[-16.742,13.839],[-16.734,13.873],[-16.744,13.902],[-16.743,13.961],[-16.703,13.99],[-16.646,13.983],[-16.637,13.975],[-16.638,13.962],[-16.599,13.997],[-16.576,13.989],[-16.577,14.011],[-16.542,14.037],[-16.568,14.03],[-16.581,14.033],[-16.591,14.044],[-16.546,14.071],[-16.489,14.128],[-16.457,14.137],[-16.441,14.157],[-16.429,14.161],[-16.405,14.145],[-16.372,14.163],[-16.297,14.157],[-16.249,14.123],[-16.235,14.104],[-16.232,14.059],[-16.248,14.021],[-16.209,13.974],[-16.197,13.906],[-16.201,13.883],[-16.238,13.842],[-16.241,13.805],[-16.227,13.786],[-16.166,13.738],[-16.154,13.678],[-16.094,13.612],[-16.085,13.585]]],[[[-16.38,14.168],[-16.428,14.17],[-16.455,14.18],[-16.453,14.202],[-16.469,14.194],[-16.475,14.179],[-16.472,14.164],[-16.46,14.154],[-16.473,14.147],[-16.513,14.151],[-16.522,14.126],[-16.557,14.072],[-16.6,14.054],[-16.632,14.01],[-16.654,14.003],[-16.721,14.016],[-16.697,14.065],[-16.68,14.069],[-16.652,14.1],[-16.666,14.096],[-16.679,14.076],[-16.728,14.065],[-16.716,14.051],[-16.717,14.042],[-16.769,13.962],[-16.773,13.934],[-16.769,13.839],[-16.775,13.839],[-16.781,14.031],[-16.786,14.06],[-16.811,14.125],[-16.77,14.147],[-16.778,14.174],[-16.746,14.208],[-16.759,14.228],[-16.745,14.286],[-16.695,14.427],[-16.648,14.433],[-16.618,14.46],[-16.594,14.471],[-16.546,14.53],[-16.531,14.581],[-16.519,14.599],[-16.462,14.607],[-16.377,14.603],[-16.333,14.581],[-16.264,14.592],[-16.197,14.578],[-16.164,14.554],[-16.14,14.545],[-16.031,14.526],[-15.933,14.535],[-15.874,14.555],[-15.816,14.61],[-15.808,14.63],[-15.803,14.715],[-15.796,14.729],[-15.773,14.742],[-15.724,14.707],[-15.675,14.718],[-15.656,14.716],[-15.633,14.693],[-15.468,14.628],[-15.445,14.609],[-15.431,14.574],[-15.594,14.544],[-15.661,14.491],[-15.684,14.479],[-15.71,14.477],[-15.773,14.405],[-15.831,14.365],[-15.954,14.37],[-15.975,14.361],[-15.993,14.342],[-16.014,14.335],[-16.099,14.349],[-16.104,14.359],[-16.113,14.36],[-16.257,14.34],[-16.278,14.326],[-16.306,14.294],[-16.356,14.221],[-16.366,14.187],[-16.38,14.168]]]],"Louga":[[[[-16.735,15.477],[-16.67,15.562],[-16.539,15.774],[-16.534,15.784],[-16.541,15.806],[-16.378,15.809],[-16.34,15.818],[-16.304,15.839],[-16.257,15.845],[-16.226,15.905],[-16.168,15.938],[-16.151,15.969],[-16.116,15.997],[-15.922,16.103],[-15.891,16.147],[-15.841,16.182],[-15.808,16.155],[-15.591,15.921],[-15.434,15.932],[-15.427,15.999],[-14.854,15.997],[-14.757,15.95],[-14.705,15.954],[-14.592,15.877],[-14.541,15.805],[-14.621,15.732],[-14.661,15.707],[-14.741,15.678],[-14.748,15.496],[-14.785,15.442],[-14.618,15.358],[-14.534,15.369],[-14.567,15.231],[-14.585,15.123],[-14.542,15.102],[-14.621,15.079],[-14.734,15.065],[-14.803,15.039],[-14.774,14.941],[-14.73,14.847],[-14.766,14.774],[-14.81,14.745],[-14.824,14.665],[-14.863,14.638],[-14.898,14.677],[-14.915,14.673],[-15.02,14.613],[-15.063,14.6],[-15.097,14.603],[-15.249,14.633],[-15.3,14.651],[-15.416,14.72],[-15.442,14.747],[-15.453,14.829],[-15.459,14.838],[-15.547,14.835],[-15.593,14.855],[-15.621,14.857],[-15.674,14.856],[-15.7,14.848],[-15.718,14.862],[-15.73,14.902],[-15.776,14.927],[-15.792,14.953],[-15.81,14.963],[-15.817,15.001],[-16.077,15.002],[-16.1,14.996],[-16.132,14.969],[-16.182,14.991],[-16.199,15.01],[-16.212,15.053],[-16.252,15.104],[-16.264,15.139],[-16.261,15.185],[-16.268,15.209],[-16.286,15.242],[-16.318,15.236],[-16.332,15.307],[-16.344,15.318],[-16.386,15.313],[-16.433,15.325],[-16.451,15.321],[-16.51,15.247],[-16.528,15.203],[-16.591,15.134],[-16.609,15.128],[-16.623,15.145],[-16.617,15.181],[-16.621,15.191],[-16.662,15.219],[-16.691,15.26],[-16.717,15.271],[-16.723,15.282],[-16.72,15.343],[-16.695,15.401],[-16.69,15.427],[-16.7,15.451],[-16.735,15.477]]]],"Thiès":[[[[-17.182,14.897],[-17.139,14.928],[-16.98,15.112],[-16.948,15.158],[-16.878,15.233],[-16.735,15.477],[-16.7,15.451],[-16.69,15.427],[-16.695,15.401],[-16.72,15.343],[-16.723,15.282],[-16.717,15.271],[-16.691,15.26],[-16.662,15.219],[-16.621,15.191],[-16.617,15.181],[-16.623,15.145],[-16.614,15.131],[-16.595,15.131],[-16.528,15.203],[-16.51,15.247],[-16.451,15.321],[-16.433,15.325],[-16.386,15.313],[-16.344,15.318],[-16.332,15.307],[-16.318,15.236],[-16.286,15.242],[-16.268,15.209],[-16.261,15.185],[-16.262,15.129],[-16.247,15.096],[-16.212,15.053],[-16.199,15.01],[-16.182,14.991],[-16.132,14.969],[-16.136,14.956],[-16.193,14.926],[-16.226,14.924],[-16.242,14.931],[-16.254,14.961],[-16.281,14.973],[-16.315,15.0],[-16.363,14.994],[-16.399,15.012],[-16.64,14.944],[-16.659,14.935],[-16.674,14.904],[-16.678,14.878],[-16.645,14.856],[-16.643,14.786],[-16.613,14.76],[-16.618,14.724],[-16.592,14.702],[-16.587,14.686],[-16.621,14.636],[-16.62,14.624],[-16.603,14.609],[-16.531,14.581],[-16.546,14.53],[-16.594,14.471],[-16.618,14.46],[-16.648,14.433],[-16.695,14.427],[-16.745,14.286],[-16.759,14.228],[-16.746,14.208],[-16.778,14.174],[-16.77,14.147],[-16.811,14.125],[-16.83,14.154],[-16.869,14.166],[-16.879,14.175],[-16.899,14.26],[-16.926,14.291],[-16.945,14.301],[-16.94,14.356],[-16.957,14.39],[-16.988,14.421],[-17.059,14.454],[-17.111,14.53],[-17.148,14.613],[-17.131,14.76],[-17.153,14.804],[-17.154,14.833],[-17.182,14.897]]]],"Dakar":[[[[-17.148,14.613],[-17.178,14.653],[-17.207,14.679],[-17.244,14.699],[-17.282,14.707],[-17.331,14.736],[-17.417,14.738],[-17.44,14.722],[-17.419,14.709],[-17.427,14.694],[-17.44,14.688],[-17.427,14.675],[-17.44,14.654],[-17.536,14.757],[-17.371,14.8],[-17.322,14.839],[-17.182,14.897],[-17.16,14.851],[-17.153,14.804],[-17.129,14.75],[-17.148,14.613]]]],"Diourbel":[[[[-15.674,14.856],[-15.593,14.855],[-15.547,14.835],[-15.481,14.841],[-15.456,14.834],[-15.442,14.747],[-15.416,14.72],[-15.38,14.698],[-15.401,14.607],[-15.431,14.574],[-15.445,14.609],[-15.468,14.628],[-15.633,14.693],[-15.656,14.716],[-15.675,14.718],[-15.724,14.707],[-15.773,14.742],[-15.796,14.729],[-15.803,14.715],[-15.808,14.63],[-15.816,14.61],[-15.874,14.555],[-15.933,14.535],[-16.031,14.526],[-16.154,14.549],[-16.188,14.573],[-16.231,14.588],[-16.274,14.591],[-16.333,14.581],[-16.377,14.603],[-16.462,14.607],[-16.506,14.603],[-16.519,14.599],[-16.531,14.581],[-16.603,14.609],[-16.62,14.624],[-16.621,14.636],[-16.595,14.667],[-16.588,14.692],[-16.596,14.706],[-16.617,14.719],[-16.616,14.764],[-16.643,14.786],[-16.645,14.856],[-16.671,14.869],[-16.679,14.882],[-16.669,14.92],[-16.659,14.935],[-16.64,14.944],[-16.399,15.012],[-16.363,14.994],[-16.315,15.0],[-16.281,14.973],[-16.254,14.961],[-16.242,14.931],[-16.218,14.922],[-16.193,14.926],[-16.145,14.951],[-16.1,14.996],[-16.077,15.002],[-15.832,15.003],[-15.817,15.001],[-15.813,14.967],[-15.792,14.953],[-15.78,14.93],[-15.73,14.902],[-15.718,14.862],[-15.7,14.848],[-15.674,14.856]]]]},"label_points":{"Kédougou":[-12.447,12.881],"Tambacounda":[-13.301,14.046],"Kolda":[-14.428,13.123],"Sédhiou":[-15.566,12.909],"Ziguinchor":[-16.355,12.475],"Saint-Louis":[-14.843,16.165],"Matam":[-13.693,15.277],"Kaffrine":[-15.214,14.206],"Kaolack":[-16.001,13.979],"Fatick":[-16.526,14.29],"Louga":[-15.687,15.385],"Thiès":[-16.898,14.799],"Dakar":[-17.331,14.754],"Diourbel":[-16.034,14.768]},"zones":{"South-East":[[[[-14.86,13.766],[-14.838,13.809],[-14.813,13.84],[-14.716,13.863],[-14.629,13.939],[-14.619,13.999],[-14.595,14.01],[-14.586,14.024],[-14.577,14.082],[-14.589,14.146],[-14.593,14.222],[-14.634,14.323],[-14.633,14.511],[-14.509,14.54],[-14.461,14.573],[-14.413,14.623],[-14.357,14.644],[-14.313,14.604],[-14.304,14.585],[-14.269,14.563],[-13.834,14.402],[-13.785,14.394],[-13.511,14.398],[-13.238,14.587],[-13.219,14.595],[-12.949,14.546],[-12.755,14.566],[-12.731,14.585],[-12.649,14.706],[-12.573,14.894],[-12.57,14.91],[-12.582,14.949],[-12.69,15.092],[-12.632,15.105],[-12.623,15.102],[-12.602,15.079],[-12.519,15.028],[-12.485,15.017],[-12.472,14.998],[-12.448,14.907],[-12.383,14.846],[-12.32,14.817],[-12.264,14.775],[-12.256,14.746],[-12.237,14.728],[-12.223,14.703],[-12.18,14.693],[-12.192,14.667],[-12.19,14.656],[-12.165,14.642],[-12.166,14.625],[-12.198,14.588],[-12.204,14.554],[-12.23,14.537],[-12.236,14.519],[-12.22,14.435],[-12.222,14.392],[-12.116,14.355],[-12.109,14.342],[-12.113,14.33],[-12.128,14.324],[-12.11,14.298],[-12.058,14.274],[-12.034,14.25],[-11.998,14.166],[-11.995,14.144],[-12.025,14.027],[-12.025,13.985],[-12.015,13.959],[-11.963,13.909],[-11.957,13.89],[-11.963,13.829],[-11.979,13.785],[-12.012,13.751],[-12.097,13.704],[-12.07,13.673],[-12.037,13.598],[-11.997,13.544],[-11.939,13.503],[-11.918,13.475],[-11.89,13.465],[-11.883,13.454],[-11.9,13.382],[-11.898,13.371],[-11.855,13.351],[-11.845,13.324],[-11.828,13.307],[-11.775,13.343],[-11.767,13.383],[-11.757,13.4],[-11.741,13.408],[-11.727,13.406],[-11.704,13.38],[-11.65,13.384],[-11.644,13.356],[-11.613,13.361],[-11.606,13.32],[-11.554,13.257],[-11.572,13.217],[-11.569,13.199],[-11.539,13.163],[-11.533,13.112],[-11.45,13.075],[-11.459,13.055],[-11.436,13.017],[-11.423,13.009],[-11.433,12.997],[-11.438,12.959],[-11.412,12.961],[-11.402,12.989],[-11.393,12.981],[-11.384,12.933],[-11.403,12.916],[-11.421,12.922],[-11.427,12.918],[-11.415,12.897],[-11.429,12.866],[-11.425,12.85],[-11.432,12.835],[-11.408,12.804],[-11.405,12.782],[-11.415,12.76],[-11.402,12.732],[-11.41,12.719],[-11.437,12.712],[-11.448,12.686],[-11.462,12.673],[-11.461,12.658],[-11.442,12.648],[-11.437,12.637],[-11.442,12.596],[-11.437,12.571],[-11.44,12.559],[-11.467,12.544],[-11.449,12.535],[-11.431,12.509],[-11.425,12.526],[-11.417,12.529],[-11.401,12.481],[-11.39,12.498],[-11.378,12.48],[-11.388,12.404],[-11.416,12.405],[-11.482,12.428],[-11.515,12.432],[-11.643,12.418],[-11.72,12.389],[-11.758,12.383],[-11.84,12.387],[-11.924,12.419],[-11.998,12.386],[-12.079,12.408],[-12.104,12.407],[-12.192,12.349],[-12.361,12.306],[-12.377,12.314],[-12.406,12.357],[-12.423,12.369],[-12.467,12.384],[-12.505,12.39],[-12.571,12.364],[-12.594,12.362],[-12.603,12.382],[-12.648,12.426],[-12.688,12.436],[-12.773,12.435],[-12.784,12.453],[-12.839,12.466],[-12.849,12.476],[-12.849,12.495],[-12.874,12.516],[-12.913,12.536],[-12.95,12.536],[-12.97,12.477],[-12.981,12.467],[-13.052,12.472],[-13.067,12.48],[-13.078,12.496],[-13.083,12.516],[-13.064,12.566],[-13.059,12.603],[-13.064,12.622],[-13.076,12.636],[-13.127,12.635],[-13.145,12.644],[-13.155,12.633],[-13.169,12.639],[-13.202,12.633],[-13.222,12.648],[-13.255,12.642],[-13.284,12.653],[-13.339,12.639],[-13.36,12.65],[-13.357,12.665],[-13.365,12.685],[-13.378,12.691],[-13.407,12.682],[-13.443,12.684],[-13.466,12.695],[-13.481,12.709],[-13.473,12.771],[-13.488,12.806],[-13.477,12.84],[-13.485,12.852],[-13.471,12.852],[-13.465,12.861],[-13.473,12.892],[-13.5,12.915],[-13.512,12.946],[-13.507,12.965],[-13.512,12.996],[-13.531,13.006],[-13.529,13.029],[-13.57,13.066],[-13.572,13.09],[-13.6,13.115],[-13.607,13.13],[-13.614,13.164],[-13.615,13.233],[-13.648,13.246],[-13.658,13.273],[-13.69,13.294],[-13.688,13.325],[-13.705,13.341],[-13.692,13.369],[-13.701,13.374],[-13.751,13.368],[-13.77,13.375],[-13.771,13.384],[-13.76,13.4],[-13.757,13.428],[-13.729,13.464],[-13.764,13.478],[-13.795,13.468],[-13.808,13.474],[-13.81,13.486],[-13.801,13.501],[-13.798,13.534],[-13.83,13.536],[-13.844,13.515],[-13.883,13.504],[-13.901,13.542],[-13.918,13.555],[-13.989,13.579],[-14.034,13.56],[-14.093,13.555],[-14.139,13.532],[-14.197,13.519],[-14.271,13.477],[-14.33,13.456],[-14.366,13.455],[-14.47,13.522],[-14.487,13.547],[-14.51,13.609],[-14.523,13.626],[-14.585,13.66],[-14.627,13.663],[-14.667,13.653],[-14.741,13.615],[-14.796,13.645],[-14.831,13.736],[-14.86,13.766]]]],"West":[[[[-17.536,14.757],[-17.371,14.8],[-17.322,14.839],[-17.173,14.901],[-17.139,14.928],[-16.98,15.112],[-16.948,15.158],[-16.878,15.233],[-16.735,15.477],[-16.7,15.451],[-16.69,15.427],[-16.695,15.401],[-16.72,15.343],[-16.723,15.282],[-16.717,15.271],[-16.691,15.26],[-16.662,15.219],[-16.621,15.191],[-16.617,15.181],[-16.623,15.145],[-16.614,15.131],[-16.595,15.131],[-16.528,15.203],[-16.51,15.247],[-16.451,15.321],[-16.433,15.325],[-16.386,15.313],[-16.344,15.318],[-16.332,15.307],[-16.318,15.236],[-16.286,15.242],[-16.268,15.209],[-16.261,15.185],[-16.261,15.129],[-16.247,15.096],[-16.212,15.053],[-16.199,15.01],[-16.182,14.991],[-16.132,14.969],[-16.136,14.956],[-16.193,14.926],[-16.226,14.924],[-16.242,14.931],[-16.254,14.961],[-16.281,14.973],[-16.315,15.0],[-16.363,14.994],[-16.4,15.012],[-16.64,14.944],[-16.659,14.935],[-16.674,14.904],[-16.678,14.878],[-16.645,14.856],[-16.643,14.786],[-16.613,14.76],[-16.618,14.724],[-16.592,14.702],[-16.587,14.686],[-16.621,14.636],[-16.62,14.624],[-16.603,14.609],[-16.531,14.581],[-16.546,14.53],[-16.594,14.471],[-16.618,14.46],[-16.648,14.433],[-16.695,14.427],[-16.745,14.286],[-16.759,14.228],[-16.746,14.208],[-16.778,14.174],[-16.77,14.147],[-16.811,14.125],[-16.83,14.154],[-16.869,14.166],[-16.879,14.175],[-16.899,14.26],[-16.926,14.291],[-16.945,14.301],[-16.94,14.356],[-16.957,14.39],[-16.988,14.421],[-17.059,14.454],[-17.07,14.465],[-17.111,14.53],[-17.138,14.599],[-17.178,14.653],[-17.207,14.679],[-17.244,14.699],[-17.282,14.707],[-17.331,14.736],[-17.417,14.738],[-17.44,14.722],[-17.419,14.709],[-17.427,14.694],[-17.439,14.688],[-17.427,14.675],[-17.44,14.654],[-17.536,14.757]]]],"South":[[[[-16.803,12.814],[-16.795,12.839],[-16.755,12.903],[-16.762,12.951],[-16.748,13.04],[-16.752,13.088],[-16.708,13.157],[-16.674,13.164],[-15.833,13.157],[-15.827,13.162],[-15.819,13.334],[-15.807,13.34],[-15.737,13.346],[-15.692,13.36],[-15.596,13.356],[-15.52,13.387],[-15.379,13.363],[-15.342,13.363],[-15.277,13.38],[-15.248,13.399],[-15.228,13.425],[-15.218,13.515],[-15.204,13.537],[-15.16,13.581],[-15.138,13.59],[-15.015,13.496],[-14.95,13.463],[-14.863,13.447],[-14.829,13.427],[-14.79,13.417],[-14.732,13.359],[-14.698,13.345],[-14.662,13.341],[-14.592,13.353],[-14.53,13.307],[-14.47,13.298],[-14.442,13.269],[-14.395,13.243],[-14.342,13.234],[-14.285,13.239],[-14.231,13.229],[-14.202,13.23],[-14.117,13.282],[-14.099,13.282],[-14.057,13.297],[-13.901,13.315],[-13.851,13.336],[-13.823,13.378],[-13.819,13.429],[-13.843,13.477],[-13.883,13.504],[-13.844,13.515],[-13.834,13.533],[-13.825,13.537],[-13.796,13.53],[-13.801,13.501],[-13.81,13.486],[-13.801,13.469],[-13.77,13.478],[-13.73,13.466],[-13.757,13.428],[-13.76,13.4],[-13.771,13.384],[-13.77,13.375],[-13.751,13.368],[-13.701,13.374],[-13.692,13.369],[-13.705,13.341],[-13.688,13.325],[-13.69,13.294],[-13.658,13.274],[-13.648,13.246],[-13.615,13.233],[-13.614,13.164],[-13.607,13.13],[-13.6,13.115],[-13.572,13.09],[-13.57,13.066],[-13.529,13.029],[-13.531,13.007],[-13.512,12.996],[-13.507,12.965],[-13.512,12.946],[-13.5,12.915],[-13.473,12.892],[-13.465,12.861],[-13.471,12.852],[-13.485,12.851],[-13.477,12.84],[-13.488,12.806],[-13.473,12.771],[-13.481,12.709],[-13.466,12.695],[-13.443,12.684],[-13.407,12.682],[-13.378,12.691],[-13.365,12.685],[-13.357,12.665],[-13.36,12.65],[-13.405,12.662],[-13.728,12.673],[-15.195,12.679],[-15.25,12.664],[-15.308,12.623],[-15.369,12.595],[-15.425,12.555],[-15.677,12.439],[-15.712,12.432],[-15.894,12.443],[-15.973,12.437],[-16.055,12.461],[-16.09,12.464],[-16.166,12.451],[-16.182,12.443],[-16.222,12.455],[-16.412,12.362],[-16.463,12.361],[-16.533,12.348],[-16.669,12.357],[-16.706,12.348],[-16.728,12.333],[-16.744,12.379],[-16.799,12.431],[-16.801,12.494],[-16.781,12.523],[-16.732,12.552],[-16.679,12.55],[-16.645,12.571],[-16.618,12.605],[-16.58,12.633],[-16.561,12.612],[-16.547,12.574],[-16.534,12.566],[-16.533,12.581],[-16.522,12.593],[-16.505,12.598],[-16.388,12.552],[-16.371,12.55],[-16.351,12.56],[-16.327,12.589],[-16.31,12.598],[-16.247,12.588],[-16.212,12.606],[-16.193,12.578],[-16.115,12.606],[-16.058,12.616],[-16.03,12.633],[-16.008,12.632],[-15.99,12.623],[-15.94,12.581],[-15.864,12.578],[-15.84,12.557],[-15.806,12.565],[-15.769,12.593],[-15.748,12.584],[-15.73,12.591],[-15.716,12.582],[-15.699,12.584],[-15.673,12.544],[-15.645,12.537],[-15.588,12.558],[-15.539,12.615],[-15.51,12.636],[-15.507,12.65],[-15.53,12.666],[-15.521,12.674],[-15.54,12.727],[-15.543,12.758],[-15.535,12.783],[-15.523,12.79],[-15.48,12.79],[-15.466,12.796],[-15.446,12.825],[-15.426,12.805],[-15.407,12.8],[-15.394,12.81],[-15.392,12.837],[-15.407,12.822],[-15.453,12.831],[-15.493,12.807],[-15.549,12.794],[-15.555,12.78],[-15.555,12.722],[-15.544,12.684],[-15.542,12.649],[-15.628,12.568],[-15.653,12.559],[-15.672,12.571],[-15.667,12.601],[-15.697,12.614],[-15.768,12.625],[-15.816,12.595],[-15.833,12.591],[-15.848,12.608],[-15.86,12.612],[-15.89,12.6],[-15.913,12.601],[-15.953,12.625],[-15.953,12.644],[-15.981,12.653],[-16.001,12.687],[-16.013,12.696],[-16.022,12.729],[-16.028,12.729],[-16.045,12.69],[-16.049,12.656],[-16.077,12.639],[-16.118,12.632],[-16.155,12.612],[-16.228,12.619],[-16.269,12.605],[-16.324,12.61],[-16.347,12.598],[-16.371,12.565],[-16.412,12.578],[-16.445,12.596],[-16.467,12.622],[-16.49,12.632],[-16.493,12.649],[-16.467,12.666],[-16.467,12.673],[-16.483,12.674],[-16.492,12.682],[-16.501,12.705],[-16.508,12.625],[-16.516,12.625],[-16.566,12.674],[-16.57,12.693],[-16.599,12.664],[-16.615,12.66],[-16.635,12.669],[-16.638,12.694],[-16.591,12.762],[-16.597,12.795],[-16.604,12.763],[-16.638,12.728],[-16.651,12.705],[-16.647,12.632],[-16.677,12.62],[-16.721,12.584],[-16.755,12.572],[-16.773,12.583],[-16.781,12.609],[-16.777,12.678],[-16.761,12.713],[-16.735,12.733],[-16.701,12.722],[-16.699,12.736],[-16.713,12.747],[-16.734,12.751],[-16.752,12.745],[-16.784,12.713],[-16.796,12.715],[-16.788,12.732],[-16.796,12.749],[-16.77,12.793],[-16.775,12.811],[-16.783,12.811],[-16.794,12.795],[-16.8,12.799],[-16.803,12.814]]]],"Centre":[[[[-16.811,14.125],[-16.77,14.147],[-16.778,14.174],[-16.746,14.207],[-16.759,14.228],[-16.745,14.286],[-16.695,14.427],[-16.648,14.433],[-16.618,14.46],[-16.594,14.471],[-16.546,14.53],[-16.531,14.58],[-16.603,14.609],[-16.62,14.624],[-16.621,14.636],[-16.588,14.686],[-16.593,14.702],[-16.618,14.724],[-16.613,14.76],[-16.643,14.786],[-16.645,14.856],[-16.678,14.878],[-16.674,14.904],[-16.659,14.935],[-16.64,14.944],[-16.399,15.012],[-16.363,14.994],[-16.315,15.0],[-16.281,14.973],[-16.254,14.961],[-16.242,14.931],[-16.218,14.922],[-16.193,14.926],[-16.145,14.951],[-16.1,14.996],[-16.077,15.002],[-15.832,15.003],[-15.817,15.001],[-15.813,14.967],[-15.792,14.953],[-15.78,14.93],[-15.73,14.902],[-15.718,14.862],[-15.7,14.848],[-15.674,14.856],[-15.621,14.857],[-15.593,14.855],[-15.547,14.835],[-15.459,14.838],[-15.453,14.829],[-15.442,14.747],[-15.416,14.72],[-15.3,14.651],[-15.249,14.633],[-15.063,14.6],[-15.02,14.613],[-14.932,14.666],[-14.898,14.677],[-14.844,14.622],[-14.639,14.557],[-14.631,14.538],[-14.634,14.323],[-14.593,14.222],[-14.577,14.065],[-14.591,14.016],[-14.619,13.999],[-14.629,13.939],[-14.716,13.863],[-14.813,13.84],[-14.838,13.809],[-14.86,13.766],[-14.879,13.781],[-14.916,13.792],[-15.017,13.797],[-15.076,13.819],[-15.098,13.82],[-15.171,13.794],[-15.246,13.747],[-15.267,13.742],[-15.304,13.782],[-15.348,13.789],[-15.394,13.771],[-15.436,13.741],[-15.467,13.708],[-15.497,13.648],[-15.502,13.588],[-15.518,13.583],[-16.561,13.587],[-16.576,13.612],[-16.576,13.64],[-16.509,13.688],[-16.529,13.701],[-16.507,13.728],[-16.509,13.74],[-16.529,13.742],[-16.531,13.707],[-16.542,13.694],[-16.556,13.695],[-16.564,13.674],[-16.579,13.66],[-16.599,13.654],[-16.618,13.66],[-16.636,13.678],[-16.64,13.696],[-16.63,13.76],[-16.576,13.839],[-16.55,13.825],[-16.557,13.846],[-16.522,13.852],[-16.513,13.813],[-16.502,13.802],[-16.487,13.805],[-16.501,13.873],[-16.516,13.88],[-16.487,14.001],[-16.509,13.971],[-16.526,13.91],[-16.549,13.869],[-16.57,13.873],[-16.58,13.86],[-16.638,13.846],[-16.649,13.837],[-16.659,13.811],[-16.682,13.794],[-16.7,13.769],[-16.71,13.774],[-16.715,13.806],[-16.737,13.82],[-16.742,13.839],[-16.734,13.873],[-16.744,13.902],[-16.747,13.951],[-16.737,13.968],[-16.703,13.99],[-16.646,13.983],[-16.637,13.975],[-16.637,13.962],[-16.599,13.997],[-16.577,13.989],[-16.577,14.011],[-16.545,14.036],[-16.568,14.03],[-16.581,14.033],[-16.591,14.044],[-16.545,14.071],[-16.489,14.128],[-16.457,14.137],[-16.441,14.157],[-16.429,14.161],[-16.405,14.145],[-16.366,14.168],[-16.428,14.17],[-16.455,14.18],[-16.453,14.201],[-16.469,14.194],[-16.475,14.179],[-16.472,14.164],[-16.46,14.154],[-16.473,14.147],[-16.513,14.151],[-16.522,14.126],[-16.557,14.072],[-16.6,14.054],[-16.632,14.01],[-16.654,14.003],[-16.721,14.016],[-16.697,14.065],[-16.68,14.069],[-16.654,14.099],[-16.666,14.096],[-16.679,14.076],[-16.727,14.065],[-16.716,14.051],[-16.717,14.042],[-16.769,13.962],[-16.773,13.934],[-16.769,13.839],[-16.775,13.839],[-16.781,14.031],[-16.786,14.06],[-16.811,14.125]]]],"North":[[[[-16.735,15.477],[-16.67,15.562],[-16.534,15.784],[-16.542,15.809],[-16.52,15.846],[-16.51,15.901],[-16.508,16.01],[-16.47,16.106],[-16.466,16.174],[-16.459,16.194],[-16.444,16.206],[-16.407,16.215],[-16.385,16.235],[-16.335,16.456],[-16.314,16.493],[-16.274,16.523],[-16.251,16.528],[-16.195,16.523],[-16.156,16.548],[-16.132,16.555],[-16.11,16.547],[-16.074,16.505],[-16.054,16.495],[-16.007,16.496],[-15.986,16.506],[-15.958,16.498],[-15.897,16.518],[-15.862,16.51],[-15.813,16.514],[-15.768,16.499],[-15.735,16.498],[-15.688,16.481],[-15.672,16.481],[-15.644,16.495],[-15.61,16.521],[-15.551,16.515],[-15.525,16.522],[-15.491,16.565],[-15.464,16.581],[-15.449,16.58],[-15.427,16.553],[-15.414,16.548],[-15.386,16.552],[-15.328,16.572],[-15.233,16.566],[-15.125,16.583],[-15.093,16.6],[-15.096,16.614],[-15.118,16.643],[-15.108,16.665],[-15.079,16.665],[-15.052,16.634],[-15.037,16.633],[-15.019,16.646],[-15.007,16.676],[-14.974,16.691],[-14.95,16.672],[-14.931,16.643],[-14.913,16.641],[-14.882,16.648],[-14.711,16.636],[-14.656,16.646],[-14.608,16.629],[-14.555,16.641],[-14.493,16.632],[-14.472,16.636],[-14.426,16.66],[-14.407,16.661],[-14.336,16.632],[-14.331,16.624],[-14.335,16.586],[-14.278,16.581],[-14.268,16.548],[-14.214,16.535],[-14.073,16.401],[-14.04,16.382],[-14.016,16.356],[-13.978,16.338],[-13.971,16.324],[-13.979,16.3],[-13.963,16.272],[-13.968,16.246],[-13.964,16.23],[-13.908,16.21],[-13.884,16.194],[-13.859,16.142],[-13.836,16.121],[-13.819,16.119],[-13.795,16.131],[-13.747,16.182],[-13.729,16.19],[-13.711,16.189],[-13.723,16.146],[-13.72,16.136],[-13.669,16.106],[-13.635,16.122],[-13.607,16.123],[-13.569,16.138],[-13.485,16.155],[-13.483,16.141],[-13.5,16.111],[-13.502,16.095],[-13.491,16.088],[-13.455,16.102],[-13.44,16.102],[-13.422,16.077],[-13.389,16.052],[-13.384,16.019],[-13.37,15.985],[-13.316,15.93],[-13.316,15.882],[-13.294,15.855],[-13.28,15.823],[-13.286,15.77],[-13.251,15.743],[-13.228,15.711],[-13.229,15.697],[-13.249,15.657],[-13.238,15.624],[-13.178,15.628],[-13.11,15.596],[-13.092,15.575],[-13.095,15.515],[-13.084,15.493],[-13.039,15.487],[-12.996,15.514],[-12.98,15.516],[-12.964,15.509],[-12.955,15.493],[-12.951,15.465],[-12.933,15.438],[-12.931,15.397],[-12.919,15.365],[-12.847,15.313],[-12.828,15.284],[-12.833,15.27],[-12.884,15.271],[-12.894,15.263],[-12.888,15.251],[-12.857,15.222],[-12.806,15.223],[-12.787,15.214],[-12.779,15.2],[-12.785,15.166],[-12.78,15.147],[-12.755,15.139],[-12.717,15.103],[-12.69,15.092],[-12.582,14.949],[-12.571,14.903],[-12.649,14.706],[-12.743,14.571],[-12.906,14.547],[-12.949,14.546],[-13.219,14.595],[-13.238,14.587],[-13.511,14.399],[-13.785,14.394],[-13.834,14.402],[-14.269,14.563],[-14.304,14.585],[-14.313,14.604],[-14.357,14.644],[-14.413,14.623],[-14.461,14.573],[-14.509,14.54],[-14.633,14.511],[-14.632,14.55],[-14.639,14.557],[-14.844,14.622],[-14.898,14.677],[-14.932,14.666],[-15.02,14.613],[-15.063,14.6],[-15.249,14.633],[-15.3,14.651],[-15.416,14.72],[-15.442,14.747],[-15.453,14.829],[-15.459,14.838],[-15.547,14.835],[-15.593,14.855],[-15.621,14.857],[-15.674,14.856],[-15.7,14.848],[-15.718,14.862],[-15.73,14.902],[-15.776,14.927],[-15.792,14.953],[-15.81,14.963],[-15.817,15.001],[-16.077,15.002],[-16.1,14.996],[-16.132,14.969],[-16.177,14.989],[-16.199,15.01],[-16.212,15.053],[-16.252,15.104],[-16.264,15.139],[-16.261,15.185],[-16.268,15.209],[-16.286,15.242],[-16.318,15.236],[-16.332,15.307],[-16.344,15.318],[-16.386,15.313],[-16.433,15.325],[-16.451,15.321],[-16.51,15.247],[-16.528,15.203],[-16.591,15.134],[-16.609,15.128],[-16.623,15.145],[-16.617,15.181],[-16.621,15.191],[-16.662,15.219],[-16.691,15.26],[-16.717,15.271],[-16.723,15.282],[-16.72,15.343],[-16.695,15.401],[-16.69,15.427],[-16.7,15.451],[-16.735,15.477]]]]},"neighbours":{"GIN":[[[[-14.783,11.4],[-14.711,11.498],[-14.687,11.511],[-14.645,11.515],[-14.562,11.506],[-14.521,11.513],[-14.339,11.611],[-14.326,11.621],[-14.308,11.655],[-14.29,11.669],[-14.162,11.661],[-14.135,11.642],[-14.119,11.638],[-14.059,11.644],[-14.015,11.637],[-13.945,11.676],[-13.924,11.665],[-13.904,11.665],[-13.886,11.676],[-13.871,11.704],[-13.87,11.744],[-13.846,11.737],[-13.843,11.717],[-13.82,11.708],[-13.822,11.694],[-13.816,11.687],[-13.77,11.689],[-13.73,11.71],[-13.72,11.766],[-13.735,11.902],[-13.724,12.003],[-13.735,12.019],[-13.78,12.029],[-13.808,12.044],[-13.842,12.091],[-13.87,12.101],[-13.918,12.138],[-13.973,12.152],[-13.966,12.194],[-13.944,12.219],[-13.883,12.246],[-13.84,12.277],[-13.813,12.284],[-13.796,12.279],[-13.809,12.256],[-13.741,12.257],[-13.712,12.268],[-13.695,12.293],[-13.683,12.339],[-13.684,12.392],[-13.661,12.459],[-13.675,12.496],[-13.725,12.558],[-13.735,12.585],[-13.728,12.673],[-13.405,12.662],[-13.339,12.639],[-13.284,12.653],[-13.255,12.642],[-13.222,12.648],[-13.202,12.633],[-13.169,12.639],[-13.155,12.633],[-13.145,12.644],[-13.127,12.635],[-13.076,12.636],[-13.064,12.622],[-13.059,12.603],[-13.064,12.566],[-13.083,12.516],[-13.078,12.496],[-13.067,12.48],[-13.052,12.472],[-12.981,12.467],[-12.97,12.477],[-12.949,12.536],[-12.913,12.536],[-12.874,12.516],[-12.849,12.495],[-12.849,12.476],[-12.839,12.466],[-12.783,12.453],[-12.773,12.435],[-12.688,12.436],[-12.648,12.426],[-12.603,12.382],[-12.594,12.362],[-12.571,12.364],[-12.505,12.39],[-12.467,12.384],[-12.423,12.369],[-12.406,12.356],[-12.377,12.314],[-12.361,12.306],[-12.192,12.349],[-12.104,12.407],[-12.079,12.408],[-11.998,12.386],[-11.924,12.419],[-11.84,12.387],[-11.758,12.383],[-11.602,12.425],[-11.515,12.432],[-11.482,12.428],[-11.416,12.405],[-11.388,12.404],[-11.389,12.384],[-11.432,12.383],[-11.447,12.375],[-11.451,12.352],[-11.445,12.289],[-11.463,12.25],[-11.492,12.221],[-11.508,12.192],[-11.49,12.151],[-11.471,12.135],[-11.41,12.108],[-11.392,12.094],[-11.351,12.04],[-11.315,12.013],[-11.275,11.996],[-11.256,11.996],[-11.191,12.014],[-11.166,12.049],[-11.159,12.078],[-11.136,12.085],[-11.117,12.114],[-11.073,12.149],[-11.052,12.203],[-11.015,12.204],[-10.952,12.219],[-10.928,12.213],[-10.91,12.2],[-10.87,12.138],[-10.805,12.096],[-10.799,12.068],[-10.811,12.042],[-10.808,12.023],[-10.781,11.996],[-10.74,11.92],[-10.711,11.89],[-10.67,11.892],[-10.655,11.899],[-10.634,11.925],[-10.622,11.958],[-10.6,11.977],[-10.6,11.4],[-14.783,11.4]]]],"MLI":[[[[-12.247,14.767],[-12.203,14.766],[-12.139,14.785],[-12.07,14.734],[-12.05,14.767],[-12.011,14.763],[-11.991,14.768],[-11.974,14.78],[-11.936,14.824],[-11.877,14.842],[-11.837,14.893],[-11.82,14.9],[-11.831,14.926],[-11.813,14.996],[-11.814,15.029],[-11.833,15.049],[-11.851,15.042],[-11.861,15.049],[-11.845,15.125],[-11.846,15.179],[-11.827,15.233],[-11.817,15.296],[-11.783,15.366],[-11.768,15.437],[-11.755,15.469],[-11.729,15.496],[-11.727,15.541],[-11.644,15.526],[-11.615,15.541],[-11.579,15.578],[-11.524,15.6],[-11.515,15.61],[-11.513,15.625],[-11.527,15.643],[-11.458,15.633],[-11.435,15.623],[-11.41,15.602],[-11.298,15.451],[-11.045,15.27],[-11.013,15.24],[-10.985,15.187],[-10.948,15.152],[-10.915,15.103],[-10.891,15.177],[-10.844,15.235],[-10.827,15.288],[-10.811,15.303],[-10.782,15.308],[-10.771,15.325],[-10.747,15.341],[-10.739,15.362],[-10.736,15.423],[-10.725,15.433],[-10.709,15.434],[-10.649,15.425],[-10.6,15.425],[-10.6,11.977],[-10.622,11.958],[-10.634,11.925],[-10.655,11.899],[-10.67,11.892],[-10.711,11.89],[-10.74,11.92],[-10.781,11.996],[-10.812,12.033],[-10.799,12.068],[-10.805,12.096],[-10.87,12.138],[-10.91,12.2],[-10.928,12.213],[-10.972,12.219],[-11.015,12.204],[-11.052,12.203],[-11.073,12.149],[-11.117,12.114],[-11.136,12.085],[-11.159,12.078],[-11.166,12.049],[-11.191,12.014],[-11.256,11.996],[-11.295,12.003],[-11.335,12.026],[-11.392,12.094],[-11.41,12.108],[-11.471,12.135],[-11.49,12.151],[-11.508,12.192],[-11.492,12.221],[-11.463,12.25],[-11.445,12.289],[-11.451,12.352],[-11.447,12.375],[-11.432,12.383],[-11.389,12.384],[-11.378,12.48],[-11.39,12.498],[-11.401,12.48],[-11.417,12.529],[-11.425,12.526],[-11.43,12.508],[-11.449,12.535],[-11.468,12.544],[-11.44,12.559],[-11.437,12.571],[-11.442,12.596],[-11.437,12.637],[-11.442,12.648],[-11.461,12.658],[-11.462,12.673],[-11.448,12.686],[-11.437,12.712],[-11.41,12.719],[-11.402,12.732],[-11.415,12.76],[-11.405,12.782],[-11.408,12.804],[-11.433,12.836],[-11.425,12.85],[-11.429,12.866],[-11.415,12.897],[-11.427,12.919],[-11.403,12.916],[-11.388,12.927],[-11.389,12.971],[-11.402,12.989],[-11.412,12.961],[-11.439,12.959],[-11.433,12.997],[-11.423,13.009],[-11.436,13.017],[-11.459,13.055],[-11.45,13.075],[-11.533,13.112],[-11.539,13.163],[-11.569,13.199],[-11.572,13.217],[-11.554,13.257],[-11.606,13.32],[-11.613,13.361],[-11.644,13.356],[-11.65,13.384],[-11.704,13.38],[-11.727,13.406],[-11.741,13.408],[-11.757,13.4],[-11.767,13.383],[-11.775,13.343],[-11.828,13.307],[-11.845,13.324],[-11.855,13.351],[-11.898,13.371],[-11.9,13.382],[-11.883,13.454],[-11.89,13.465],[-11.918,13.475],[-11.939,13.503],[-11.997,13.544],[-12.015,13.564],[-12.037,13.598],[-12.07,13.673],[-12.098,13.704],[-12.012,13.751],[-11.979,13.785],[-11.963,13.829],[-11.957,13.89],[-11.963,13.909],[-12.015,13.959],[-12.025,13.985],[-12.025,14.027],[-11.995,14.144],[-11.998,14.166],[-12.034,14.25],[-12.058,14.274],[-12.11,14.298],[-12.128,14.324],[-12.113,14.33],[-12.109,14.342],[-12.116,14.355],[-12.222,14.391],[-12.22,14.435],[-12.236,14.519],[-12.23,14.537],[-12.204,14.554],[-12.198,14.588],[-12.166,14.625],[-12.165,14.642],[-12.19,14.656],[-12.192,14.667],[-12.18,14.693],[-12.223,14.703],[-12.237,14.728],[-12.257,14.746],[-12.264,14.775],[-12.247,14.767]]]],"GNB":[[[[-13.736,12.608],[-13.725,12.558],[-13.675,12.496],[-13.661,12.459],[-13.684,12.392],[-13.683,12.339],[-13.695,12.293],[-13.712,12.268],[-13.741,12.257],[-13.809,12.256],[-13.796,12.279],[-13.813,12.284],[-13.84,12.277],[-13.883,12.246],[-13.944,12.219],[-13.966,12.194],[-13.973,12.152],[-13.918,12.138],[-13.87,12.101],[-13.842,12.091],[-13.808,12.044],[-13.78,12.029],[-13.735,12.019],[-13.724,12.003],[-13.735,11.902],[-13.72,11.766],[-13.73,11.71],[-13.77,11.689],[-13.816,11.687],[-13.822,11.694],[-13.82,11.708],[-13.843,11.717],[-13.846,11.737],[-13.87,11.744],[-13.871,11.704],[-13.886,11.676],[-13.904,11.665],[-13.924,11.665],[-13.945,11.676],[-14.015,11.637],[-14.059,11.644],[-14.119,11.638],[-14.135,11.642],[-14.162,11.661],[-14.29,11.669],[-14.308,11.655],[-14.326,11.621],[-14.339,11.611],[-14.521,11.513],[-14.562,11.506],[-14.645,11.515],[-14.687,11.511],[-14.711,11.498],[-14.783,11.4],[-15.287,11.4],[-15.268,11.433],[-15.309,11.4],[-15.362,11.4],[-15.357,11.43],[-15.331,11.463],[-15.336,11.468],[-15.353,11.463],[-15.366,11.451],[-15.378,11.416],[-15.394,11.4],[-15.491,11.4],[-15.453,11.46],[-15.471,11.461],[-15.477,11.468],[-15.473,11.494],[-15.457,11.489],[-15.439,11.502],[-15.432,11.481],[-15.421,11.505],[-15.412,11.508],[-15.405,11.494],[-15.398,11.494],[-15.412,11.529],[-15.405,11.536],[-15.354,11.525],[-15.349,11.528],[-15.357,11.543],[-15.35,11.563],[-15.343,11.563],[-15.335,11.552],[-15.31,11.554],[-15.309,11.536],[-15.29,11.543],[-15.29,11.556],[-15.316,11.584],[-15.336,11.577],[-15.336,11.584],[-15.295,11.605],[-15.275,11.578],[-15.258,11.57],[-15.24,11.577],[-15.252,11.579],[-15.254,11.605],[-15.277,11.612],[-15.281,11.625],[-15.24,11.612],[-15.226,11.619],[-15.234,11.591],[-15.209,11.574],[-15.202,11.575],[-15.192,11.591],[-15.213,11.622],[-15.194,11.643],[-15.186,11.646],[-15.148,11.629],[-15.144,11.598],[-15.125,11.605],[-15.131,11.619],[-15.116,11.618],[-15.09,11.591],[-15.144,11.584],[-15.125,11.57],[-15.129,11.554],[-15.123,11.545],[-15.115,11.546],[-15.105,11.573],[-15.045,11.575],[-15.024,11.583],[-15.007,11.612],[-15.039,11.599],[-15.08,11.598],[-15.081,11.613],[-15.062,11.619],[-15.066,11.633],[-15.062,11.646],[-15.073,11.649],[-15.082,11.643],[-15.09,11.619],[-15.104,11.634],[-15.125,11.638],[-15.125,11.646],[-15.103,11.666],[-15.11,11.687],[-15.117,11.687],[-15.131,11.653],[-15.143,11.65],[-15.172,11.66],[-15.161,11.675],[-15.135,11.69],[-15.131,11.708],[-15.149,11.695],[-15.172,11.701],[-15.176,11.684],[-15.192,11.68],[-15.189,11.665],[-15.197,11.662],[-15.207,11.667],[-15.213,11.68],[-15.228,11.659],[-15.236,11.664],[-15.231,11.681],[-15.207,11.694],[-15.219,11.708],[-15.213,11.728],[-15.186,11.722],[-15.199,11.745],[-15.226,11.755],[-15.241,11.712],[-15.24,11.694],[-15.259,11.677],[-15.274,11.677],[-15.261,11.701],[-15.273,11.706],[-15.275,11.718],[-15.268,11.748],[-15.284,11.734],[-15.289,11.711],[-15.323,11.708],[-15.309,11.701],[-15.309,11.694],[-15.281,11.67],[-15.285,11.656],[-15.294,11.653],[-15.309,11.666],[-15.323,11.632],[-15.336,11.684],[-15.35,11.68],[-15.343,11.666],[-15.357,11.666],[-15.353,11.64],[-15.336,11.619],[-15.343,11.605],[-15.372,11.63],[-15.392,11.66],[-15.395,11.642],[-15.371,11.591],[-15.392,11.598],[-15.396,11.585],[-15.405,11.584],[-15.412,11.595],[-15.412,11.612],[-15.419,11.612],[-15.426,11.57],[-15.439,11.577],[-15.439,11.557],[-15.449,11.564],[-15.469,11.606],[-15.453,11.622],[-15.428,11.67],[-15.426,11.694],[-15.476,11.654],[-15.494,11.666],[-15.488,11.708],[-15.509,11.705],[-15.538,11.676],[-15.551,11.716],[-15.551,11.733],[-15.525,11.782],[-15.48,11.83],[-15.463,11.84],[-15.452,11.873],[-15.432,11.885],[-15.419,11.871],[-15.41,11.884],[-15.398,11.885],[-15.313,11.871],[-15.189,11.871],[-15.162,11.893],[-15.136,11.899],[-15.09,11.94],[-15.067,11.94],[-15.063,11.925],[-15.087,11.87],[-15.086,11.845],[-15.069,11.804],[-15.003,11.784],[-14.982,11.754],[-14.959,11.749],[-14.936,11.753],[-14.925,11.769],[-14.951,11.761],[-14.997,11.81],[-15.026,11.814],[-15.047,11.823],[-15.07,11.861],[-15.042,11.896],[-15.023,11.951],[-14.994,11.957],[-15.007,11.978],[-15.029,11.982],[-15.047,11.976],[-15.08,11.988],[-15.094,11.984],[-15.141,11.961],[-15.192,11.913],[-15.215,11.91],[-15.313,11.933],[-15.359,11.963],[-15.381,11.968],[-15.426,11.961],[-15.456,11.94],[-15.497,11.937],[-15.529,11.906],[-15.555,11.893],[-15.593,11.845],[-15.68,11.804],[-15.744,11.792],[-15.811,11.753],[-15.83,11.748],[-15.81,11.83],[-15.791,11.863],[-15.734,11.858],[-15.734,11.865],[-15.753,11.866],[-15.776,11.886],[-15.796,11.885],[-15.812,11.874],[-15.818,11.838],[-15.85,11.775],[-15.86,11.769],[-15.88,11.774],[-15.884,11.787],[-15.878,11.824],[-15.894,11.816],[-15.906,11.762],[-15.959,11.735],[-15.97,11.768],[-15.946,11.824],[-15.932,11.88],[-15.922,11.897],[-15.898,11.91],[-15.891,11.934],[-15.865,11.972],[-15.822,11.998],[-15.782,12.003],[-15.768,11.979],[-15.721,11.967],[-15.706,12.008],[-15.714,12.016],[-15.743,11.995],[-15.761,11.995],[-15.782,12.016],[-15.837,12.014],[-15.864,12.029],[-15.871,12.013],[-15.89,12.003],[-15.933,11.995],[-15.926,11.981],[-15.933,11.975],[-15.946,11.984],[-15.945,11.953],[-15.968,11.927],[-15.982,11.921],[-16.056,11.92],[-16.093,11.906],[-16.125,11.885],[-16.159,11.913],[-16.183,11.908],[-16.243,11.925],[-16.255,11.937],[-16.275,11.981],[-16.323,11.995],[-16.338,12.017],[-16.355,12.087],[-16.357,12.101],[-16.32,12.172],[-16.283,12.194],[-16.277,12.173],[-16.27,12.171],[-16.261,12.18],[-16.269,12.214],[-16.257,12.23],[-16.148,12.291],[-16.104,12.338],[-16.063,12.351],[-16.09,12.352],[-16.108,12.345],[-16.155,12.3],[-16.228,12.283],[-16.272,12.252],[-16.306,12.214],[-16.325,12.222],[-16.316,12.283],[-16.35,12.197],[-16.365,12.192],[-16.385,12.201],[-16.391,12.194],[-16.406,12.208],[-16.408,12.194],[-16.416,12.188],[-16.443,12.187],[-16.458,12.172],[-16.466,12.184],[-16.501,12.197],[-16.508,12.218],[-16.539,12.256],[-16.567,12.277],[-16.576,12.283],[-16.604,12.277],[-16.624,12.304],[-16.673,12.338],[-16.688,12.338],[-16.721,12.324],[-16.728,12.333],[-16.706,12.348],[-16.669,12.357],[-16.533,12.348],[-16.463,12.361],[-16.412,12.362],[-16.222,12.455],[-16.182,12.443],[-16.166,12.451],[-16.091,12.464],[-16.055,12.461],[-15.973,12.437],[-15.894,12.443],[-15.712,12.432],[-15.677,12.439],[-15.425,12.555],[-15.369,12.595],[-15.308,12.623],[-15.25,12.664],[-15.195,12.679],[-13.728,12.673],[-13.736,12.608]]],[[[-15.663,11.492],[-15.652,11.488],[-15.658,11.474],[-15.653,11.466],[-15.68,11.44],[-15.686,11.454],[-15.705,11.443],[-15.721,11.443],[-15.735,11.45],[-15.74,11.46],[-15.735,11.469],[-15.714,11.468],[-15.693,11.502],[-15.686,11.494],[-15.665,11.509],[-15.66,11.505],[-15.663,11.492]]],[[[-16.214,11.44],[-16.302,11.454],[-16.255,11.483],[-16.237,11.508],[-16.183,11.518],[-16.166,11.516],[-16.161,11.506],[-16.164,11.496],[-16.214,11.44]]],[[[-16.282,11.493],[-16.314,11.481],[-16.337,11.502],[-16.391,11.502],[-16.419,11.488],[-16.412,11.502],[-16.418,11.514],[-16.412,11.522],[-16.426,11.529],[-16.389,11.548],[-16.31,11.553],[-16.283,11.584],[-16.268,11.577],[-16.248,11.584],[-16.241,11.57],[-16.245,11.552],[-16.282,11.493]]],[[[-16.022,11.512],[-16.001,11.569],[-15.975,11.583],[-15.952,11.607],[-15.94,11.605],[-15.931,11.579],[-15.922,11.578],[-15.918,11.591],[-15.906,11.578],[-15.909,11.552],[-15.898,11.543],[-15.91,11.523],[-15.913,11.498],[-15.898,11.454],[-15.966,11.427],[-15.974,11.447],[-15.981,11.447],[-15.987,11.433],[-16.001,11.447],[-16.016,11.428],[-16.035,11.42],[-16.079,11.457],[-16.084,11.468],[-16.022,11.512]]],[[[-15.474,11.619],[-15.488,11.595],[-15.472,11.566],[-15.491,11.563],[-15.521,11.57],[-15.542,11.55],[-15.57,11.55],[-15.57,11.543],[-15.548,11.535],[-15.552,11.52],[-15.566,11.51],[-15.576,11.522],[-15.622,11.522],[-15.63,11.529],[-15.633,11.538],[-15.607,11.55],[-15.586,11.578],[-15.562,11.577],[-15.56,11.592],[-15.529,11.625],[-15.512,11.62],[-15.473,11.632],[-15.474,11.619]]],[[[-16.148,11.877],[-16.08,11.873],[-16.035,11.893],[-16.037,11.899],[-15.995,11.899],[-15.986,11.887],[-15.989,11.862],[-16.015,11.824],[-16.029,11.769],[-16.041,11.755],[-16.052,11.762],[-16.084,11.755],[-16.096,11.763],[-16.121,11.8],[-16.155,11.808],[-16.166,11.871],[-16.148,11.877]]]],"MRT":[[[[-10.6,15.425],[-10.709,15.434],[-10.725,15.433],[-10.736,15.423],[-10.739,15.362],[-10.747,15.341],[-10.771,15.325],[-10.782,15.308],[-10.811,15.303],[-10.827,15.288],[-10.844,15.235],[-10.891,15.177],[-10.915,15.103],[-10.948,15.152],[-10.985,15.187],[-11.013,15.24],[-11.045,15.27],[-11.298,15.451],[-11.41,15.602],[-11.435,15.623],[-11.458,15.633],[-11.527,15.643],[-11.513,15.625],[-11.515,15.61],[-11.524,15.6],[-11.579,15.578],[-11.615,15.541],[-11.644,15.526],[-11.727,15.541],[-11.729,15.496],[-11.755,15.469],[-11.768,15.437],[-11.783,15.366],[-11.817,15.296],[-11.827,15.233],[-11.846,15.179],[-11.845,15.125],[-11.861,15.049],[-11.851,15.042],[-11.833,15.049],[-11.814,15.029],[-11.813,14.996],[-11.831,14.926],[-11.82,14.9],[-11.837,14.893],[-11.877,14.842],[-11.936,14.824],[-11.974,14.78],[-11.991,14.768],[-12.011,14.763],[-12.05,14.767],[-12.07,14.734],[-12.139,14.785],[-12.203,14.766],[-12.247,14.767],[-12.32,14.817],[-12.382,14.846],[-12.448,14.907],[-12.472,14.998],[-12.485,15.017],[-12.519,15.028],[-12.602,15.079],[-12.623,15.102],[-12.632,15.105],[-12.69,15.092],[-12.717,15.103],[-12.755,15.139],[-12.78,15.146],[-12.785,15.166],[-12.779,15.2],[-12.787,15.214],[-12.806,15.223],[-12.857,15.222],[-12.888,15.251],[-12.894,15.263],[-12.884,15.271],[-12.833,15.27],[-12.828,15.284],[-12.847,15.313],[-12.919,15.365],[-12.931,15.397],[-12.933,15.438],[-12.951,15.465],[-12.955,15.493],[-12.964,15.509],[-12.98,15.516],[-12.996,15.514],[-13.039,15.487],[-13.084,15.493],[-13.095,15.515],[-13.092,15.575],[-13.11,15.596],[-13.178,15.628],[-13.238,15.623],[-13.249,15.657],[-13.229,15.697],[-13.228,15.711],[-13.251,15.743],[-13.286,15.77],[-13.28,15.823],[-13.294,15.855],[-13.316,15.882],[-13.316,15.93],[-13.37,15.985],[-13.384,16.019],[-13.389,16.052],[-13.422,16.077],[-13.44,16.102],[-13.455,16.102],[-13.492,16.088],[-13.502,16.095],[-13.5,16.111],[-13.483,16.141],[-13.485,16.155],[-13.569,16.138],[-13.607,16.123],[-13.635,16.122],[-13.669,16.106],[-13.72,16.136],[-13.723,16.146],[-13.711,16.189],[-13.729,16.19],[-13.747,16.182],[-13.795,16.131],[-13.819,16.119],[-13.836,16.121],[-13.859,16.142],[-13.884,16.194],[-13.908,16.21],[-13.964,16.23],[-13.968,16.246],[-13.963,16.272],[-13.979,16.3],[-13.971,16.324],[-13.978,16.338],[-14.016,16.356],[-14.04,16.382],[-14.073,16.401],[-14.214,16.535],[-14.268,16.548],[-14.278,16.581],[-14.335,16.586],[-14.331,16.624],[-14.336,16.632],[-14.407,16.661],[-14.426,16.66],[-14.472,16.636],[-14.493,16.632],[-14.555,16.641],[-14.608,16.629],[-14.656,16.646],[-14.711,16.636],[-14.882,16.648],[-14.912,16.641],[-14.931,16.643],[-14.95,16.672],[-14.974,16.691],[-15.007,16.676],[-15.019,16.646],[-15.037,16.633],[-15.052,16.634],[-15.079,16.665],[-15.108,16.665],[-15.118,16.643],[-15.096,16.614],[-15.093,16.6],[-15.125,16.583],[-15.232,16.566],[-15.328,16.572],[-15.386,16.552],[-15.415,16.548],[-15.427,16.553],[-15.449,16.58],[-15.464,16.581],[-15.491,16.565],[-15.525,16.522],[-15.551,16.515],[-15.61,16.521],[-15.644,16.495],[-15.672,16.481],[-15.688,16.481],[-15.735,16.498],[-15.768,16.499],[-15.813,16.514],[-15.862,16.51],[-15.897,16.518],[-15.958,16.498],[-15.986,16.506],[-16.007,16.496],[-16.054,16.495],[-16.074,16.505],[-16.11,16.547],[-16.132,16.555],[-16.156,16.548],[-16.195,16.523],[-16.251,16.528],[-16.274,16.523],[-16.314,16.493],[-16.335,16.456],[-16.385,16.235],[-16.407,16.215],[-16.444,16.206],[-16.459,16.194],[-16.466,16.174],[-16.47,16.106],[-16.508,16.01],[-16.51,15.902],[-16.52,15.846],[-16.542,15.809],[-16.527,15.977],[-16.539,16.267],[-16.529,16.333],[-16.494,16.402],[-16.486,16.452],[-16.467,16.493],[-16.467,16.6],[-16.459,16.644],[-16.379,16.841],[-16.358,16.875],[-16.343,16.942],[-16.278,17.041],[-16.193,17.219],[-16.187,17.244],[-16.173,17.262],[-16.127,17.4],[-10.6,17.4],[-10.6,15.425]]]],"GMB":[[[[-16.754,13.065],[-16.784,13.083],[-16.787,13.09],[-16.776,13.113],[-16.797,13.177],[-16.795,13.236],[-16.803,13.262],[-16.82,13.281],[-16.824,13.295],[-16.817,13.311],[-16.83,13.339],[-16.81,13.38],[-16.763,13.405],[-16.704,13.475],[-16.679,13.496],[-16.671,13.479],[-16.652,13.468],[-16.635,13.477],[-16.617,13.478],[-16.591,13.461],[-16.585,13.443],[-16.613,13.439],[-16.624,13.427],[-16.609,13.414],[-16.596,13.385],[-16.588,13.355],[-16.591,13.339],[-16.546,13.29],[-16.495,13.288],[-16.45,13.264],[-16.42,13.256],[-16.413,13.222],[-16.419,13.208],[-16.4,13.224],[-16.378,13.215],[-16.406,13.246],[-16.415,13.267],[-16.44,13.27],[-16.44,13.276],[-16.411,13.276],[-16.385,13.284],[-16.375,13.264],[-16.35,13.282],[-16.258,13.325],[-16.23,13.316],[-16.228,13.256],[-16.21,13.251],[-16.151,13.276],[-16.151,13.284],[-16.173,13.279],[-16.199,13.259],[-16.211,13.313],[-16.222,13.32],[-16.227,13.334],[-16.224,13.36],[-16.216,13.373],[-16.176,13.405],[-16.161,13.429],[-16.135,13.42],[-15.995,13.413],[-15.941,13.425],[-15.891,13.427],[-15.871,13.441],[-15.764,13.437],[-15.706,13.455],[-15.669,13.476],[-15.645,13.458],[-15.621,13.455],[-15.59,13.468],[-15.565,13.503],[-15.543,13.51],[-15.5,13.488],[-15.47,13.496],[-15.451,13.491],[-15.419,13.455],[-15.384,13.444],[-15.341,13.441],[-15.305,13.455],[-15.295,13.496],[-15.302,13.496],[-15.304,13.481],[-15.322,13.458],[-15.352,13.457],[-15.416,13.473],[-15.465,13.51],[-15.491,13.503],[-15.556,13.528],[-15.576,13.523],[-15.611,13.489],[-15.628,13.489],[-15.672,13.503],[-15.77,13.465],[-16.09,13.44],[-16.142,13.455],[-16.178,13.441],[-16.273,13.363],[-16.281,13.36],[-16.289,13.366],[-16.285,13.395],[-16.289,13.407],[-16.306,13.392],[-16.302,13.372],[-16.316,13.366],[-16.316,13.359],[-16.302,13.353],[-16.322,13.343],[-16.391,13.337],[-16.522,13.359],[-16.504,13.393],[-16.504,13.417],[-16.52,13.451],[-16.557,13.486],[-16.548,13.548],[-16.549,13.564],[-16.561,13.587],[-15.518,13.583],[-15.502,13.588],[-15.497,13.648],[-15.468,13.708],[-15.436,13.741],[-15.394,13.772],[-15.348,13.789],[-15.304,13.782],[-15.267,13.742],[-15.246,13.747],[-15.171,13.794],[-15.098,13.82],[-15.076,13.819],[-15.017,13.797],[-14.916,13.792],[-14.879,13.781],[-14.831,13.736],[-14.803,13.652],[-14.741,13.615],[-14.667,13.653],[-14.627,13.664],[-14.586,13.66],[-14.543,13.641],[-14.51,13.609],[-14.487,13.547],[-14.47,13.522],[-14.366,13.455],[-14.33,13.456],[-14.271,13.477],[-14.197,13.519],[-14.139,13.532],[-14.093,13.555],[-14.034,13.56],[-13.988,13.579],[-13.942,13.567],[-13.901,13.542],[-13.885,13.506],[-13.843,13.477],[-13.819,13.429],[-13.823,13.378],[-13.851,13.336],[-13.901,13.315],[-13.976,13.308],[-14.016,13.298],[-14.057,13.297],[-14.099,13.282],[-14.117,13.282],[-14.202,13.23],[-14.231,13.229],[-14.285,13.239],[-14.369,13.236],[-14.431,13.261],[-14.47,13.298],[-14.53,13.307],[-14.592,13.353],[-14.662,13.341],[-14.698,13.345],[-14.732,13.359],[-14.79,13.417],[-14.829,13.427],[-14.863,13.447],[-14.95,13.463],[-15.016,13.496],[-15.138,13.59],[-15.16,13.581],[-15.204,13.537],[-15.218,13.515],[-15.228,13.425],[-15.248,13.399],[-15.277,13.38],[-15.342,13.363],[-15.379,13.363],[-15.52,13.387],[-15.596,13.356],[-15.693,13.36],[-15.737,13.346],[-15.807,13.34],[-15.819,13.334],[-15.827,13.162],[-15.833,13.157],[-16.674,13.164],[-16.708,13.157],[-16.726,13.123],[-16.749,13.099],[-16.754,13.065]]]]}}""")


# ############################################################################################
# PART 2. TABLES OF THE ARTICLE (Tables 1-3, Excel); names prefixed tab_
# ############################################################################################
# =====================================================================================
# 1. Reading and preparing the data
# =====================================================================================
def tab_locate(src):
    """Return a folder containing the three .DTA files (extracting them from the .zip if needed)."""
    if not src.lower().endswith('.zip'):
        return src
    tmp = tempfile.mkdtemp(prefix='dhs_')
    with zipfile.ZipFile(src) as z:
        for name in z.namelist():
            u = name.upper()
            if u.endswith(('SNHR8RFL.DTA', 'SNPR8RFL.DTA', 'SNIR8RFL.DTA', 'SNHR8RDT.ZIP', 'SNPR8RDT.ZIP', 'SNIR8RDT.ZIP')):
                z.extract(name, tmp)
    for inner in glob.glob(os.path.join(tmp, '**', '*.[zZ][iI][pP]'), recursive=True):
        with zipfile.ZipFile(inner) as z:
            for name in z.namelist():
                if name.upper().endswith('.DTA'):
                    z.extract(name, tmp)
    return tmp


def tab_read_data(src):
    import pyreadstat
    folder = tab_locate(src)
    hr_cols = (['hv001', 'hv002', 'hv005', 'hv006', 'hv009', 'hv013', 'hv021', 'hv022', 'hv024', 'hv025',
                'hv270', 'hv271'] + [f'{v}_{k}' for v in ['hml10', 'hml21', 'hml4'] for k in K])
    pr_cols = ['hml12', 'hml18', 'hv001', 'hv002', 'hv005', 'hv021', 'hv022', 'hv101', 'hv103', 'hv104', 'hv105',
               'hv106', 'hv218', 'hvidx']
    ir_cols = (['b19_01', 'v005', 'v006', 'v012', 'v021', 'v022', 'v024', 'v025', 'v106', 'v190', 'v191', 'v201']
               + [f'{v}_{k}' for v in ['m80', 'm13', 'm14', 'm2a', 'm2b', 'm2c', 'm2n', 'm49a', 'ml1']
                  for k in range(1, 7)])
    hr, _ = pyreadstat.read_dta(find(folder, 'SNHR8RFL.DTA'), usecols=hr_cols)
    pr, _ = pyreadstat.read_dta(find(folder, 'SNPR8RFL.DTA'), usecols=pr_cols)
    ir, _ = pyreadstat.read_dta(find(folder, 'SNIR8RFL.DTA'), usecols=ir_cols)
    return hr, pr, ir


def tab_build(hr, pr, ir):
    itn = np.column_stack([(hr[f'hml10_{k}'] == 1).values for k in K])
    age = np.column_stack([pd.to_numeric(hr[f'hml4_{k}'], errors='coerce').astype(float).values for k in K])
    h = pd.DataFrame({'hhid': hhid(hr), 'psu': hr.hv021, 'strata': hr.hv022, 'w': hr.hv005 / 1e6,
                      'region': hr.hv024.map(REG), 'res': hr.hv025.map({1: 'Urban', 2: 'Rural'}),
                      'wq': hr.hv270, 'wscore': hr.hv271, 'month': hr.hv006, 'defacto': hr.hv013,
                      'n_itn': itn.sum(1)})
    h['zone'] = h.region.map(ZONE)
    h['own'] = (h.n_itn > 0).astype(float)
    h['enough'] = np.where(h.defacto > 0, (2 * h.n_itn >= h.defacto).astype(float), np.nan)
    h['access_hh'] = np.where(h.defacto > 0, np.minimum(1, 2 * h.n_itn / h.defacto.replace(0, np.nan)), np.nan)
    h['itn_age_mean'] = np.nanmean(np.where(itn & (age < 96), age, np.nan), axis=1)

    p = pd.DataFrame({'hhid': hhid(pr), 'line': pr.hvidx, 'psu': pr.hv021, 'strata': pr.hv022, 'w': pr.hv005 / 1e6,
                      'defacto_flag': pr.hv103, 'sex': pr.hv104.map({1: 'Male', 2: 'Female'}), 'age': pr.hv105,
                      'educ': pr.hv106, 'preg': pr.hml18, 'hml12': pr.hml12, 'head_line': pr.hv218})
    p = p.merge(h.drop(columns=['psu', 'strata', 'w']), on='hhid', how='left', validate='m:1')
    p['use'] = p.hml12.isin([1, 2]).astype(float)
    head = p[p.line == p.head_line][['hhid', 'educ']].rename(columns={'educ': 'head_educ'})
    p = p.merge(head, on='hhid', how='left')
    p = p[p.defacto_flag == 1].copy()
    p['agegrp'] = pd.cut(p.age, [-1, 4, 14, 49, 97], labels=['0-4', '5-14', '15-49', '50+']).astype(object)
    p.loc[p.age >= 98, 'agegrp'] = np.nan
    p['hhsize'] = pd.cut(p.defacto, [0, 4, 7, 10, 200], labels=['1-4', '5-7', '8-10', '11+']).astype(object)
    p['head_ed'] = p.head_educ.map({0: 'None', 1: 'Primary', 2: 'Secondary+', 3: 'Secondary+'})
    p['season'] = p.month.map(SEASON)
    p['netage'] = pd.cut(p.itn_age_mean, [-1, 11.99, 23.99, 35.99, 1000],
                         labels=['<12', '12-23', '24-35', '36+']).astype(object)
    p['wqs'] = 'Q' + p.wq.astype(int).astype(str)

    blk = np.full(len(ir), np.nan)                 # maternity section of the most recent live birth
    for k in range(6, 0, -1):
        blk = np.where(ir[f'm80_{k}'] == 1, k, blk)
    for v in ['m13', 'm14', 'm2a', 'm2b', 'm2c', 'm2n', 'm49a', 'ml1']:
        vals = np.full(len(ir), np.nan)
        for k in range(1, 7):
            vals = np.where(blk == k, ir[f'{v}_{k}'], vals)
        ir[v] = vals
    ir['w'] = ir.v005 / 1e6; ir['psu'] = ir.v021; ir['strata'] = ir.v022
    ir['region'] = ir.v024.map(REG); ir['zone'] = ir.region.map(ZONE)
    ir['res'] = ir.v025.map({1: 'Urban', 2: 'Rural'}); ir['wscore'] = ir.v191
    ir['wqs'] = 'Q' + ir.v190.astype(int).astype(str)
    ir['educ'] = ir.v106.map({0: 'None', 1: 'Primary', 2: 'Secondary', 3: 'Higher'})
    ir['agegrp'] = pd.cut(ir.v012, [14, 19, 34, 49], labels=['15-19', '20-34', '35-49']).astype(object)
    ir['parity'] = pd.cut(ir.v201, [0, 1, 3, 20], labels=['1', '2-3', '4+']).astype(object)
    ir['season'] = ir.v006.map(SEASON)
    ir['eligible'] = (ir.b19_01 < 24) & ir.m14.notna()
    ir['anc1'] = (ir.m2n == 0).astype(float)
    ir['anc4'] = ((ir.m14 >= 4) & (ir.m14 < 98)).astype(float)
    ir['skilled'] = ((ir.m2a == 1) | (ir.m2b == 1) | (ir.m2c == 1)).astype(float)
    ir['ancvis'] = np.select([ir.m14 == 0, (ir.m14 >= 1) & (ir.m14 <= 3), (ir.m14 >= 4) & (ir.m14 < 98),
                              ir.m14 == 98], ['0', '1-3', '4+', 'DK'], None)
    ir.loc[ir.m2n == 1, 'ancvis'] = '0'
    ir['sp1'] = (ir.m49a == 1).astype(float)
    ir['doses'] = np.where(ir.m49a == 1, ir.ml1, 0)
    ir['sp2'] = ((ir.doses >= 2) & (ir.doses < 98)).astype(float)
    ir['sp3'] = ((ir.doses >= 3) & (ir.doses < 98)).astype(float)
    ir['ftri'] = np.select([ir.m13 <= 3, (ir.m13 >= 4) & (ir.m13 <= 9)], ['1st trimester', 'Later'], 'DK')
    ir.loc[ir.anc1 == 0, 'ftri'] = None
    v = ir.m14
    ir['elig3'] = np.where(ir.anc1 == 1, (((ir.m13 >= 4) & (ir.m13 <= 9) & (v >= 3) & (v < 98))
                                          | ((ir.m13 <= 3) & (v >= 4) & (v < 98))).astype(float), np.nan)

    nets = []
    for k in K:
        t = pd.DataFrame({'psu': hr.hv021, 'strata': hr.hv022, 'w': hr.hv005 / 1e6, 'itn': hr[f'hml10_{k}'],
                          'used': hr[f'hml21_{k}']})
        nets.append(t[t.itn == 1])
    nets = pd.concat(nets, ignore_index=True)
    return h, p, ir, nets


# =====================================================================================
# 2. Design-based estimators
# =====================================================================================


def tab_mean(d, y, mask=None):
    """Weighted proportion, Taylor-linearised SE, logit-transformed 95% CI."""
    y = np.asarray(y, float)
    m = np.ones_like(y, bool) if mask is None else np.asarray(mask, bool)
    m = m & ~np.isnan(y)
    w = np.where(m, d.w, 0.0); yy = np.where(m, y, 0.0); W = w.sum(); est = (w * yy).sum() / W
    se = np.sqrt(d.var_total(w * (yy - est) / W)[0, 0]); t = stats.t.ppf(0.975, d.df)
    if 0 < est < 1 and se > 0:
        L, sl = np.log(est / (1 - est)), se / (est * (1 - est))
        lo, hi = 1 / (1 + np.exp(-(L - t * sl))), 1 / (1 + np.exp(-(L + t * sl)))
    else:
        lo = hi = np.nan
    return dict(est=est, lo=lo, hi=hi, n=int(m.sum()))


def tab_logit(d, X, y, mask):
    """Survey-weighted logistic regression with design-based (linearised) variance."""
    m = mask & ~np.isnan(y) & ~np.isnan(X).any(1)
    w = np.where(m, d.w, 0.0); Xm = np.where(m[:, None], X, 0.0); ym = np.where(m, y, 0.0)
    b = np.zeros(X.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-(Xm @ b)))
        H = (Xm * (w * p * (1 - p))[:, None]).T @ Xm
        step = np.linalg.solve(H, Xm.T @ (w * (ym - p))); b += step
        if np.abs(step).max() < 1e-10:
            break
    p = 1 / (1 + np.exp(-(Xm @ b)))
    Ainv = np.linalg.inv(H)
    V = Ainv @ d.var_total(Xm * (w * (ym - p))[:, None]) @ Ainv
    se = np.sqrt(np.diag(V)); t = stats.t.ppf(0.975, d.df)
    pv = 2 * stats.t.sf(np.abs(b / se), d.df)
    return dict(b=b, V=V, OR=np.exp(b), lo=np.exp(b - t * se), hi=np.exp(b + t * se), p=pv, n=int(m.sum()))


def tab_wald_p(res, idx, df):
    b = res['b'][idx]; V = res['V'][np.ix_(idx, idx)]; k = len(idx)
    F = b @ np.linalg.solve(V, b) / k
    return float(stats.f.sf(F, k, df - k + 1))


# ---------------- concentration indices (Erreygers), rescaling bootstrap


def erreygers(y, score, w):
    R = frank(score, w); W = w.sum(); mu = (w * y).sum() / W
    C = 2 * ((w * (y - mu) * (R - (w * R).sum() / W)).sum() / W) / mu
    return 4 * mu * C


def tab_ci_E(df, y, mask, mult):
    keep = mask & df[y].notna().values
    d = df[keep]; yy = d[y].values.astype(float); w = d.w.values; sc = d.wscore.values.astype(float)
    E = erreygers(yy, sc, w)
    M, key = mult
    pos = key.get_indexer(pd.MultiIndex.from_arrays([d.strata, d.psu]))
    bs = []
    for b in range(M.shape[0]):
        wb = w * M[b, pos]; k = wb > 0
        bs.append(erreygers(yy[k], sc[k], wb[k]))
    se = np.std(bs, ddof=1)
    return (E, E - 1.96 * se, E + 1.96 * se)


def tab_concentration(p, ir, B):
    """Same random stream as the article: persons first, then women."""
    rng = np.random.default_rng(SEED); C = {}
    pp = p.copy(); pp['w'] = pp.w.astype(float)
    mult = multipliers(rng, pp.strata.values, pp.psu.values, B)
    allp = np.ones(len(pp), bool); pw = ((pp.preg == 1) & (pp.sex == 'Female')).values
    C['own'] = tab_ci_E(pp, 'own', allp, mult); C['access'] = tab_ci_E(pp, 'access_hh', allp, mult)
    C['use'] = tab_ci_E(pp, 'use', allp, mult); C['use_enough'] = tab_ci_E(pp, 'use', (pp.enough == 1).values, mult)
    C['preg_own'] = tab_ci_E(pp, 'own', pw, mult); C['preg_access'] = tab_ci_E(pp, 'access_hh', pw, mult)
    C['preg_use'] = tab_ci_E(pp, 'use', pw, mult)
    w = ir[ir.eligible].copy(); w['w'] = w.w.astype(float)
    mult = multipliers(rng, w.strata.values, w.psu.values, B)
    al = np.ones(len(w), bool)
    for v in ['anc1', 'anc4', 'sp1', 'sp3']:
        C[v] = tab_ci_E(w, v, al, mult)
    C['sp3_anc4'] = tab_ci_E(w, 'sp3', (w.anc4 == 1).values, mult)
    return C


# =====================================================================================
# 3. Formatting (as in the article)
# =====================================================================================


def tab_fp(r):                          # 46.4 (41.9 to 51.0)
    return f"{100 * r['est']:.1f} ({100 * r['lo']:.1f} to {100 * r['hi']:.1f})"


# =====================================================================================
# 4. Tables
# =====================================================================================
def table1(p, ir):
    dp = Design(p.w, p.strata, p.psu)
    rows = [('header', 'A. ITN analyses: de facto household population (n=' + fn(len(p)) + ')', 'n', '% (weighted)',
             'In household with ≥1 ITN, % (95% CI)', 'Access, % (95% CI)', 'Slept under ITN, % (95% CI)')]
    A = [('Ecogeographical zone', 'zone', [('West', 'West'), ('Centre', 'Centre'), ('North', 'North'), ('South', 'South'), ('South-East', 'South-East')]),
         ('Place of residence', 'res', [('Urban', 'Urban'), ('Rural', 'Rural')]),
         ('Wealth quintile', 'wqs', [('Q1', 'Q1 (poorest)'), ('Q2', 'Q2'), ('Q3', 'Q3'), ('Q4', 'Q4'), ('Q5', 'Q5 (wealthiest)')]),
         ('Sex', 'sex', [('Female', 'Female'), ('Male', 'Male')]),
         ('Age group, y', 'agegrp', [('0-4', '0–4'), ('5-14', '5–14'), ('15-49', '15–49'), ('50+', '≥50')]),
         ('Household size (de facto members)', 'hhsize', [('1-4', '1–4'), ('5-7', '5–7'), ('8-10', '8–10'), ('11+', '≥11')]),
         ('Education of household head', 'head_ed', [('None', 'None'), ('Primary', 'Primary'), ('Secondary+', 'Secondary or higher')]),
         ('Period of interview', 'season', [('Jan-Mar', 'January–March'), ('Apr-May', 'April–May'), ('Jun-Aug', 'June–August')])]
    for title, v, levels in A:
        rows.append(('section', title))
        for code, lab in levels:
            m = (p[v] == code).values
            rows.append(('row', lab, fn(int(m.sum())), f'{100 * np.average(m, weights=p.w):.1f}',
                         tab_fp(tab_mean(dp, p.own, m)), tab_fp(tab_mean(dp, p.access_hh, m)), tab_fp(tab_mean(dp, p.use, m))))
        mm = p[v].isna().values
        if mm.sum():
            rows.append(('row', 'Missing', fn(int(mm.sum())), f'{100 * np.average(mm, weights=p.w):.1f}', '', '', ''))
    rows.append(('blank',))
    e = ir[ir.eligible]; di = Design(ir.w, ir.strata, ir.psu)
    rows.append(('header', f'B. IPTp-SP analyses: women with a live birth in the past 24 months (n={fn(len(e))})', 'n',
                 '% (weighted)', '≥4 ANC visits, % (95% CI)', '≥1 SP dose, % (95% CI)', '≥3 SP doses, % (95% CI)'))
    B = [('Ecogeographical zone', 'zone', [('West', 'West'), ('Centre', 'Centre'), ('North', 'North'), ('South', 'South'), ('South-East', 'South-East')]),
         ('Place of residence', 'res', [('Urban', 'Urban'), ('Rural', 'Rural')]),
         ('Wealth quintile', 'wqs', [('Q1', 'Q1 (poorest)'), ('Q2', 'Q2'), ('Q3', 'Q3'), ('Q4', 'Q4'), ('Q5', 'Q5 (wealthiest)')]),
         ("Woman's education", 'educ', [('None', 'None'), ('Primary', 'Primary'), ('Secondary', 'Secondary'), ('Higher', 'Higher')]),
         ('Age group, y', 'agegrp', [('15-19', '15–19'), ('20-34', '20–34'), ('35-49', '35–49')]),
         ('Parity', 'parity', [('1', '1'), ('2-3', '2–3'), ('4+', '≥4')]),
         ('Number of ANC visits', 'ancvis', [('0', 'None'), ('1-3', '1–3'), ('4+', '≥4'), ('DK', 'Unknown')]),
         ('Timing of first ANC visit', 'ftri', [(None, 'No ANC visit'), ('1st trimester', 'First trimester'),
                                               ('Later', 'Second or third trimester'), ('DK', 'Unknown')]),
         ('Period of interview', 'season', [('Jan-Mar', 'January–March'), ('Apr-May', 'April–May'), ('Jun-Aug', 'June–August')])]
    for title, v, levels in B:
        rows.append(('section', title))
        for code, lab in levels:
            if code is None:
                M = (ir.eligible & (ir.anc1 == 0)).values; m = (e.anc1 == 0).values
            else:
                M = (ir.eligible & (ir[v] == code)).values; m = (e[v] == code).values
            no_anc4 = v == 'ancvis' or code is None
            rows.append(('row', lab, fn(int(m.sum())), f'{100 * np.average(m, weights=e.w):.1f}',
                         '–' if no_anc4 else tab_fp(tab_mean(di, ir.anc4, M)), tab_fp(tab_mean(di, ir.sp1, M)), tab_fp(tab_mean(di, ir.sp3, M))))
    note = ('n, unweighted number; percentages are weighted. Row percentages in the last three columns are the coverage '
            'of each indicator within the category, with logit-transformed 95% CIs accounting for the survey design. '
            'ANC, antenatal care; IPTp-SP, intermittent preventive treatment in pregnancy with sulfadoxine-pyrimethamine; '
            'ITN, insecticide-treated net; Q, quintile; SP, sulfadoxine-pyrimethamine. Women whose number of ANC visits '
            "was unknown attended ANC; ‘None’ for ANC visits and ‘No ANC visit’ for timing represent the "
            f"{int((e.anc1 == 0).sum())} women with no ANC visit; 25 of them reported taking SP, 24 of whom reported "
            'receiving it at an ANC visit, an inconsistency between the ANC and SP questions.')
    return ('Table 1. Characteristics of the two analytical populations, 2023 Senegal Continuous Demographic and Health Survey',
            rows, note, [58, 9, 12, 34, 22, 26])


def table2(h, p, ir, nets, C):
    dh = Design(h.w, h.strata, h.psu); dp = Design(p.w, p.strata, p.psu)
    di = Design(ir.w, ir.strata, ir.psu); dn = Design(nets.w, nets.strata, nets.psu)
    E = ir.eligible.values; pw = ((p.preg == 1) & (p.sex == 'Female')).values
    r = lambda lab, den, res, e='–': ('row', lab, den, fn(res['n']), tab_fp(res), fe(e) if e != '–' else '–')
    rows = [('header', 'Indicator', 'Denominator', 'n', '% (95% CI)', 'E (95% CI)'),
            ('section', 'A. ITN cascade, de facto household population'),
            r('Households owning ≥1 ITN (household level)', 'All households', tab_mean(dh, h.own)),
            r('Households with ≥1 ITN per 2 people (household level)', 'All households', tab_mean(dh, h.enough)),
            r('Lives in a household with ≥1 ITN', 'De facto population', tab_mean(dp, p.own), C['own']),
            r('Access to an ITN', 'De facto population', tab_mean(dp, p.access_hh), C['access']),
            r('Slept under an ITN last night', 'De facto population', tab_mean(dp, p.use), C['use']),
            r('Slept under an ITN last night', 'Households with ≥1 ITN per 2 people', tab_mean(dp, p.use, p.enough == 1),
              C['use_enough']),
            r('ITNs used last night (net level)', 'ITNs in net roster', tab_mean(dn, (nets.used == 1).astype(float))),
            ('section', 'B. ITN cascade, currently pregnant women aged 15–49 y'),
            r('Lives in a household with ≥1 ITN', 'Pregnant women', tab_mean(dp, p.own, pw), C['preg_own']),
            r('Access to an ITN', 'Pregnant women', tab_mean(dp, p.access_hh, pw), C['preg_access']),
            r('Slept under an ITN last night', 'Pregnant women', tab_mean(dp, p.use, pw), C['preg_use']),
            r('Slept under an ITN last night', 'In households owning ≥1 ITN', tab_mean(dp, p.use, pw & (p.own == 1).values)),
            ('section', 'C. IPTp-SP cascade, women with a live birth in the past 24 months'),
            r('≥1 ANC visit (any provider)', 'All women', tab_mean(di, ir.anc1, E), C['anc1']),
            r('≥1 ANC visit (skilled provider)', 'All women', tab_mean(di, ir.skilled, E)),
            r('≥4 ANC visits', 'All women', tab_mean(di, ir.anc4, E), C['anc4']),
            r('≥1 SP dose', 'All women', tab_mean(di, ir.sp1, E), C['sp1']),
            r('≥2 SP doses', 'All women', tab_mean(di, ir.sp2, E)),
            r('≥3 SP doses (IPTp-SP3)', 'All women', tab_mean(di, ir.sp3, E), C['sp3']),
            r('≥1 SP dose', '≥1 ANC visit', tab_mean(di, ir.sp1, E & (ir.anc1 == 1))),
            r('≥4 ANC visits', '≥1 ANC visit', tab_mean(di, ir.anc4, E & (ir.anc1 == 1))),
            r('≥3 SP doses (IPTp-SP3)', '≥4 ANC visits', tab_mean(di, ir.sp3, E & (ir.anc4 == 1)), C['sp3_anc4']),
            r('Missed opportunity: <3 SP doses', '≥4 ANC visits', tab_mean(di, 1 - ir.sp3, E & (ir.anc4 == 1))),
            r('Missed opportunity: <3 SP doses', '≥3 SP-eligible ANC contacts', tab_mean(di, 1 - ir.sp3, E & (ir.elig3 == 1)))]
    note = ('n, unweighted denominator; percentages are weighted and 95% CIs account for the survey design and are '
            'computed on the logit scale. E, Erreygers-corrected concentration index ranked on the continuous wealth index '
            'score (negative, concentration among poorer people; positive, among wealthier people), with 95% CIs from a '
            'design-based bootstrap (1000 replicates); – , not computed. Household-level net sufficiency excludes '
            f'{int((h.defacto == 0).sum())} households with no de facto members (n = {fn(int((h.defacto > 0).sum()))}). '
            'Access: proportion of the population that could sleep under an ITN if each ITN were used by up to two people '
            '(capped at 100% per household). SP-eligible contacts: first ANC visit at ≥4 months of pregnancy and ≥3 visits, '
            'or first visit in the first trimester and ≥4 visits. ANC, antenatal care; IPTp-SP, intermittent preventive '
            'treatment in pregnancy with sulfadoxine-pyrimethamine; ITN, insecticide-treated net; SP, sulfadoxine-pyrimethamine.')
    return ('Table 2. Coverage at each step of the ITN and IPTp-SP cascades, with Erreygers-corrected concentration indices',
            rows, note, [52, 36, 9, 22, 28])


def tab_model(df, mask, y, spec):
    """spec: list of (title, var, [(code, label), ...]); first level = reference."""
    d = df
    ok = mask.copy()
    for _, v, _ in spec:
        ok = ok & d[v].notna().values
    des = Design(d.w, d.strata, d.psu)
    cols, groups = [np.ones(len(d))], {}
    for _, v, lev in spec:
        groups[v] = []
        for code, _ in lev[1:]:
            cols.append((d[v] == code).astype(float).values); groups[v].append(len(cols) - 1)
    X = np.column_stack(cols); yy = d[y].values.astype(float)
    adj = tab_logit(des, X, yy, ok)
    rows = []
    for title, v, lev in spec:
        crude = tab_logit(des, np.column_stack([np.ones(len(d))] + [(d[v] == c).astype(float).values for c, _ in lev[1:]]), yy, ok)
        rows.append(('section', f'{title} (p={fpv(tab_wald_p(adj, groups[v], des.df)).replace("p=<", "<")})'.replace('(p=<', '(p<')))
        for i, (code, lab) in enumerate(lev):
            m = ok & (d[v] == code).values
            n, pct = int(m.sum()), f'{100 * np.average(yy[m], weights=d.w[m]):.1f}'
            if i == 0:
                rows.append(('row', lab, fn(n), pct, '1.00 (ref)', '1.00 (ref)', ''))
            else:
                j = groups[v][i - 1]
                rows.append(('row', lab, fn(n), pct,
                             f'{fo(crude["OR"][i])} ({fo(crude["lo"][i], True)} to {fo(crude["hi"][i], True)})',
                             f'{fo(adj["OR"][j])} ({fo(adj["lo"][j], True)} to {fo(adj["hi"][j], True)})', fpv(adj['p'][j])))
    return adj['n'], int(mask.sum()) - adj['n'], rows


def table3(p, ir):
    zone = ('Ecogeographical zone', 'zone', [('West', 'West'), ('Centre', 'Centre'), ('North', 'North'), ('South', 'South'), ('South-East', 'South-East')])
    res = ('Place of residence', 'res', [('Rural', 'Rural'), ('Urban', 'Urban')])
    wq = ('Wealth quintile', 'wqs', [('Q1', 'Q1 (poorest)'), ('Q2', 'Q2'), ('Q3', 'Q3'), ('Q4', 'Q4'), ('Q5', 'Q5 (wealthiest)')])
    season = ('Period of interview', 'season', [('Jan-Mar', 'January–March'), ('Apr-May', 'April–May'), ('Jun-Aug', 'June–August')])
    specA = [zone, res, wq, ('Sex', 'sex', [('Female', 'Female'), ('Male', 'Male')]),
             ('Age group, y', 'agegrp', [('0-4', '0–4'), ('5-14', '5–14'), ('15-49', '15–49'), ('50+', '≥50')]),
             ('Household size (de facto members)', 'hhsize', [('1-4', '1–4'), ('5-7', '5–7'), ('8-10', '8–10'), ('11+', '≥11')]),
             ('Education of household head', 'head_ed', [('None', 'None'), ('Primary', 'Primary'), ('Secondary+', 'Secondary or higher')]),
             ("Mean age of household's ITNs, months", 'netage', [('<12', '<12'), ('12-23', '12–23'), ('24-35', '24–35'), ('36+', '≥36')]),
             season]
    specB = [zone, res, wq, ("Woman's education", 'educ', [('None', 'None'), ('Primary', 'Primary'), ('Secondary', 'Secondary'), ('Higher', 'Higher')]),
             ('Age group, y', 'agegrp', [('20-34', '20–34'), ('15-19', '15–19'), ('35-49', '35–49')]),
             ('Parity', 'parity', [('1', '1'), ('2-3', '2–3'), ('4+', '≥4')]),
             ('Number of ANC visits', 'ancvis', [('1-3', '1–3'), ('4+', '≥4'), ('DK', 'Unknown')]),
             ('Timing of first ANC visit', 'ftri', [('Later', 'Second or third trimester'), ('1st trimester', 'First trimester'), ('DK', 'Unknown')]),
             season]
    nA, exA, rA = tab_model(p, (p.enough == 1).values, 'use', specA)
    w = ir[ir.eligible & (ir.anc1 == 1)].copy()
    nB, exB, rB = tab_model(w, np.ones(len(w), bool), 'sp3', specB)
    rows = [('header', 'Variable (category)', 'n', '% with outcome', 'cOR (95% CI)', 'aOR (95% CI)', 'p'),
            ('model', f'Model A. ITN use among de facto members of households with ≥1 ITN per 2 people (n={fn(nA)})')] + rA + \
           [('model', f'Model B. IPTp-SP3 among women with ≥1 ANC visit (n={fn(nB)})')] + rB
    note = ('n, unweighted; % with outcome, weighted percentage using an ITN (Model A) or receiving ≥3 SP doses (Model B) in '
            'the category among model observations. cOR, crude odds ratio; aOR, odds ratio adjusted for all other variables '
            'in the same model; ref, reference category. Confidence limits that would round to 1.00 are shown to three '
            'decimals. p-values in variable rows are design-adjusted Wald tests for the variable; p-values in category rows '
            'refer to the aOR. Estimates account for survey weights, stratification and clustering. '
            f'Model A excludes {exA} people with missing covariates; Model B has no missing covariates' +
            ('.' if exB == 0 else f' ({exB} excluded).') +
            ' Because of these exclusions, % with outcome can differ slightly from the estimates for all eligible people '
            f'given in the Results (e.g., n = {fn(int((p.enough == 1).sum()))} for people with sufficient nets). '
            'The supplementary model including all members of ITN-owning households (Model A2) is shown in Supplementary '
            'Table S11, and a sensitivity analysis using month of interview instead of period in Supplementary Table S5. '
            'ANC, antenatal care; IPTp-SP3, ≥3 doses of sulfadoxine-pyrimethamine; ITN, insecticide-treated net; Q, quintile.')
    return ('Table 3. Factors associated with ITN use and with IPTp-SP3: survey-weighted logistic regression',
            rows, note, [62, 9, 14, 22, 22, 9])


# =====================================================================================
# 5. Excel output
# =====================================================================================
def tab_write_excel(tables, path):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    wb = Workbook(); wb.remove(wb.active)
    thin = Side(style='thin', color='000000'); hair = Side(style='hair', color='BFBFBF')
    for sheet, (title, rows, note, widths) in tables.items():
        ws = wb.create_sheet(sheet)
        ncol = len(widths)
        for i, wdt in enumerate(widths):
            ws.column_dimensions[chr(65 + i)].width = wdt
        ws.cell(1, 1, title).font = Font(name='Arial', bold=True, size=11)
        ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=ncol)
        r = 3
        for row in rows:
            kind = row[0]
            if kind == 'blank':
                r += 1; continue
            vals = list(row[1:]) + [''] * (ncol - len(row) + 1)
            for c, v in enumerate(vals, start=1):
                cell = ws.cell(r, c, v)
                cell.font = Font(name='Arial', size=9, bold=kind in ('header', 'section', 'model'))
                cell.alignment = Alignment(horizontal='left' if c == 1 else 'center', vertical='center',
                                           wrap_text=kind == 'header', indent=1 if (kind == 'row' and c == 1) else 0)
                if kind == 'header':
                    cell.fill = PatternFill('solid', fgColor='EAF0F8')
                    cell.border = Border(top=thin, bottom=thin)
                elif kind == 'model':
                    cell.fill = PatternFill('solid', fgColor='F2F2F2')
                else:
                    cell.border = Border(bottom=hair)
            if kind == 'header':
                ws.row_dimensions[r].height = 30
            r += 1
        for c in range(1, ncol + 1):
            ws.cell(r - 1, c).border = Border(bottom=thin)
        cell = ws.cell(r + 1, 1, note)
        cell.font = Font(name='Arial', size=8, italic=True)
        cell.alignment = Alignment(wrap_text=True, vertical='top')
        ws.merge_cells(start_row=r + 1, start_column=1, end_row=r + 1, end_column=ncol)
        ws.row_dimensions[r + 1].height = 15 * (len(note) // 150 + 2)
        ws.freeze_panes = 'A4'
        ws.sheet_view.showGridLines = False
        ws.page_setup.orientation = 'landscape'; ws.page_setup.fitToWidth = 1; ws.page_setup.fitToHeight = 0
        ws.sheet_properties.pageSetUpPr.fitToPage = True
    wb.save(path)


# ############################################################################################
# PART 3. SUPPLEMENTARY TABLES (S1-S13, Excel); names prefixed sup_
# ############################################################################################
ZONES = ['West', 'Centre', 'North', 'South', 'South-East']
REG_ORDER = ['Dakar', 'Thiès', 'Diourbel', 'Fatick', 'Kaolack', 'Kaffrine', 'Saint-Louis', 'Louga', 'Matam',
             'Ziguinchor', 'Kolda', 'Sédhiou', 'Tambacounda', 'Kédougou']
MONTH = {1: 'Feb', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul-Aug', 8: 'Jul-Aug'}

# Titles, footnotes and the two text tables (S1, S3), as in the Supplementary data file
CONST = json.loads(r"""{"caps": {"S1": "Supplementary Table S1. STROBE checklist for cross-sectional studies", "S2": "Supplementary Table S2. Ecogeographical zones, administrative regions and timing of fieldwork (percentage of interviewed households by month of interview)", "S3": "Supplementary Table S3. DHS recode variables used in the analysis", "S4": "Supplementary Table S4. Number of SP doses by number of ANC visits, and reconciliation of the ANC1+ estimate", "S5": "Supplementary Table S5. Sensitivity analysis: month of interview instead of period of interview in the regression models", "S6": "Supplementary Table S6. ITN indicators by region (de facto household population)", "S7": "Supplementary Table S7. IPTp-SP indicators by region (women with a live birth in the past 24 months)", "S8": "Supplementary Table S8. ITN and IPTp-SP indicators by ecogeographical zone", "S9": "Supplementary Table S9. Concentration indices: standard (C), Wagstaff-normalised (W) and Erreygers-corrected (E), overall, by stratum and after standardisation", "S10": "Supplementary Table S10. Reported reasons for not using ITNs the previous night, and source of ITNs", "S11": "Supplementary Table S11. Model A2: ITN use among all de facto members of households owning ≥1 ITN (original specification with additional covariates)", "S12": "Supplementary Table S12. ITN use by age group and by wealth quintile, according to household net sufficiency", "S13": "Supplementary Table S13. Variance inflation factors for the regression models"}, "static": {"S1": [["Item", "Section", "Recommendation", "Where reported"], ["1", "Title and abstract", "Indicate design; informative summary", "Title; Abstract"], ["2", "Background/rationale", "Scientific background and rationale", "Introduction, paragraphs 1–3"], ["3", "Objectives", "Specific objectives", "Introduction, paragraph 3"], ["4", "Study design", "Key elements of design", "Methods: Study design and setting"], ["5", "Setting", "Settings, locations, dates", "Methods: Study design and setting; Data source and sampling; Table S2"], ["6", "Participants", "Eligibility, sources and selection", "Methods: Analytical populations; Figure 2"], ["7", "Variables", "Outcomes, exposures, confounders", "Methods: Indicators; Covariates; Table S3"], ["8", "Data sources/measurement", "Sources and measurement", "Methods: Data source and sampling; Indicators; Table S3"], ["9", "Bias", "Efforts to address bias", "Methods: Statistical analysis; Discussion, limitations"], ["10", "Study size", "How size was arrived at", "Methods: Data source and sampling; Figure 2"], ["11", "Quantitative variables", "Handling and groupings", "Methods: Indicators; Covariates; Supplementary Methods"], ["12", "Statistical methods", "Methods, subgroups, missing data, sampling strategy, sensitivity analyses", "Methods: Statistical analysis; Supplementary Methods; Figure 2; Tables S4, S5, S9 and S11"], ["13", "Participants", "Numbers at each stage; flow diagram", "Results, first section; Figure 2"], ["14", "Descriptive data", "Characteristics; missing data", "Table 1; Figure 2"], ["15", "Outcome data", "Numbers of outcome events", "Tables 1–3"], ["16", "Main results", "Unadjusted and adjusted estimates with CIs", "Table 3; Tables S5 and S11"], ["17", "Other analyses", "Subgroups, interactions, sensitivity", "Results; Tables S4–S12; Figure S1"], ["18", "Key results", "Summary with reference to objectives", "Discussion, paragraph 1"], ["19", "Limitations", "Limitations, bias", "Discussion, final paragraph"], ["20", "Interpretation", "Cautious overall interpretation", "Discussion; Conclusions"], ["21", "Generalisability", "External validity", "Discussion"], ["22", "Funding", "Funding source", "Funding statement"]], "S3": [["File", "Variable(s)", "Content / use"], ["HR (household)", "hv001, hv002", "Cluster and household number (linkage to PR)"], ["", "hv013", "Number of de facto household members"], ["", "hv006", "Month of interview"], ["", "hv024, hv025", "Region; place of residence"], ["", "hv270, hv271", "Wealth index quintile; wealth index factor score"], ["", "hml10_1–hml10_7", "Net is an insecticide-treated net (ITN), net roster"], ["", "hml21_1–hml21_7", "Someone slept under the net last night"], ["", "hml22_1–hml22_7", "Source of the net (campaign, ANC, immunization visit)"], ["", "hml24_1–hml24_7", "Reason the net was not used last night"], ["", "hml4_1–hml4_7", "Months since the net was obtained"], ["PR (household member)", "hv103", "Slept in the household last night (de facto member)"], ["", "hv104, hv105", "Sex; age"], ["", "hv101, hv218, hv106", "Relationship to head; line number of head; education (used for education of the household head)"], ["", "hml12", "Type of net slept under last night (ITN use: categories 1 and 2)"], ["", "hml18", "Pregnancy status from the individual interview"], ["IR (women)", "b19_01", "Age in months of the most recent child (live birth in the past 24 months)"], ["", "m2a_1–m2n_1", "ANC provider for the most recent birth (any ANC; skilled provider)"], ["", "m80_1–m80_6", "Pregnancy outcome of each maternity section (1, live birth; 3, stillbirth; 5, miscarriage/abortion); the section with m80 = 1 (most recent live birth) was used for ANC and SP variables"], ["", "m14_1", "Number of ANC visits"], ["", "m13_1", "Months of pregnancy at first ANC visit"], ["", "m49a_1", "Took SP/Fansidar during the pregnancy"], ["", "ml1_1, ml2_1", "Number of SP doses; source of SP"], ["", "s433a_1", "Received a free mosquito net during the pregnancy (country-specific; asked of women who took SP)"], ["", "v012, v106, v201", "Age; education; parity"], ["", "v024, v025, v190, v191, v006", "Region; residence; wealth quintile and score; month of interview"], ["Survey design", "hv005/v005", "Household / women’s sample weight (divided by 1 000 000)"], ["", "hv021/v021", "Primary sampling unit (cluster)"], ["", "hv022/v022", "Sampling stratum (region × residence; 28 strata)"]]}, "notes": ["Checklist based on von Elm et al., Lancet 2007.", "Unweighted percentages of interviewed households by month of interview (2023); weighted, 15.2% of households were interviewed in July or August (13.4% unweighted). Zones were defined by the authors following the north–south rainfall and transmission gradient; they are not DHS sampling strata. The rainy season in Senegal generally runs from about June–July to October.", "Files: SNHR8RFL, SNPR8RFL and SNIR8RFL (2023 Senegal Continuous DHS, Standard Recode DHS-8).", "Weighted row percentages. SP doses are counted regardless of source (DHS-8 definition) and are self-reported; the doses reported by women with no ANC visit (n=85) may reflect SP obtained outside ANC or recall error. Women who did not know whether they had taken SP (n=88) are counted as having received 0 doses. Women whose most recent pregnancy ended in a miscarriage or stillbirth after their most recent live birth (n=22) are included with the maternity section of that live birth (Supplementary Methods). Reconciliation: ANC1+ = 98.2% when any ANC visit is counted (women who did not know the number of visits attended ANC); 88.9% when ‘don’t know’ is coded as no visit, the coding that produced the value reported in the previous version (89.1%, computed from the children’s file). ANC from a skilled provider: 97.3% (published: 97.3%); ANC4+: 68.4% (published: 68.4%). Refined missed opportunity: 76.8% of ANC attenders had ≥3 estimated SP-eligible contacts; 59.9% of them received <3 doses (57.8% with the stricter definition, n=1938).", "Same models and covariates as in Table 3, with month of interview replacing period of interview; the single household interviewed in January is grouped with February and the 262 households interviewed in July with August because of small numbers. % with outcome: weighted percentage using an ITN (Model A) or receiving ≥3 SP doses (Model B); n unweighted. The other adjusted odds ratios changed by less than 10% (maximum 8.8% in Model A, for mean net age ≥36 months, and 5.2% in Model B); in Model B the design-adjusted Wald test for wealth was p=0.042 (p=0.028 in Table 3). cOR, crude odds ratio; aOR, adjusted odds ratio.", "Percentages with logit-transformed 95% CIs. Use-to-access ratio = population use ÷ population access (aggregate ratio).", "Percentages with logit-transformed 95% CIs. CI not estimable: no woman without ANC in the Diourbel sample. Regional estimates are based on small unweighted denominators (152 to 406 women; 113 to 279 women with ≥4 visits) and should be interpreted with caution.", "Weighted percentages with logit-transformed 95% CIs; n unweighted. Zones as defined in Supplementary Table S2.", "Indices ranked on the continuous DHS wealth index score (survey-weighted fractional rank). W = C/(1−μ); E = 4μC. 95% CIs and p-values from a rescaling bootstrap of primary sampling units within strata (1000 replicates). Need-standardised: indirect standardisation for zone, residence, age group and sex (ITN) or zone, residence, age group and parity (IPTp-SP3); 297 people with unknown age excluded from standardised ITN indices. Stratified indices rank individuals within the stratum. Last seven rows: sensitivity analyses (separate bootstrap run with the same settings). An unknown number of ANC visits was reported by 6–7% of ANC attenders in the two poorest quintiles and 12–14% in the two wealthiest. In the previous version, the standard index was computed on wealth quintiles with a tie correction (ITN use −0.12; IPTp-SP3 +0.05).", "Net-level weighted percentages among ITNs not used the previous night (one reason recorded per net), with logit-transformed 95% CIs. The denominator includes the 222 unused ITNs (4.3%) with no recorded reason. Source of ITNs: weighted percentages of all ITNs in the net roster, with logit-transformed 95% CIs.", "n=50 260 of 53 691 members of ITN-owning households (exclusions: unknown age 161, unknown education of household head 1086, unknown net age 2184). Net sufficiency is partly definitional for use (members of households with fewer than one ITN per two people cannot all sleep under an ITN); its odds ratio is shown for comparison with the previous version and is not interpreted as a determinant.", "Upper panel: ITN use by age group (years) in the whole de facto population, in ITN-owning households without enough ITNs and in households with at least one ITN per two people. Lower panel: ITN use and access in the whole de facto population, and ITN use in households with at least one ITN per two people, by wealth quintile. n, unweighted.", "VIFs computed on the unweighted design matrix of each model. All VIFs were below 5 in Models A and B; in Model A2, the VIF for households of 11 or more members was 5.8, reflecting the correlation between household size and net sufficiency."]}""")


# =====================================================================================
# 1. Data
# =====================================================================================
def sup_locate(src):
    if not src.lower().endswith('.zip'):
        return src
    tmp = tempfile.mkdtemp(prefix='dhs_')
    with zipfile.ZipFile(src) as z:
        for name in z.namelist():
            if name.upper().endswith(('SNHR8RFL.DTA', 'SNPR8RFL.DTA', 'SNIR8RFL.DTA',
                                      'SNHR8RDT.ZIP', 'SNPR8RDT.ZIP', 'SNIR8RDT.ZIP')):
                z.extract(name, tmp)
    for inner in glob.glob(os.path.join(tmp, '**', '*.[zZ][iI][pP]'), recursive=True):
        with zipfile.ZipFile(inner) as z:
            for name in z.namelist():
                if name.upper().endswith('.DTA'):
                    z.extract(name, tmp)
    return tmp


def sup_read_data(src):
    import pyreadstat
    folder = sup_locate(src)
    hr_cols = (['hv001', 'hv002', 'hv005', 'hv006', 'hv013', 'hv021', 'hv022', 'hv024', 'hv025', 'hv270', 'hv271']
               + [f'{v}_{k}' for v in ['hml10', 'hml21', 'hml22', 'hml24', 'hml4'] for k in K])
    pr_cols = ['hml12', 'hml18', 'hv001', 'hv002', 'hv005', 'hv021', 'hv022', 'hv103', 'hv104', 'hv105', 'hv106',
               'hv218', 'hvidx']
    ir_cols = (['b19_01', 'v005', 'v006', 'v012', 'v021', 'v022', 'v024', 'v025', 'v106', 'v190', 'v191', 'v201']
               + [f'{v}_{k}' for v in ['m80', 'm13', 'm14', 'm2n', 'm49a', 'ml1'] for k in range(1, 7)])
    hr, _ = pyreadstat.read_dta(find(folder, 'SNHR8RFL.DTA'), usecols=hr_cols)
    pr, _ = pyreadstat.read_dta(find(folder, 'SNPR8RFL.DTA'), usecols=pr_cols)
    ir, _ = pyreadstat.read_dta(find(folder, 'SNIR8RFL.DTA'), usecols=ir_cols)
    return hr, pr, ir


def sup_build(hr, pr, ir):
    itn = np.column_stack([(hr[f'hml10_{k}'] == 1).values for k in K])
    age = np.column_stack([pd.to_numeric(hr[f'hml4_{k}'], errors='coerce').astype(float).values for k in K])
    h = pd.DataFrame({'hhid': hhid(hr), 'psu': hr.hv021, 'strata': hr.hv022, 'w': hr.hv005 / 1e6,
                      'region': hr.hv024.map(REG), 'res': hr.hv025.map({1: 'Urban', 2: 'Rural'}), 'wq': hr.hv270,
                      'wscore': hr.hv271, 'month': hr.hv006, 'defacto': hr.hv013, 'n_itn': itn.sum(1)})
    h['zone'] = h.region.map(ZONE)
    h['own'] = (h.n_itn > 0).astype(float)
    h['enough'] = np.where(h.defacto > 0, (2 * h.n_itn >= h.defacto).astype(float), np.nan)
    h['access_hh'] = np.where(h.defacto > 0, np.minimum(1, 2 * h.n_itn / h.defacto.replace(0, np.nan)), np.nan)
    h['itn_age_mean'] = np.nanmean(np.where(itn & (age < 96), age, np.nan), axis=1)

    p = pd.DataFrame({'hhid': hhid(pr), 'line': pr.hvidx, 'psu': pr.hv021, 'strata': pr.hv022, 'w': pr.hv005 / 1e6,
                      'defacto_flag': pr.hv103, 'sex': pr.hv104.map({1: 'Male', 2: 'Female'}), 'age': pr.hv105,
                      'educ': pr.hv106, 'preg': pr.hml18, 'hml12': pr.hml12, 'head_line': pr.hv218})
    p = p.merge(h.drop(columns=['psu', 'strata', 'w']), on='hhid', how='left', validate='m:1')
    p['use'] = p.hml12.isin([1, 2]).astype(float)
    head = p[p.line == p.head_line][['hhid', 'educ']].rename(columns={'educ': 'head_educ'})
    p = p.merge(head, on='hhid', how='left')
    p = p[p.defacto_flag == 1].copy()
    p['agegrp'] = pd.cut(p.age, [-1, 4, 14, 49, 97], labels=['0-4', '5-14', '15-49', '50+']).astype(object)
    p.loc[p.age >= 98, 'agegrp'] = np.nan
    p['hhsize'] = pd.cut(p.defacto, [0, 4, 7, 10, 200], labels=['1-4', '5-7', '8-10', '11+']).astype(object)
    p['head_ed'] = p.head_educ.map({0: 'None', 1: 'Primary', 2: 'Secondary+', 3: 'Secondary+'})
    p['season'] = p.month.map(SEASON); p['month_i'] = p.month.map(MONTH)
    p['netage'] = pd.cut(p.itn_age_mean, [-1, 11.99, 23.99, 35.99, 1000],
                         labels=['<12', '12-23', '24-35', '36+']).astype(object)
    p['wqs'] = 'Q' + p.wq.astype(int).astype(str)

    blk = np.full(len(ir), np.nan)
    for k in range(6, 0, -1):
        blk = np.where(ir[f'm80_{k}'] == 1, k, blk)
    for v in ['m13', 'm14', 'm2n', 'm49a', 'ml1']:
        vals = np.full(len(ir), np.nan)
        for k in range(1, 7):
            vals = np.where(blk == k, ir[f'{v}_{k}'], vals)
        ir[v] = vals
    ir['w'] = ir.v005 / 1e6; ir['psu'] = ir.v021; ir['strata'] = ir.v022
    ir['region'] = ir.v024.map(REG); ir['zone'] = ir.region.map(ZONE)
    ir['res'] = ir.v025.map({1: 'Urban', 2: 'Rural'}); ir['wscore'] = ir.v191
    ir['wqs'] = 'Q' + ir.v190.astype(int).astype(str)
    ir['educ'] = ir.v106.map({0: 'None', 1: 'Primary', 2: 'Secondary', 3: 'Higher'})
    ir['agegrp'] = pd.cut(ir.v012, [14, 19, 34, 49], labels=['15-19', '20-34', '35-49']).astype(object)
    ir['parity'] = pd.cut(ir.v201, [0, 1, 3, 20], labels=['1', '2-3', '4+']).astype(object)
    ir['season'] = ir.v006.map(SEASON); ir['month_i'] = ir.v006.map(MONTH)
    ir['eligible'] = (ir.b19_01 < 24) & ir.m14.notna()
    ir['anc1'] = (ir.m2n == 0).astype(float)
    ir['anc4'] = ((ir.m14 >= 4) & (ir.m14 < 98)).astype(float)
    ir['ancvis'] = np.select([ir.m14 == 0, (ir.m14 >= 1) & (ir.m14 <= 3), (ir.m14 >= 4) & (ir.m14 < 98),
                              ir.m14 == 98], ['0', '1-3', '4+', 'DK'], None)
    ir.loc[ir.m2n == 1, 'ancvis'] = '0'
    ir['sp1'] = (ir.m49a == 1).astype(float)
    ir['doses'] = np.where(ir.m49a == 1, ir.ml1, 0)
    ir['sp3'] = ((ir.doses >= 3) & (ir.doses < 98)).astype(float)
    ir['ftri'] = np.select([ir.m13 <= 3, (ir.m13 >= 4) & (ir.m13 <= 9)], ['1st trimester', 'Later'], 'DK')
    ir.loc[ir.anc1 == 0, 'ftri'] = None

    nets = []
    for k in K:
        t = pd.DataFrame({'psu': hr.hv021, 'strata': hr.hv022, 'w': hr.hv005 / 1e6, 'itn': hr[f'hml10_{k}'],
                          'used': hr[f'hml21_{k}'], 'reason': hr[f'hml24_{k}'], 'src': hr[f'hml22_{k}'],
                          'region': hr.hv024.map(REG)})
        nets.append(t[t.itn == 1])
    nets = pd.concat(nets, ignore_index=True)
    nets['zone'] = nets.region.map(ZONE)
    return h, p, ir, nets


# =====================================================================================
# 2. Estimators
# =====================================================================================


def sup_mean(d, y, mask=None):
    y = np.asarray(y, float)
    m = np.ones_like(y, bool) if mask is None else np.asarray(mask, bool)
    m = m & ~np.isnan(y)
    w = np.where(m, d.w, 0.0); yy = np.where(m, y, 0.0); W = w.sum(); est = (w * yy).sum() / W
    se = np.sqrt(d.var_total(w * (yy - est) / W)[0, 0]); t = stats.t.ppf(0.975, d.df)
    if 0 < est < 1 and se > 0:
        L, sl = np.log(est / (1 - est)), se / (est * (1 - est))
        lo, hi = 1 / (1 + np.exp(-(L - t * sl))), 1 / (1 + np.exp(-(L + t * sl)))
    else:
        lo = hi = np.nan
    return dict(est=est, lo=lo, hi=hi, n=int(m.sum()))


def sup_logit(d, X, y, mask):
    m = mask & ~np.isnan(y) & ~np.isnan(X).any(1)
    w = np.where(m, d.w, 0.0); Xm = np.where(m[:, None], X, 0.0); ym = np.where(m, y, 0.0)
    b = np.zeros(X.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-(Xm @ b)))
        H = (Xm * (w * p * (1 - p))[:, None]).T @ Xm
        step = np.linalg.solve(H, Xm.T @ (w * (ym - p))); b += step
        if np.abs(step).max() < 1e-10:
            break
    p = 1 / (1 + np.exp(-(Xm @ b)))
    Ainv = np.linalg.inv(H)
    V = Ainv @ d.var_total(Xm * (w * (ym - p))[:, None]) @ Ainv
    se = np.sqrt(np.diag(V)); t = stats.t.ppf(0.975, d.df)
    return dict(b=b, V=V, OR=np.exp(b), lo=np.exp(b - t * se), hi=np.exp(b + t * se),
                p=2 * stats.t.sf(np.abs(b / se), d.df), n=int(m.sum()))


def sup_wald_p(res, idx, df):
    b = res['b'][idx]; V = res['V'][np.ix_(idx, idx)]; k = len(idx)
    return float(stats.f.sf(b @ np.linalg.solve(V, b) / k, k, df - k + 1))


def vif(X):
    """Variance inflation factors (unweighted), as statsmodels: regress each column on all others."""
    out = []
    for j in range(1, X.shape[1]):
        others = np.delete(X, j, axis=1)
        beta, *_ = np.linalg.lstsq(others, X[:, j], rcond=None)
        resid = X[:, j] - others @ beta
        r2 = 1 - (resid @ resid) / ((X[:, j] - X[:, j].mean()) @ (X[:, j] - X[:, j].mean()))
        out.append(1 / (1 - r2))
    return out


# ---------------- concentration indices


def sup_cindex(y, score, w):
    R = frank(score, w); W = w.sum(); mu = (w * y).sum() / W
    C = 2 * ((w * (y - mu) * (R - (w * R).sum() / W)).sum() / W) / mu
    return mu, C, C / (1 - mu), 4 * mu * C           # mean, C, W (Wagstaff), E (Erreygers)


def run(df, y, mask, mult, ystd=None):
    keep = mask & df[y].notna().values
    d = df[keep]
    yy = d[y].values.astype(float) if ystd is None else ystd[keep]
    w, sc = d.w.values, d.wscore.values.astype(float)
    mu, C, W, E = sup_cindex(yy, sc, w)
    M, key = mult
    pos = key.get_indexer(pd.MultiIndex.from_arrays([d.strata, d.psu]))
    bs = []
    for b in range(M.shape[0]):
        wb = w * M[b, pos]; k = wb > 0
        bs.append(sup_cindex(yy[k], sc[k], wb[k])[1:])
    se = np.std(np.array(bs), axis=0, ddof=1)
    res = dict(n=int(len(d)), mu=mu)
    for name, v, s in zip('CWE', (C, W, E), se):
        res[name] = (v, v - 1.96 * s, v + 1.96 * s)
    res['p'] = 2 * stats.norm.sf(abs(E / se[2]))
    return res


def sup_std_pred(df, mask, y, needs):
    d = df[mask]
    X = pd.get_dummies(d[needs], drop_first=True, dtype=float); X.insert(0, 'const', 1.0)
    r = sup_logit(Design(d.w, d.strata, d.psu), X.values, d[y].values.astype(float), np.ones(len(d), bool))
    ph = 1 / (1 + np.exp(-(X.values @ r['b'])))
    full = np.full(len(df), np.nan)
    full[np.where(mask)[0]] = d[y].values - ph + np.average(d[y], weights=d.w)
    return full


# =====================================================================================
# 3. Formatting (as in the Supplementary data file)
# =====================================================================================


def sup_fp(r):
    if np.isnan(r['lo']):
        return f"{100 * r['est']:.1f}" + (' (CI not estimable)' if r['est'] >= 1 else '')
    return f"{100 * r['est']:.1f} ({100 * r['lo']:.1f} to {100 * r['hi']:.1f})"


def table(key, rows, widths):
    return (CONST['caps'][key], rows, CONST['notes'][int(key[1:]) - 1], widths)


# =====================================================================================
# 4. Regression models (shared by S5, S11 and S13)
# =====================================================================================
ZONE_SPEC = ('Ecogeographical zone', 'zone', [(z, z) for z in ZONES])
RES_SPEC = ('Place of residence', 'res', [('Rural', 'Rural'), ('Urban', 'Urban')])
WQ_SPEC = ('Wealth quintile', 'wqs', [('Q1', 'Q1 (poorest)'), ('Q2', 'Q2'), ('Q3', 'Q3'), ('Q4', 'Q4'), ('Q5', 'Q5 (wealthiest)')])
SEASON_SPEC = ('Period of interview', 'season', [('Jan-Mar', 'January–March'), ('Apr-May', 'April–May'), ('Jun-Aug', 'June–August')])
MONTH_SPEC = ('Month of interview', 'month_i', [('Feb', 'February (incl. 1 household interviewed in January)'), ('Mar', 'March'),
                                               ('Apr', 'April'), ('May', 'May'), ('Jun', 'June'), ('Jul-Aug', 'July–August')])
A_SPEC = [ZONE_SPEC, RES_SPEC, WQ_SPEC, ('Sex', 'sex', [('Female', 'Female'), ('Male', 'Male')]),
          ('Age group, y', 'agegrp', [('0-4', '0–4'), ('5-14', '5–14'), ('15-49', '15–49'), ('50+', '≥50')]),
          ('Household size (de facto members)', 'hhsize', [('1-4', '1–4'), ('5-7', '5–7'), ('8-10', '8–10'), ('11+', '≥11')]),
          ('Education of household head', 'head_ed', [('None', 'None'), ('Primary', 'Primary'), ('Secondary+', 'Secondary or higher')]),
          ("Mean age of household's ITNs, months", 'netage', [('<12', '<12'), ('12-23', '12–23'), ('24-35', '24–35'), ('36+', '≥36')]),
          SEASON_SPEC]
ENOUGH_SPEC = ('≥1 ITN per 2 people in household', 'enough', [(0.0, 'No'), (1.0, 'Yes')])
B_SPEC = [ZONE_SPEC, RES_SPEC, WQ_SPEC,
          ("Woman's education", 'educ', [('None', 'None'), ('Primary', 'Primary'), ('Secondary', 'Secondary'), ('Higher', 'Higher')]),
          ('Age group, y', 'agegrp', [('20-34', '20–34'), ('15-19', '15–19'), ('35-49', '35–49')]),
          ('Parity', 'parity', [('1', '1'), ('2-3', '2–3'), ('4+', '≥4')]),
          ('Number of ANC visits', 'ancvis', [('1-3', '1–3'), ('4+', '≥4'), ('DK', 'Unknown')]),
          ('Timing of first ANC visit', 'ftri', [('Later', 'Second or third trimester'), ('1st trimester', 'First trimester'), ('DK', 'Unknown')]),
          SEASON_SPEC]


def sup_model(d, mask, y, spec):
    ok = mask.copy()
    for _, v, _ in spec:
        ok = ok & d[v].notna().values
    des = Design(d.w, d.strata, d.psu)
    cols, groups, terms = [np.ones(len(d))], {}, []
    for title, v, lev in spec:
        groups[v] = []
        for code, lab in lev[1:]:
            cols.append((d[v] == code).astype(float).values); groups[v].append(len(cols) - 1)
            terms.append(f'{title}: {lab}')
    X = np.column_stack(cols); yy = d[y].values.astype(float)
    adj = sup_logit(des, X, yy, ok)
    blocks = {}
    for title, v, lev in spec:
        crude = sup_logit(des, np.column_stack([np.ones(len(d))] + [(d[v] == c).astype(float).values for c, _ in lev[1:]]), yy, ok)
        p = fpv(sup_wald_p(adj, groups[v], des.df))
        rows = [('section', f'{title} (p{p})' if p.startswith('<') else f'{title} (p={p})')]
        for i, (code, lab) in enumerate(lev):
            m = ok & (d[v] == code).values
            n, pct = int(m.sum()), f'{100 * np.average(yy[m], weights=d.w[m]):.1f}'
            if i == 0:
                rows.append(('row', lab, fn(n), pct, '1.00 (ref)', '1.00 (ref)', ''))
            else:
                j = groups[v][i - 1]
                rows.append(('row', lab, fn(n), pct,
                             f'{fo(crude["OR"][i])} ({fo(crude["lo"][i], True)} to {fo(crude["hi"][i], True)})',
                             f'{fo(adj["OR"][j])} ({fo(adj["lo"][j], True)} to {fo(adj["hi"][j], True)})', fpv(adj['p'][j])))
        blocks[v] = rows
    return dict(n=adj['n'], blocks=blocks, vif=dict(zip(terms, vif(X[ok]))))


# =====================================================================================
# 5. Tables
# =====================================================================================
def s1():
    rows = [('header',) + tuple(CONST['static']['S1'][0])] + [('row',) + tuple(r) for r in CONST['static']['S1'][1:]]
    return table('S1', rows, [8, 26, 52, 60])


def s3():
    rows = [('header',) + tuple(CONST['static']['S3'][0])] + [('row',) + tuple(r) for r in CONST['static']['S3'][1:]]
    return table('S3', rows, [22, 30, 100])


def s2(h):
    months = list(range(1, 9))
    rows = [('header', 'Zone', 'Region', 'Households, n', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug')]
    for rg in REG_ORDER:
        m = h.region == rg
        pct = h[m].month.value_counts(normalize=True)
        first = REG_ORDER.index(rg) == 0 or ZONE[REG_ORDER[REG_ORDER.index(rg) - 1]] != ZONE[rg]
        rows.append(('plain', ZONE[rg] if first else '', rg, fn(int(m.sum())),
                     *[f'{100 * pct.get(k, 0):.0f}' for k in months]))
    pct = h.month.value_counts(normalize=True)
    rows.append(('plain', 'Senegal', '', fn(len(h)), *[f'{100 * pct.get(k, 0):.0f}' for k in months]))
    return table('S2', rows, [12, 14, 13] + [7] * 8)


def s4(ir):
    e = ir[ir.eligible]
    dcat = np.select([e.doses == 0, e.doses == 1, e.doses == 2, (e.doses >= 3) & (e.doses < 98)], ['0', '1', '2', '3+'], 'DK')
    ct = pd.crosstab(e.ancvis, dcat, values=e.w, aggfunc='sum', normalize='index') * 100
    n = e.ancvis.value_counts()
    rows = [('header', 'ANC attendance', 'n', '0 doses, %', '1 dose, %', '2 doses, %', '≥3 doses, %')]
    for code, lab in [('0', 'No ANC visit'), ('1-3', '1–3 visits'), ('4+', '≥4 visits'), ('DK', 'Unknown number of visits')]:
        rows.append(('row', lab, fn(int(n[code])), *[f'{ct.loc[code].get(c, 0):.1f}' for c in ['0', '1', '2', '3+']]))
    return table('S4', rows, [28, 9, 13, 13, 13, 13])


def s5(p, ir):
    A = sup_model(p, (p.enough == 1).values, 'use', A_SPEC[:-1] + [MONTH_SPEC])
    w = ir[ir.eligible & (ir.anc1 == 1)]
    B = sup_model(w, np.ones(len(w), bool), 'sp3', B_SPEC[:-1] + [MONTH_SPEC])
    rows = [('header', 'Variable (category)', 'n', '% with outcome', 'cOR (95% CI)', 'aOR (95% CI)', 'p'),
            ('model', f'Model A. ITN use among de facto members of households with ≥1 ITN per 2 people (n={fn(A["n"])})')] + \
        A['blocks']['month_i'] + \
        [('model', f'Model B. IPTp-SP3 among women with ≥1 ANC visit (n={fn(B["n"])})')] + B['blocks']['month_i']
    return table('S5', rows, [50, 9, 14, 22, 22, 9])


def s6(h, p, nets):
    dh, dp, dn = Design(h.w, h.strata, h.psu), Design(p.w, p.strata, p.psu), Design(nets.w, nets.strata, nets.psu)
    rows = [('header', 'Region', 'House-holds, n', 'Persons, n', 'Households owning ≥1 ITN, %',
             'Population in household with ≥1 ITN, %', 'Access, %', 'Use, %', 'Use-to-access ratio',
             'Use in households with ≥1 ITN per 2 people, % [n]', 'ITNs used last night, %')]

    def line(lab, mh, mp, mn):
        a, u = sup_mean(dp, p.access_hh, mp), sup_mean(dp, p.use, mp)
        ue = sup_mean(dp, p.use, mp & (p.enough == 1).values)
        return ('plain', lab, fn(int(mh.sum())), fn(int(mp.sum())), sup_fp(sup_mean(dh, h.own, mh)), sup_fp(sup_mean(dp, p.own, mp)),
                sup_fp(a), sup_fp(u), f"{u['est'] / a['est']:.2f}", f"{sup_fp(ue)} [{ue['n']}]",
                sup_fp(sup_mean(dn, (nets.used == 1).astype(float), mn)))
    for rg in REG_ORDER:
        rows.append(line(rg, (h.region == rg).values, (p.region == rg).values, (nets.region == rg).values))
    rows.append(line('Senegal', np.ones(len(h), bool), np.ones(len(p), bool), np.ones(len(nets), bool)))
    return table('S6', rows, [13, 10, 10, 20, 20, 20, 20, 10, 26, 20])


def s7(ir):
    di = Design(ir.w, ir.strata, ir.psu); E = ir.eligible.values
    rows = [('header', 'Region', 'Women, n', '≥1 ANC visit, %', '≥4 ANC visits, %', '≥1 SP dose, %', '≥3 SP doses, %',
             '≥3 SP doses among women with ≥4 visits, % [n]')]

    def line(lab, m):
        M = E & m; r = sup_mean(di, ir.sp3, M & (ir.anc4 == 1).values)
        return ('plain', lab, fn(int(M.sum())), sup_fp(sup_mean(di, ir.anc1, M)), sup_fp(sup_mean(di, ir.anc4, M)), sup_fp(sup_mean(di, ir.sp1, M)),
                sup_fp(sup_mean(di, ir.sp3, M)), f"{sup_fp(r)} [{r['n']}]")
    for rg in REG_ORDER:
        rows.append(line(rg, (ir.region == rg).values))
    rows.append(line('Senegal', np.ones(len(ir), bool)))
    return table('S7', rows, [13, 10, 22, 22, 22, 22, 28])


def s8(p, ir, nets):
    dp, dn = Design(p.w, p.strata, p.psu), Design(nets.w, nets.strata, nets.psu)
    di = Design(ir.w, ir.strata, ir.psu); E = ir.eligible.values
    rows = [('header', 'A. ITN (de facto population)', 'Persons, n', 'In household with ≥1 ITN, %', 'Access, %', 'Use, %',
             'Use with ≥1 ITN per 2 people, % [n]', 'ITNs used last night, %')]

    def la(lab, mp, mn):
        ue = sup_mean(dp, p.use, mp & (p.enough == 1).values)
        return ('plain', lab, fn(int(mp.sum())), sup_fp(sup_mean(dp, p.own, mp)), sup_fp(sup_mean(dp, p.access_hh, mp)),
                sup_fp(sup_mean(dp, p.use, mp)), f"{sup_fp(ue)} [{ue['n']}]", sup_fp(sup_mean(dn, (nets.used == 1).astype(float), mn)))
    for z in ZONES:
        rows.append(la(z, (p.zone == z).values, (nets.zone == z).values))
    rows.append(la('Senegal', np.ones(len(p), bool), np.ones(len(nets), bool)))
    rows += [('blank',), ('header', 'B. IPTp-SP (women with a live birth in the past 24 months)', 'Women, n',
                          '≥1 ANC visit, %', '≥4 ANC visits, %', '≥1 SP dose, %', '≥3 SP doses, %',
                          '≥3 SP doses among women with ≥4 visits, % [n]')]

    def lb(lab, m):
        M = E & m; r = sup_mean(di, ir.sp3, M & (ir.anc4 == 1).values)
        return ('plain', lab, fn(int(M.sum())), sup_fp(sup_mean(di, ir.anc1, M)), sup_fp(sup_mean(di, ir.anc4, M)),
                sup_fp(sup_mean(di, ir.sp1, M)), sup_fp(sup_mean(di, ir.sp3, M)), f"{sup_fp(r)} [{r['n']}]")
    for z in ZONES:
        rows.append(lb(z, (ir.zone == z).values))
    rows.append(lb('Senegal', np.ones(len(ir), bool)))
    return table('S8', rows, [40, 11, 22, 22, 22, 22, 28])


def s9(p, ir, B):
    pp = p.copy(); pp['w'] = pp.w.astype(float)
    w = ir[ir.eligible].copy(); w['w'] = w.w.astype(float)
    rng = np.random.default_rng(SEED)                  # article: persons, then women
    Mp = multipliers(rng, pp.strata.values, pp.psu.values, B)
    Mw = multipliers(rng, w.strata.values, w.psu.values, B)
    Mw2 = multipliers(np.random.default_rng(SEED), w.strata.values, w.psu.values, B)   # sensitivity analyses
    allp = np.ones(len(pp), bool); pw = ((pp.preg == 1) & (pp.sex == 'Female')).values
    pa = pp.agegrp.notna().values; al = np.ones(len(w), bool); dk = (w.ancvis == 'DK').values
    w['anc4_dk1'] = np.where(dk, 1.0, w.anc4)
    R = [('Population in household with ≥1 ITN', run(pp, 'own', allp, Mp)),
         ('Population access to an ITN', run(pp, 'access_hh', allp, Mp)),
         ('ITN use', run(pp, 'use', allp, Mp)),
         ('ITN use, households with ≥1 ITN per 2 people', run(pp, 'use', (pp.enough == 1).values, Mp)),
         ('Pregnant women: household with ≥1 ITN', run(pp, 'own', pw, Mp)),
         ('Pregnant women: access', run(pp, 'access_hh', pw, Mp)),
         ('Pregnant women: ITN use', run(pp, 'use', pw, Mp))]
    R += [(f'ITN use, {r.lower()}', run(pp, 'use', (pp.res == r).values, Mp)) for r in ['Urban', 'Rural']]
    R += [(f'ITN use, {z}', run(pp, 'use', (pp.zone == z).values, Mp)) for z in ZONES]
    for v, lab in [('own', 'Population in household with ≥1 ITN'), ('access_hh', 'Population access'), ('use', 'ITN use')]:
        R.append((f'{lab} (need-standardised)', run(pp, v, pa, Mp, sup_std_pred(pp, pa, v, ['zone', 'res', 'agegrp', 'sex']))))
    R += [(lab, run(w, v, al, Mw)) for v, lab in [('anc1', 'ANC1+'), ('anc4', 'ANC4+'), ('sp1', 'SP1+'), ('sp3', 'IPTp-SP3')]]
    R.append(('IPTp-SP3 among ANC4+', run(w, 'sp3', (w.anc4 == 1).values, Mw)))
    R += [(f'IPTp-SP3, {r.lower()}', run(w, 'sp3', (w.res == r).values, Mw)) for r in ['Urban', 'Rural']]
    R += [(f'IPTp-SP3, {z}', run(w, 'sp3', (w.zone == z).values, Mw)) for z in ZONES]
    R.append(('IPTp-SP3 (need-standardised)', run(w, 'sp3', al, Mw, sup_std_pred(w, al, 'sp3', ['zone', 'res', 'agegrp', 'parity']))))
    R.append(('IPTp-SP3 (standardised for age group and parity only)',
              run(w, 'sp3', al, Mw2, sup_std_pred(w, al, 'sp3', ['agegrp', 'parity']))))
    R.append(('IPTp-SP3 (standardised for age group, parity and residence)',
              run(w, 'sp3', al, Mw2, sup_std_pred(w, al, 'sp3', ['agegrp', 'parity', 'res']))))
    R.append(('ANC4+, excluding women with an unknown number of visits', run(w, 'anc4', ~dk, Mw2)))
    R.append(('ANC4+, unknown number of visits counted as ≥4', run(w, 'anc4_dk1', al, Mw2)))
    R.append(('IPTp-SP3 among ANC4+, unknown number of visits counted as ≥4', run(w, 'sp3', (w.anc4_dk1 == 1).values, Mw2)))
    R.append(('ITN use (standardised for age group and sex only)', run(pp, 'use', pa, Mp, sup_std_pred(pp, pa, 'use', ['agegrp', 'sex']))))
    R.append(('ITN use (standardised for zone, age group and sex)',
              run(pp, 'use', pa, Mp, sup_std_pred(pp, pa, 'use', ['zone', 'agegrp', 'sex']))))
    rows = [('header', 'Outcome / population', 'n', 'Mean, %', 'C (95% CI)', 'W (95% CI)', 'E (95% CI)', 'p (E)')]
    for lab, r in R:
        rows.append(('plain', lab, fn(r['n']), f"{100 * r['mu']:.1f}", fe(r['C']), fe(r['W']), fe(r['E']), fpv(r['p'])))
    return table('S9', rows, [52, 9, 9, 26, 26, 26, 9])


def s10(nets):
    dn = Design(nets.w, nets.strata, nets.psu)
    un = nets[nets.used != 1]; du = Design(un.w, un.strata, un.psu); dak = (un.region == 'Dakar').values
    lab = {1: 'Too hot', 2: "Don't like shape/colour/size", 3: "Don't like smell", 4: 'Unable to hang net',
           5: 'Slept outdoors', 6: 'Usual user did not sleep here', 7: 'No mosquitoes/no malaria',
           8: 'Extra net/saving for later', 9: 'Protect crops/plants', 10: 'Used for fishing',
           11: 'Used on mattress against bedbugs', 96: 'Other'}
    est = {c: sup_mean(du, (un.reason == c).astype(float)) for c in lab}
    rows = [('header', f'Reason (unused ITNs, n={fn(len(un))})', 'Senegal, % (95% CI)', f'Dakar, % (95% CI) [n={int(dak.sum())}]')]
    for c in sorted(lab, key=lambda c: -est[c]['est']):
        rows.append(('plain', lab[c], sup_fp(est[c]), sup_fp(sup_mean(du, (un.reason == c).astype(float), dak))))
    rows.append(('plain', 'Missing reason', f'{100 * un.reason.isna().mean():.1f} (unweighted)', ''))
    rows.append(('section', f'Source of ITNs (all ITNs, n={fn(len(nets))})'))
    for c, l in [(0, 'Not from campaign, ANC or immunization visit'), (1, 'Mass campaign'), (2, 'Antenatal care visit'),
                 (3, 'Immunization visit')]:
        rows.append(('plain', l, sup_fp(sup_mean(dn, (nets.src == c).astype(float))), ''))
    return table('S10', rows, [44, 24, 28])


def s11(p):
    A2 = sup_model(p, (p.own == 1).values, 'use', A_SPEC + [ENOUGH_SPEC])
    rows = [('header', 'Variable (category)', 'n', '% using ITN', 'cOR (95% CI)', 'aOR (95% CI)', 'p')]
    for _, v, _ in A_SPEC + [ENOUGH_SPEC]:
        rows += A2['blocks'][v]
    return table('S11', rows, [46, 9, 12, 22, 22, 9]), A2


def s12(p):
    dp = Design(p.w, p.strata, p.psu)
    rows = [('header', 'Age group, y', 'All de facto population, % (95% CI)',
             'ITN-owning households without enough ITNs, % (95% CI) [n]', 'Households with ≥1 ITN per 2 people, % (95% CI) [n]')]
    for code, lab in [('0-4', '0–4'), ('5-14', '5–14'), ('15-49', '15–49'), ('50+', '≥50')]:
        m = (p.agegrp == code).values
        a = sup_mean(dp, p.use, m & (p.own == 1).values & (p.enough == 0).values)
        b = sup_mean(dp, p.use, m & (p.own == 1).values & (p.enough == 1).values)
        rows.append(('plain', lab, sup_fp(sup_mean(dp, p.use, m)), f"{sup_fp(a)} [{a['n']}]", f"{sup_fp(b)} [{b['n']}]"))
    rows.append(('header', 'Wealth quintile', 'Use, all de facto population, % (95% CI)',
                 'Access, all de facto population, % (95% CI)', 'Use, households with ≥1 ITN per 2 people, % (95% CI) [n]'))
    for code, lab in [('Q1', 'Q1 (poorest)'), ('Q2', 'Q2'), ('Q3', 'Q3'), ('Q4', 'Q4'), ('Q5', 'Q5 (wealthiest)')]:
        m = (p.wqs == code).values
        b = sup_mean(dp, p.use, m & (p.enough == 1).values)
        rows.append(('plain', lab, sup_fp(sup_mean(dp, p.use, m)), sup_fp(sup_mean(dp, p.access_hh, m)), f"{sup_fp(b)} [{b['n']}]"))
    return table('S12', rows, [18, 30, 40, 40])


def s13(p, ir, A2):
    A = sup_model(p, (p.enough == 1).values, 'use', A_SPEC)
    w = ir[ir.eligible & (ir.anc1 == 1)]
    B = sup_model(w, np.ones(len(w), bool), 'sp3', B_SPEC)
    order = list(A2['vif'].keys())
    order = [t for t in order if not t.startswith('Period')] + [t for t in order if t.startswith('Period')]
    order.remove('≥1 ITN per 2 people in household: Yes'); order.append('≥1 ITN per 2 people in household: Yes')
    order += [t for t in B['vif'] if t not in order]
    rows = [('header', 'Term', 'Model A', 'Model A2', 'Model B')]
    for t in order:
        lab = t.replace('Wealth quintile: Q5 (wealthiest)', 'Wealth quintile: Q5 (wealthiest)')
        rows.append(('plain', lab, *[f'{m["vif"][t]:.2f}' if t in m['vif'] else '' for m in (A, A2, B)]))
    return table('S13', rows, [52, 12, 12, 12])


# =====================================================================================
# 6. Excel output
# =====================================================================================
def sup_write_excel(tables, path):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter
    wb = Workbook(); wb.remove(wb.active)
    thin = Side(style='thin', color='000000'); hair = Side(style='hair', color='BFBFBF')
    for sheet, (title, rows, note, widths) in tables.items():
        ws = wb.create_sheet(sheet); ncol = len(widths)
        for i, wdt in enumerate(widths):
            ws.column_dimensions[get_column_letter(i + 1)].width = wdt
        ws.cell(1, 1, title).font = Font(name='Arial', bold=True, size=11)
        ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=ncol)
        r = 3
        for row in rows:
            kind = row[0]
            if kind == 'blank':
                r += 1; continue
            vals = list(row[1:]) + [''] * (ncol - len(row) + 1)
            for c, v in enumerate(vals, start=1):
                cell = ws.cell(r, c, v)
                cell.font = Font(name='Arial', size=9, bold=kind in ('header', 'section', 'model'))
                cell.alignment = Alignment(horizontal='left' if c == 1 or sheet in ('S1', 'S3') else 'center',
                                           vertical='center', wrap_text=kind == 'header' or sheet in ('S1', 'S3'),
                                           indent=1 if (kind == 'row' and c == 1 and sheet not in ('S1', 'S3')) else 0)
                if kind == 'header':
                    cell.fill = PatternFill('solid', fgColor='EAF0F8'); cell.border = Border(top=thin, bottom=thin)
                elif kind == 'model':
                    cell.fill = PatternFill('solid', fgColor='F2F2F2')
                else:
                    cell.border = Border(bottom=hair)
            if kind == 'header':
                ws.row_dimensions[r].height = 42
            r += 1
        for c in range(1, ncol + 1):
            ws.cell(r - 1, c).border = Border(bottom=thin)
        cell = ws.cell(r + 1, 1, note)
        cell.font = Font(name='Arial', size=8, italic=True)
        cell.alignment = Alignment(wrap_text=True, vertical='top')
        ws.merge_cells(start_row=r + 1, start_column=1, end_row=r + 1, end_column=ncol)
        ws.row_dimensions[r + 1].height = 13 * (len(note) // max(60, int(sum(widths) * 1.1)) + 2)
        ws.freeze_panes = 'A4'; ws.sheet_view.showGridLines = False
        ws.page_setup.orientation = 'landscape'; ws.page_setup.fitToWidth = 1; ws.page_setup.fitToHeight = 0
        ws.sheet_properties.pageSetUpPr.fitToPage = True
    wb.save(path)


# ############################################################################################
# PART 4. OTHER RESULTS QUOTED IN THE TEXT (Excel); names prefixed txt_
# ############################################################################################
# Results that are quoted in the text of the article (or in the notes of the supplementary tables) but do not
# appear as such in a table. They are written to Results_in_text.xlsx, with the place where they are reported.

def txt_extra_ir(folder, ir):
    """Source of SP (ml2) and free net during pregnancy (s433a), taken from the maternity section of the most
    recent live birth, as for all other ANC and SP variables."""
    import pyreadstat
    cols = [f'{v}_{k}' for v in ['m80', 'ml2', 's433a'] for k in range(1, 7)]
    x, _ = pyreadstat.read_dta(find(folder, 'SNIR8RFL.DTA'), usecols=cols)
    blk = np.full(len(x), np.nan)
    for k in range(6, 0, -1):
        blk = np.where(x[f'm80_{k}'] == 1, k, blk)
    for v in ['ml2', 's433a']:
        vals = np.full(len(x), np.nan)
        for k in range(1, 7):
            vals = np.where(blk == k, x[f'{v}_{k}'], vals)
        ir[v] = vals
    return ir


def txt_ratio(d, num, den, mask=None):
    """Ratio of two weighted means (population use / population access) with a linearised 95% CI."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    m = np.ones(len(num), bool) if mask is None else np.asarray(mask, bool)
    w = np.where(m, d.w, 0.0); Y, X = (w * num).sum(), (w * den).sum(); r = Y / X
    se = np.sqrt(d.var_total(w * (num - r * den) / X)[0, 0]); t = stats.t.ppf(0.975, d.df)
    return dict(est=r, lo=r - t * se, hi=r + t * se, n=int(m.sum()))


def txt_fit(d, mask, y, spec):
    """Adjusted odds ratios by term and design-adjusted Wald p-value of each variable (same model as sup_model)."""
    ok = mask.copy()
    for _, v, _ in spec:
        ok = ok & d[v].notna().values
    des = Design(d.w, d.strata, d.psu)
    cols, groups, terms = [np.ones(len(d))], {}, []
    for title, v, lev in spec:
        groups[v] = []
        for code, lab in lev[1:]:
            cols.append((d[v] == code).astype(float).values); groups[v].append(len(cols) - 1); terms.append((v, code))
    adj = sup_logit(des, np.column_stack(cols), d[y].values.astype(float), ok)
    OR = {t: adj['OR'][i + 1] for i, t in enumerate(terms)}
    return OR, {v: sup_wald_p(adj, g, des.df) for v, g in groups.items()}


def run_text_results(folder, path):
    hr, pr, ir = sup_read_data(folder)
    h, p, ir, nets = sup_build(hr, pr, ir)
    ir = txt_extra_ir(folder, ir)
    dp = Design(p.w, p.strata, p.psu)
    di = Design(ir.w, ir.strata, ir.psu); E = ir.eligible.values
    rows = []

    def pct(where, label, r):
        rows.append((where, label, f"{100 * r['est']:.1f}", f"{100 * r['lo']:.1f} to {100 * r['hi']:.1f}", r['n']))

    def val(where, label, v, n=''):
        rows.append((where, label, v, '', n))

    own, enough, use, acc = p.own.values, p.enough.values, p.use.values, p.access_hh.values
    # --- ITN cascade
    w = p.w.values; sea = p.season.values
    val('Methods, Data source', 'De facto population interviewed January–May, %',
        f"{100 * w[np.isin(sea, ['Jan-Mar', 'Apr-May'])].sum() / w.sum():.1f}", len(p))
    pct('Results, ITN cascade', 'Population in households with ≥1 ITN per 2 people', sup_mean(dp, enough))
    pct('Results, ITN cascade', 'ITN use, ITN-owning households without enough ITNs',
        sup_mean(dp, use, (own == 1) & (enough == 0)))
    r = txt_ratio(dp, use, acc)
    rows.append(('Results, ITN cascade', 'National use-to-access ratio (linearised 95% CI)', f"{r['est']:.2f}",
                 f"{r['lo']:.2f} to {r['hi']:.2f}", r['n']))
    nage = []
    for k in K:
        t = pd.DataFrame({'age': hr[f'hml4_{k}'], 'w': hr.hv005 / 1e6, 'psu': hr.hv021, 'strata': hr.hv022})
        nage.append(t[(hr[f'hml10_{k}'] == 1).values])
    nage = pd.concat(nage, ignore_index=True); dn = Design(nage.w, nage.strata, nage.psu)
    pct('Results, ITN cascade', 'ITNs less than 1 year old, among ITNs with a known age',
        sup_mean(dn, (nage.age < 12).astype(float).values, (nage.age < 98).values))
    for s, lab in [('Jan-Mar', 'January–March'), ('Apr-May', 'April–May'), ('Jun-Aug', 'June–August')]:
        pct('Results, ITN cascade', f'ITN use with ≥1 ITN per 2 people, interviews in {lab}',
            sup_mean(dp, use, (enough == 1) & (sea == s)))
    pw = ((p.preg == 1) & (p.sex == 'Female')).values
    pct('Results, ITN cascade', 'ITN use, pregnant women in households with ≥1 ITN per 2 people',
        sup_mean(dp, use, pw & (enough == 1)))
    took = E & (ir.sp1 == 1).values
    val('Results, ITN cascade', 'Women who took SP during their most recent pregnancy, n', int(took.sum()))
    pct('Results, ITN cascade', 'Received a free mosquito net during that pregnancy (women who took SP)',
        sup_mean(di, (ir.s433a == 1).astype(float).values, took & ir.s433a.notna().values))
    pct('Methods, Indicators', 'Women who took SP and received it at an ANC visit',
        sup_mean(di, (ir.ml2 == 1).astype(float).values, took))
    # --- age gap by net sufficiency (Results, Factors; Discussion)
    ag = p.agegrp.values
    for s, lab in [(0, 'without enough ITNs (ITN-owning households)'), (1, 'with ≥1 ITN per 2 people')]:
        m = (own == 1) & (enough == s)
        a, b = sup_mean(dp, use, m & (ag == '0-4'))['est'], sup_mean(dp, use, m & (ag == '5-14'))['est']
        val('Results, Factors associated', f'ITN use, age 0–4 minus age 5–14 y, {lab}, percentage points',
            f'{100 * (a - b):.1f}')
    val('Results, Factors associated', 'Urban share of the West zone population, %',
        f"{100 * sup_mean(dp, (p.res == 'Urban').astype(float).values, (p.zone == 'West').values)['est']:.0f}",
        int((p.zone == 'West').sum()))
    # --- IPTp-SP
    anc = E & (ir.anc1 == 1).values; v, m13 = ir.m14.values, ir.m13.values
    elig = ((m13 >= 4) & (m13 <= 9) & (v >= 3) & (v < 98)) | ((m13 <= 3) & (v >= 4) & (v < 98))
    strict = ((m13 >= 4) & (m13 <= 9) & (v >= 3) & (v < 98)) | ((m13 <= 3) & (v >= 5) & (v < 98))
    pct('Results, IPTp-SP cascade', 'ANC attenders with ≥3 estimated SP-eligible contacts',
        sup_mean(di, elig.astype(float), anc))
    pct('Table S4 note', '<3 SP doses among women with ≥3 SP-eligible contacts, stricter definition',
        sup_mean(di, 1 - ir.sp3.values, anc & strict))
    pct('Table S4 note', 'ANC1+ when an unknown number of visits is coded as no visit',
        sup_mean(di, ((v >= 1) & (v < 98)).astype(float), E))
    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        pct('Table S9 note', f'Unknown number of ANC visits among ANC attenders, {q}',
            sup_mean(di, (ir.ancvis == 'DK').astype(float).values, anc & (ir.wqs == q).values))
    # --- Supplementary Methods
    x, _ = __import__('pyreadstat').read_dta(find(folder, 'SNIR8RFL.DTA'), usecols=[f'm80_{k}' for k in range(1, 7)])
    blk = np.full(len(x), np.nan)
    for k in range(6, 0, -1):
        blk = np.where(x[f'm80_{k}'] == 1, k, blk)
    val('Supplementary Methods', 'Women with a recent live birth: maternity section used = first, n', int((E & (blk == 1)).sum()))
    val('Supplementary Methods', 'Women with a recent live birth: maternity section used = second, n', int((E & (blk == 2)).sum()))
    val('Supplementary Methods', 'ITNs obtained more than 3 years earlier (hml4 = 96), unweighted %',
        f"{100 * (nage.age == 96).mean():.1f}", len(nage))
    val('Supplementary Methods', "ITNs of unknown age (hml4 = 98, don't know), unweighted %",
        f"{100 * (nage.age >= 98).mean():.1f}", len(nage))
    d7 = E & (ir.doses >= 7).values & (ir.doses < 98).values
    val('Supplementary Methods', 'Women reporting 7 or more SP doses, n', int(d7.sum()))
    pct('Supplementary Methods', 'IPTp-SP3 excluding women reporting 7 or more SP doses', sup_mean(di, ir.sp3.values, E & ~d7))
    a4 = E & (ir.anc4 == 1).values; dz = ir.doses.values
    for lab, m in [('0', dz == 0), ('1', dz == 1), ('2', dz == 2), ('3', dz == 3), ('4', dz == 4), ('5 or more', (dz >= 5) & (dz < 98))]:
        val('Supplementary Methods', f'Women with ≥4 ANC visits: {lab} SP dose(s), weighted %',
            f"{100 * sup_mean(di, m.astype(float), a4)['est']:.1f}", int(a4.sum()))
    # --- other notes of the supplementary tables
    m78 = h.month.isin([7, 8]).values
    val('Table S2 note', 'Households interviewed in July or August, weighted %', f'{100 * h.w[m78].sum() / h.w.sum():.1f}')
    val('Table S2 note', 'Households interviewed in July or August, unweighted %', f'{100 * m78.mean():.1f}', len(h))
    un = nets[nets.used != 1]
    val('Table S10 note', 'Unused ITNs with no recorded reason, n (%)',
        f'{int(un.reason.isna().sum())} ({100 * un.reason.isna().mean():.1f}%)', len(un))
    ok = own == 1
    val('Table S11 note', 'Members of ITN-owning households, n', int(ok.sum()))
    for lab, col in [('unknown age', 'agegrp'), ('unknown education of household head', 'head_ed'),
                     ('unknown net age', 'netage')]:
        ex = ok & p[col].isna().values; ok = ok & ~ex
        val('Table S11 note', f'Model A2 exclusions: {lab}, n', int(ex.sum()))
    val('Table S11 note', 'Model A2 sample, n', int(ok.sum()))
    # sensitivity analysis with month of interview (Table S5 note)
    wa = ir[ir.eligible & (ir.anc1 == 1)]
    for lab, d, mask, y, spec in [('A', p, (p.enough == 1).values, 'use', A_SPEC),
                                  ('B', wa, np.ones(len(wa), bool), 'sp3', B_SPEC)]:
        o1, _ = txt_fit(d, mask, y, spec)
        o2, pw2 = txt_fit(d, mask, y, spec[:-1] + [MONTH_SPEC])
        ch = {t: abs(o2[t] / o1[t] - 1) for t in o1 if t[0] != 'season'}
        t = max(ch, key=ch.get)
        val('Table S5 note', f'Model {lab}: largest relative change of the other aORs (vs Table 3) with month of interview, %',
            f'{100 * ch[t]:.1f} ({t[0]} = {t[1]})')
        if lab == 'B':
            val('Table S5 note', 'Model B with month of interview: Wald test for wealth, p', fpv(pw2['wqs']))
    import openpyxl
    from openpyxl.styles import Font, Alignment
    wb = openpyxl.Workbook(); ws = wb.active; ws.title = 'Results in text'
    ws.append(['Results quoted in the text of the article and in the notes of the supplementary tables']); ws['A1'].font = Font(bold=True)
    ws.append([]); ws.append(['Where reported', 'Result', 'Estimate', '95% CI', 'n (unweighted)'])
    for c in ws[3]:
        c.font = Font(bold=True)
    for r in rows:
        ws.append(list(r))
    ws.append([]); ws.append(['Percentages are weighted, with logit-transformed 95% CIs accounting for the survey design.'])
    for col, wdt in zip('ABCDE', [28, 80, 22, 16, 15]):
        ws.column_dimensions[col].width = wdt
    for row in ws.iter_rows(min_row=4):
        for c in row:
            c.alignment = Alignment(vertical='top', wrap_text=True)
    wb.save(path)
    print(f'  written: {os.path.abspath(path)}')


# ############################################################################################
# MAIN
# ############################################################################################
def run_figures(folder, out, reps, png):
    os.makedirs(out, exist_ok=True)
    print(f'Font: {FONT}')
    hr, pr, ir = fig_read_data(folder)
    h, p, ir, counts = fig_build(hr, pr, ir)
    # covariates needed for the regression-sample counts in Figure 2
    pr2 = pd.DataFrame({'hhid': pr.hv001.astype(int).astype(str) + '_' + pr.hv002.astype(int).astype(str),
                        'line': pr.hvidx, 'educ': pr.hv106, 'head_line': pr.hv218})
    head = pr2[pr2.line == pr2.head_line][['hhid', 'educ']].rename(columns={'educ': 'head_educ'})
    p = p.merge(head, on='hhid', how='left')
    p['head_ed'] = p.head_educ.map({0: 'None', 1: 'Primary', 2: 'Secondary+', 3: 'Secondary+'})
    age = np.column_stack([pd.to_numeric(hr[f'hml4_{k}'], errors='coerce').astype(float).values for k in K])
    itn = np.column_stack([(hr[f'hml10_{k}'] == 1).values for k in K])
    netage = pd.Series(np.nanmean(np.where(itn & (age < 96), age, np.nan), axis=1),
                       index=hr.hv001.astype(int).astype(str) + '_' + hr.hv002.astype(int).astype(str))
    p['netage'] = p.hhid.map(netage)
    R = estimates(h, p, ir)
    counts.update(regression_counts(p, ir))
    counts.update(n_preg=R['n_preg'], n_anc0=R['n_anc0'], n_dk=R['n_dk'])
    print(f'  concentration indices ({reps} bootstrap replicates)...'); C = fig_concentration(p, ir, reps)
    print(f'  writing figures to {os.path.abspath(out)}')
    figure1(out, png); figure2(out, png, counts); figure3(out, png, R); figure4(out, png, R)
    figure5(out, png, R); figure6(out, png, R); figure7(out, png, C); figureS1(out, png, p, ir)


def run_tables(folder, path, reps):
    hr, pr, ir = tab_read_data(folder)
    h, p, ir, nets = tab_build(hr, pr, ir)
    print(f'  concentration indices ({reps} bootstrap replicates)...'); C = tab_concentration(p, ir, reps)
    tables = {'Table 1': table1(p, ir), 'Table 2': table2(h, p, ir, nets, C), 'Table 3': table3(p, ir)}
    tab_write_excel(tables, path)
    print(f'  written: {os.path.abspath(path)}')


def run_supplementary(folder, path, reps):
    hr, pr, ir = sup_read_data(folder)
    h, p, ir, nets = sup_build(hr, pr, ir)
    T = {'S1': s1(), 'S2': s2(h), 'S3': s3(), 'S4': s4(ir), 'S5': s5(p, ir), 'S6': s6(h, p, nets), 'S7': s7(ir),
         'S8': s8(p, ir, nets)}
    print(f'  Table S9 (concentration indices, {reps} bootstrap replicates)...')
    T['S9'] = s9(p, ir, reps)
    T['S10'] = s10(nets)
    T['S11'], A2 = s11(p)
    T['S12'] = s12(p); T['S13'] = s13(p, ir, A2)
    sup_write_excel(T, path)
    print(f'  written: {os.path.abspath(path)}')


def main():
    ap = argparse.ArgumentParser(description='Reproduce all figures, tables and supplementary tables.')
    ap.add_argument('dhs', help='DHS .zip file or unzipped folder')
    ap.add_argument('out', nargs='?', default='outputs', help='output folder (default: outputs)')
    ap.add_argument('--reps', type=int, default=1000, help='bootstrap replicates (default 1000)')
    ap.add_argument('--png', action='store_true', help='also write PNG previews of the figures')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    print('Reading the DHS download...')
    folder = tab_locate(a.dhs)                 # unzips once if a .zip file was given
    print('Part 1. Figures'); run_figures(folder, os.path.join(a.out, 'figures'), a.reps, a.png)
    print('Part 2. Tables of the article'); run_tables(folder, os.path.join(a.out, 'Tables_article.xlsx'), a.reps)
    print('Part 3. Supplementary tables')
    run_supplementary(folder, os.path.join(a.out, 'Supplementary_tables.xlsx'), a.reps)
    print('Part 4. Other results quoted in the text')
    run_text_results(folder, os.path.join(a.out, 'Results_in_text.xlsx'))
    print(f'Done. All outputs are in {os.path.abspath(a.out)}')


if __name__ == '__main__':
    main()
