"""Render the published SVG figures deterministically from evaluated CSV files.

Run from repository root: python src/visualize_results.py
This plots stored metrics; it does not retrain a model.
2025 4-Fold reconstruction and 2026 team evaluation are distinct experiments.
"""
import csv
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / 'reports' / 'metrics'
FIGURES = ROOT / 'reports' / 'figures'
FIGURES.mkdir(parents=True, exist_ok=True)
COLORS = {'Poisson · 19-feature reproduction': '#13795b',
          '3-week average': '#d99b29', 'Weekday-hour mean': '#7a899d'}
LABELS = {'Poisson · 19-feature reproduction': 'LightGBM Poisson',
          '3-week average': '3-week average', 'Weekday-hour mean': 'Weekday-hour mean'}


def read_csv(name):
    with (METRICS / name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def f1(value):
    return f'{float(value):.1f}'


def t(x, y, label, size=16, color='#203047', bold=False):
    weight = ' font-weight="700"' if bold else ''
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}"'
            f'{weight}>{escape(str(label))}</text>')


def line(x1, y1, x2, y2, color='#e3eaf1', width=1):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"'
            f' stroke="{color}" stroke-width="{width}"/>')


def base(title, subtitle):
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 520"'
            ' role="img" aria-label="' + escape(title) + '">'
            '<rect width="960" height="520" fill="#f8fafc"/>'
            '<rect x="18" y="18" width="924" height="484" rx="18"'
            ' fill="#fff" stroke="#e5ebf0"/>'
            '<g font-family="Arial,Helvetica,sans-serif">'
            + t(54, 70, title, 27, bold=True)
            + t(54, 98, subtitle, 14, '#63758a'))


def save(name, svg):
    (FIGURES / name).write_text(svg + '</g></svg>', encoding='utf-8')


def fold_chart(scores):
    s = base('2025 expanding-window CV: MAE by fold',
             'Independent reproduction | Lower MAE is better')
    xs = [140, 367, 593, 820]
    for i in range(8):
        v = 1.00 + 0.05*i
        y = f1(405 - (v-1.0)*700)
        s += line(125, y, 840, y) + t(68, f1(float(y)+5), f'{v:.2f}', 13, '#718096')
    for i, x in enumerate(xs):
        s += t(x-20, 440, f'Fold {i+1}', 14)
    for method, color in COLORS.items():
        rows = sorted((r for r in scores if r['method'] == method), key=lambda r:r['fold'])
        if len(rows) != 4:
            raise ValueError('Expected four folds per method: '+method)
        pts = [(x, f1(405 - (float(r['mae'])-1.0)*700)) for x, r in zip(xs, rows)]
        points = ' '.join(f'{x},{y}' for x,y in pts)
        s += f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="4"/>'
        for x,y in pts:
            s += f'<circle cx="{x}" cy="{y}" r="6" fill="{color}"/>'
    for method, x in zip(COLORS,[85,355,605]):
        s += f'<circle cx="{x}" cy="478" r="6" fill="{COLORS[method]}"/>'
        s += t(x+16, 484, LABELS[method], 14)
    save('2025_cv_mae_by_fold.svg', s)


def mean_chart(mean):
    s = base('2025 CV: Mean MAE and fold variability',
             'Error bars = sample SD across four folds, NOT confidence intervals')
    by = {r['method']:r for r in mean}
    for j in range(6):
        v = 1.0 + 0.1*j
        x = f1(300 + (v-1.0)*900)
        s += line(x, 143, x, 435) + t(f1(float(x)-15), 457, f'{v:.1f}', 13, '#718096')
    for idx,(method,color) in enumerate(COLORS.items()):
        r = by[method]
        y = 185 + idx*110
        val,sd = float(r['mae_mean']), float(r['mae_sd'])
        xx,lo,hi = [f1(300+(v-1.0)*900) for v in [val, val-sd, val+sd]]
        s += t(54,y+5,LABELS[method],15)
        s += line(lo,y,hi,y,color,5)+line(lo,y-12,lo,y+12,color,3)+line(hi,y-12,hi,y+12,color,3)
        s += f'<circle cx="{xx}" cy="{y}" r="9" fill="{color}"/>'
        s += t(755,y+5,f'{val:.3f} ± {sd:.3f}',15)
    save('2025_cv_mae_mean_sd.svg',s)


def holdout_mae(hold):
    s=base('2026 H1 holdout: MAE',
           'Team-reported final model | Separate from 2025 independent CV')
    by = {(r['metric'],r['method']):float(r['value']) for r in hold}
    for idx,(method,color) in enumerate([('LightGBM Poisson','#13795b'),('3-week average','#d99b29')]):
        v=by[('mae',method)]; y=190+idx*125; width=f1(v*445)
        s+=t(62,y+30,method,19)
        s+=f'<rect x="310" y="{y}" width="{width}" height="54" rx="8" fill="{color}"/>'
        s+=t(f1(322+float(width)),y+35,f'{v:.3f}',20)
    s+=t(305,469,'MAE: observed pickups per zone-hour (lower is better)',15,'#63758a')
    save('2026_holdout_mae.svg',s)


def share_chart(hold):
    s=base('2026 H1: Demand share evaluation',
           'Team-reported final model | Modest differences in share metrics')
    by={(r['metric'],r['method']):float(r['value']) for r in hold}
    for idx,(metric,title,scale) in enumerate([('tvd','TVD (lower is better)',0.40),('top5','Top-5 observed pickup share (higher is better)',0.80)]):
        y=173+idx*145
        s+=t(52,y-24,title,18)
        for j,(method,color) in enumerate([('Weekday-hour average','#7a899d'),('LightGBM Poisson','#13795b')]):
            val=by[(metric,method)]; yy=y+j*42; width=f1(val/scale*475)
            s+=t(56,yy+21,method,14)
            s+=f'<rect x="280" y="{yy}" width="{width}" height="29" rx="6" fill="{color}"/>'
            label=f'{val:.3f}' if metric=='tvd' else f'{val*100:.2f}%'
            s+=t(f1(292+float(width)),yy+21,label,15)
    save('2026_share_metrics.svg',s)


def main():
    scores=read_csv('2025_4fold_model_scores.csv')
    mean=read_csv('2025_4fold_mean_sd.csv')
    hold=read_csv('2026_team_holdout_summary.csv')
    fold_chart(scores)
    mean_chart(mean)
    holdout_mae(hold)
    share_chart(hold)
    print('Generated four SVG figures in',FIGURES)


if __name__=='__main__':
    main()
