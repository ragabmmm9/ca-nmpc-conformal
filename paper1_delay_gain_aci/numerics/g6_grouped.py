"""G6: critical gain of the linearised loop when the level is driven by the equal-weight average of indicators with delays in a set G
(delta_{t+1}=delta_t-(k/|G|) sum_{j in G} delta_{t-j} ; char. poly z^{d+1}-z^d+(k/|G|) sum_j z^{d-j}, d=max G)."""
import numpy as np, csv, os
OUT = os.path.dirname(os.path.abspath(__file__))
def maxroot(k, G):
    d = max(G); c = np.zeros(d + 2); c[0] = 1.0; c[1] -= 1.0
    for j in G: c[d + 1 - (d - j)] += k / len(G)   # coefficient of z^{d-j}: index (d+1)-(d-j)
    return np.abs(np.roots(c)).max()
def first_exit(G):
    lo, hi = 1e-6, 3.0
    grid = np.linspace(lo, hi, 6000)
    mr = np.array([maxroot(k, G) for k in grid])
    i = int(np.argmax(mr > 1 + 1e-9)); a, b = grid[i - 1], grid[i]
    for _ in range(60):
        m = 0.5 * (a + b)
        if maxroot(m, G) > 1 + 1e-9: b = m
        else: a = m
    return 0.5 * (a + b)
rows = []
for name, G in [("1-4", range(1, 5)), ("5-10", range(5, 11)), ("11-25", range(11, 26))]:
    G = list(G); ks = first_exit(G); d = max(G)
    rows.append(dict(group=name, kstar_group=ks, kstar_longest=2 * np.sin(np.pi / (2 * (2 * d + 1))), ratio_at_eta005=0.05 / ks))
with open(os.path.join(OUT, "G6_grouped.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
for r in rows: print(r)
