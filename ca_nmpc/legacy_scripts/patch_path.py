p=open('p2.py').read()
p=p.replace("Y_LO, Y_HI = -1.75, 5.25","Y_LO, Y_HI = -3.5, 3.5       # 7 m wide road\nPATH_A, PATH_L = 1.0, 55.0   # lane-centre path: gentle S-curve  y = A sin(2 pi x / L)  (continuous lateral excitation, ~2 m/s^2 peak at 13 m/s)\n_KX = 2 * np.pi / PATH_L\ndef path_y(x): return PATH_A * np.sin(_KX * x)")
old=p[p.index("    obs = np.array([[45.0"):p.index("    v0 = V_REF")]
new='''    x1 = 50.0 + rng.uniform(-2, 2); x2 = 86.0 + rng.uniform(-2, 2)
    obs = np.array([[x1, path_y(x1) + rng.uniform(-0.2, 0.2)],      # O1 sits on the lane centre -> evade to the left (+y)
                    [x2, 1.4 + rng.uniform(-0.2, 0.2)]])            # O2 sits left of centre   -> evade to the right (-y)
'''
p=p.replace(old,new)
p=p.replace("""            cost += w * (1.0 * X[1, j] ** 2 + 4.0 * X[2, j] ** 2 + 0.5 * (X[3, j] - V_REF) ** 2 + 1.0 * X[5, j] ** 2)""",
"""            xj = X[0, j]; yref = PATH_A * ca.sin(_KX * xj); dyref = PATH_A * _KX * ca.cos(_KX * xj)
            psiref = ca.atan(dyref); rref = X[3, j] * (-PATH_A * _KX ** 2 * ca.sin(_KX * xj)) / (1 + dyref ** 2) ** 1.5
            cost += w * (1.0 * (X[1, j] - yref) ** 2 + 4.0 * (X[2, j] - psiref) ** 2 + 0.5 * (X[3, j] - V_REF) ** 2 + 1.0 * (X[5, j] - rref) ** 2)""")
p=p.replace("# A 2-lane road, two static obstacles that force an evasive double lane-change (**limited perception range: 32 m**, so the manoeuvre must be planned ~2.5 s ahead), and four conditions:",
"# A 7 m road with a gentle S-shaped lane centre (so lateral dynamics are excited from the start and the shift becomes *observable*), two static obstacles that force an evasive left-then-right manoeuvre (**limited perception range: 32 m**, so it must be planned ~2.5 s ahead), and four conditions:")
open('p2.py','w').write(p)
q=open('p3.py').read()
q=q.replace("rms_y=float(np.sqrt(np.mean(P[:, 1] ** 2)))","rms_y=float(np.sqrt(np.mean((P[:, 1] - path_y(P[:, 0])) ** 2)))")
open('p3.py','w').write(q)
print("patched")
