import itertools
import sympy as sp

# Variables and multipliers
x1, x2, x3 = sp.symbols('x1 x2 x3', real=True)
l1, l2, l3, l4 = sp.symbols('lambda1 lambda2 lambda3 lambda4', real=True)
X = [x1, x2, x3]
L = [l1, l2, l3, l4]

# Objective
f = 3*x1**2 + 5*x2**2 - x2*x3 + 2*x3**2

# Constraints in standard form g_i(x) <= 0
g = [
    x1 + x2 + x3 + 4,        # g1: x1 + x2 + x3 <= -4
    4*x1**2 - 5,             # g2: 4 x1^2 <= 5
    3*x2 + 8*x1 + 50,        # g3: 3 x2 + 8 x1 <= -50
    -x3**2 + 6,              # g4: -x3^2 <= -6
]

# Lagrangian and stationarity equations
Lag = f + sum(li*gi for li, gi in zip(L, g))
stationarity = [sp.diff(Lag, xi) for xi in X]

print("Lagrangian:", sp.expand(Lag))
print("Stationarity:")
for xi, eq in zip(X, stationarity):
    print(f"  dL/d{xi} =", sp.expand(eq), "= 0")
print()

tol = 1e-9
candidates = []

# Enumerate all 2^4 = 16 activity sets (1 = active: g_i = 0, 0 = inactive: lambda_i = 0)
for active in itertools.product([0, 1], repeat=4):
    eqs = list(stationarity)
    for i, a in enumerate(active):
        eqs.append(g[i] if a else L[i])
    sols = sp.solve(eqs, X + L, dict=True)
    label = "".join("A" if a else "I" for a in active)
    if not sols:
        print(f"Set {label}: no solution")
        continue
    for s in sols:
        vals = {k: complex(sp.N(v)) for k, v in s.items()}
        if any(abs(v.imag) > tol for v in vals.values()):
            print(f"Set {label}: complex solution, rejected")
            continue
        vals = {k: v.real for k, v in vals.items()}
        primal = all(float(gi.subs(vals)) <= tol for gi in g)
        dual = all(vals[li] >= -tol for li in L)
        status = "FEASIBLE" if (primal and dual) else (
            "rejected (primal)" if not primal else "rejected (dual: lambda<0)")
        fval = float(f.subs(vals))
        print(f"Set {label}: x=({vals[x1]:.4f}, {vals[x2]:.4f}, {vals[x3]:.4f}), "
              f"lambda=({vals[l1]:.4f}, {vals[l2]:.4f}, {vals[l3]:.4f}, {vals[l4]:.4f}), "
              f"f={fval:.4f} -> {status}")
        if primal and dual:
            candidates.append((fval, label, vals))

print("\nFeasible KKT candidates:")
for fval, label, v in sorted(candidates, key=lambda c: c[0]):
    print(f"  {label}: x=({v[x1]:.3f}, {v[x2]:.3f}, {v[x3]:.3f}), "
          f"lambda=({v[l1]:.3f}, {v[l2]:.3f}, {v[l3]:.3f}, {v[l4]:.3f}), f={fval:.3f}")

best = min(candidates, key=lambda c: c[0])
fval, label, v = best
print(f"\nGlobal minimum (activity set {label}):")
print(f"  x* = ({v[x1]:.3f}, {v[x2]:.3f}, {v[x3]:.3f})")
print(f"  f(x*) = {fval:.3f}")
