"""Exact check of the finite inputs of Section 6 of the paper (two different factors).

All arithmetic is exact (python-flint integer matrices and Python fractions).
The script checks:
  * the graphs Gamma_2 and Gamma_3: 12 edges, maximum degree 3, minimum degree
    at least 2;
  * the stars g_{e,c}: p = 12N = 72, and each row meets at most 3 stars;
  * the identity (6.1), v_T = b_s + sum_{t in T\\s} sum_c g_{(t,T),c}, for all
    s in T;
  * the row labels, the hitting sets behind (6.2), the intersection property
    behind (6.3), and the common label of any three 3-sets used for n = 6;
  * the functional Phi_i: it vanishes on every star and equals 1 on every v_T
    and every b_s;
  * the facet verification of Section 6.3: rank E'_k = 12(N-1) + rank E_k, the
    affine hull of C_k has dimension rank E'_k, every coordinate row F gives a
    facet (the stars vanishing at F have rank E'_k - 1, some v_T vanishes at F
    and some v_T is positive at F), and distinct rows give distinct facets;
  * the constants in Proposition 11 and Theorem 2.

Notation in the code: A[s], X[T] and R[e, c] are the vectors b_s, v_T and
g_{e,c}; H and H' are the incidence matrices E_k and E'_k; B is the polytope
C_k. In check_constants, R and L are the budget H and the scale Lambda of
Proposition 11.

Requirements: Python 3 and python-flint (pip install python-flint).
Run: python3 verify_unbounded_note.py   (about 15 s; prints ALL CHECKS PASSED)
"""
from fractions import Fraction as Fr
from itertools import combinations, product
import sys

import flint

N = 6
LEFT = list(range(4))


def build(k, N=N):
    right = [frozenset(T) for T in combinations(LEFT, k)]
    nodes = [("L", s) for s in LEFT] + [("R", T) for T in right]
    edges = [(s, T) for T in right for s in sorted(T)]
    inc = {f: [] for f in nodes}
    for ei, (s, T) in enumerate(edges):
        inc[("L", s)].append(ei)
        inc[("R", T)].append(ei)
    rows = []
    for f in nodes:
        for sig in product(range(N), repeat=len(inc[f])):
            rows.append((f, dict(zip(inc[f], sig))))
    stars = [(e, c) for e in range(len(edges)) for c in range(N)]

    def star(e, c):
        return [1 if (e in sig and sig[e] == c) else 0 for (f, sig) in rows]

    def lift(coarse):  # coarse: dict node -> value
        return [coarse.get(f, 0) for (f, sig) in rows]

    A = {s: lift({("L", s): 1}) for s in LEFT}
    X = {}
    for T in right:
        d = {("L", s): 1 for s in T}
        d[("R", T)] = k - 1
        X[T] = lift(d)
    R = {(e, c): star(e, c) for (e, c) in stars}
    return dict(k=k, right=right, nodes=nodes, edges=edges, inc=inc, rows=rows,
                stars=stars, A=A, X=X, R=R)


def rank_q(vectors):
    if not vectors:
        return 0
    return flint.fmpz_mat([list(v) for v in vectors]).rank()


def check_graph(G):
    k = G["k"]
    assert len(G["edges"]) == 12
    degs = [len(G["inc"][f]) for f in G["nodes"]]
    assert max(degs) == 3 and min(degs) >= 2, degs
    assert len(G["stars"]) == 72
    nrows = len(G["rows"])
    hits = [sum(G["R"][s][i] for s in G["stars"]) for i in range(nrows)]
    assert max(hits) <= 3
    # identity (6.1)
    for T in G["right"]:
        for s in T:
            rhs = list(G["A"][s])
            for t in T:
                if t == s:
                    continue
                e = G["edges"].index((t, T))
                for c in range(N):
                    rhs = [a + b for a, b in zip(rhs, G["R"][(e, c)])]
            assert rhs == G["X"][T], (k, T, s)
    # label property: for every admissible labeling of right-node rows
    for T in G["right"]:
        for i, (f, sig) in enumerate(G["rows"]):
            if G["X"][T][i] > 0:
                if f[0] == "L":
                    assert f[1] in T
                else:
                    assert f[1] == T  # every label in U=T lies in T
    return dict(rows=nrows, degrees=sorted(set(degs)))


def check_functional(G, trials=20):
    import random
    rnd = random.Random(1)
    idx = {(f, tuple(sorted(sig.items()))): i for i, (f, sig) in enumerate(G["rows"])}
    for _ in range(trials):
        gamma = [rnd.randrange(N) for _ in G["edges"]]
        coeff = {}
        for f in G["nodes"]:
            sig = tuple(sorted((e, gamma[e]) for e in G["inc"][f]))
            coeff[idx[(f, sig)]] = 1 if f[0] == "L" else -1
        D = lambda v: sum(c * v[i] for i, c in coeff.items())
        assert all(D(G["R"][s]) == 0 for s in G["stars"])
        assert all(D(G["X"][T]) == 1 for T in G["right"])
        assert all(D(G["A"][s]) == 1 for s in LEFT)
    return len(coeff)


def coarse_rank(G):
    nodes, edges = G["nodes"], G["edges"]
    H = [[1 if (f == ("L", s) or f == ("R", T)) else 0 for (s, T) in edges] for f in nodes]
    return rank_q(H)


def check_facets(G):
    stars = G["stars"]
    Rv = [G["R"][s] for s in stars]
    rH = coarse_rank(G)
    rHp = rank_q(Rv)
    assert rHp == 12 * (N - 1) + rH, (rHp, rH)
    # dim aff B = rank H': differences X_T - X_T0 lie in im H' and stars span im H'
    T0 = G["right"][0]
    diffs = [[a - b for a, b in zip(G["X"][T], G["X"][T0])] for T in G["right"][1:]]
    assert rank_q(Rv + diffs) == rHp
    nfacets = 0
    patterns = set()
    for i, (f, sig) in enumerate(G["rows"]):
        off = [G["R"][s] for s in stars if G["R"][s][i] == 0]
        on = [s for s in stars if G["R"][s][i] == 1]
        assert len(on) == len(G["inc"][f])
        assert rank_q(off) == rHp - 1
        zeroT = [T for T in G["right"] if G["X"][T][i] == 0]
        posT = [T for T in G["right"] if G["X"][T][i] > 0]
        assert zeroT and posT
        patterns.add(frozenset(on))
        nfacets += 1
    assert len(patterns) == len(G["rows"])  # distinct 0/1 star patterns
    return dict(rank_H=rH, rank_Hp=rHp, dim_B=rHp, coordinate_facets=nfacets)


def check_set_systems():
    two = [set(T) for T in combinations(LEFT, 2)]
    three = [set(T) for T in combinations(LEFT, 3)]
    hit2 = min(len(H) for r in range(1, 5) for H in combinations(LEFT, r)
               if all(set(H) & T for T in two))
    hit3 = min(len(H) for r in range(1, 5) for H in combinations(LEFT, r)
               if all(set(H) & T for T in three))
    assert (hit2, hit3) == (3, 2)
    assert all(T & U for T in two for U in three)
    assert all(T & U & V for T in three for U in three for V in three)
    # four 3-sets can have empty intersection (so triples are the largest groups)
    assert not set.intersection(*three)
    return dict(transversal_2sets=hit2, transversal_3sets=hit3)


def check_constants():
    p = 72
    out = {}
    for n in (1, 2, 6, 10**6):
        R = (p + 3) * n
        L = 10**5 * R**3
        tau = Fr(1, 100 * R**2)
        beta = Fr(2) / (L * tau)
        assert beta == Fr(1, 500 * R)
        eps = Fr(2, L) + R * tau
        assert eps <= Fr(1, 90 * R)
        assert R * (2 * beta + tau) < 1
        bound = 2 * R * max(10 * beta, 10 * eps / (1 - eps))
        assert bound <= Fr(20, 89) < 1
    # general R >= 1: eps <= 1/(90R) iff 2e-5/R^2 <= 1/900
    assert Fr(2, 10**5) <= Fr(1, 900)
    for k, rho, req in ((2, Fr(1, 2), 3), (3, Fr(3, 4), 2)):
        vals = [d * rho**(d - 1) for d in range(1, 60)]
        mx = max(vals)
        out[f"max_d d*rho^(d-1), k={k}"] = (mx, [d for d in range(1, 60) if d * rho**(d - 1) == mx])
        ck = Fr(req) / mx
        out[f"c_{k}"] = ck
        assert 1 - (3 + ck) / 6 >= 0
    assert out["c_2"] == 3 and out["c_3"] == Fr(32, 27)
    assert 3 + Fr(32, 27) - 4 == Fr(5, 27)
    # n = 6 numbers
    n = 6
    assert (p + 3) * n == 450
    lb3 = (p + Fr(32, 27)) * n
    assert 439 < lb3 < 440 and 6 * p + 8 == 440
    assert (2 * p + 4) * n == 888 and 450 + 440 - 888 == 2
    # the constant K in Theorem 2
    assert 6 * (1 + p) == 438 and 4 * (1 + p) == 292 and 2 * p + 5 == 149
    out["eta_order"] = f"5/(27K) ~ 10^{{-{len(str(149 + 2*(2**438 + 2**292)))-1}}}"
    return out


if __name__ == "__main__":
    print("set systems:", check_set_systems())
    print("constants:", check_constants())
    for k in (2, 3):
        G = build(k)
        print(f"k={k} graph:", check_graph(G))
        print(f"k={k} functional rows used:", check_functional(G))
        print(f"k={k} facets:", check_facets(G))
    print("ALL CHECKS PASSED")
