"""Build the integer counterexample and check its factorization and facet data exactly."""
import argparse
from fractions import Fraction
from itertools import combinations, product
import json
from math import gcd
from pathlib import Path

from sympy import Matrix, __version__ as sympy_version


LARGE = 10**6
CLUSTERS = 6
NODES = 12


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def add(a, b):
    return tuple(x+y for x,y in zip(a,b))


def subtract(a, b):
    return tuple(x-y for x,y in zip(a,b))


def directions(points, support):
    return [subtract(points[i],points[support[0]]) for i in support[1:]]


def all_facets(base, triangle):
    """A facet of the 11-dimensional sum needs at least 9 maximizing base points."""
    found = {}
    examined = 0
    for count in range(9,13):
        for face in combinations(range(13),count):
            base_directions = directions(base,face)
            for t_count in range(1,4):
                for t_face in combinations(range(3),t_count):
                    equations = base_directions+directions(triangle,t_face)
                    if len(equations)<10:
                        continue
                    examined += 1
                    kernel = Matrix(equations).to_DM().nullspace()
                    if kernel.shape[0]!=1:
                        continue
                    normal = [int(v) for v in kernel.to_list()[0]]
                    divisor = gcd(*normal)
                    for sign in (1,-1):
                        a = tuple(sign*v//divisor for v in normal)
                        if a in found:
                            continue
                        b_values = [dot(a,x) for x in base]
                        t_values = [dot(a,x) for x in triangle]
                        b, t = max(b_values), max(t_values)
                        if b_values[face[0]]==b and t_values[t_face[0]]==t:
                            found[a]=LARGE*b+t
    return found, examined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir",type=Path,required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True,exist_ok=True)
    edges = [(2*j+a,2*((j+1)%CLUSTERS)+b) for j,a,b in product(range(6),range(2),range(2))]
    R = Matrix([[int(v in e) for v in range(NODES)] for e in edges])
    A = Matrix([[int(i//4==2*t) for t in range(3)] for i in range(24)])
    X = Matrix([[int((i//4-2*t)%6 in (2,3,4)) for t in range(3)] for i in range(24)])
    assert R.rank()==11 and R[:,:11].rank()==11
    def unit_on(cluster):
        return Matrix([int(v//2==cluster%6) for v in range(12)])
    options = {}
    for t in range(3):
        for s,c in [((t+1)%3,2*t+4),((t+2)%3,2*t+3)]:
            options[t,s]=unit_on(c)
            assert X[:,t]==A[:,s]+R*options[t,s]
    assert all(min(row)==0 for row in X.tolist())
    assert max(X)==1 and min(X)==0

    zero=Matrix.zeros(12,1)
    eye=Matrix.eye(12)
    shifts=[zero]+[LARGE*eye[:,v] for v in range(12)]
    labels=list(product(range(3),range(13)))
    S=Matrix.hstack(*(X[:,t]+R*shifts[v] for t,v in labels))
    single_dictionary=R.row_join(A[:,:2])
    single_coefficients=[]
    for t,v in labels:
        s=next(s for s in (0,1) if (t,s) in options)
        single_coefficients.append((options[t,s]+shifts[v]).col_join(eye[:2,s]))
    assert single_dictionary*Matrix.hstack(*single_coefficients)==S
    W=Matrix.diag(R,R).row_join(A.col_join(A))
    right=[]
    target=[]
    trit_eye=Matrix.eye(3)
    for i,(t,v) in enumerate(labels):
        for j,(u,w) in enumerate(labels):
            s=next(s for s in range(3) if (t,s) in options and (u,s) in options)
            right.append((options[t,s]+shifts[v]).col_join(options[u,s]+shifts[w]).col_join(trit_eye[:,s]))
            target.append(S[:,i].col_join(S[:,j]))
    H=Matrix.hstack(*right)
    assert min(W)>=0 and min(H)>=0 and W*H==Matrix.hstack(*target)

    # All cycles used in the lower bound and all possible exceptional star sets.
    cycles=[]
    for choice in product(range(2),repeat=6):
        cycle=[4*j+2*choice[j]+choice[(j+1)%6] for j in range(6)]
        weights=Matrix([[(-1)**cycle.index(i) if i in cycle else 0 for i in range(24)]])
        assert weights*R==Matrix.zeros(1,12) and weights*X==Matrix.ones(1,3)
        cycles.append({2*j+choice[j] for j in range(6)})
    exceptional=[{v} for v in range(12)]+[set(e) for e in edges]
    assert all(any(not excluded&cycle for cycle in cycles) for excluded in exceptional)
    r=13
    tau=Fraction(1,5000)
    beta=1/(LARGE*tau)
    epsilon=Fraction(1,LARGE)+r*tau
    cycle_bound=max(6*beta,6*epsilon/(1-epsilon))
    assert r*(tau+2*beta)<1 and r*cycle_bound<1
    for missing in range(1,13):
        double_atoms=4 if missing==1 else 2*missing
        assert NODES-missing+double_atoms>r

    basis=R[:,:11]
    def coordinates(vec):
        v,free=basis.gauss_jordan_solve(vec)
        assert free.rows==0 and all(a.is_Integer for a in v) and basis*v==vec
        return tuple(map(int,v))
    base=[coordinates(Matrix.zeros(24,1))]+[coordinates(R[:,v]) for v in range(12)]
    triangle=[coordinates(X[:,t]-X[:,0]) for t in range(3)]
    points=[add(triangle[t],tuple(LARGE*c for c in base[v])) for t,v in labels]
    assert all(X[:,0]+basis*Matrix(y)==S[:,i] for i,y in enumerate(points))
    assert Matrix(directions(points,tuple(range(len(points))))).rank()==11
    contact={}
    for f in range(24):
        equality=tuple(i for i in range(S.cols) if S[f,i]==0)
        assert Matrix(directions(points,equality)).rank()==10
        a=tuple(-int(c) for c in basis[f,:])
        assert a not in contact
        contact[a]=(int(X[f,0]),f)
    facets,examined=all_facets(base,triangle)
    assert all(facets[a]==b for a,(b,f) in contact.items())
    hs=[]
    for a,b in sorted(facets.items()):
        values=[dot(a,y) for y in points]
        eq=tuple(i for i,v in enumerate(values) if v==b)
        assert max(values)==b and Matrix(directions(points,eq)).rank()==10
        hs.append(dict(a=a,b=b,coordinate_row=contact[a][1] if a in contact else None))
    k=len(hs)-24
    data=dict(dimension_of_base=11,scale=LARGE,edge_endpoints=edges,
              candidate_points_y=points,facets_of_base=hs,
              polytope_definition="For every coordinate_row facet retain a*y <= b. For each other facet use one new variable z and impose 0 <= z <= b-a*y. Use the order of facets_of_base for z.")
    checks=dict(sympy=sympy_version,base_candidates=len(points),base_dimension=11,
                retained_facets=24,total_facets=len(hs),split_facets=k,
                dimension_of_counterexample_factor=11+k,facets_of_counterexample_factor=24+2*k,
                single_matrix_shape=S.shape,single_matrix_rank=S.rank(),
                single_factorization_width=single_dictionary.cols,
                paired_matrix_shape=(W.rows,H.cols),paired_factorization_width=W.cols,
                paired_integer_entries_compared=W.rows*H.cols,checked_cycles=len(cycles),
                checked_exceptional_sets=len(exceptional),candidate_normal_systems=examined,
                coverage_error=str(r*(tau+2*beta)),cycle_error=str(r*cycle_bound),
                extension_complexity_factor=14+2*k,
                extension_complexity_square_upper_bound=27+4*k,
                twice_extension_complexity_factor=28+4*k)
    for name, record in [("polytope.json",data),("checks.json",checks)]:
        with (args.outdir/name).open("x",encoding="utf-8") as fd:
            json.dump(record,fd,indent=2)
            fd.write("\n")
    print(json.dumps(checks,indent=2))


if __name__=="__main__":
    main()
