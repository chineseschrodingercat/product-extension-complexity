"""Independent GMP hull conversion and graph-potential check of the square extension."""
from argparse import ArgumentParser
from collections import deque
from fractions import Fraction
from itertools import product
import json
from math import gcd,lcm
from pathlib import Path

import cdd.gmp as cdd


def primitive(row):
    fractions=list(map(Fraction,row))
    denominator=lcm(*(x.denominator for x in fractions))
    numbers=[int(denominator*x) for x in fractions]
    divisor=gcd(*numbers)
    return tuple(v//divisor for v in numbers)


def potentials(values,edges):
    """Solve u_v+u_w=values[e], u>=0 by graph search and one gauge interval."""
    neighbors=[[] for _ in range(12)]
    for (a,b),w in zip(edges,values):
        neighbors[a].append((b,w));neighbors[b].append((a,w))
    u=[None]*12
    sign=[0]*12
    u[0]=0;sign[0]=1
    queue=deque([0])
    while queue:
        v=queue.popleft()
        for w,value in neighbors[v]:
            opposite=value-u[v]
            if u[w] is None:
                u[w]=opposite;sign[w]=-sign[v];queue.append(w)
            elif u[w]!=opposite:
                return None
    lower=max(-x for x,s in zip(u,sign) if s==1)
    upper=min(x for x,s in zip(u,sign) if s==-1)
    if lower>upper:
        return None
    solution=[x+lower*s for x,s in zip(u,sign)]
    assert min(solution)>=0 and [solution[v]+solution[w] for v,w in edges]==values
    return solution


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("polytope",type=Path)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--full",action="store_true",help="also convert the 24-dimensional counterexample directly from halfspaces to vertices")
    args=parser.parse_args()
    given=json.loads(args.polytope.read_text(encoding="utf-8"))
    scale=10**6
    # Build vertices directly in the eleven free node-potential coordinates.
    origins=[[0]*11,[1,1,0,0,0,0,-1,-1,0,0,0],[0,0,1,1,0,0,0,0,-1,-1,0]]
    steps=[[0]*11]+[[int(i==j) for i in range(11)] for j in range(11)]
    steps.append([(-1)**(j//2) for j in range(11)])
    points=[tuple(v+scale*w for v,w in zip(a,b)) for a,b in product(origins,steps)]
    assert set(points)==set(map(tuple,given["candidate_points_y"]))
    vmat=cdd.matrix_from_array([[1,*p] for p in points],rep_type=cdd.RepType.GENERATOR)
    hmat=cdd.copy_inequalities(cdd.polyhedron_from_matrix(vmat))
    assert not hmat.lin_set
    hull={primitive(row) for row in hmat.array}
    expected={primitive([f["b"],*[-v for v in f["a"]]]) for f in given["facets_of_base"]}
    assert hull==expected and len(hull)==37
    # Separate reverse conversion from just the defining inequalities.
    rebuilt=cdd.matrix_from_array(sorted(hull),rep_type=cdd.RepType.INEQUALITY)
    generators=cdd.copy_generators(cdd.polyhedron_from_matrix(rebuilt))
    assert not generators.lin_set and all(row[0]==1 for row in generators.array)
    vertices={tuple(row[1:]) for row in generators.array}
    assert vertices<=set(points)
    missing=[row for row in given["facets_of_base"] if row["coordinate_row"] is None]
    from collections import Counter
    positive_slacks=Counter(sum(sum(a*b for a,b in zip(f["a"],v))<f["b"] for f in missing)
                            for v in vertices)
    assert positive_slacks=={1:3,2:36}
    wedge_vertices=sum(count*2**active for active,count in positive_slacks.items())
    corners=((0,0,0),(0,-1,0),(0,-1,-1))
    hand_inequalities=set()
    for j,even in enumerate((0,2,4)):
        for k,odd in enumerate((1,3,5)):
            for a,b in product(range(2),repeat=2):
                nodes=(2*even+a,2*odd+b)
                hand_inequalities.add(primitive([-corners[j][k],*[int(i in nodes) for i in range(11)]]))
    hand_inequalities.add(primitive([scale,*([-1]*11)]))
    assert hand_inequalities==hull
    # Coordinate rows in the explicit H-model must agree with the unscaled graph slacks.
    edge_order=[(2*j+x,2*((j+1)%6)+y) for j in range(6) for x in (0,1) for y in (0,1)]
    assert [tuple(v) for v in given["edge_endpoints"]]==edge_order
    row_by_edge={f["coordinate_row"]:f for f in given["facets_of_base"] if f["coordinate_row"] is not None}
    options=[]
    for p in points:
        y=(*p,0)
        x=[int(e//4 in (2,3,4))+y[i]+y[j] for e,(i,j) in enumerate(edge_order)]
        assert x==[row_by_edge[e]["b"]-sum(a*v for a,v in zip(row_by_edge[e]["a"],p)) for e in range(24)]
        selectors={}
        for t in range(3):
            z=potentials([val-int(e//4==2*t) for e,val in enumerate(x)],edge_order)
            if z is not None:
                selectors[t]=z
        assert len(selectors)>=2
        options.append(selectors)
    overlaps=[set(i)&set(j) for i,j in product(options,repeat=2)]
    assert all(overlaps)
    report=dict(backend="pycddlib 3.0.2 cdd.gmp double description, exact rational arithmetic",
                verified_facets=len(hull),true_base_vertices=len(vertices),
                true_wedge_vertices=wedge_vertices,base_dimension=11,retained_facets=len(row_by_edge),
                positive_omitted_slack_histogram=dict(sorted(positive_slacks.items())),
                displayed_cluster_table_matches_exact_hull=True,
                split_facets=len(missing),wedge_dimension=11+len(missing),
                single_columns_checked=len(options),product_columns_checked=len(overlaps),
                minimum_shared_selectors=min(map(len,overlaps)),
                factor_extension_complexity=14+2*len(missing),
                square_extension_size=27+4*len(missing))
    if args.full:
        actual_h=[]
        for f in row_by_edge.values():
            actual_h.append([f["b"],*[-v for v in f["a"]],*([0]*13)])
        for i,f in enumerate(missing):
            e=[int(j==i) for j in range(13)]
            actual_h.append([0,*([0]*11),*e])
            actual_h.append([f["b"],*[-v for v in f["a"]],*[-v for v in e]])
        matrix=cdd.matrix_from_array(actual_h,rep_type=cdd.RepType.INEQUALITY)
        actual_v=cdd.copy_generators(cdd.polyhedron_from_matrix(matrix))
        assert not actual_v.lin_set and all(row[0]==1 for row in actual_v.array)
        from_formula=set()
        for v in vertices:
            slacks=[f["b"]-sum(a*b for a,b in zip(f["a"],v)) for f in missing]
            for z in product(*[(0,s) if s else (0,) for s in slacks]):
                from_formula.add((*v,*z))
        assert {tuple(row[1:]) for row in actual_v.array}==from_formula
        report["direct_full_space_vertices"]=len(from_formula)
        report["full_halfspaces"]=len(actual_h)
    with args.out.open("x",encoding="utf-8") as fd:
        json.dump(report,fd,indent=2);fd.write("\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
