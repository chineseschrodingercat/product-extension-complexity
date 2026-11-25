"""Independent finite checks of the fixed incidence data and bounds in the proof."""
from argparse import ArgumentParser
from collections import Counter
from fractions import Fraction
from itertools import combinations, product
import json
from pathlib import Path
from random import Random

import numpy as np
import sympy as sp


COPIES = 6
COLORS = tuple(range(4))


def exact_rank(matrix):
    return sp.Matrix(matrix.tolist()).to_DM().rank()


def incidence_data(k):
    words = tuple(combinations(COLORS,k))
    edges = [(s,4+t) for t,word in enumerate(words) for s in word]
    coarse = np.array([[int(f in edge) for edge in edges]
                       for f in range(4+len(words))],dtype=np.int64)
    stars = []
    origins = []
    selectors = []
    designations = []
    row_at = {}
    for f,coarse_row in enumerate(coarse):
        incident = tuple(np.flatnonzero(coarse_row))
        for sigma in product(range(COPIES),repeat=len(incident)):
            row_at[f,sigma] = len(stars)
            row = np.zeros(12*COPIES,dtype=np.int64)
            row[[int(e)*COPIES+c for e,c in zip(incident,sigma)]] = 1
            stars.append(row)
            selectors.append([int(f==s) for s in COLORS])
            origins.append([int(f in word) if f<4 else (k-1)*int(f==4+t)
                            for t,word in enumerate(words)])
            designations.append(f if f<4 else words[f-4][0])
    M,A,X = (np.array(x,dtype=np.int64) for x in (stars,selectors,origins))
    return words,edges,coarse,M,A,X,np.array(designations),row_at


def check_geometry_and_choices(k,n):
    words,edges,H,M,A,X,labels,lookup = incidence_data(k)
    L = 100000*(75*n)**3
    fits_int64 = L+2 < np.iinfo(np.int64).max//100
    pure = M.shape[1]
    assert pure==72 and M.shape[0]==(1080 if k==2 else 1728)
    rank0,rank1 = exact_rank(H),exact_rank(M)
    assert rank1==12*(COPIES-1)+rank0
    assert len({tuple(np.flatnonzero(row)) for row in M})==len(M)
    # Within each coarse node clone permutations identify every row. Color
    # permutations act transitively on nodes of each side of the bipartition.
    facet_ranks = []
    for f in (0,4):
        mask = H[f].astype(bool)
        row = M[lookup[f,(0,)*int(mask.sum())]]
        facet_ranks.append(exact_rank(M[:,row==0]))
    assert facet_ranks==[rank1-1]*2
    for f,row in enumerate(H):
        assert exact_rank(H[:,row==0])==rank0-1
    for t,word in enumerate(words):
        assert set(labels[X[:,t]>0])<=set(word)
    assert (X.min(axis=1)==0).all() and (X.max(axis=1)>0).all()

    assignments = []
    identity = np.eye(pure,dtype=np.int64)
    assert np.array_equal(M@identity,M)
    for t,word in enumerate(words):
        choices = {}
        # Compute every coefficient directly from the edge list, including all
        # 72 shifted input columns. This does not need a nonnegative rank solver.
        for s in word:
            weights = np.repeat([int(b==4+t and color!=s) for color,b in edges],COPIES)
            assert np.array_equal(M@weights+A[:,s],X[:,t])
            # For arbitrary integer scales the constant and linear coefficients
            # have just been checked separately. Also evaluate the whole array
            # when its entries and intermediate sums fit in int64.
            if fits_int64:
                columns = np.column_stack((weights,weights[:,None]+L*identity))
                target = np.column_stack((X[:,t], X[:,t,None]+L*M))
                assert np.array_equal(M@columns+A[:,s,None],target)
            choices[s]=True
        assignments.extend([frozenset(choices)]*(pure+1))

    # Verify the functional on a coarse transversal, and whenever one selected
    # clone is changed. Its identity in general is also local to each edge.
    signed=np.array([1]*4+[-1]*len(words),dtype=np.int64)
    rng=Random(2134+k)
    transversals=[[0]*12]+[[c if e==j else 0 for e in range(12)]
                            for j,c in product(range(12),range(1,COPIES))]
    transversals += [[rng.randrange(COPIES) for _ in range(12)] for _ in range(128)]
    for gamma in transversals:
        rows=[lookup[f,tuple(gamma[e] for e in np.flatnonzero(h))]
              for f,h in enumerate(H)]
        assert np.all(signed@M[rows,:]==0) and np.all(signed@X[rows,:]==1)
    for e in range(12):
        # Independent row-slice overlap used for the deficient-star charge.
        for f in np.flatnonzero(H[:,e]):
            rows=[index for (node,sigma),index in lookup.items() if node==f and M[index,e*COPIES]]
            counts=M[rows,:].sum(axis=0)
            counts[e*COPIES]=0
            assert counts.max()*COPIES<=len(rows)
    return assignments,dict(k=k,coarse_rank=rank0,dimension=rank1,
                            retained_facet_rows=len(M),potential_vertex_columns=len(assignments),
                            two_facet_direction_ranks=facet_ranks,
                            exact_representations=len(assignments)*k,
                            graph_functionals_checked=len(transversals))


def conditional_checks(k):
    words=tuple(combinations(COLORS,k))
    q=Fraction(k,4)
    for count in range(5-k):
        for labels in product(COLORS,repeat=count):
            assert any(set(labels).isdisjoint(T) for T in words)
    for d in range(1,7):
        # Permuting the four labels independently makes each designated label 0.
        total=len(words)**(d-1)
        hits=sum(all(0 in T for T in other) for other in product(words,repeat=d-1))
        assert Fraction(hits,total)==q**(d-1)
    values=[d*q**(d-1) for d in range(1,12)]
    maximum=max(values)
    assert maximum==(1 if k==2 else Fraction(27,16))
    # (d+1)q/d decreases with d and is already <=1 by d=4.
    assert 5*q<=4
    c=Fraction(5-k,1)/maximum
    assert 1-(3+c)/COPIES>=0
    return dict(k=k,maximum=str(maximum),required_extra_cost_per_good_block=str(c))


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--n",type=int,default=6)
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    assert args.n>=1
    left,info2=check_geometry_and_choices(2,args.n)
    right,info3=check_geometry_and_choices(3,args.n)
    histogram=Counter(len(a&b) for a,b in product(left,right))
    assert sum(histogram.values())==438*292 and min(histogram)>=1
    R=75*args.n
    L=100000*R**3
    tau=Fraction(1,100*R**2)
    beta=2/(L*tau)
    eps=Fraction(2,L)+R*tau
    assert beta==Fraction(1,500*R) and eps<=Fraction(1,90*R)
    assert R*(2*beta+tau)<1
    margin=2*R*max(10*beta,10*eps/(1-eps))
    assert margin<=Fraction(20,89)<1
    c2,c3=Fraction(3),Fraction(32,27)
    gain=c2+c3-4
    assert gain==Fraction(5,27)
    def ceiling(x):
        return -((-x.numerator)//x.denominator)
    report=dict(n=args.n,N=COPIES,L=L,proof_budget=R,exact_error_bound=str(margin),
                geometry=[info2,info3],intersection_sizes_single_coordinate=dict(histogram),
                conditional_counts=[conditional_checks(k) for k in (2,3)],
                partial_rank_lower_bounds=[ceiling((72+c)*args.n) for c in (c2,c3)],
                partial_joint_upper_bound=148*args.n,
                binary_deficit_lower_bound=ceiling(gain*args.n),
                note="Finite checks of the proof inputs. The inequalities for every real factorization and every n are proved in the paper.")
    with args.out.open("x",encoding="utf-8") as f:
        json.dump(report,f,indent=2)
        f.write("\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
