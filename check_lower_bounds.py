"""Exact finite checks for the four claims in the simultaneous lower bounds."""
from argparse import ArgumentParser
from fractions import Fraction
from itertools import combinations, combinations_with_replacement, product
import json
from pathlib import Path


def graph_checks(d):
    edges = [(12*b+2*j+i, 12*b+2*((j+1)%6)+k)
             for b,j,i,k in product(range(d), range(6), range(2), range(2))]
    stars = [frozenset(e for e,pair in enumerate(edges) if v in pair)
             for v in range(12*d)]
    assert all(len(star)==4 for star in stars)
    assert all(len(a & b)<=1 for a,b in combinations(stars,2))
    budget = 13*d
    # For b=1,2 compute the exact number of forced edge-centered columns.
    # For b>=3, each of 4b incidences can be counted at most twice.
    for b in range(1,12*d+1):
        needed = (min(len(frozenset.union(*ss)) for ss in combinations(stars,b))
                  if b<=2 else 2*b)
        assert 12*d-b+needed>budget
    cycles = []
    for block in range(d):
        for clone in product(range(2),repeat=6):
            nodes = tuple(12*block+2*j+clone[j] for j in range(6))
            cycles.append(frozenset(nodes))
            ordered_edges = [24*block+4*j+2*clone[j]+clone[(j+1)%6] for j in range(6)]
            for star in stars:
                assert sum((-1)**j for j,e in enumerate(ordered_edges) if e in star)==0
            for t in range(3):
                assert sum((-1)**j for j in range(6) if (j-2*t)%6 in (2,3,4))==1
    exceptions = [frozenset([v]) for v in range(12*d)]+list(map(frozenset,edges))
    cases = 0
    for k in range(d+1):
        for groups in combinations_with_replacement(exceptions,k):
            forbidden = frozenset().union(*groups)
            assert any(cycle.isdisjoint(forbidden) for cycle in cycles)
            cases += 1
    return dict(blocks=d, excluded_width=budget, checked_exception_patterns=cases,
                alternating_cycles=len(cycles))


def star_cover_check():
    """Every column support of S is a union of stars, so twelve rectangles cover supp S."""
    edges = [(2*j+i, 2*((j+1)%6)+k) for j,i,k in product(range(6), range(2), range(2))]
    stars = [frozenset(e for e,pair in enumerate(edges) if v in pair) for v in range(12)]
    def support_X(t):
        return frozenset(e for e in range(24) if (e//4-2*t)%6 in (2,3,4))
    supports = [support_X(t) for t in range(3)]
    supports += [support_X(t) | stars[v] for t in range(3) for v in range(12)]
    for sup in supports:
        assert frozenset().union(*[st for st in stars if st <= sup]) == sup
    return dict(columns_checked=len(supports), rectangle_cover_single=len(stars),
                rectangle_cover_square=2*len(stars))


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    tau=Fraction(1,5000)
    beta=1/(10**6*tau)
    epsilon=Fraction(1,10**6)+26*tau
    coverage=26*(tau+2*beta)
    cycle=26*max(6*beta,6*epsilon/(1-epsilon))
    assert coverage<1 and cycle==Fraction(811356,994799)<1
    # A generator positive in both blocks is supported on small columns in
    # one of these rectangles (choose any one positive coordinate per block).
    allowed=[{t for t in range(3) if (e-2*t)%6 in (2,3,4)} for e in range(6)]
    masks={frozenset(product(a,b)) for a,b in product(allowed,repeat=2)}
    best=max(len(a|b) for a,b in combinations_with_replacement(masks,2))
    assert best==7
    selectors=[set(product(set(range(3))-{k},repeat=2)) for k in range(3)]
    assert len(set.union(*selectors))==9
    result=dict(graph_checks=[graph_checks(d) for d in (1,2)],
                star_cover=star_cover_check(),
                exact_probe_bound=str(coverage),exact_cycle_bound=str(cycle),
                maximum_cells_two_cross_block_rectangles=best,
                cells_covered_three_explicit_selectors=9)
    with args.out.open("x",encoding="utf-8") as f:
        json.dump(result,f,indent=2)
        f.write("\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
