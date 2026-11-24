"""Exhaust the four-block conditional-count LP dual, independently of incidence data."""
from fractions import Fraction
from itertools import combinations, product
import argparse
import json
from pathlib import Path


def exhaustive_dual(k):
    colors=range(4)
    words=list(combinations(colors,k))
    n=4
    # A test fixes the tested block and every other word. Its demand is 5-k.
    # An atom's profile has a label or a blank in each block.
    tests=[(i,choices) for i in range(n) for choices in product(words,repeat=n-1)]
    best=0
    attained=set()
    for profile in product((-1,0,1,2,3),repeat=n):
        occupied=sum(s>=0 for s in profile)
        if not occupied:
            continue
        covered=0
        for i,choices in tests:
            if profile[i]<0:
                continue
            remaining=profile[:i]+profile[i+1:]
            covered+=all(s<0 or s in T for s,T in zip(remaining,choices))
        if covered>best:
            best=covered
            attained={occupied}
        elif covered==best:
            attained.add(occupied)
    ratio=Fraction(best,len(words)**(n-1))
    certified=Fraction(len(tests)*(5-k),best)
    expected=Fraction(3) if k==2 else Fraction(32,27)
    assert certified/n==expected
    return dict(k=k,tests=len(tests),profiles=5**4-1,maximum_tests_served=best,
                maximizing_arities=sorted(attained),capacity_per_profile=str(ratio),
                dual_lower_bound_per_block=str(certified/n))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    checked=0
    for extras in range(501):
        missing_max=3*extras//6
        bad_max=extras//6
        for c in (Fraction(3),Fraction(32,27)):
            assert extras-missing_max-c*bad_max>=0
            checked+=1
    small=list(map(set,combinations(range(4),3)))
    for triple in product(small,repeat=3):
        assert set.intersection(*triple)
    with args.out.open("x",encoding="utf-8") as f:
        result=dict(dual=[exhaustive_dual(2),exhaustive_dual(3)],
                    exceptional_budget_endpoint_checks=checked,
                    triple_choices_verified=len(small)**3,
                    six_block_small_factor_exact=6*72+8)
        json.dump(result,f,indent=2);f.write("\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
