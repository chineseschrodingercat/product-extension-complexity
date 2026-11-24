"""Exact finite checks for the separated-alphabet proof. Uses the standard library."""
import argparse
import json
from fractions import Fraction
from itertools import combinations, product
from math import comb
from pathlib import Path


def family(j):
    return list(combinations(range(2 * j - 1), j))


def check_node_deletions(j):
    sets = family(j)
    alphabet = 2 * j - 1
    neighbors = [set() for _ in range(alphabet + len(sets))]
    for index, subset in enumerate(sets, start=alphabet):
        for label in subset:
            neighbors[label].add(index)
            neighbors[index].add(label)

    for excluded in range(len(neighbors)):
        start = 0 if excluded != 0 else 1
        reached = {start}
        pending = [start]
        while pending:
            for node in neighbors[pending.pop()] - reached - {excluded}:
                reached.add(node)
                pending.append(node)
        assert len(reached) == len(neighbors) - 1
    return {
        "j": j,
        "coarse_nodes_checked": len(neighbors),
        "coarse_edges": sum(map(len, sets)),
        "maximum_degree": max(map(len, neighbors)),
    }


def check_first_pair():
    sizes = (2, 4)
    alphabets = [list(range(2 * j - 1)) for j in sizes]
    inputs = list(product(*(family(j) for j in sizes)))
    profiles = [
        (x, y) for x, y in product([None] + alphabets[0], [None] + alphabets[1])
        if (x, y) != (None, None)
    ]
    # One bit per (input, demanded block). Duplicate profiles add no coverage.
    masks = {}
    for profile in profiles:
        mask = 0
        for index, word in enumerate(inputs):
            if all(label is None or label in subset for label, subset in zip(profile, word)):
                for block, label in enumerate(profile):
                    if label is not None:
                        mask |= 1 << (2 * index + block)
        masks[profile] = mask
    target = (1 << (2 * len(inputs))) - 1
    tested = 0
    for a, b, c, d, e in combinations(masks.values(), 5):
        assert (a | b | c | d | e) != target
        tested += 1
    assert tested == comb(len(profiles), 5)
    separate = [(x, None) for x in range(2)] + [(None, y) for y in range(4)]
    witness = 0
    for profile in separate:
        witness |= masks[profile]
    assert witness == target
    # Each twin profile specifies the very same label in the matching blocks.
    paired_tests = 0
    for left, right in product(inputs, repeat=2):
        for block, alphabet in enumerate(alphabets):
            assert any(s in left[block] and s in right[block] for s in alphabet)
        paired_tests += 1
    return {
        "j_list": list(sizes), "possible_profiles": len(profiles),
        "single_inputs": len(inputs), "demands": 2 * len(inputs),
        "five_profile_candidates_exhausted": tested,
        "individual_cover_number": len(separate),
        "square_inputs_checked": paired_tests,
        "shared_square_profiles_sufficient": sum(map(len, alphabets)),
    }


def check_counting_constraints():
    # Any too-small cover must satisfy all three necessary conditions below.
    # Integer loops validate the arithmetic in the boundary cases j=2C, 2C+1.
    tried = 0
    for old_cost in range(1, 16):
        for j in (2 * old_cost, 2 * old_cost + 1):
            for earlier in range(old_cost + j):
                for last in range(old_cost + j - earlier):
                    for mixed in range(old_cost + j - earlier - last):
                        tried += 1
                        fixed_last = earlier + max(0, mixed - (j - 1)) >= old_cost
                        hits_last = last + mixed >= j
                        mean_passes = 3 * last + 2 * mixed >= 3 * j
                        assert not (fixed_last and hits_last and mean_passes)
    return tried


def check_fractional_witness():
    sets = list(combinations(range(4), 3))
    profiles = list(product(range(4), repeat=3))
    constraints = 0
    for demanded in range(3):
        others = [i for i in range(3) if i != demanded]
        for fixed in product(sets, repeat=2):
            passes = sum(all(p[i] in s for i, s in zip(others, fixed)) for p in profiles)
            assert Fraction(passes, 18) == 2
            constraints += 1
    total = Fraction(len(profiles), 18)
    assert total == 3 * Fraction(32, 27)
    return {"input_sets": len(sets), "constraints_checked": constraints,
            "profiles": len(profiles), "weight_per_profile": "1/18", "total": str(total)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    graphs = [check_node_deletions(j) for j in (2, 4)]
    pair = check_first_pair()
    clones = max(g["maximum_degree"] for g in graphs) + max(pair["j_list"])
    stars = clones * sum(g["coarse_edges"] for g in graphs)
    result = {
        "first_two_blocks": pair,
        "node_deletions": graphs,
        "counting_triples_excluded": check_counting_constraints(),
        "three_block_fractional_witness_for_four_label_family": check_fractional_witness(),
        "clone_count_per_edge": clones, "individual_stars": stars,
        "rank_from_proved_lemmas": stars + pair["individual_cover_number"],
        "square_rank_upper_bound_from_construction": 2 * stars + pair["shared_square_profiles_sufficient"],
    }
    with args.out.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2)
        out.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
