# Extension complexity is not additive under Cartesian products

Code and finite certificates accompanying the paper by Minhao Liu. [Repository](https://github.com/chineseschrodingercat/product-extension-complexity)

The manuscript proves its lower bounds for arbitrary real nonnegative factorizations. The scripts below verify the finite inputs and explicit certificates. Run them from the directory containing this README, with a fresh output directory. A working configuration uses Python 3.13, SymPy 1.14.0, NumPy, pycddlib 3.0.2 with GMP, and python-flint 0.9.0. The logical results in the reference JSON files can be compared directly after parsing.

## The example with complexities 40 and 79

~~~sh
python verify_construction.py --outdir replay
python check_lower_bounds.py --out replay/lower_bounds.json
python independent_check.py replay/polytope.json --full --out replay/full_hull.json
python check_full_slack.py replay/polytope.json --out replay/full_slack.json
~~~

- **Coordinates.** The supplied `polytope.json` lists 39 points and the 37 facet inequalities of the 11-dimensional base. Keep the 24 coordinate facets. For each of the remaining 13, introduce a variable between zero and its slack. The resulting 24-dimensional polytope has 50 facets and 150 vertices.
- **Lower bounds.** The standard-library checker verifies the deficient-star counts, cycle identities, 2,701 exceptional-assignment patterns and rational error estimates. It checks the star covers of sizes 12 and 24. It also enumerates the rectangles on the nine small targets. Two rectangles cover at most seven cells.
- **Hull and full slack.** The independent cddlib calculation recovers the same 150 vertices from the 50 halfspaces. Integer products of widths 40 and 79 reproduce both complete slack matrices, including all 2,250,000 entries for the square. The `results` directory contains reference outputs. The field `extension_complexity_square_upper_bound` reports the explicit factorization. The matching lower bound is proved in the manuscript.

## Unbounded deficits

~~~sh
python unbounded/verify_family.py --n 6 --out replay/checks_n6.json
python unbounded/verify_family.py --n 27 --out replay/checks_n27.json
python unbounded/verify_counting.py --out replay/counting_certificate.json
python unbounded/verify_unbounded_note.py
python unbounded/check_identical_factors.py --out replay/identical_factors.json
~~~

- **Two different factors.** The SymPy family checker verifies the graph ranks, star and label identities, conditional capacities, and error estimates at the requested parameter. The independent four-block program exhausts label profiles and computes the dual conditional bound in rational arithmetic. A separate python-flint script, `unbounded/verify_unbounded_note.py`, tests every coordinate row individually, including all 2,808 facet-direction rank calculations in the two bases. Its reference output is `unbounded/flint_reference_output.txt`.
- **Identical factors.** The standard-library checker enumerates all 169,911 candidate covers of size five on the first two alphabets, of sizes three and seven, and verifies a cover of size six. It tests the paired construction on all 11,025 input pairs, connectivity after every coarse-node deletion in both graphs, and the inequalities used in the counting lemma for selected parameters. It also verifies the three-block fractional certificate of total weight 32/9 in the four-label construction. The rank entries in the output apply the proved lemmas to the checked combinatorial counts.
- **Reference outputs.** The JSON and text files in `unbounded` accompany these checks. The general conclusions about every parameter are analytic. The constructions with unbounded deficit specify polytopes through rational point hulls and omitted-facet wedges.
