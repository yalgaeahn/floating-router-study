# Study numerical conventions

### Floating-island Fors assignment

The floating-island router has a model-specific Exact reference. At every flux point
the complete retained tensor-product Hamiltonian is diagonalized, and
`abs(eigenvectors)` defines the bare–dressed similarity matrix. All retained bare
states are receivers and all dressed states are proposers in a deterministic,
dressed-proposer Gale–Shapley match. The computational labels `00`, `10`, `01`, and
`11` are read from that full matching. No anchor, sweep direction, adjacent-point
eigenvector, or parity block participates in this default assignment.

The stored diagnostics include the full receiver-to-proposer indices, assigned bare
overlap and squared overlap, row/column/confidence margins, computational-branch
overlaps, one-excitation subspace weight, and Löwdin error. A point is `ambiguous`
when either the minimum computational overlap or one-excitation subspace weight is
below 0.5, or a reported result is non-finite. It is `reliable` when both weights are
at least 0.75 and `borderline` otherwise. The thresholds are configurable for
diagnostic studies, with 0.5 and 0.75 as the public defaults.

The full dressed eigenvalue spectrum is stored relative to the physical ground at
each point. One- and two-excitation manifold plots select the dressed eigenvalue
assigned to fixed bare tensor-product labels by that same full-basis match. An
assigned eigen-index change breaks the plotted line, so the figure does not imply
continuation or branch tracking between flux points. Solid, dashed, and `x` styles
denote assigned bare weights above 0.75, between 0.5 and 0.75, and below 0.5.
The interaction figure shades ambiguous Exact regions but retains their numerical
values as `x` markers; SW-invalid values are likewise retained as `x` markers.

The same pointwise Gale–Shapley rule is used throughout the codebase; no
adjacent-coordinate continuation or graph tracking is applied.
