---
paths:
  - "src/SpatialFilters/**"
  - "src/FilteredKineticEnergyEquation.jl"
  - "src/SubFilterKineticEnergyEquation.jl"
  - "src/FilteredAvailablePotentialEnergyEquation.jl"
  - "src/SubFilterAvailablePotentialEnergyEquation.jl"
  - "test/test_spatial_filters.jl"
  - "test/test_filtered_*.jl"
  - "test/test_subfilter_*.jl"
  - "docs/src/filters.md"
  - "docs/src/filtered_*.md"
  - "docs/src/subfilter_*.md"
  - "docs/examples/spatial_filtering.jl"
  - "docs/examples/kelvin_helmholtz.jl"
  - "docs/examples/rayleigh_taylor_instability.jl"
---
# Spatial filters and the filtered and subfilter budgets

## Filters
Size a 1D kernel's in-range check with `stencil_length(grid, d, ψ)`, the operand's own extent, never `size(grid, d)`, which misses the last face of a `Face` operand along a `Bounded` direction (#293).
Measure that extent on the host and carry it into the kernel in `SizedBoundary`; on a GPU the kernel's `ψ` is the adapted operand, which has no location, so recomputing the extent there makes every filtered direction one cell long (#296).
Check a filter change on a field that varies in space: a constant field hides both the one-cell shift and the lost taps, which is why CPU-only CI never caught the extent bug.
The recursive (fused) kernel methods reach the operand through `fargs[end]`, since it is the last argument at every level of the recursion.
A filter is staged (one 1D pass per direction) only when it is the direct operand of a `Field`; inside another operation it falls back to the fused `Nᵈ`-point kernel with no warning (`filters.md` §Performance notes).
`test_perf_invariants.jl` requires the staged 3D wide-stencil path to be at least twice as fast as the fused one, so a change that disables staging fails there.

## Filtered and subfilter diagnostics
Materialize a filtered field with `Field(...)` before reusing it, as `filtered_velocities` does, so it is computed once and takes the staged path.
A filtered diagnostic that reuses a full-field kernel goes through a forwarding kernel with its own name (`filtered_ape_ccc` → `local_ape_ccc`, `filtered_upsilon_ccc` → `upsilon_ccc`), so its alias gets its own type and `@diagnostic_show` display.
A subfilter diagnostic is `filter(full) − filtered` on the same discretization, and for APE on the same profile, which is what makes the decomposition exact; it is returned as a `KernelFunctionOperation` so it gets its own alias and display.
A filtered dissipation rate contracts the filtered flux (`τ̄ᵢⱼ = filter(τᵢⱼ)`, `q̄ᵢ = filter(qᵢ)`) rather than a flux recomputed from the filtered field; the two agree only for a constant viscosity or diffusivity.
`subfilter_covariance` takes a pre-filtered factor through `filtered_a`, so several covariances can share one filtered field.
An identity-scale Gaussian (σ ≪ Δx, `N = 3`) makes every filtered diagnostic equal its full-field counterpart and every subfilter one vanish, to the bit; the filtered and subfilter suites use it to check the kernels without reimplementing them.
