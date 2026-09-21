---
paths:
  - "src/PotentialEnergyEquation.jl"
  - "src/BackgroundPotentialEnergyEquation.jl"
  - "src/AvailablePotentialEnergyEquation.jl"
  - "src/FilteredAvailablePotentialEnergyEquation.jl"
  - "src/SubFilterAvailablePotentialEnergyEquation.jl"
  - "test/test_pe_diagnostics.jl"
  - "test/test_ape_diagnostics.jl"
  - "test/test_filtered_ape_diagnostics.jl"
  - "test/test_subfilter_ape_diagnostics.jl"
  - "docs/src/*potential_energy_equation.md"
  - "docs/src/validation/reference_state.md"
  - "docs/examples/baroclinic_adjustment.jl"
  - "docs/examples/lock_release.jl"
---
# Potential, background and available potential energy

## Budget terms
A term that weights buoyancy by `z` or sorts along it needs `BuoyancyTracer` and a `NegativeZDirection` gravity; check both at construction with `validate_buoyancy_is_a_tracer` and `validate_gravity_is_z_aligned`.
`DiffusiveVerticalBuoyancyFlux` and `PotentialToKineticEnergyConversion` skip the gravity check on purpose, since neither depends on the direction of gravity.
Every `(model, z✶)` method calls `validate_gravity_is_z_aligned` itself, because a `z✶` built from a bare `Field` carries no model for `reference_height` to check; `test_tilted_gravity_is_rejected` pins each one.
A `BackgroundField` buoyancy adds `-z∂ⱼ(uⱼB)`, which has no diagnostic yet, so the `-z ×` split closes only without one; `Tendency` still includes it, since it comes off the model's own kernel.
The exact `-z ×` split and the two integral identities are tested only in `test/test_pe_diagnostics.jl`; `baroclinic_adjustment.jl` closes the integrated budget with the two conversions and never exercises the `-z ×` terms.
Keep `baroclinic_adjustment.jl` a double front: a single Eady front needs a uniform background gradient, which makes `∫eₚ dV` infinite (#279).

## Reference state
`reference_height` returns a `Field` over a `SortedReferenceState` that `compute!` rebuilds, the way `Scan` backs `Integral`; sorting couples every cell to every other, so it cannot be a `KernelFunctionOperation`.
No method runs on an `ImmersedBoundaryGrid`, and `HeavisideIntegral` is the only method that accepts a stretched grid (`validate_grid_for_method`).
Profile arrays live on the field's architecture: set one element with a one-element broadcast (`@views f[1:1] .= x`) and check order with `is_nondecreasing`, never `setindex!` or `issorted`, which are scalar indexing on a GPU (#296).
Read b✶(z) through `reference_buoyancy_at_height(z✶)`, which reuses the sort behind `z✶`, so b✶(z) stays the derivative of the `Ψ` that eₐ integrates rather than a second reconstruction.
A diagnostic that is a map on the model grid (Υ, bᵣ, wbᵣ, εₐ) rejects `VerticalSort` through `validate_reference_height_grid` and defaults to `HeavisideIntegral`.
`AvailablePotentialToKineticEnergyConversion` (wbᵣ) forms its product on the same face as `PotentialToKineticEnergyConversion` (wb), which is what makes wb = wbᵣ + wb✶ hold cell by cell; keep a new conversion on that discretization.

## Filtered and subfilter APE
The filtered buoyancy is looked up in a profile it did not produce, so `method` must be a `ProfileLookup`; `shared_profile_lookup` turns the default into a `VerticalSort` column of the model's buoyancy.
Build `z✶` and `z✶ˡ` against one lookup (`filtered_buoyancy_and_lookup`, `subfilter_reference_heights`); with two lookups `filter(eₐ) − eₐˡ` is a difference of unrelated quantities, not a decomposition.
`b_rˡ = b̄ − b✶(z)` keeps the reference profile unfiltered, which is what differentiating eₐˡ in z produces; `filter(b_r)` differs from it once the filter acts vertically.
`eₐˢ ≥ 0` holds pointwise only for a filter with no vertical component (Jensen); `test_subfilter_ape_signs` marks the vertical and 3D cases `broken` for that reason.
