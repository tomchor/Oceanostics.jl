# Flow diagnostics

The `FlowDiagnostics` module provides a collection of commonly-used diagnostics for
characterizing the state of ocean and turbulent flows. These are not budget terms
from a specific governing equation but rather derived quantities that summarize
important flow properties at a glance.

The module includes:

- **Stability parameters**: the gradient Richardson number (``Ri``) comparing
  stratification to shear, useful for predicting shear instabilities (``Ri < 0.25``).
- **Rotation diagnostics**: the Rossby number (``Ro``), which measures relative
  vorticity against planetary vorticity and indicates whether flow dynamics are
  geostrophically balanced or ageostrophic.
- **Potential vorticity**: the Ertel PV (``\boldsymbol{\omega}_{\text{tot}} \cdot \nabla b``),
  a materially conserved quantity under adiabatic, inviscid conditions that is
  fundamental for understanding large-scale ocean dynamics. A thermal-wind-balance
  approximation and a directional decomposition are also provided.
- **Velocity gradient tensor diagnostics**: the full strain rate tensor ``S_{ij}``
  and its modulus (``\|S_{ij}\|``), the vorticity tensor modulus (``\|\Omega_{ij}\|``),
  and the ``Q``-criterion for vortex identification.
- **Mixed layer depth**: computed by scanning downward from the surface to find
  where buoyancy or density departs from the surface value by more than a
  user-specified threshold.
- **Bottom cell value**: extracts the value of any diagnostic at the bottommost
  active cell (respecting immersed boundaries).

## Example

```jldoctest flow_diag
julia> using Oceananigans, Oceanostics

julia> using Oceanostics: FlowDiagnostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b, coriolis=FPlane(f=1e-4));

julia> Ri = FlowDiagnostics.RichardsonNumber(model)
KernelFunctionOperation at (Center, Center, Face)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: richardson_number_ccf (generic function with 1 method)
└── arguments: ("Field", "Field", "Field", "Field", "Tuple")

julia> Ro = FlowDiagnostics.RossbyNumber(model)
KernelFunctionOperation at (Face, Face, Face)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: rossby_number_fff (generic function with 1 method)
└── arguments: ("Field", "Field", "Field", "NamedTuple")

julia> EPV = FlowDiagnostics.ErtelPotentialVorticity(model)
KernelFunctionOperation at (Face, Face, Face)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: ertel_potential_vorticity_fff (generic function with 1 method)
└── arguments: ("Field", "Field", "Field", "Field", "Int64", "Int64", "Float64")

julia> S = FlowDiagnostics.StrainRateTensorModulus(model)
KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: strain_rate_tensor_modulus_ccc (generic function with 1 method)
└── arguments: ("Field", "Field", "Field")
```

## Richardson number

The gradient Richardson number measures the ratio of stratification to shear:

```math
Ri = \frac{\partial b / \partial z}{|\partial \mathbf{u}_h / \partial z|^2}
```

where ``z`` is the true vertical (anti-parallel to gravity). Computed at `(Center, Center, Face)`.

```@docs
Oceanostics.FlowDiagnostics.RichardsonNumber
```

## Rossby number

The Rossby number measures the ratio of relative vorticity to planetary vorticity
in the direction of the rotation axis:

```math
Ro = \frac{\omega^z}{f}
```

Computed at `(Face, Face, Face)`.

```@docs
Oceanostics.FlowDiagnostics.RossbyNumber
```

## Potential vorticity

### Ertel potential vorticity

The full Ertel potential vorticity:

```math
\text{EPV} = \boldsymbol{\omega}_{\text{tot}} \cdot \nabla b
```

where ``\boldsymbol{\omega}_{\text{tot}}`` is the absolute (relative + planetary) vorticity.
Computed at `(Face, Face, Face)`.

Passing `thermal_wind = true` to `ErtelPotentialVorticity` returns the simplified PV that assumes
thermal wind balance:

```math
\text{EPV} = (f + \omega^z)\,\partial_z b - f\left[(\partial_z U)^2 + (\partial_z V)^2\right]
```

The result is a subtype of both `ErtelPotentialVorticity` and `ThermalWindPotentialVorticity`, so
the thermal-wind variant can still be identified by type.

```@docs
Oceanostics.FlowDiagnostics.ErtelPotentialVorticity
Oceanostics.FlowDiagnostics.ThermalWindPotentialVorticity
```

### Directional Ertel potential vorticity

The contribution to the Ertel PV from a single user-specified direction.

```@docs
Oceanostics.FlowDiagnostics.DirectionalErtelPotentialVorticity
```

## Velocity gradient tensor invariants

### Strain rate tensor

The (symmetric) strain rate tensor, returned as a `NamedTuple` of its independent components. Each
component is a `KernelFunctionOperation` evaluated at its natural location on the staggered grid
(the diagonal components at `(Center, Center, Center)`; the off-diagonals at the edge locations
`(Face, Face, Center)`, `(Face, Center, Face)`, and `(Center, Face, Face)`). The `dims` keyword
selects a sub-dimensional tensor — component ``S_{ij}`` is included only when both ``i`` and ``j``
are in `dims` — so `dims=(1, 3)` returns the 2D strain rate tensor in the ``x``–``z`` plane
(``S_{11}``, ``S_{33}``, ``S_{13}``).

```math
S_{ij} = \tfrac{1}{2}(\partial_j u_i + \partial_i u_j)
```

```@docs
Oceanostics.FlowDiagnostics.StrainRateTensor
```

### Strain rate tensor modulus

```math
\|S_{ij}\| = \sqrt{S_{ij} S_{ij}}, \qquad S_{ij} = \tfrac{1}{2}(\partial_j u_i + \partial_i u_j)
```

```@docs
Oceanostics.FlowDiagnostics.StrainRateTensorModulus
```

### Principal strain rates

The principal strain rates are the eigenvalues ``\lambda_1 \ge \lambda_2 \ge \lambda_3`` of the strain
rate tensor, returned as a `NamedTuple` at `(Center, Center, Center)`. They quantify stretching
(``\lambda > 0``) and compression (``\lambda < 0``) along the principal axes and are rotation
invariants of the tensor:

```math
\lambda_1 + \lambda_2 + \lambda_3 = \nabla\cdot\mathbf{u}, \qquad
\lambda_1^2 + \lambda_2^2 + \lambda_3^2 = S_{ij} S_{ij}
```

so for incompressible flow they sum to zero and their root-sum-of-squares equals ``\|S_{ij}\|``.

```@docs
Oceanostics.FlowDiagnostics.PrincipalStrainRates
```

### Vorticity tensor modulus

```math
\|\Omega_{ij}\| = \sqrt{\Omega_{ij} \Omega_{ij}}, \qquad \Omega_{ij} = \tfrac{1}{2}(\partial_j u_i - \partial_i u_j)
```

```@docs
Oceanostics.FlowDiagnostics.VorticityTensorModulus
```

### Q-criterion

```math
Q = \tfrac{1}{2}(\Omega_{ij}\Omega_{ij} - S_{ij}S_{ij})
```

Used to identify vortices in fluid flow; positive values indicate vortex-dominated regions.

```@docs
Oceanostics.FlowDiagnostics.QVelocityGradientTensorInvariant
```

## Mixed layer depth

```@docs
Oceanostics.FlowDiagnostics.MixedLayerDepth
Oceanostics.FlowDiagnostics.BuoyancyAnomalyCriterion
Oceanostics.FlowDiagnostics.DensityAnomalyCriterion
```

## Bottom cell value

```@docs
Oceanostics.FlowDiagnostics.BottomCellValue
```
