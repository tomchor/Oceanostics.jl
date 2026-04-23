# Flow diagnostics

The `FlowDiagnostics` module provides common diagnostics for characterizing
ocean and turbulent flows.

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

```@docs
Oceanostics.FlowDiagnostics.ErtelPotentialVorticity
```

### Thermal wind potential vorticity

A simplified PV assuming thermal wind balance:

```math
\text{TWPV} = (f + \omega^z)\,\partial_z b - f\left[(\partial_z U)^2 + (\partial_z V)^2\right]
```

```@docs
Oceanostics.FlowDiagnostics.ThermalWindPotentialVorticity
```

### Directional Ertel potential vorticity

The contribution to the Ertel PV from a single user-specified direction.

```@docs
Oceanostics.FlowDiagnostics.DirectionalErtelPotentialVorticity
```

## Velocity gradient tensor invariants

### Strain rate tensor modulus

```math
\|S_{ij}\| = \sqrt{S_{ij} S_{ij}}, \qquad S_{ij} = \tfrac{1}{2}(\partial_j u_i + \partial_i u_j)
```

```@docs
Oceanostics.FlowDiagnostics.StrainRateTensorModulus
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
