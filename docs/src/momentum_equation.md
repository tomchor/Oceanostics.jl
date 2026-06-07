# Momentum equation

The `UMomentumEquation`, `VMomentumEquation`, and `WMomentumEquation` modules provide
diagnostics for every term on the right-hand side of the momentum equations. The momentum
equation for the ``i``-th velocity component ``u_i`` is

```math
\partial_t u_i = \underbrace{-\partial_j(u_j u_i)}_{\text{advection}}
               + \underbrace{\hat{g}_i \, b}_{\text{buoyancy}}
               + \underbrace{-\epsilon_{ijk} f_j u_k}_{\text{Coriolis}}
               + \underbrace{-\partial_i p}_{\text{pressure}}
               + \underbrace{-\partial_j \tau_{ij}}_{\text{viscous}}
               + \underbrace{(\nabla \times \mathbf{u}^S) \times \mathbf{u}}_{\text{Stokes shear}}
               + \underbrace{\partial_t u_i^S}_{\text{Stokes tendency}}
               + \underbrace{F_{u_i}}_{\text{forcing}}
```

where ``\hat{g}_i`` is the ``i``-th component of the gravitational unit vector, ``b`` is the
buoyancy, ``f_j`` is the Coriolis frequency, ``p`` is the pressure, ``\tau_{ij}`` is the
viscous/subgrid stress tensor, ``\mathbf{u}^S`` is the Stokes drift, and ``F_{u_i}`` is the
forcing. This decomposition lets the user compute each contribution independently, build
diagnostics like budget closure, or analyse the energetics of individual processes.

Each module wraps the corresponding Oceananigans velocity-tendency kernel and provides
diagnostics at the natural grid location for that velocity component:

| Module | Location | Tendency wrapped |
| --- | --- | --- |
| `UMomentumEquation` | `(Face, Center, Center)` | `u_velocity_tendency` (NH) / `hydrostatic_free_surface_u_velocity_tendency` (HFS) |
| `VMomentumEquation` | `(Center, Face, Center)` | `v_velocity_tendency` (NH) / `hydrostatic_free_surface_v_velocity_tendency` (HFS) |
| `WMomentumEquation` | `(Center, Center, Face)` | `w_velocity_tendency` (NH only) |

Constructors accept either a full Oceananigans `model` object (the convenience form) or
the individual arguments expected by the underlying kernel function. The `model`-only form
is enough for most analysis workflows.

## Example

```jldoctest momentum_eq
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b, coriolis=FPlane(f=1e-4));

julia> ADV = UMomentumEquation.Advection(model)
UAdvection (KernelFunctionOperation) at (Face, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: div_𝐯u (generic function with 4 methods)
└── arguments: ("Centered", "NamedTuple", "Field")
└── computes: advection of u-momentum  ∂ⱼ(uⱼu)

julia> BUOY = UMomentumEquation.BuoyancyAcceleration(model)
UBuoyancyAcceleration (KernelFunctionOperation) at (Face, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: x_dot_g_bᶠᶜᶜ (generic function with 3 methods)
└── arguments: ("BuoyancyForce", "NamedTuple")
└── computes: buoyancy acceleration (x)  ĝₓ b

julia> COR = VMomentumEquation.CoriolisAcceleration(model)
VCoriolisAcceleration (KernelFunctionOperation) at (Center, Face, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: y_f_cross_U (generic function with 11 methods)
└── arguments: ("FPlane", "NamedTuple")
└── computes: Coriolis acceleration (y)  (f⃗ × u⃗)_y
```

## Budget closure

The combined RHS — `Tendency(model)` — wraps Oceananigans' internal `*_velocity_tendency`
kernel directly. The sum of the individual term diagnostics, taken with the correct
signs from Oceananigans' source, equals `Tendency` to machine precision. For
`NonhydrostaticModel` the u/v formula is

```math
\text{Tendency} = -\text{Advection} + \text{BuoyancyAcceleration}
                  -\text{CoriolisAcceleration}
                  -\text{PressureGradient}
                  -\text{TotalViscousDissipation}
                  +\text{StokesShear} + \text{StokesTendency} + \text{Forcing}
```

The w-momentum equation is similar but with one subtlety: Oceananigans' `w_velocity_tendency`
has no explicit `-\partial_z p` line. The vertical hydrostatic balance is treated as a
property of the pressure-projection step, and whether the buoyancy term appears in the
tendency depends on whether the model uses hydrostatic/nonhydrostatic pressure splitting:

| `hydrostatic_pressure_anomaly` | Buoyancy term in `Tendency` | Budget formula |
| --- | --- | --- |
| `Field` (default, splitting on) | absorbed by ``pHY'`` | `-ADV - COR - TVISC + SS + ST + FORC` |
| `nothing` (splitting off) | included as ``+b`` | `-ADV + BUOY - COR - TVISC + SS + ST + FORC` |

## `HydrostaticFreeSurfaceModel` caveats

- `WMomentumEquation.Tendency`, `WMomentumEquation.Forcing(model, Val(:w))`, and the
  U/V/W `StokesShear` and `StokesTendency` throw an `ArgumentError` on
  `HydrostaticFreeSurfaceModel`: `w` is diagnosed from continuity rather than evolved by a
  prognostic equation, and HFS has no `stokes_drift` field.
- `Advection(::HydrostaticFreeSurfaceModel)` wraps `div_𝐯u`/`div_𝐯v` (the flux-form
  kernel). HFS's default `VectorInvariant` momentum advection uses `U_dot_∇u` instead, so
  to use the `Advection` diagnostic with HFS you must build the model with a flux-form
  scheme, e.g. `momentum_advection = Centered()`.
- For HFS, `Tendency` wraps `hydrostatic_free_surface_*_velocity_tendency`, which differs
  from the NH tendency: it has no Stokes terms, includes a barotropic free-surface
  pressure gradient, and a grid-slope contribution. Budget closure for HFS is not yet
  supported by this module — it would require additional diagnostics for those terms.

## `Forcing` dispatch quirk

`Forcing` aliases `KernelFunctionOperation` directly (no narrowing on the kernel function),
so its constructor methods are shared across the three modules. To avoid clobbering
`UMomentumEquation.Forcing(model)` when the V and W modules are also loaded, the V/W
single-argument convenience form takes an explicit `Val` tag:

```julia
UMomentumEquation.Forcing(model)              # u-momentum forcing
VMomentumEquation.Forcing(model, Val(:v))     # v-momentum forcing
WMomentumEquation.Forcing(model, Val(:w))     # w-momentum forcing
```

The full explicit form `Forcing(model, forcing, clock, model_fields, Val(:u/:v/:w))` is
available in all three modules.

## U-momentum

```@docs
Oceanostics.UMomentumEquation.Advection
Oceanostics.UMomentumEquation.BuoyancyAcceleration
Oceanostics.UMomentumEquation.CoriolisAcceleration
Oceanostics.UMomentumEquation.PressureGradient
Oceanostics.UMomentumEquation.ViscousDissipation
Oceanostics.UMomentumEquation.ImmersedViscousDissipation
Oceanostics.UMomentumEquation.TotalViscousDissipation
Oceanostics.UMomentumEquation.StokesShear
Oceanostics.UMomentumEquation.StokesTendency
Oceanostics.UMomentumEquation.Forcing
Oceanostics.UMomentumEquation.Tendency
```

## V-momentum

```@docs
Oceanostics.VMomentumEquation.Advection
Oceanostics.VMomentumEquation.BuoyancyAcceleration
Oceanostics.VMomentumEquation.CoriolisAcceleration
Oceanostics.VMomentumEquation.PressureGradient
Oceanostics.VMomentumEquation.ViscousDissipation
Oceanostics.VMomentumEquation.ImmersedViscousDissipation
Oceanostics.VMomentumEquation.TotalViscousDissipation
Oceanostics.VMomentumEquation.StokesShear
Oceanostics.VMomentumEquation.StokesTendency
Oceanostics.VMomentumEquation.Forcing
Oceanostics.VMomentumEquation.Tendency
```

## W-momentum

```@docs
Oceanostics.WMomentumEquation.Advection
Oceanostics.WMomentumEquation.BuoyancyAcceleration
Oceanostics.WMomentumEquation.CoriolisAcceleration
Oceanostics.WMomentumEquation.ViscousDissipation
Oceanostics.WMomentumEquation.ImmersedViscousDissipation
Oceanostics.WMomentumEquation.TotalViscousDissipation
Oceanostics.WMomentumEquation.StokesShear
Oceanostics.WMomentumEquation.StokesTendency
Oceanostics.WMomentumEquation.Forcing
Oceanostics.WMomentumEquation.Tendency
```
