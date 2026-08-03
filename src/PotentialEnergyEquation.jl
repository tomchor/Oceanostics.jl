module PotentialEnergyEquation

using DocStringExtensions

export PotentialEnergy
# Short name inside the module, prefixed alias for `using Oceanostics`, as elsewhere in the package.
export DiffusiveBuoyancyFlux, PotentialEnergyDiffusiveBuoyancyFlux
# `wb` is the one term the kinetic and potential energy budgets share, so it is defined in
# `KineticEnergyEquation` and re-exported here under both its own name and a budget-neutral alias.
export PotentialToKineticEnergyConversion, KineticEnergyConversion

using Oceananigans: fields
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Models: seawater_density
using Oceananigans.Models: model_geopotential_height
using Oceananigans.Grids: Center, Face
using Oceananigans.Grids: NegativeZDirection
using Oceananigans.BuoyancyFormulations: BuoyancyForce, BuoyancyTracer, SeawaterBuoyancy, LinearEquationOfState
using Oceananigans.BuoyancyFormulations: buoyancy_perturbationᶜᶜᶜ, Zᶜᶜᶜ
using Oceananigans.Models: ShallowWaterModel
using Oceananigans.Operators: ℑzᵃᵃᶜ
using Oceananigans.TurbulenceClosures: diffusive_flux_z
using Oceanostics: validate_location, CustomKFO
using SeawaterPolynomials: BoussinesqEquationOfState

using ..KineticEnergyEquation: PotentialToKineticEnergyConversion

# `uᵢbᵢ` is the source of kinetic energy and, with the sign flipped, the buoyancy conversion term of the
# `e_p` equation. It is defined in `KineticEnergyEquation`, whose local alias `PotentialEnergyConversion`
# names the reservoir the energy comes from; this one names where it goes. Each module names the other
# side, so the term reads correctly whichever budget is being written. It is exported from this module
# and from `AvailablePotentialEnergyEquation`, but not from `Oceanostics`, where unprefixed it would say
# nothing about which budget it belongs to.
const KineticEnergyConversion = PotentialToKineticEnergyConversion

const NoBuoyancyModel = Union{Nothing, ShallowWaterModel}
const BuoyancyTracerModel = BuoyancyForce{<:BuoyancyTracer, g} where g
const LinearSeawaterBuoyancy = SeawaterBuoyancy{FT, <:LinearEquationOfState, T, S} where {FT, T, S}
const BuoyancyLinearEOSModel = BuoyancyForce{<:LinearSeawaterBuoyancy, g} where {g}
const BoussinesqSeawaterBuoyancy = SeawaterBuoyancy{FT, <:BoussinesqEquationOfState, T, S} where {FT, T, S}
const BuoyancyBoussinesqEOSModel = BuoyancyForce{<:BoussinesqSeawaterBuoyancy, g} where {g}

# Inline functions for potential energy calculation
@inline minus_bz_ccc(i, j, k, grid, b) = -b[i, j, k] * Zᶜᶜᶜ(i, j, k, grid)
@inline minus_bz_ccc(i, j, k, grid, b::LinearSeawaterBuoyancy, C) = -buoyancy_perturbationᶜᶜᶜ(i, j, k, grid, b, C) * Zᶜᶜᶜ(i, j, k, grid)
@inline minus_bz_ccc(i, j, k, grid, ρ, p) = (p.g / p.ρ₀) * ρ[i, j, k] * Zᶜᶜᶜ(i, j, k, grid)

# Type aliases for major functions
const PotentialEnergy = CustomKFO{<:typeof(minus_bz_ccc)}

validate_gravity_unit_vector(gravity_unit_vector::NegativeZDirection) = nothing
validate_gravity_unit_vector(gravity_unit_vector) =
    throw(ArgumentError("`PotentialEnergy` is curently only defined for models that have a `NegativeZDirection` gravity unit vector."))

"""
    $(SIGNATURES)

Return a `KernelFunctionOperation` to compute the `PotentialEnergy` per unit volume,
```math
eₚ = \\frac{gρ}{ρ₀}z = -bz
```
at each grid `location` in `model`. `PotentialEnergy` is defined for both `BuoyancyTracer`
and `SeawaterBuoyancy`. See the relevant Oceananigans.jl documentation on
[buoyancy models](https://clima.github.io/OceananigansDocumentation/dev/model_setup/buoyancy_and_equation_of_state/)
for more information about available options.

The optional keyword argument `geopotential_height` is only used
if one wishes to calculate `eₚ` with a potential density referenced to `geopotential_height`,
rather than in-situ density, when using a `BoussinesqEquationOfState`.

Example
=======

Usage with a `BuoyancyTracer` buoyacny model
```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=100, z=(-1000, 0), topology=(Flat, Flat, Bounded))
1×1×100 RectilinearGrid{Float64, Flat, Flat, Bounded} on CPU with 0×0×3 halo
├── Flat x
├── Flat y
└── Bounded  z ∈ [-1000.0, 0.0] regularly spaced with Δz=10.0

julia> model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=(:b,))
NonhydrostaticModel{CPU, RectilinearGrid}(time = 0 seconds, iteration = 0)
├── grid: 1×1×100 RectilinearGrid{Float64, Flat, Flat, Bounded} on CPU with 0×0×3 halo
├── timestepper: RungeKutta3TimeStepper
├── advection scheme: Centered(order=2)
├── tracers: b
├── closure: Nothing
├── buoyancy: BuoyancyTracer with ĝ = NegativeZDirection()
└── coriolis: Nothing

julia> PotentialEnergyEquation.PotentialEnergy(model)
PotentialEnergy KernelFunctionOperation at (Center, Center, Center)
├── grid: 1×1×100 RectilinearGrid{Float64, Flat, Flat, Bounded} on CPU with 0×0×3 halo
├── kernel_function: minus_bz_ccc (generic function with 3 methods)
└── arguments: ("Field",)
└── computes: potential energy per unit volume  eₚ = -bz
```

The default behaviour of `PotentialEnergy` uses the *in-situ density* in the calculation
when the equation of state is a `BoussinesqEquationOfState`:
```jldoctest
julia> using Oceananigans, SeawaterPolynomials.TEOS10, Oceanostics

julia> grid = RectilinearGrid(size=100, z=(-1000, 0), topology=(Flat, Flat, Bounded))
1×1×100 RectilinearGrid{Float64, Flat, Flat, Bounded} on CPU with 0×0×3 halo
├── Flat x
├── Flat y
└── Bounded  z ∈ [-1000.0, 0.0] regularly spaced with Δz=10.0

julia> tracers = (:T, :S)
(:T, :S)

julia> eos = TEOS10EquationOfState()
BoussinesqEquationOfState{Float64}:
├── seawater_polynomial: TEOS10SeawaterPolynomial{Float64}
└── reference_density: 1020.0

julia> buoyancy = SeawaterBuoyancy(equation_of_state=eos)
SeawaterBuoyancy{Float64}:
├── gravitational_acceleration: 9.80665
└── equation_of_state: BoussinesqEquationOfState{Float64}

julia> model = NonhydrostaticModel(grid; buoyancy, tracers)
NonhydrostaticModel{CPU, RectilinearGrid}(time = 0 seconds, iteration = 0)
├── grid: 1×1×100 RectilinearGrid{Float64, Flat, Flat, Bounded} on CPU with 0×0×3 halo
├── timestepper: RungeKutta3TimeStepper
├── advection scheme: Centered(order=2)
├── tracers: (T, S)
├── closure: Nothing
├── buoyancy: SeawaterBuoyancy with g=9.80665 and BoussinesqEquationOfState{Float64} with ĝ = NegativeZDirection()
└── coriolis: Nothing

julia> PotentialEnergyEquation.PotentialEnergy(model)
PotentialEnergy KernelFunctionOperation at (Center, Center, Center)
├── grid: 1×1×100 RectilinearGrid{Float64, Flat, Flat, Bounded} on CPU with 0×0×3 halo
├── kernel_function: minus_bz_ccc (generic function with 3 methods)
└── arguments: ("KernelFunctionOperation", "NamedTuple")
└── computes: potential energy per unit volume  eₚ = -bz
```

To use a reference density set a constant value for the keyword argument `geopotential_height`
and pass this the function. For example,
```jldoctest
julia> using Oceananigans, SeawaterPolynomials.TEOS10, Oceanostics

julia> grid = RectilinearGrid(size=100, z=(-1000, 0), topology=(Flat, Flat, Bounded));

julia> tracers = (:T, :S);

julia> eos = TEOS10EquationOfState();

julia> buoyancy = SeawaterBuoyancy(equation_of_state=eos);

julia> model = NonhydrostaticModel(grid; buoyancy, tracers);

julia> geopotential_height = 0; # density variable will be σ₀

julia> PotentialEnergyEquation.PotentialEnergy(model)
PotentialEnergy KernelFunctionOperation at (Center, Center, Center)
├── grid: 1×1×100 RectilinearGrid{Float64, Flat, Flat, Bounded} on CPU with 0×0×3 halo
├── kernel_function: minus_bz_ccc (generic function with 3 methods)
└── arguments: ("KernelFunctionOperation", "NamedTuple")
└── computes: potential energy per unit volume  eₚ = -bz
```
"""
@inline function PotentialEnergy(model; location = (Center, Center, Center),
                                 geopotential_height = model_geopotential_height(model))

    validate_location(location, "PotentialEnergy")
    isnothing(model.buoyancy) ? nothing : validate_gravity_unit_vector(model.buoyancy.gravity_unit_vector)

    return PotentialEnergy(model, model.buoyancy, geopotential_height)
end

@inline PotentialEnergy(model, buoyancy_model::NoBuoyancyModel, geopotential_height) =
    throw(ArgumentError("Cannot calculate gravitational potential energy without a Buoyancy model."))

@inline function PotentialEnergy(model, buoyancy_model::BuoyancyTracerModel, geopotential_height)

    grid = model.grid
    b = model.tracers.b

    return KernelFunctionOperation{Center, Center, Center}(minus_bz_ccc, grid, b)
end

@inline function PotentialEnergy(model, buoyancy_model::BuoyancyLinearEOSModel, geopotential_height)

    grid = model.grid
    C = model.tracers
    b = buoyancy_model.formulation

    return KernelFunctionOperation{Center, Center, Center}(minus_bz_ccc, grid, b, C)
end

@inline function PotentialEnergy(model, buoyancy_model::BuoyancyBoussinesqEOSModel, geopotential_height)

    grid = model.grid
    ρ = seawater_density(model; geopotential_height)
    parameters = (g = model.buoyancy.formulation.gravitational_acceleration,
                  ρ₀ = model.buoyancy.formulation.equation_of_state.reference_density)

    return KernelFunctionOperation{Center, Center, Center}(minus_bz_ccc, grid, ρ, parameters)
end

#+++ Buoyancy as a diffused tracer
# `Φ` here, and `ε_A` over in `AvailablePotentialEnergyEquation`, both read `κ∇b` off the closure's own
# diffusive flux, which needs the buoyancy to be a tracer the closure diffuses. `SeawaterBuoyancy` would
# need the fluxes of temperature and salinity combined through the equation of state. Both helpers live
# here because this is the module the two diagnostics have in common.
# `Oceanostics.validate_dissipative_closure` is the wrong tool here: it admits only
# `AbstractScalarDiffusivity`, so it would reject Smagorinsky, AMD and every other closure that defines
# `diffusive_flux_*` and works fine. What actually breaks is having no closure at all, which is a legal
# model and which otherwise surfaces as a `MethodError` from inside the kernel naming none of the
# caller's code.
validate_closure_supplies_a_flux(diagnostic, model) =
    isnothing(model.closure) &&
        throw(ArgumentError("`$diagnostic` reads `κ∇b` off the model's closure, but this model has no \
                             closure, so there is no diffusive flux to read. Build the model with one, \
                             e.g. `closure = ScalarDiffusivity(κ=...)`."))

validate_buoyancy_is_a_diffused_tracer(diagnostic, model) =
    model.buoyancy isa BuoyancyTracerModel ||
        throw(ArgumentError("`$diagnostic` needs the buoyancy to be a tracer the closure diffuses, so that `κ∇b` is \
                             the closure's own diffusive flux, but this model's buoyancy is a \
                             $(summary(model.buoyancy)). Only `BuoyancyTracer` is supported for now."))

# The arguments every diagnostic built on `diffusive_flux_*` passes through to the closure.
buoyancy_diffusive_flux_arguments(model) =
    (model.closure,
     model.closure_fields,
     Val(findfirst(n -> n === :b, propertynames(model.tracers))),
     model.tracers.b,
     model.clock,
     fields(model),
     model.buoyancy)
#---

#+++ Diffusive buoyancy flux
# `Φ = κ ∂b/∂z = -q₃`, the vertical diffusive buoyancy flux taken upward. It is the diffusive conversion
# term of the `e_p` equation, the only way diffusion changes the potential energy of a closed domain,
# and it is also the second of the two parts `ε_A` is written out of. It comes off the closure's own
# `diffusive_flux_z`, exactly as `ε_A` does, so the two always carry the same `κ`. The flux lives on the
# `z` face, so it is interpolated to the cell center.
@inline diffusive_buoyancy_flux_ccc(i, j, k, grid, args...) = -ℑzᵃᵃᶜ(i, j, k, grid, diffusive_flux_z, args...)

const DiffusiveBuoyancyFlux = CustomKFO{<:typeof(diffusive_buoyancy_flux_ccc)}
const PotentialEnergyDiffusiveBuoyancyFlux = DiffusiveBuoyancyFlux

"""
    $(SIGNATURES)

Return a `KernelFunctionOperation` computing the vertical diffusive buoyancy flux, taken upward,

```
    Φ = κ ∂b/∂z = -q₃ ,
```

which is the work diffusion does against gravity as it smooths the stratification. It is the diffusive
conversion term of the `e_p` equation, and the only way diffusion can change the potential energy of a
closed domain: the rest of the diffusive contribution is a flux divergence that integrates to zero. The
result lives at `(Center, Center, Center)`, per unit mass (units `m² s⁻³`).

`Φ` is also the second of the two parts
[`AvailablePotentialEnergyDissipationRate`](@ref Oceanostics.AvailablePotentialEnergyEquation.AvailablePotentialEnergyDissipationRate)
is written out of,

```
    ε_A = κ (∂z✶/∂b) |∇b|² - Φ ,
```

and it is the part that carries no available energy with it: `Φ` enters the `E_p` and `E_b` budgets
identically, so it cancels from their difference, which is `E_a`. A statically stable, horizontally
uniform stratification is its own reference state and has `ε_A = 0` cell by cell, so there `Φ` accounts
for the whole of the diapycnal mixing rate. Adding it back to `ε_A` recovers that rate in general, and
its volume integral is the rate
[`BackgroundPotentialEnergy`](@ref Oceanostics.BackgroundPotentialEnergyEquation.BackgroundPotentialEnergy)
grows:

```
    d/dt ∫e_b dV = ∫(ε_A + Φ) dV .
```

For a **constant** `κ`, `Φ` volume-integrates to a boundary term, since no buoyancy crosses the top or
the bottom of a closed domain:

```
    ∫Φ dV = κ A [b(z_top) - b(z_bottom)]        (constant κ only)
```

with `A` the domain's horizontal area, so the flow enters only through the buoyancy difference across
it. That collapse is exact rather than approximate: the discrete sum telescopes, which is also what
makes the cells against a wall report half of what the interior does, since a no-flux wall zeroes the
flux on the outer face and the cell center averages the two faces bounding it.

A `κ` that varies in space breaks the telescoping, and the boundary form is then simply wrong. What the
integral always is, is the flux summed over the interior `z` faces,

```
    ∫Φ dV = A Σ κ [b(above) - b(below)] ,
```

which collapses only when `κ` factors out of the sum. Under a depth-dependent diffusivity or any closure
that computes `κ` from the flow, the integral depends on the interior arrangement and can come out with
the opposite sign to `κ A [b(z_top) - b(z_bottom)]`, so do not reach for the boundary form as a check
there. `Φ` is returned as a field rather than as either number, since it is the pointwise partner of
`ε_A`.

`κ ∂b/∂z` is taken from the closure's own diffusive flux rather than from a diffusivity supplied here,
so this follows whatever closure the model runs with, including ones that compute `κ` from the flow.
That needs the buoyancy to be a tracer the closure diffuses, so this is defined for `BuoyancyTracer`
models only.

Not to be confused with `TracerEquation.ZDiffusiveFlux(model, :b)`, which is the same closure call read
raw: that one is `q₃` itself, down-gradient and on the `z` face. This is `-q₃`, interpolated to the cell
center, which are the sign and the location the energy budgets want. Reach for the tracer diagnostic
when you want the flux as a flux, and this one when you want it as a term of the `e_p` equation.

```jldoctest
using Oceananigans, Oceanostics

grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1), topology=(Periodic, Periodic, Bounded))
model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b, closure=ScalarDiffusivity(κ=1e-4))

PotentialEnergyDiffusiveBuoyancyFlux(model)   # or `DiffusiveBuoyancyFlux` inside the module

# output

PotentialEnergyDiffusiveBuoyancyFlux KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: diffusive_buoyancy_flux_ccc (generic function with 1 method)
└── arguments: ("ScalarDiffusivity", "Nothing", "Val", "Field", "Clock", "NamedTuple", "BuoyancyForce")
└── computes: diffusive buoyancy flux  Φ = κ ∂b/∂z = -q₃
```
"""
function DiffusiveBuoyancyFlux(model; location = (Center, Center, Center))
    validate_location(location, "DiffusiveBuoyancyFlux")
    validate_buoyancy_is_a_diffused_tracer("DiffusiveBuoyancyFlux", model)
    validate_closure_supplies_a_flux("DiffusiveBuoyancyFlux", model)

    return KernelFunctionOperation{Center, Center, Center}(diffusive_buoyancy_flux_ccc, model.grid,
                                                           buoyancy_diffusive_flux_arguments(model)...)
end
#---

end # module
