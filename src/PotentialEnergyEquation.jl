module PotentialEnergyEquation

using DocStringExtensions

export PotentialEnergy
# Short name inside the module, prefixed alias for `using Oceanostics`, as elsewhere in the package.
export DiffusiveVerticalBuoyancyFlux, PotentialEnergyDiffusiveVerticalBuoyancyFlux
export Tendency, PotentialEnergyTendency
export Advection, PotentialEnergyAdvection
export BuoyancyAdvection, PotentialEnergyBuoyancyAdvection
export Diffusion, PotentialEnergyDiffusion
export BuoyancyDiffusion, PotentialEnergyBuoyancyDiffusion
export Forcing, PotentialEnergyForcing
# `wb` is the one term the kinetic and potential energy budgets share, so it is defined in
# `KineticEnergyEquation` and re-exported here under both its own name and a budget-neutral alias.
export PotentialToKineticEnergyConversion, KineticEnergyConversion

using Oceananigans: NonhydrostaticModel, fields
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Models: seawater_density
using Oceananigans.Models: model_geopotential_height
using Oceananigans.Models.NonhydrostaticModels: div_Uc, tracer_tendency
using Oceananigans.Grids: Center, Face
using Oceananigans.Grids: NegativeZDirection
using Oceananigans.BuoyancyFormulations: BuoyancyForce, BuoyancyTracer, SeawaterBuoyancy, LinearEquationOfState
using Oceananigans.BuoyancyFormulations: buoyancy_perturbationᶜᶜᶜ, Zᶜᶜᶜ, Zᶜᶜᶠ
using Oceananigans.Models: ShallowWaterModel
using Oceananigans.Operators: ℑzᵃᵃᶜ, δxᶜᵃᵃ, δyᵃᶜᵃ, δzᵃᵃᶜ, V⁻¹ᶜᶜᶜ, Axᶠᶜᶜ, Ayᶜᶠᶜ, Azᶜᶜᶠ
using Oceananigans.TurbulenceClosures: diffusive_flux_x, diffusive_flux_y, diffusive_flux_z, ∇_dot_qᶜ
using Oceananigans.Utils: sum_of_velocities
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

# Measuring buoyancy against depth is what `eₚ = -bz` does, so it holds only when gravity points along
# `-z`. Callers name themselves, since diagnostics in more than one module land here.
validate_gravity_unit_vector(diagnostic, gravity_unit_vector::NegativeZDirection) = nothing
validate_gravity_unit_vector(diagnostic, gravity_unit_vector) =
    throw(ArgumentError("`$diagnostic` measures buoyancy against depth, which assumes gravity points along \
                         `NegativeZDirection`, but this model's gravity unit vector is $(gravity_unit_vector). \
                         Only `NegativeZDirection` is supported for now."))

#+++ Potential energy
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
    isnothing(model.buoyancy) ? nothing : validate_gravity_unit_vector("PotentialEnergy", model.buoyancy.gravity_unit_vector)

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
#---

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

# The terms of the `eₚ` equation are `-z` times the terms of the equation the model steps for `b`, so
# they need `b` to be one of the model's tracers. That is weaker than the condition above, which also
# wants the closure to diffuse it, so advection, forcing and the tendency get their own check.
validate_buoyancy_is_a_tracer(diagnostic, model) =
    model.buoyancy isa BuoyancyTracerModel ||
        throw(ArgumentError("`$diagnostic` is a term of the `eₚ = -bz` equation, which is `-z` times the equation \
                             the model steps for the tracer `b`, but this model's buoyancy is a \
                             $(summary(model.buoyancy)). Only `BuoyancyTracer` is supported for now."))

# Every term below reads `b`'s slot in `model.tracers`, which `validate_buoyancy_is_a_tracer` has
# already established is there.
buoyancy_tracer_index(model) = Val(findfirst(n -> n === :b, propertynames(model.tracers)))

# The arguments every diagnostic built on `diffusive_flux_*` passes through to the closure.
buoyancy_diffusive_flux_arguments(model) =
    (model.closure,
     model.closure_fields,
     buoyancy_tracer_index(model),
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

const DiffusiveVerticalBuoyancyFlux = CustomKFO{<:typeof(diffusive_buoyancy_flux_ccc)}
const PotentialEnergyDiffusiveVerticalBuoyancyFlux = DiffusiveVerticalBuoyancyFlux

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

PotentialEnergyDiffusiveVerticalBuoyancyFlux(model)   # or `DiffusiveVerticalBuoyancyFlux` inside the module

# output

PotentialEnergyDiffusiveVerticalBuoyancyFlux KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: diffusive_buoyancy_flux_ccc (generic function with 1 method)
└── arguments: ("ScalarDiffusivity", "Nothing", "Val", "Field", "Clock", "NamedTuple", "BuoyancyForce")
└── computes: diffusive vertical buoyancy flux  Φ = κ ∂b/∂z = -q₃
```
"""
function DiffusiveVerticalBuoyancyFlux(model; location = (Center, Center, Center))
    validate_location(location, "DiffusiveVerticalBuoyancyFlux")
    validate_buoyancy_is_a_diffused_tracer("DiffusiveVerticalBuoyancyFlux", model)
    validate_closure_supplies_a_flux("DiffusiveVerticalBuoyancyFlux", model)

    return KernelFunctionOperation{Center, Center, Center}(diffusive_buoyancy_flux_ccc, model.grid,
                                                           buoyancy_diffusive_flux_arguments(model)...)
end

# `Φ = -q₃` is the vertical component of the diffusive flux rather than the flux vector, and the name
# says so as of the rename. The old short name stays behind a deprecation for a release, so a
# downstream script gets a pointer to the new one instead of an `UndefVarError`. The prefixed alias is
# deprecated in `Oceanostics` itself rather than here: a binding deprecation does not survive being
# re-exported through another module, so attaching it there is what makes `Oceanostics.<old name>` warn.
Base.@deprecate_binding DiffusiveBuoyancyFlux DiffusiveVerticalBuoyancyFlux
#---

#+++ The rest of the `eₚ` equation
# `eₚ = -bz` and `z` does not change in time, so every term of the `eₚ` equation is `-z` times the
# matching term of the equation the model steps for `b`. `Tendency`, `BuoyancyAdvection`,
# `BuoyancyDiffusion` and `Forcing` are built that way, on Oceananigans' own kernel for that term, which
# is the convention `KineticEnergyEquation` follows too (`uᵢ∂ⱼ(uⱼuᵢ)` rather than `∂ⱼ(uⱼK)`): the split
# is then the model's own tendency taken apart term by term, rather than a continuum rearrangement of
# it, so it holds at the discrete level.
#
# `Tendency` comes off `tracer_tendency`, so it always carries every term the model steps. The other
# three cover most but not all of them, and the split closes exactly only when none of the following is
# in play. In each case `Tendency` still includes the term and the other three do not, so the shortfall
# shows up as a residual with no error raised:
#
#   * a `BackgroundField` buoyancy `B`. The model prognoses the perturbation and its equation picks up
#     `-∂ⱼ(uⱼB)`, which weighted by `-z` is a source of `eₚ` that does not drop out of a volume integral.
#     It has no diagnostic here yet.
#   * an `AdvectiveForcing`, or a biogeochemistry with a drift velocity. `tracer_tendency` folds both
#     into the advecting flow through `with_advective_forcing` and `biogeochemical_drift_velocity`,
#     while `BuoyancyAdvection`'s `velocities` default is only `sum_of_velocities(velocities,
#     background)`. An `AdvectiveForcing` is doubly invisible: it also evaluates to zero when called
#     pointwise, which is how `Forcing` calls it. Pass `velocities` explicitly to cover the drift.
#   * an `ImmersedBoundaryGrid`. `tracer_tendency` adds `-immersed_∇_dot_qᶜ`, the flux through immersed
#     faces, and no diagnostic here accounts for it.
#   * a biogeochemistry with a transition term for `b`, which `tracer_tendency` adds and nothing here
#     mirrors.
#
# Pulling `z` inside the derivative splits each of the two flux terms into a transport of `eₚ` and a
# conversion:
#
#     BuoyancyAdvection = z∂ⱼ(uⱼb) = -Advection - wb ,      Advection = ∂ⱼ(uⱼeₚ)
#     BuoyancyDiffusion = z∂ⱼqⱼ    =  Diffusion  + Φ ,      Diffusion = ∂ⱼ(zqⱼ) ,  Φ = -q₃
#
# `Advection` and `Diffusion` are the transports, and both are built here as genuine flux divergences
# rather than by rearranging their `Buoyancy*` partners, so each telescopes and integrates to zero to
# roundoff over a periodic or closed domain. `Diffusion` builds its divergence from the unconditional
# `diffusive_flux_*`, whereas `BuoyancyDiffusion` reaches the closure through `∇_dot_qᶜ`, which uses the
# conditional `_diffusive_flux_*` that `ImmersedBoundaries` overrides to zero across immersed faces. Off
# an immersed grid the two are the same flux; on one they are not, and both the telescoping and the
# `BuoyancyDiffusion = Diffusion + Φ` identity below stop holding. That leaves
#
#     ∫BuoyancyAdvection dV = -∫wb dV        and        ∫BuoyancyDiffusion dV = ∫Φ dV ,
#
# which is why an integrated budget is usually written with `PotentialToKineticEnergyConversion` and
# `DiffusiveVerticalBuoyancyFlux` in place of the two `Buoyancy*` terms. Those identities come from the
# continuum product rule, so they hold to the truncation error of the discretization rather than
# exactly.

#+++ Tendency
@inline minus_z_∂ₜb_ccc(i, j, k, grid, args...) = -Zᶜᶜᶜ(i, j, k, grid) * tracer_tendency(i, j, k, grid, args...)

const PotentialEnergyTendency = CustomKFO{<:typeof(minus_z_∂ₜb_ccc)}
const Tendency = PotentialEnergyTendency

"""
    $(SIGNATURES)

Return a `KernelFunctionOperation` computing the tendency of the potential energy `eₚ = -bz`,

```
    ∂ₜeₚ = -z ∂ₜb ,
```

where `∂ₜb` is Oceananigans' own tracer tendency for the buoyancy. Since that kernel is the one the
model steps, this is the whole right-hand side of the `eₚ` equation in one term: advection by the total
(perturbation plus background) flow, advection of a background buoyancy field, diffusion, and forcing.
The individual terms are [`PotentialEnergyBuoyancyAdvection`](@ref),
[`PotentialEnergyBuoyancyDiffusion`](@ref) and [`PotentialEnergyForcing`](@ref), and they sum to this
one cell by cell — but only when the model has none of the features listed below. Note it is those two
`Buoyancy*` terms that sum, not [`PotentialEnergyAdvection`](@ref) and [`PotentialEnergyDiffusion`](@ref)
— the latter are the transports alone, and differ from them by the two conversions `wb` and `Φ`.

Because this term is `tracer_tendency` itself, it carries everything the model steps, including four
things the other three diagnostics do not. With any of them present the split falls short by exactly
that term, silently:

  * a `BackgroundField` buoyancy `B`, contributing `-z ∂ⱼ(uⱼB)`, which has no diagnostic yet;
  * an `AdvectiveForcing` or a biogeochemical drift velocity, which `tracer_tendency` folds into the
    advecting flow but [`PotentialEnergyBuoyancyAdvection`](@ref) does not pick up by default;
  * an `ImmersedBoundaryGrid`, whose `immersed_∇_dot_qᶜ` no diagnostic here mirrors;
  * a biogeochemical transition term for `b`.

Defined for `BuoyancyTracer` models, where `b` is one of the model's tracers.

```jldoctest
using Oceananigans, Oceanostics

grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1))
model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b, closure=ScalarDiffusivity(κ=1e-4))

PotentialEnergyTendency(model)   # or `Tendency` inside the module

# output

PotentialEnergyTendency KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: minus_z_∂ₜb_ccc (generic function with 1 method)
└── arguments: ("Val", "Val", "Centered", "ScalarDiffusivity", "Nothing", "BuoyancyForce", "Nothing", "Oceananigans.Models.NonhydrostaticModels.BackgroundFields", "NamedTuple", "NamedTuple", "NamedTuple", "Nothing", "Clock", "Returns")
└── computes: potential energy tendency  ∂ₜeₚ = -z ∂ₜb
```
"""
function PotentialEnergyTendency(model::NonhydrostaticModel; location = (Center, Center, Center))
    validate_location(location, "PotentialEnergyTendency")
    validate_buoyancy_is_a_tracer("PotentialEnergyTendency", model)

    dependencies = (buoyancy_tracer_index(model),
                    Val(:b),
                    model.advection,
                    model.closure,
                    model.tracers.b.boundary_conditions.immersed,
                    model.buoyancy,
                    model.biogeochemistry,
                    model.background_fields,
                    model.velocities,
                    model.tracers,
                    model.auxiliary_fields,
                    model.closure_fields,
                    model.clock,
                    model.forcing.b)

    return KernelFunctionOperation{Center, Center, Center}(minus_z_∂ₜb_ccc, model.grid, dependencies...)
end
#---

#+++ Advection of eₚ
# `∂ⱼ(uⱼeₚ)`, the transport of the potential energy itself, formed by handing `eₚ` to the model's own
# advection scheme exactly as it would a tracer. Being a flux divergence it telescopes, so its volume
# integral over a periodic or closed domain vanishes to roundoff rather than to truncation error.
@inline div_U_eₚ_ccc(i, j, k, grid, advection, U, eₚ) = div_Uc(i, j, k, grid, advection, U, eₚ)

const PotentialEnergyAdvection = CustomKFO{<:typeof(div_U_eₚ_ccc)}
const Advection = PotentialEnergyAdvection

"""
    $(SIGNATURES)

Return a `KernelFunctionOperation` computing the advection of the potential energy itself,

```
    ADV = ∂ⱼ(uⱼeₚ) ,
```

with `eₚ = -bz` handed to the model's own advection scheme the way a tracer would be. `uⱼ` defaults to
the *total* velocity, perturbation plus background, which is what the model advects with; pass
`velocities` to override it. It enters the `eₚ` equation with a minus sign, as
[`TracerEquation.Advection`](@ref Oceanostics.TracerEquation.Advection) does for a tracer.

This is a transport and nothing else: it is a flux divergence, so over a periodic or closed domain it
integrates to zero to roundoff. What it does *not* include is the conversion `wb` that arises from
weighting the buoyancy equation by `-z`; the two together make
[`PotentialEnergyBuoyancyAdvection`](@ref), which is the term that sums with the others to
[`PotentialEnergyTendency`](@ref):

```
    z ∂ⱼ(uⱼb) = -∂ⱼ(uⱼeₚ) - wb .
```

```jldoctest
using Oceananigans, Oceanostics

grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1))
model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)

PotentialEnergyAdvection(model)   # or `Advection` inside the module

# output

PotentialEnergyAdvection KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: div_U_eₚ_ccc (generic function with 1 method)
└── arguments: ("Centered", "NamedTuple", "KernelFunctionOperation")
└── computes: potential energy advection  ∂ⱼ(uⱼeₚ)
```
"""
function PotentialEnergyAdvection(model::NonhydrostaticModel;
                                  velocities = sum_of_velocities(model.velocities, model.background_fields.velocities),
                                  location = (Center, Center, Center))
    validate_location(location, "PotentialEnergyAdvection")
    validate_buoyancy_is_a_tracer("PotentialEnergyAdvection", model)

    return KernelFunctionOperation{Center, Center, Center}(div_U_eₚ_ccc, model.grid,
                                                           model.advection, velocities, PotentialEnergy(model))
end
#---

#+++ Buoyancy advection
@inline z_div_Uc_ccc(i, j, k, grid, advection, U, c) = Zᶜᶜᶜ(i, j, k, grid) * div_Uc(i, j, k, grid, advection, U, c)

const PotentialEnergyBuoyancyAdvection = CustomKFO{<:typeof(z_div_Uc_ccc)}
const BuoyancyAdvection = PotentialEnergyBuoyancyAdvection

"""
    $(SIGNATURES)

Return a `KernelFunctionOperation` computing the advective term of the `eₚ = -bz` equation,

```
    ADV = z ∂ⱼ(uⱼb) ,
```

the advection of the buoyancy weighted by `-z`. `uⱼ` defaults to the *total* velocity, perturbation
plus background, which is what the model advects with; pass `velocities` to override it.

Pulling `z` inside the derivative writes this as a transport of `eₚ` plus the conversion term,
`ADV = -∂ⱼ(uⱼeₚ) - wb`, so over a periodic or closed domain its volume integral is
`-∫wb dV`, the (negated)
[`PotentialToKineticEnergyConversion`](@ref Oceanostics.KineticEnergyEquation.PotentialEnergyConversion).

A background buoyancy field is advected by a term of its own, which this one does not include and which
has no diagnostic here yet.

The default `velocities` is the perturbation plus background flow. That is what the model advects with
in the ordinary case, but not when an `AdvectiveForcing` or a biogeochemical drift velocity is in play:
`tracer_tendency` folds those in too, through `with_advective_forcing` and
`biogeochemical_drift_velocity`. Pass `velocities` explicitly to match the model in that case.

```jldoctest
using Oceananigans, Oceanostics

grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1))
model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)

PotentialEnergyBuoyancyAdvection(model)   # or `BuoyancyAdvection` inside the module

# output

PotentialEnergyBuoyancyAdvection KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: z_div_Uc_ccc (generic function with 1 method)
└── arguments: ("Centered", "NamedTuple", "Field")
└── computes: potential energy buoyancy advection  z ∂ⱼ(uⱼb)
```
"""
function PotentialEnergyBuoyancyAdvection(model::NonhydrostaticModel;
                                          velocities = sum_of_velocities(model.velocities, model.background_fields.velocities),
                                          location = (Center, Center, Center))
    validate_location(location, "PotentialEnergyBuoyancyAdvection")
    validate_buoyancy_is_a_tracer("PotentialEnergyBuoyancyAdvection", model)

    return KernelFunctionOperation{Center, Center, Center}(z_div_Uc_ccc, model.grid,
                                                           model.advection, velocities, model.tracers.b)
end
#---

#+++ Diffusive transport of eₚ
# `∂ⱼ(zqⱼ)`, the diffusive transport of the potential energy, built as a flux divergence the way
# Oceananigans builds `∇_dot_qᶜ`: the closure's flux on each face, weighted by `z` there and by the face
# area, differenced across the cell and divided by its volume. Forming it this way rather than as
# `z∂ⱼqⱼ + q₃` is what makes it telescope, so its volume integral vanishes to roundoff.
@inline Ax_z_qx_fcc(i, j, k, grid, args...) = Axᶠᶜᶜ(i, j, k, grid) * Zᶜᶜᶜ(i, j, k, grid) * diffusive_flux_x(i, j, k, grid, args...)
@inline Ay_z_qy_cfc(i, j, k, grid, args...) = Ayᶜᶠᶜ(i, j, k, grid) * Zᶜᶜᶜ(i, j, k, grid) * diffusive_flux_y(i, j, k, grid, args...)
@inline Az_z_qz_ccf(i, j, k, grid, args...) = Azᶜᶜᶠ(i, j, k, grid) * Zᶜᶜᶠ(i, j, k, grid) * diffusive_flux_z(i, j, k, grid, args...)

@inline div_z_q_ccc(i, j, k, grid, args...) =
    V⁻¹ᶜᶜᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Ax_z_qx_fcc, args...) +
                             δyᵃᶜᵃ(i, j, k, grid, Ay_z_qy_cfc, args...) +
                             δzᵃᵃᶜ(i, j, k, grid, Az_z_qz_ccf, args...))

const PotentialEnergyDiffusion = CustomKFO{<:typeof(div_z_q_ccc)}
const Diffusion = PotentialEnergyDiffusion

"""
    $(SIGNATURES)

Return a `KernelFunctionOperation` computing the diffusive transport of the potential energy,

```
    DIFF = ∂ⱼ(z qⱼ) ,
```

where `qⱼ` is the closure's own diffusive flux of buoyancy (`qⱼ = -κ ∂ⱼb` for Fickian diffusion). It is
the diffusive counterpart of [`PotentialEnergyAdvection`](@ref): a flux divergence, so over a periodic
or closed domain it integrates to zero to roundoff.

What it does *not* include is the vertical flux `-q₃` that arises from weighting the buoyancy equation
by `-z`; the two together make [`PotentialEnergyBuoyancyDiffusion`](@ref), which is the term that sums
with the others to [`PotentialEnergyTendency`](@ref):

```
    z ∂ⱼqⱼ = ∂ⱼ(z qⱼ) - q₃ = ∂ⱼ(z qⱼ) + Φ ,
```

with `Φ` the [`DiffusiveVerticalBuoyancyFlux`](@ref).

Like the other diffusive terms this reads `κ∇b` off the closure, so it needs the buoyancy to be a
tracer the closure diffuses.

Two limits are worth knowing. The divergence is assembled from the unconditional `diffusive_flux_*`,
while [`PotentialEnergyBuoyancyDiffusion`](@ref) reaches the closure through Oceananigans' `∇_dot_qᶜ`,
which uses the conditional `_diffusive_flux_*` that `ImmersedBoundaries` overrides to zero across
immersed faces. Off an immersed grid the two are the same flux; on one this term includes fluxes its
partner zeroes, and neither the telescoping nor the identity above survives. And the horizontal fluxes
are weighted by `z` at the cell centre rather than at the `x`- and `y`-faces they live on, which is
exact wherever `znode` does not vary with `i` and `j` — every `RectilinearGrid` and
`LatitudeLongitudeGrid` with an immutable vertical coordinate — but not under a
`MutableVerticalDiscretization`.

```jldoctest
using Oceananigans, Oceanostics

grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1))
model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b, closure=ScalarDiffusivity(κ=1e-4))

PotentialEnergyDiffusion(model)   # or `Diffusion` inside the module

# output

PotentialEnergyDiffusion KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: div_z_q_ccc (generic function with 1 method)
└── arguments: ("ScalarDiffusivity", "Nothing", "Val", "Field", "Clock", "NamedTuple", "BuoyancyForce")
└── computes: potential energy diffusive transport  ∂ⱼ(z qⱼ)
```
"""
function PotentialEnergyDiffusion(model; location = (Center, Center, Center))
    validate_location(location, "PotentialEnergyDiffusion")
    validate_buoyancy_is_a_diffused_tracer("PotentialEnergyDiffusion", model)
    validate_closure_supplies_a_flux("PotentialEnergyDiffusion", model)

    return KernelFunctionOperation{Center, Center, Center}(div_z_q_ccc, model.grid,
                                                           model.closure, model.closure_fields,
                                                           buoyancy_tracer_index(model), model.tracers.b,
                                                           model.clock, fields(model), model.buoyancy)
end
#---

#+++ Buoyancy diffusion
@inline z_∇_dot_qᶜ_ccc(i, j, k, grid, args...) = Zᶜᶜᶜ(i, j, k, grid) * ∇_dot_qᶜ(i, j, k, grid, args...)

const PotentialEnergyBuoyancyDiffusion = CustomKFO{<:typeof(z_∇_dot_qᶜ_ccc)}
const BuoyancyDiffusion = PotentialEnergyBuoyancyDiffusion

"""
    $(SIGNATURES)

Return a `KernelFunctionOperation` computing the diffusive term of the `eₚ = -bz` equation,

```
    DIFF = z ∂ⱼqⱼ ,
```

where `qⱼ` is the closure's own diffusive flux of buoyancy (`qⱼ = -κ ∂ⱼb` for Fickian diffusion).

Pulling `z` inside the derivative writes this as a transport plus the vertical flux,
`DIFF = ∂ⱼ(zqⱼ) - q₃`, so over a periodic or closed domain its volume integral is `∫Φ dV`, the
[`DiffusiveVerticalBuoyancyFlux`](@ref). Diffusion in the horizontal drops out of that integral entirely, since
`z` does not vary along it.

Like `Φ`, this reads `κ∇b` off the closure, so it needs the buoyancy to be a tracer the closure
diffuses.

```jldoctest
using Oceananigans, Oceanostics

grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1))
model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b, closure=ScalarDiffusivity(κ=1e-4))

PotentialEnergyBuoyancyDiffusion(model)   # or `BuoyancyDiffusion` inside the module

# output

PotentialEnergyBuoyancyDiffusion KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: z_∇_dot_qᶜ_ccc (generic function with 1 method)
└── arguments: ("ScalarDiffusivity", "Nothing", "Val", "Field", "Clock", "NamedTuple", "BuoyancyForce")
└── computes: potential energy buoyancy diffusion  z ∂ⱼqⱼ
```
"""
function PotentialEnergyBuoyancyDiffusion(model; location = (Center, Center, Center))
    validate_location(location, "PotentialEnergyBuoyancyDiffusion")
    validate_buoyancy_is_a_diffused_tracer("PotentialEnergyBuoyancyDiffusion", model)
    validate_closure_supplies_a_flux("PotentialEnergyBuoyancyDiffusion", model)

    return KernelFunctionOperation{Center, Center, Center}(z_∇_dot_qᶜ_ccc, model.grid,
                                                           model.closure, model.closure_fields,
                                                           buoyancy_tracer_index(model), model.tracers.b,
                                                           model.clock, fields(model), model.buoyancy)
end
#---

#+++ Forcing
@inline minus_z_Fᵇ_ccc(i, j, k, grid, forcing, clock, model_fields) =
    -Zᶜᶜᶜ(i, j, k, grid) * forcing(i, j, k, grid, clock, model_fields)

const PotentialEnergyForcing = CustomKFO{<:typeof(minus_z_Fᵇ_ccc)}
const Forcing = PotentialEnergyForcing

"""
    $(SIGNATURES)

Return a `KernelFunctionOperation` computing the forcing term of the `eₚ = -bz` equation,

```
    FORC = -z Fᵇ ,
```

where `Fᵇ` is whatever forcing is applied to the buoyancy. Unlike the transport terms this does not
drop out of a volume integral, so a forced run needs it in the budget.

This calls the forcing pointwise, as `Fᵇ(i, j, k, grid, clock, fields)`. An `AdvectiveForcing` returns
zero there by construction — it acts through the advecting velocity instead — so this term reports
nothing for one, and [`PotentialEnergyBuoyancyAdvection`](@ref) will not see it either unless it is
handed matching `velocities`.

```jldoctest
using Oceananigans, Oceanostics

grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1))
model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b,
                            forcing = (; b = Forcing((x, y, z, t) -> 1e-8)))

PotentialEnergyForcing(model)

# output

PotentialEnergyForcing KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: minus_z_Fᵇ_ccc (generic function with 1 method)
└── arguments: ("Oceananigans.Forcings.ContinuousForcing", "Clock", "NamedTuple")
└── computes: potential energy forcing  -z Fᵇ
```
"""
function PotentialEnergyForcing(model::NonhydrostaticModel; location = (Center, Center, Center))
    validate_location(location, "PotentialEnergyForcing")
    validate_buoyancy_is_a_tracer("PotentialEnergyForcing", model)

    return KernelFunctionOperation{Center, Center, Center}(minus_z_Fᵇ_ccc, model.grid,
                                                           model.forcing.b, model.clock, fields(model))
end
#---
#---

end # module
