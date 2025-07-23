module PotentialEnergyEquation

using DocStringExtensions

export PotentialEnergy

using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Models: seawater_density
using Oceananigans.Models: model_geopotential_height
using Oceananigans.Grids: Center, Face
using Oceananigans.Grids: NegativeZDirection
using Oceananigans.BuoyancyFormulations: BuoyancyForce, BuoyancyTracer, SeawaterBuoyancy, LinearEquationOfState
using Oceananigans.BuoyancyFormulations: buoyancy_perturbationᶜᶜᶜ, Zᶜᶜᶜ
using Oceananigans.Models: ShallowWaterModel
using Oceanostics: validate_location, CustomKFO
using SeawaterPolynomials: BoussinesqEquationOfState

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
Eₚ = \\frac{gρ}{ρ₀}z = -bz
```
at each grid `location` in `model`. `PotentialEnergy` is defined for both `BuoyancyTracer`
and `SeawaterBuoyancy`. See the relevant Oceananigans.jl documentation on
[buoyancy models](https://clima.github.io/OceananigansDocumentation/dev/model_setup/buoyancy_and_equation_of_state/)
for more information about available options.

The optional keyword argument `geopotential_height` is only used
if ones wishes to calculate `Eₚ` with a potential density referenced to `geopotential_height`,
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

julia> model = NonhydrostaticModel(; grid, buoyancy=BuoyancyTracer(), tracers=(:b,))
NonhydrostaticModel{CPU, RectilinearGrid}(time = 0 seconds, iteration = 0)
├── grid: 1×1×100 RectilinearGrid{Float64, Flat, Flat, Bounded} on CPU with 0×0×3 halo
├── timestepper: RungeKutta3TimeStepper
├── advection scheme: Centered(order=2)
├── tracers: b
├── closure: Nothing
├── buoyancy: BuoyancyTracer with ĝ = NegativeZDirection()
└── coriolis: Nothing

julia> PotentialEnergyEquation.PotentialEnergy(model)
KernelFunctionOperation at (Center, Center, Center)
├── grid: 1×1×100 RectilinearGrid{Float64, Flat, Flat, Bounded} on CPU with 0×0×3 halo
├── kernel_function: minus_bz_ccc (generic function with 3 methods)
└── arguments: ("Field",)
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

julia> model = NonhydrostaticModel(; grid, buoyancy, tracers)
NonhydrostaticModel{CPU, RectilinearGrid}(time = 0 seconds, iteration = 0)
├── grid: 1×1×100 RectilinearGrid{Float64, Flat, Flat, Bounded} on CPU with 0×0×3 halo
├── timestepper: RungeKutta3TimeStepper
├── advection scheme: Centered(order=2)
├── tracers: (T, S)
├── closure: Nothing
├── buoyancy: SeawaterBuoyancy with g=9.80665 and BoussinesqEquationOfState{Float64} with ĝ = NegativeZDirection()
└── coriolis: Nothing

julia> PotentialEnergyEquation.PotentialEnergy(model)
KernelFunctionOperation at (Center, Center, Center)
├── grid: 1×1×100 RectilinearGrid{Float64, Flat, Flat, Bounded} on CPU with 0×0×3 halo
├── kernel_function: minus_bz_ccc (generic function with 3 methods)
└── arguments: ("KernelFunctionOperation", "NamedTuple")
```

To use a reference density set a constant value for the keyword argument `geopotential_height`
and pass this the function. For example,
```jldoctest
julia> using Oceananigans, SeawaterPolynomials.TEOS10, Oceanostics

julia> grid = RectilinearGrid(size=100, z=(-1000, 0), topology=(Flat, Flat, Bounded));

julia> tracers = (:T, :S);

julia> eos = TEOS10EquationOfState();

julia> buoyancy = SeawaterBuoyancy(equation_of_state=eos);

julia> model = NonhydrostaticModel(; grid, buoyancy, tracers);

julia> geopotential_height = 0; # density variable will be σ₀

julia> PotentialEnergyEquation.PotentialEnergy(model)
KernelFunctionOperation at (Center, Center, Center)
├── grid: 1×1×100 RectilinearGrid{Float64, Flat, Flat, Bounded} on CPU with 0×0×3 halo
├── kernel_function: minus_bz_ccc (generic function with 3 methods)
└── arguments: ("KernelFunctionOperation", "NamedTuple")
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

end # module
