module PotentialEnergyEquationTerms

using DocStringExtensions

export PotentialEnergy, BackgroundPotentialEnergy, OneDReferenceField

using Oceananigans.AbstractOperations: KernelFunctionOperation, volume, Az, GridMetricOperation
using Oceananigans.Models: seawater_density
using Oceananigans.Models: model_geopotential_height
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid
using Oceananigans.Grids
using Oceananigans.Grids: Center, NegativeZDirection, interior, CenterField
using Oceananigans.BuoyancyModels: Buoyancy, BuoyancyTracer, SeawaterBuoyancy, LinearEquationOfState
using Oceananigans.BuoyancyModels: buoyancy_perturbationᶜᶜᶜ, Zᶜᶜᶜ
using Oceananigans.Models: ShallowWaterModel
using Oceananigans.Fields: Field, compute!, field, set!
using Oceanostics: validate_location
using SeawaterPolynomials: BoussinesqEquationOfState

const NoBuoyancyModel = Union{Nothing, ShallowWaterModel}
const BuoyancyTracerModel = Buoyancy{<:BuoyancyTracer, g} where g
const BuoyancyLinearEOSModel = Buoyancy{<:SeawaterBuoyancy{FT, <:LinearEquationOfState, T, S} where {FT, T, S}, g} where {g}
const BuoyancyBoussinesqEOSModel = Buoyancy{<:SeawaterBuoyancy{FT, <:BoussinesqEquationOfState, T, S} where {FT, T, S}, g} where {g}

validate_gravity_unit_vector(gravity_unit_vector::NegativeZDirection) = nothing
validate_gravity_unit_vector(gravity_unit_vector) =
    throw(ArgumentError("`PotentialEnergy` and `BackgroundPotentialEnergy` are curently only defined for models that have a `NegativeZDirection` gravity unit vector."))
validate_grid(grid) = nothing
validate_grid(grid::ImmersedBoundaryGrid) =
    throw(ArgumentError("`PotentialEnergy` and `BackgroundPotentialEnergy` are not currently defined for $(grid)."))

"""
    $(SIGNATURES)

Return a `KernelFunctionOperation` to compute the `PotentialEnergy` per unit volume,
```math
Eₚ = \\frac{gρz}{ρ₀}
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
julia> using Oceananigans

julia> using Oceanostics.PotentialEnergyEquationTerms: PotentialEnergy

julia> grid = RectilinearGrid(size=100, z=(-1000, 0), topology=(Flat, Flat, Bounded));

julia> model = NonhydrostaticModel(; grid, buoyancy=BuoyancyTracer(), tracers=:b);

julia> PotentialEnergy(model)
KernelFunctionOperation at (Center, Center, Center)
├── grid: 1×1×100 RectilinearGrid{Float64, Flat, Flat, Bounded} on CPU with 0×0×3 halo
├── kernel_function: bz_ccc (generic function with 2 methods)
└── arguments: ("1×1×100 Field{Center, Center, Center} on RectilinearGrid on CPU",)
```

The default behaviour of `PotentialEnergy` uses the *in-situ density* in the calculation
when the equation of state is a `BoussinesqEquationOfState`:
```jldoctest
julia> using Oceananigans, SeawaterPolynomials.TEOS10

julia> using Oceanostics.PotentialEnergyEquationTerms: PotentialEnergy

julia> grid = RectilinearGrid(size=100, z=(-1000, 0), topology=(Flat, Flat, Bounded));

julia> tracers = (:T, :S);

julia> eos = TEOS10EquationOfState();

julia> buoyancy = SeawaterBuoyancy(equation_of_state=eos);

julia> model = NonhydrostaticModel(; grid, buoyancy, tracers);

julia> PotentialEnergy(model)
KernelFunctionOperation at (Center, Center, Center)
├── grid: 1×1×100 RectilinearGrid{Float64, Flat, Flat, Bounded} on CPU with 0×0×3 halo
├── kernel_function: g′z_ccc (generic function with 1 method)
└── arguments: ("KernelFunctionOperation at (Center, Center, Center)", "(g=9.80665, ρ₀=1020.0)")
```

To use a reference density set a constant value for the keyword argument `geopotential_height`
and pass this the function. For example,
```jldoctest
julia> using Oceananigans, SeawaterPolynomials.TEOS10;

julia> using Oceanostics.PotentialEnergyEquationTerms: PotentialEnergy;

julia> grid = RectilinearGrid(size=100, z=(-1000, 0), topology=(Flat, Flat, Bounded));

julia> tracers = (:T, :S);

julia> eos = TEOS10EquationOfState();

julia> buoyancy = SeawaterBuoyancy(equation_of_state=eos);

julia> model = NonhydrostaticModel(; grid, buoyancy, tracers);

julia> geopotential_height = 0; # density variable will be σ₀

julia> PotentialEnergy(model)
KernelFunctionOperation at (Center, Center, Center)
├── grid: 1×1×100 RectilinearGrid{Float64, Flat, Flat, Bounded} on CPU with 0×0×3 halo
├── kernel_function: g′z_ccc (generic function with 1 method)
└── arguments: ("KernelFunctionOperation at (Center, Center, Center)", "(g=9.80665, ρ₀=1020.0)")
```
"""
@inline function PotentialEnergy(model; location = (Center, Center, Center),
                                 geopotential_height = model_geopotential_height(model))

    validate_location(location, "PotentialEnergy")
    isnothing(model.buoyancy) ? nothing : validate_gravity_unit_vector(model.buoyancy.gravity_unit_vector)
    validate_grid(model.grid)

    return PotentialEnergy(model, model.buoyancy, geopotential_height)
end

@inline PotentialEnergy(model, buoyancy_model::NoBuoyancyModel, geopotential_height) =
    throw(ArgumentError("Cannot calculate gravitational potential energy without a Buoyancy model."))

@inline function PotentialEnergy(model, buoyancy_model::BuoyancyTracerModel, geopotential_height)

    grid = model.grid
    b = model.tracers.b

    return KernelFunctionOperation{Center, Center, Center}(bz_ccc, grid, b)
end

@inline bz_ccc(i, j, k, grid, b) = b[i, j, k] * Zᶜᶜᶜ(i, j, k, grid)

@inline function PotentialEnergy(model, buoyancy_model::BuoyancyLinearEOSModel, geopotential_height)

    grid = model.grid
    C = model.tracers
    b = buoyancy_model.model

    return KernelFunctionOperation{Center, Center, Center}(bz_ccc, grid, b, C)
end

@inline bz_ccc(i, j, k, grid, b, C) = buoyancy_perturbationᶜᶜᶜ(i, j, k, grid, b, C) * Zᶜᶜᶜ(i, j, k, grid)

@inline function PotentialEnergy(model, buoyancy_model::BuoyancyBoussinesqEOSModel, geopotential_height)

    grid = model.grid
    ρ = seawater_density(model; geopotential_height)
    parameters = (g = model.buoyancy.model.gravitational_acceleration,
                  ρ₀ = model.buoyancy.model.equation_of_state.reference_density)

    return KernelFunctionOperation{Center, Center, Center}(g′z_ccc, grid, ρ, parameters)
end

@inline g′z_ccc(i, j, k, grid, ρ, p) = (p.g / p.ρ₀) * ρ[i, j, k] * Zᶜᶜᶜ(i, j, k, grid)

## Grid metrics from https://github.com/tomchor/Oceanostics.jl/issues/163#issuecomment-2012623824

function MetricField(loc, grid, metric)
    metric_operation = GridMetricOperation(loc, metric, grid)
    metric_field = Field(metric_operation)
    return compute!(metric_field)
end

VolumeField(grid, loc=(Center, Center, Center)) = MetricField(loc, grid, volume)
  AreaField(grid, loc=(Center, Center, Nothing)) = MetricField(loc, grid, Az)

"""
    function OneDReferenceField(f::Field)
Return a `OneDReferenceField` of the data in `f` and the corresponding `z✶` coordinate.
The gridded data is first reshaped into a 1D `Array` then sorted. Returned is a new `Field`
of this sorted data as well as the `z✶` `Field` that are both on a one-dimensional grid that
has the same vertical extent as the original field (`f.grid.Lz`) but with the same _total_
number of points of the original domain (`Nx * Ny * Nz`).
The `z✶` `Field` is defined as
```math
\\frac{1}{A}\\int_{f\\mathrm{min}}^{f\\mathrm{max}} \\mathrm{d}V.
```
and is computed by cumulatively summing the 1D `Array` of grid volumes `ΔV`.
**Note:** `OneDReferenceField` is currently only appropriate for grids that have uniform
horizontal area.
"""
function OneDReferenceField(f::Field; rev = false)

    area = sum(AreaField(f.grid))
    volume_field = VolumeField(f.grid)
    v = reshape(interior(volume_field), :)
    field_data = reshape(interior(f), :)

    p = sortperm(field_data; rev)
    sorted_field_data = field_data[p]
    z✶ = cumsum(v[p]) / area

    grid_arch = f.grid.architecture
    grid_size = prod(size(f.grid))
    new_grid = RectilinearGrid(grid_arch, size = grid_size, z = (-f.grid.Lz, 0), topology=(Flat, Flat, Bounded))

    sorted_field = CenterField(new_grid)
    set!(sorted_field, reshape(sorted_field_data, size(new_grid)))
    z✶_field = CenterField(new_grid)
    set!(z✶_field, reshape(z✶, size(new_grid)))

    return sorted_field, z✶_field

end

"""
    $(SIGNATURES)

Return a `kernelFunctionOperation` to compute an approximation to the `BackgroundPotentialEnergy`
per unit volume,
```math
E_{b} = \\frac{gρz✶}{ρ₀}.
```
The `BackgroundPotentialEnergy` is the potential energy computed by adiabatically resorting
the buoyancy or density field into a reference state of minimal potential energy.
The reference state, and corresponding `z✶` cooridinate, is approximated using [OneDReferenceField](@ref).
For more informaion about the background potential energy state and its implementation see
[Winters et al. (1995)](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/available-potential-energy-and-mixing-in-densitystratified-fluids/A45F1A40521FF0A0DC82BC705AD398DA)
and [Carpenter et al. (2012)](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/simulations-of-a-doublediffusive-interface-in-the-diffusive-convection-regime/63D2ECE2AA41439E01A01F9A0D76F2E2).

Examples
========

```jldoctest
julia> using Oceananigans

julia> using Oceanostics.PotentialEnergyEquationTerms: BackgroundPotentialEnergy

julia> grid = RectilinearGrid(size=100, z=(-1000, 0), topology=(Flat, Flat, Bounded));

julia> model = NonhydrostaticModel(; grid, buoyancy=BuoyancyTracer(), tracers=(:b,));

julia> bpe = BackgroundPotentialEnergy(model)
KernelFunctionOperation at (Center, Center, Center)
├── grid: 1×1×100 RectilinearGrid{Float64, Flat, Flat, Bounded} on CPU with 0×0×3 halo
├── kernel_function: bz✶_ccc (generic function with 1 method)
└── arguments: ("1×1×100 Field{Center, Center, Center} on RectilinearGrid on CPU", "1×1×100 Field{Center, Center, Center} on RectilinearGrid on CPU")
```
"""
@inline function BackgroundPotentialEnergy(model; location = (Center, Center, Center),
                                                  geopotential_height = model_geopotential_height(model))

    validate_location(location, "BackgroundPotentialEnergy")
    isnothing(model.buoyancy) ? nothing : validate_gravity_unit_vector(model.buoyancy.gravity_unit_vector)
    validate_grid(model.grid)

    return BackgroundPotentialEnergy(model, model.buoyancy, geopotential_height)
end

@inline function BackgroundPotentialEnergy(model, buoyancy_model::BuoyancyTracerModel, geopotential_height)

    grid = model.grid
    sorted_buoyancy, z✶ = OneDReferenceField(model.tracers.b)

    return KernelFunctionOperation{Center, Center, Center}(bz✶_ccc, grid, sorted_buoyancy, z✶)
end

@inline bz✶_ccc(i, j, k, grid, b, z✶) = b[i, j, k] * z✶[i, j, k]

linear_eos_buoyancy(grid, buoyancy, tracers) = KernelFunctionOperation{Center, Center, Center}(buoyancy_perturbationᶜᶜᶜ, grid, buoyancy, tracers)

@inline function BackgroundPotentialEnergy(model, buoyancy_model::BuoyancyLinearEOSModel, geopotential_height)

    grid = model.grid
    buoyancy = model.buoyancy.model
    tracers = model.tracers
    b = linear_eos_buoyancy(grid, buoyancy, tracers)
    compute!(b)
    sorted_buoyancy, z✶ = OneDReferenceField(b)

    return KernelFunctionOperation{Center, Center, Center}(bz✶_ccc, grid, sorted_buoyancy, z✶)
end

@inline function BackgroundPotentialEnergy(model, buoyancy_model::BuoyancyBoussinesqEOSModel, geopotential_height)

    grid = model.grid
    ρ = seawater_density(model; geopotential_height)
    compute!(ρ)
    sorted_density, z✶ = OneDReferenceField(ρ)
    parameters = (g = model.buoyancy.model.gravitational_acceleration,
                  ρ₀ = model.buoyancy.model.equation_of_state.reference_density)

    return KernelFunctionOperation{Center, Center, Center}(g′z✶_ccc, grid, sorted_density, z✶, parameters)
end

@inline g′z✶_ccc(i, j, k, grid, ρ, z✶, p) = (p.g / p.ρ₀) * ρ[i, j, k] * z✶[i, j, k]

end # module
