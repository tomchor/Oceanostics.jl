module PotentialEnergyEquation

using DocStringExtensions

export PotentialEnergy, BackgroundPotentialEnergy, AvailablePotentialEnergy, sorted_reference_height

using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Architectures: architecture, on_architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: Field, FieldStatus, compute_at!, interior, set_status!
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid
using Oceananigans.Models: seawater_density
using Oceananigans.Models: model_geopotential_height
using Oceananigans.Operators: Vᶜᶜᶜ
using Oceananigans.Grids: Center, Face
using Oceananigans.Grids: NegativeZDirection, znode
using Oceananigans.BuoyancyFormulations: BuoyancyForce, BuoyancyTracer, SeawaterBuoyancy, LinearEquationOfState
using Oceananigans.BuoyancyFormulations: buoyancy_perturbationᶜᶜᶜ, Zᶜᶜᶜ
using Oceananigans.Models: ShallowWaterModel
using Oceanostics: validate_location, CustomKFO
using SeawaterPolynomials: BoussinesqEquationOfState

import Oceananigans.Fields: compute!

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

# The sorted reference state assumes the domain's horizontal cross-sectional area does not vary with
# depth, so that a cumulative volume maps onto a height by a simple division. That fails as soon as
# there is topography.
validate_sortable_grid(grid) = nothing
validate_sortable_grid(grid::ImmersedBoundaryGrid) =
    throw(ArgumentError("`BackgroundPotentialEnergy` and `AvailablePotentialEnergy` are not currently defined on \
                         an `ImmersedBoundaryGrid`, whose horizontal area varies with depth."))

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
└── computes: potential energy per unit volume  Eₚ = -bz
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
└── computes: potential energy per unit volume  Eₚ = -bz
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
└── computes: potential energy per unit volume  Eₚ = -bz
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

#+++ Buoyancy as a single materialized `Field`
# The sorting below needs the buoyancy of every cell as plain data, and the `BackgroundPotentialEnergy`
# and `AvailablePotentialEnergy` kernels then read that same data, so `b` is materialized once here for
# all three buoyancy formulations. The sign convention matches `minus_bz_ccc`: for a
# `BoussinesqEquationOfState` the buoyancy is `b = -gρ/ρ₀`, so `Eₚ = -bz` is the `(g/ρ₀)ρz` computed above.
@inline minus_gρ_over_ρ₀_ccc(i, j, k, grid, ρ, p) = @inbounds -(p.g / p.ρ₀) * ρ[i, j, k]

@inline buoyancy_field(model, buoyancy_model::NoBuoyancyModel, geopotential_height) =
    throw(ArgumentError("Cannot calculate gravitational potential energy without a Buoyancy model."))

@inline buoyancy_field(model, buoyancy_model::BuoyancyTracerModel, geopotential_height) = model.tracers.b

@inline buoyancy_field(model, buoyancy_model::BuoyancyLinearEOSModel, geopotential_height) =
    Field(KernelFunctionOperation{Center, Center, Center}(buoyancy_perturbationᶜᶜᶜ, model.grid,
                                                          buoyancy_model.formulation, model.tracers))

@inline function buoyancy_field(model, buoyancy_model::BuoyancyBoussinesqEOSModel, geopotential_height)

    ρ = seawater_density(model; geopotential_height)
    parameters = (g = buoyancy_model.formulation.gravitational_acceleration,
                  ρ₀ = buoyancy_model.formulation.equation_of_state.reference_density)

    return Field(KernelFunctionOperation{Center, Center, Center}(minus_gρ_over_ρ₀_ccc, model.grid, ρ, parameters))
end
#---

#+++ Adiabatically sorted reference state
"""
    $(TYPEDEF)

Operand of the `Field` returned by [`sorted_reference_height`](@ref). It carries the buoyancy `Field`
that gets sorted, the (time-independent) cell volumes and domain geometry, and the workspaces that
every `compute!` reuses. Sorting couples every cell in the domain to every other one, so unlike every
other diagnostic in Oceanostics it cannot be expressed as a `KernelFunctionOperation`. This plays the
same role `Oceananigans.Fields.Scan` plays for `Integral` and `Average` instead, hooking a whole-field
computation into `compute!` so the reference state is refreshed whenever the diagnostic is written out.
"""
struct SortedReferenceState{B, V, P, W, FT}
    buoyancy :: B
    cell_volume :: V
    permutation :: P
    workspace :: W
    horizontal_area :: FT
    bottom_height :: FT
end

Base.summary(s::SortedReferenceState) = string("SortedReferenceState of ", summary(s.buoyancy))

const SortedReferenceHeightField = Field{<:Any, <:Any, <:Any, <:SortedReferenceState}

"""
    $(SIGNATURES)

Sort the buoyancy of `s` and write the resulting reference height `z✶` into `z✶_field`.

Cells are ranked by buoyancy, densest first, and stacked from the bottom of the domain up. The
cumulative volume of the stack, divided by the domain's horizontal area `A`, is the height the stack
has reached, so a cell whose rank puts it between cumulative volumes `V₋` and `V₊` sits at
`z✶ = z_bottom + (V₋ + V₊) / 2A`. Ranking is a permutation of the cells, so the sorted column holds
exactly the same volume as the original one and `z✶` covers the full depth of the domain.
"""
function sort_buoyancy!(z✶_field, s::SortedReferenceState)

    sz   = size(z✶_field)
    work = s.workspace
    perm = s.permutation

    reshape(work, sz) .= interior(s.buoyancy) # flatten the buoyancy field into the workspace
    sortperm!(perm, work)                     # rank the cells by buoyancy, densest (lowest b) first

    ΔV = s.cell_volume[perm]                                       # cell volumes, in sorted order
    cumsum!(work, ΔV)                                              # volume of the column below each cell's top face
    @. ΔV = s.bottom_height + (work - ΔV / 2) / s.horizontal_area  # cell-center z✶, in sorted order
    work[perm] = ΔV                                                # scatter back to the original cell ordering

    interior(z✶_field) .= reshape(work, sz)

    return z✶_field
end

function compute!(z✶_field::SortedReferenceHeightField, time=nothing)

    s = z✶_field.operand
    compute_at!(s.buoyancy, time)
    sort_buoyancy!(z✶_field, s)
    fill_halo_regions!(z✶_field)
    set_status!(z✶_field.status, time)

    return z✶_field
end

"""
    $(SIGNATURES)

Return a `Field` holding the reference height `z✶`: the height each parcel would occupy once the
buoyancy field is rearranged adiabatically into the state of minimum potential energy, following
[Winters et al. (1995)](https://doi.org/10.1017/S002211209500125X). It is the building block of
[`BackgroundPotentialEnergy`](@ref) and [`AvailablePotentialEnergy`](@ref), and both accept a `z✶` you
built yourself so a pair of them can share one sort:

```julia
z✶ = sorted_reference_height(model)
E_b = BackgroundPotentialEnergy(model, z✶)
E_a = AvailablePotentialEnergy(model, z✶)
```

Unlike the pointwise diagnostics elsewhere in Oceanostics, `z✶` is defined by a sort of every cell in
the domain, and it is re-sorted on every `compute!`, so writing it (or anything built on it) out during
a simulation tracks the evolving flow, at a cost that grows like `N log N` in the number of cells. It
holds three `Nx*Ny*Nz` workspace arrays for its lifetime and allocates one more per sort.

The domain's horizontal cross-sectional area is taken to be independent of depth (true of a
`RectilinearGrid`, false as soon as there is topography), and an `ImmersedBoundaryGrid` is rejected for
that reason. Cells of exactly equal buoyancy get consecutive slots in the sorted column rather than a
shared height, so `z✶` spreads over the depth the tied group fills: half a cell either side of `z` for
a horizontally uniform stratification, where the cells tied at each level fill exactly that level. The
spread cancels in the volume integrals the sorted state is meant for, but it does make a cell-by-cell
map of `z✶` noisy at the grid scale wherever the buoyancy is uniform.

`geopotential_height` is passed through to `seawater_density` exactly as in [`PotentialEnergy`](@ref),
and defaults the same way, so the two diagnostics always sort and weight the same density. With a
nonlinear equation of state, prefer a fixed value (`geopotential_height = 0` for σ₀): sorting is only
meaningful for a variable the flow conserves, which in-situ density is not.

A second method, `sorted_reference_height(b::Field)`, sorts a `Field` you supply instead of the model's
buoyancy, which is what you want for a reference state built from a coarse-grained buoyancy `filter(b)`.
It sorts in ascending order, so `b` has to be buoyancy-like (large where the fluid is light); pass
`-ρ` rather than `ρ` for a density.

```jldoctest
using Oceananigans, Oceanostics

grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1), topology=(Periodic, Periodic, Bounded))
model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)
set!(model, b = (x, y, z) -> z)

z✶ = sorted_reference_height(model)

# a stably stratified field is already sorted, so z✶ stays within half a cell of z
maximum(abs, interior(z✶) .- reshape(znodes(grid, Center()), 1, 1, 4)) <= 0.5 * 0.25

# output

true
```
"""
function sorted_reference_height(model; geopotential_height = model_geopotential_height(model))

    isnothing(model.buoyancy) ? nothing : validate_gravity_unit_vector(model.buoyancy.gravity_unit_vector)
    validate_sortable_grid(model.grid)

    return sorted_reference_height(buoyancy_field(model, model.buoyancy, geopotential_height))
end

function sorted_reference_height(b::Field)

    grid = b.grid
    validate_sortable_grid(grid)

    arch = architecture(grid)
    FT   = eltype(grid)
    N    = prod(size(grid))

    cell_volume = on_architecture(arch, zeros(FT, N))
    reshape(cell_volume, size(grid)) .= interior(Field(KernelFunctionOperation{Center, Center, Center}(Vᶜᶜᶜ, grid)))

    z_bottom = znode(1, 1, 1, grid, Center(), Center(), Face())
    # Taking the horizontal area from the total volume (rather than from the surface area) makes the
    # sorted column fill the domain's depth exactly, which is what keeps ∫Eₚ = ∫E_b for a sorted field.
    horizontal_area = sum(cell_volume) / grid.Lz

    operand = SortedReferenceState(b, cell_volume,
                                   on_architecture(arch, zeros(Int, N)),
                                   on_architecture(arch, zeros(FT, N)),
                                   convert(FT, horizontal_area),
                                   convert(FT, z_bottom))

    return compute!(Field{Center, Center, Center}(grid; operand, status = FieldStatus()))
end
#---

#+++ Background and available potential energy
@inline minus_bz✶_ccc(i, j, k, grid, b, z✶) = @inbounds -b[i, j, k] * z✶[i, j, k]
# `Eₐ` multiplies the *displacement* `z - z✶` rather than subtracting `Eₚ - E_b`: near equilibrium the two
# energies are large and nearly equal, and differencing the heights first avoids that cancellation.
@inline minus_bδz✶_ccc(i, j, k, grid, b, z✶) = @inbounds -b[i, j, k] * (Zᶜᶜᶜ(i, j, k, grid) - z✶[i, j, k])

const BackgroundPotentialEnergy = CustomKFO{<:typeof(minus_bz✶_ccc)}
const AvailablePotentialEnergy = CustomKFO{<:typeof(minus_bδz✶_ccc)}

"""
    $(SIGNATURES)

Return a `KernelFunctionOperation` computing the background (or reference) potential energy per unit
volume,

```
    E_b = -b z✶ = (g/ρ₀) ρ z✶
```

the potential energy the fluid would retain if it were rearranged adiabatically into the state of
minimum potential energy ([Winters et al., 1995](https://doi.org/10.1017/S002211209500125X), eq. 22).
`z✶` is the reference height that rearrangement assigns to each parcel, computed by
[`sorted_reference_height`](@ref); pass one explicitly to share a single sort with
[`AvailablePotentialEnergy`](@ref).

`E_b` responds only to irreversible changes in the buoyancy field, so in a closed domain the continuous
equations make `Integral(E_b)` grow monotonically at the diapycnal mixing rate. Numerically it also
picks up whatever spurious diapycnal transport the advection scheme introduces, in either direction,
which is what makes it a standard measure of a scheme's mixing. The remainder `Eₚ - E_b` is the
[`AvailablePotentialEnergy`](@ref), the part that can be converted to kinetic energy. The result lives
at `(Center, Center, Center)`, per unit mass (units `m² s⁻²`), and is defined for the same buoyancy
formulations as [`PotentialEnergy`](@ref).

```jldoctest
using Oceananigans, Oceanostics

grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1), topology=(Periodic, Periodic, Bounded))
model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)

BackgroundPotentialEnergy(model)

# output

BackgroundPotentialEnergy KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: minus_bz✶_ccc (generic function with 1 method)
└── arguments: ("Field", "Field")
└── computes: background potential energy per unit volume  E_b = -bz✶
```
"""
function BackgroundPotentialEnergy(model; location = (Center, Center, Center),
                                   geopotential_height = model_geopotential_height(model))

    validate_location(location, "BackgroundPotentialEnergy")

    return BackgroundPotentialEnergy(model, sorted_reference_height(model; geopotential_height))
end

BackgroundPotentialEnergy(model, z✶::SortedReferenceHeightField) =
    KernelFunctionOperation{Center, Center, Center}(minus_bz✶_ccc, z✶.grid, z✶.operand.buoyancy, z✶)

"""
    $(SIGNATURES)

Return a `KernelFunctionOperation` computing the available potential energy per unit volume,

```
    Eₐ = Eₚ - E_b = -b (z - z✶)
```

the part of the potential energy that an adiabatic rearrangement can release into kinetic energy
([Winters et al., 1995](https://doi.org/10.1017/S002211209500125X), eq. 23). `z✶` is the reference
height computed by [`sorted_reference_height`](@ref); pass one explicitly to share a single sort with
[`BackgroundPotentialEnergy`](@ref).

The decomposition `Eₚ = E_b + Eₐ` holds cell by cell, so up to roundoff `Integral(Eₐ)` is
`Integral(PotentialEnergy(model)) - Integral(BackgroundPotentialEnergy(model))`, and it vanishes for a
statically stable, horizontally uniform stratification. Only the volume integral is guaranteed
non-negative: cell by cell, `Eₐ` goes negative wherever a parcel sits below its reference height. The
result lives at `(Center, Center, Center)`, per unit mass (units `m² s⁻²`), and is defined for the same
buoyancy formulations as [`PotentialEnergy`](@ref).

```jldoctest
using Oceananigans, Oceanostics

grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1), topology=(Periodic, Periodic, Bounded))
model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)

AvailablePotentialEnergy(model)

# output

AvailablePotentialEnergy KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: minus_bδz✶_ccc (generic function with 1 method)
└── arguments: ("Field", "Field")
└── computes: available potential energy per unit volume  Eₐ = -b(z - z✶)
```
"""
function AvailablePotentialEnergy(model; location = (Center, Center, Center),
                                  geopotential_height = model_geopotential_height(model))

    validate_location(location, "AvailablePotentialEnergy")

    return AvailablePotentialEnergy(model, sorted_reference_height(model; geopotential_height))
end

AvailablePotentialEnergy(model, z✶::SortedReferenceHeightField) =
    KernelFunctionOperation{Center, Center, Center}(minus_bδz✶_ccc, z✶.grid, z✶.operand.buoyancy, z✶)
#---

end # module
