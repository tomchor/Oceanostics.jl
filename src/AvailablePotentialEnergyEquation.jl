module AvailablePotentialEnergyEquation

using DocStringExtensions

export BackgroundPotentialEnergy, AvailablePotentialEnergy, reference_height, reference_buoyancy
export ThreeDimensionalSort, HeavisideIntegral, OneDimensionalSort

using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Architectures: architecture, on_architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: Field, CenterField, FieldStatus, compute_at!, interior, set_status!
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid
using Oceananigans.Models: seawater_density, model_geopotential_height
using Oceananigans.Operators: Vᶜᶜᶜ
using Oceananigans.Grids: Center, Face, RectilinearGrid, Flat, znode, topology
using Oceananigans.BuoyancyFormulations: buoyancy_perturbationᶜᶜᶜ, Zᶜᶜᶜ
using Oceanostics: validate_location, CustomKFO

# The buoyancy formulations this budget dispatches over are the same ones `PotentialEnergy` uses, and
# `PotentialEnergy` itself is imported so its docstring `@ref`s resolve in-module.
using ..PotentialEnergyEquation: PotentialEnergy, NoBuoyancyModel, BuoyancyTracerModel,
                                 BuoyancyLinearEOSModel, BuoyancyBoussinesqEOSModel,
                                 validate_gravity_unit_vector

import Oceananigans.Fields: compute!

# The sorted reference state assumes the domain's horizontal cross-sectional area does not vary with
# depth, so that a cumulative volume maps onto a height by a simple division. That fails as soon as
# there is topography.
validate_sortable_grid(grid) = nothing
validate_sortable_grid(grid::ImmersedBoundaryGrid) =
    throw(ArgumentError("`BackgroundPotentialEnergy` and `AvailablePotentialEnergy` are not currently defined on \
                         an `ImmersedBoundaryGrid`, whose horizontal area varies with depth."))

#+++ Buoyancy as a single materialized `Field`
# The sorting below needs the buoyancy of every cell as plain data, and the `BackgroundPotentialEnergy`
# and `AvailablePotentialEnergy` kernels then read that same data, so `b` is materialized once here for
# all three buoyancy formulations. The sign convention matches `PotentialEnergyEquation`'s
# `minus_bz_ccc`: for a `BoussinesqEquationOfState` the buoyancy is `b = -gρ/ρ₀`, so the `Eₚ = -bz` it
# computes is the same `(g/ρ₀)ρz`, and the three energies stay consistent with each other.
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

Operand of the `Field` returned by [`reference_height`](@ref). It carries the buoyancy `Field`
that gets sorted, the (time-independent) cell volumes and domain geometry, and the workspaces that
every `compute!` reuses. Sorting couples every cell in the domain to every other one, so unlike every
other diagnostic in Oceanostics it cannot be expressed as a `KernelFunctionOperation`. This plays the
same role `Oceananigans.Fields.Scan` plays for `Integral` and `Average` instead, hooking a whole-field
computation into `compute!` so the reference state is refreshed whenever the diagnostic is written out.
"""
struct SortedReferenceState{M, B, V, P, W, FT}
    method :: M
    buoyancy :: B
    cell_volume :: V
    permutation :: P
    workspace :: W
    horizontal_area :: FT
    bottom_height :: FT
end

Base.summary(s::SortedReferenceState) =
    string("SortedReferenceState (", summary(s.method), ") of ", summary(s.buoyancy))

const SortedReferenceHeightField = Field{<:Any, <:Any, <:Any, <:SortedReferenceState}

#+++ Ways of building the sorted reference state
"""
    $(TYPEDEF)

Supertype of the three strategies [`reference_height`](@ref) offers for turning a buoyancy
field into a reference height: [`ThreeDimensionalSort`](@ref), [`HeavisideIntegral`](@ref) and
[`OneDimensionalSort`](@ref). They agree on every volume integral built from `z✶`, and differ along
two axes rather than one. [`OneDimensionalSort`](@ref) is the only one that moves the answer off the
model grid, onto a sorted column; the other two both leave `z✶` on the model grid and differ instead
in where they place cells of equal buoyancy.
"""
abstract type AbstractSortingMethod end

"""
    $(TYPEDEF)

Give every cell the height of its own slot in the sorted column: rank the cells by buoyancy, stack
them from the bottom of the domain, and read off where each one lands. `z✶` comes back on the model
grid, which is what the name contrasts with [`OneDimensionalSort`](@ref); note that
[`HeavisideIntegral`](@ref) also answers on the model grid and differs over ties instead. This is the
default.

Cells that share a buoyancy take consecutive slots rather than a shared height, so `z✶` spreads over
the depth the tied group fills, which is half a cell either side of `z` for a horizontally uniform
stratification. That spread is the volume-weighted mean of what [`HeavisideIntegral`](@ref) assigns,
so every volume integral agrees with it exactly, but it does make a cell-by-cell map of `z✶` noisy at
the grid scale wherever the buoyancy is uniform.
"""
struct ThreeDimensionalSort <: AbstractSortingMethod end

"""
    $(TYPEDEF)

Evaluate the reference height as [Winters et al. (1995)](https://doi.org/10.1017/S002211209500125X)
define it in their eq. (11),

```
    z✶(x) = z_bottom + (1/A) ∫ H(ρ(x′) - ρ(x)) dV′ ,
```

where `H` is the Heaviside step function, taking the value `1/2` where the two densities are equal.
That half-weight makes every cell of a given buoyancy share one `z✶`, the mid-height of the layer
that buoyancy class occupies in the sorted column, so `z✶` is a function of buoyancy alone and is
constant on isopycnals. `z✶` comes back on the model grid.

Prefer this to [`ThreeDimensionalSort`](@ref) when you want a cell-by-cell map: a horizontally uniform,
statically stable stratification gives `z✶ = z` exactly here, hence `Eₐ = 0` cell by cell rather than
only in the integral. It costs a couple of extra passes over the sorted cells to find the tied runs.
"""
struct HeavisideIntegral <: AbstractSortingMethod end

"""
    $(TYPEDEF)

Return the sorted column itself, on its own `1×1×N` grid with `N = Nx*Ny*Nz` cells that span the
domain's full horizontal area. The cells are reshaped rather than re-counted: each holds the same
volume as a cell of the model grid, so volume integrals over the column match those over the model
grid, and `z✶` is simply the column's own cell centers. The column keeps the model grid's topology,
collapsing each horizontal direction to a single cell, so a `Flat` direction stays `Flat` rather than
turning into a spurious periodic axis in any output.

This is the representation to reach for when you want the reference state as a profile, say to plot
`b✶(z✶)` or to differentiate it into a reference stratification. The parcels' original positions are
carried along, so [`AvailablePotentialEnergy`](@ref) still works, but its result is indexed by rank
in the sorted column rather than by position in the flow. Requires every cell of the model grid to
hold the same volume, since otherwise the column's cell boundaries would move as the flow evolves.
"""
struct OneDimensionalSort{B, H, V} <: AbstractSortingMethod
    reference_buoyancy :: B
    sorted_height :: H
    source_height :: V
end

OneDimensionalSort() = OneDimensionalSort(nothing, nothing, nothing)

Base.summary(::ThreeDimensionalSort) = "ThreeDimensionalSort"
Base.summary(::HeavisideIntegral) = "HeavisideIntegral"
Base.summary(::OneDimensionalSort) = "OneDimensionalSort"

"""
    $(TYPEDEF)

Operand of the `Field` [`reference_buoyancy`](@ref) returns for [`OneDimensionalSort`](@ref). The column's
buoyancy is filled as part of sorting `z✶`, not by a computation of its own, so this exists to make
that dependency explicit: `compute!` on the buoyancy delegates to the parent reference height, which
does the sort and writes both. Without it the buoyancy would be a bare `Field` that `compute!` silently
ignores, and an output writer handed it on its own would keep writing the previous sort's profile.
"""
struct SortedBuoyancyState{Z}
    reference_height :: Z
end

Base.summary(s::SortedBuoyancyState) = string("SortedBuoyancyState of ", summary(s.reference_height))

const SortedBuoyancyField = Field{<:Any, <:Any, <:Any, <:SortedBuoyancyState}

function compute!(b✶::SortedBuoyancyField, time=nothing)

    # Sorting the parent writes this field's data as a side effect, and is status-gated, so a `z✶` that
    # has already been computed at this `time` is not sorted twice.
    compute_at!(b✶.operand.reference_height, time)
    set_status!(b✶.status, time)

    return b✶
end

"""
    $(SIGNATURES)

Return the buoyancy `Field` that pairs with the reference height `z✶` cell by cell, so that
`(z✶, reference_buoyancy(z✶))` is the reference profile the sort produced.

Under [`OneDimensionalSort`](@ref) that is the sorted profile `b✶` living on the column alongside
`z✶`. Under [`ThreeDimensionalSort`](@ref) and [`HeavisideIntegral`](@ref) it is the model's own
buoyancy, which already pairs with `z✶` cell by cell since both live on the model grid; ordering
those pairs by `z✶` recovers the same profile the column stores directly.

The returned field recomputes itself, so it is safe to hand straight to an output writer: computing it
sorts the parent `z✶` if that has not already happened at this time. It shares its data with the column
the sort fills, so it costs no extra memory and stays consistent with `z✶` by construction.
"""
reference_buoyancy(z✶::SortedReferenceHeightField) =
    reference_buoyancy(z✶, z✶.operand.method, reference_buoyancy(z✶.operand))

# On the model grid the buoyancy is the model's own field, which already recomputes itself, so it is
# handed back untouched. On the sorted column it is storage the sort writes into, so it is wrapped in a
# field that shares that storage and knows to trigger the sort.
reference_buoyancy(z✶, ::AbstractSortingMethod, buoyancy) = buoyancy

reference_buoyancy(z✶, ::OneDimensionalSort, buoyancy) =
    Field{Center, Center, Center}(buoyancy.grid; data = buoyancy.data,
                                  operand = SortedBuoyancyState(z✶), status = FieldStatus())

# The buoyancy the diagnostics read, and the height they measure a parcel's displacement from. On the
# model grid both come straight from the flow (`nothing` height means "use the grid's own `Zᶜᶜᶜ`");
# on the sorted column both have to be carried in sorted order.
reference_buoyancy(s::SortedReferenceState) = reference_buoyancy(s.method, s.buoyancy)
reference_buoyancy(::AbstractSortingMethod, buoyancy) = buoyancy
reference_buoyancy(method::OneDimensionalSort, buoyancy) = method.reference_buoyancy

sorted_height(s::SortedReferenceState) = sorted_height(s.method)
sorted_height(::AbstractSortingMethod) = nothing
sorted_height(method::OneDimensionalSort) = method.sorted_height
#---

"""
    $(SIGNATURES)

Rank the cells of `s.buoyancy` by buoyancy, densest (lowest `b`) first, leaving the flattened
buoyancy in `s.workspace` and the ranking in `s.permutation`. Returns the cell volumes in that
sorted order, which is what every [`AbstractSortingMethod`](@ref) then accumulates.
"""
function rank_by_buoyancy!(s::SortedReferenceState)

    reshape(s.workspace, size(s.buoyancy)) .= interior(s.buoyancy)
    sortperm!(s.permutation, s.workspace)

    return s.cell_volume[s.permutation]
end

"""
    $(SIGNATURES)

Sort the buoyancy of `s` and write the resulting reference height `z✶` into `z✶_field`, following
`s.method`.

Cells are ranked by buoyancy, densest first, and stacked from the bottom of the domain up. The
cumulative volume of the stack, divided by the domain's horizontal area `A`, is the height the stack
has reached, so a cell whose rank puts it between cumulative volumes `V₋` and `V₊` sits at
`z✶ = z_bottom + (V₋ + V₊) / 2A`. Ranking is a permutation of the cells, so the sorted column holds
exactly the same volume as the original one and `z✶` covers the full depth of the domain.
"""
sort_buoyancy!(z✶_field, s::SortedReferenceState) = sort_buoyancy!(z✶_field, s, s.method)

function sort_buoyancy!(z✶_field, s::SortedReferenceState, ::ThreeDimensionalSort)
    sz   = size(z✶_field)
    work = s.workspace
    perm = s.permutation
    ΔV   = rank_by_buoyancy!(s)

    cumsum!(work, ΔV)                                              # volume of the column below each cell's top face
    @. ΔV = s.bottom_height + (work - ΔV / 2) / s.horizontal_area  # cell-center z✶, in sorted order
    work[perm] = ΔV                                                # scatter back to the original cell ordering

    interior(z✶_field) .= reshape(work, sz)

    return z✶_field
end

function sort_buoyancy!(z✶_field, s::SortedReferenceState, ::HeavisideIntegral)
    sz   = size(z✶_field)
    work = s.workspace
    perm = s.permutation
    FT   = eltype(work)
    N    = length(work)
    ΔV   = rank_by_buoyancy!(s)

    b = work[perm]      # buoyancy in sorted order, which is where the ties show up
    cumsum!(work, ΔV)   # volume of the column below each cell's top face
    below = work .- ΔV  # ... and below its bottom face

    # Eq. (11) gives every cell of a tied run the same z✶: the mid-height of the layer that run fills,
    # `(volume strictly denser + volume no lighter) / 2A`. Those two volumes are the run's first
    # `below` and last `work`. Both arrays are non-decreasing, so a running maximum over the run
    # starts and a running minimum back from the run ends recover them without materializing the runs.
    starts = similar(b, Bool)
    fill!(view(starts, 1:1), true)
    view(starts, 2:N) .= view(b, 2:N) .!= view(b, 1:N-1)

    ends = similar(starts)
    view(ends, 1:N-1) .= view(starts, 2:N)
    fill!(view(ends, N:N), true)

    denser = ifelse.(starts, below, typemin(FT))
    accumulate!(max, denser, denser)

    no_lighter = ifelse.(ends, work, typemax(FT))
    reverse!(no_lighter)
    accumulate!(min, no_lighter, no_lighter)
    reverse!(no_lighter)

    @. ΔV = s.bottom_height + (denser + no_lighter) / (2 * s.horizontal_area)
    work[perm] = ΔV

    interior(z✶_field) .= reshape(work, sz)

    return z✶_field
end

function sort_buoyancy!(z✶_field, s::SortedReferenceState, method::OneDimensionalSort)
    sz   = size(z✶_field) # (1, 1, N): the sorted column, not the model grid
    work = s.workspace
    perm = s.permutation
    ΔV   = rank_by_buoyancy!(s)

    interior(method.reference_buoyancy) .= reshape(work[perm], sz)                # buoyancy, densest first
    interior(method.sorted_height)   .= reshape(method.source_height[perm], sz) # where each parcel came from

    cumsum!(work, ΔV)
    @. ΔV = s.bottom_height + (work - ΔV / 2) / s.horizontal_area
    interior(z✶_field) .= reshape(ΔV, sz) # which is exactly the column's own cell centers

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
z✶ = reference_height(model)
E_b = BackgroundPotentialEnergy(model, z✶)
E_a = AvailablePotentialEnergy(model, z✶)
```

Unlike the pointwise diagnostics elsewhere in Oceanostics, `z✶` is defined by a sort of every cell in
the domain, and it is re-sorted on every `compute!`, so writing it (or anything built on it) out during
a simulation tracks the evolving flow, at a cost that grows like `N log N` in the number of cells. It
holds three `Nx*Ny*Nz` workspace arrays for its lifetime and allocates one more per sort.

The domain's horizontal cross-sectional area is taken to be independent of depth (true of a
`RectilinearGrid`, false as soon as there is topography), and an `ImmersedBoundaryGrid` is rejected for
that reason.

`method` picks how the sorted state is built. All three agree on every volume integral, so `∫E_b dV`
and `∫Eₐ dV` do not depend on the choice:

  - [`ThreeDimensionalSort`](@ref) (the default) gives each cell the height of its own slot in the sorted
    column, on the model grid. Tied cells take consecutive slots, which spreads `z✶` over a grid cell
    wherever the buoyancy is uniform.
  - [`HeavisideIntegral`](@ref) is eq. (11) of Winters et al. verbatim, also on the model grid. Tied
    cells share the mid-height of their layer, so `z✶` is a function of buoyancy alone and a
    cell-by-cell map is clean. Use this one for local fields.
  - [`OneDimensionalSort`](@ref) returns the sorted column itself, on a `1×1×N` grid of cells that span
    the domain's horizontal area, which is the form to use for a reference profile.

```jldoctest
using Oceananigans, Oceanostics

grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1), topology=(Periodic, Periodic, Bounded))
model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)
set!(model, b = (x, y, z) -> z)

# with eq. (11), a horizontally uniform stable stratification is exactly its own sorted state
z✶ = reference_height(model, method=HeavisideIntegral())
interior(z✶) ≈ reshape(znodes(grid, Center()), 1, 1, 4) .* ones(4, 4, 4)

# output

true
```

`geopotential_height` is passed through to `seawater_density` exactly as in [`PotentialEnergy`](@ref),
and defaults the same way, so the two diagnostics always sort and weight the same density. With a
nonlinear equation of state, prefer a fixed value (`geopotential_height = 0` for σ₀): sorting is only
meaningful for a variable the flow conserves, which in-situ density is not.

A second method, `reference_height(b::Field)`, sorts a `Field` you supply instead of the model's
buoyancy, which is what you want for a reference state built from a coarse-grained buoyancy `filter(b)`.
It sorts in ascending order, so `b` has to be buoyancy-like (large where the fluid is light); pass
`-ρ` rather than `ρ` for a density.

```jldoctest
using Oceananigans, Oceanostics

grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1), topology=(Periodic, Periodic, Bounded))
model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)
set!(model, b = (x, y, z) -> z)

z✶ = reference_height(model)

# a stably stratified field is already sorted, so z✶ stays within half a cell of z
maximum(abs, interior(z✶) .- reshape(znodes(grid, Center()), 1, 1, 4)) <= 0.5 * 0.25

# output

true
```
"""
function reference_height(model; method = ThreeDimensionalSort(),
                                 geopotential_height = model_geopotential_height(model))

    isnothing(model.buoyancy) ? nothing : validate_gravity_unit_vector(model.buoyancy.gravity_unit_vector)
    validate_sortable_grid(model.grid)

    return reference_height(buoyancy_field(model, model.buoyancy, geopotential_height); method)
end

function reference_height(b::Field; method = ThreeDimensionalSort())

    grid = b.grid
    validate_sortable_grid(grid)

    arch = architecture(grid)
    FT   = eltype(grid)
    N    = prod(size(grid))

    cell_volume = flat_grid_metric(grid, Vᶜᶜᶜ)

    z_bottom = convert(FT, znode(1, 1, 1, grid, Center(), Center(), Face()))
    # Taking the horizontal area from the total volume (rather than from the surface area) makes the
    # sorted column fill the domain's depth exactly, which is what keeps ∫Eₚ = ∫E_b for a sorted field.
    horizontal_area = convert(FT, sum(cell_volume) / grid.Lz)

    method  = build_sorting_method(method, grid, cell_volume, horizontal_area, z_bottom)
    operand = SortedReferenceState(method, b, cell_volume,
                                   on_architecture(arch, zeros(Int, N)),
                                   on_architecture(arch, zeros(FT, N)),
                                   horizontal_area, z_bottom)

    return compute!(Field{Center, Center, Center}(sorting_grid(method, grid); operand, status = FieldStatus()))
end

#+++ Per-method setup
"Evaluate a grid metric (a `(i, j, k, grid)` operator) over every cell and return it as a flat vector."
function flat_grid_metric(grid, metric)

    flat = on_architecture(architecture(grid), zeros(eltype(grid), prod(size(grid))))
    reshape(flat, size(grid)) .= interior(Field(KernelFunctionOperation{Center, Center, Center}(metric, grid)))

    return flat
end

# `ThreeDimensionalSort` and `HeavisideIntegral` write `z✶` onto the model grid; `OneDimensionalSort` writes it
# onto the sorted column it allocates below.
sorting_grid(::AbstractSortingMethod, grid) = grid
sorting_grid(method::OneDimensionalSort, grid) = method.reference_buoyancy.grid

build_sorting_method(method::AbstractSortingMethod, grid, cell_volume, horizontal_area, z_bottom) = method

function build_sorting_method(::OneDimensionalSort, grid, cell_volume, horizontal_area, z_bottom)

    # The column's cell boundaries sit at the cumulative volume of the sorted cells. They may only be
    # baked into a grid if that stacking cannot change, which needs every cell to hold the same volume.
    ΔV_max, ΔV_min = maximum(cell_volume), minimum(cell_volume)
    ΔV_max - ΔV_min > sqrt(eps(eltype(cell_volume))) * ΔV_max &&
        throw(ArgumentError("`OneDimensionalSort` needs every cell of the grid to hold the same volume, but they \
                             range over [$ΔV_min, $ΔV_max]. Use `ThreeDimensionalSort()` or `HeavisideIntegral()` on a \
                             grid with variable spacing."))

    # The column collapses the two horizontal directions to a single cell and stacks the sorted cells
    # in the vertical, but it keeps the model grid's topology so that, e.g., a `Flat` horizontal stays
    # `Flat` rather than becoming a spurious periodic axis in the output. A `Flat` direction takes no
    # size entry and no coordinate; the others get a single cell spanning the domain.
    tx, ty, tz = topology(grid)
    N = length(cell_volume)
    column_size = Tuple(s for (s, T) in zip((1, 1, N), (tx, ty, tz)) if T !== Flat)

    x_kw = tx === Flat ? (;) : (; x = (0, grid.Lx))
    y_kw = ty === Flat ? (;) : (; y = (0, grid.Ly))

    column = RectilinearGrid(architecture(grid), eltype(grid);
                             size = column_size, topology = (tx, ty, tz),
                             z = (z_bottom, z_bottom + grid.Lz), x_kw..., y_kw...)

    return OneDimensionalSort(CenterField(column), CenterField(column), flat_grid_metric(grid, Zᶜᶜᶜ))
end
#---
#---

#+++ Background and available potential energy
@inline minus_bz✶_ccc(i, j, k, grid, b, z✶) = @inbounds -b[i, j, k] * z✶[i, j, k]
# `Eₐ` multiplies the displacement `z - z✶` rather than subtracting `Eₚ - E_b`: near equilibrium the two
# energies are large and nearly equal, and differencing the heights first avoids that cancellation. The
# parcel's own height is the grid's `Zᶜᶜᶜ` on the model grid, and a carried field on a sorted column.
@inline minus_bδz✶_ccc(i, j, k, grid, b, z✶) = @inbounds -b[i, j, k] * (Zᶜᶜᶜ(i, j, k, grid) - z✶[i, j, k])
@inline minus_bδz✶_ccc(i, j, k, grid, b, z, z✶) = @inbounds -b[i, j, k] * (z[i, j, k] - z✶[i, j, k])

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
[`reference_height`](@ref); pass one explicitly to share a single sort with
[`AvailablePotentialEnergy`](@ref), or pass `method` through to choose how it is built
([`ThreeDimensionalSort`](@ref), [`HeavisideIntegral`](@ref) or [`OneDimensionalSort`](@ref); `Integral(E_b)` is
the same either way).

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
function BackgroundPotentialEnergy(model; location = (Center, Center, Center), method = ThreeDimensionalSort(),
                                   geopotential_height = model_geopotential_height(model))

    validate_location(location, "BackgroundPotentialEnergy")

    return BackgroundPotentialEnergy(model, reference_height(model; method, geopotential_height))
end

BackgroundPotentialEnergy(model, z✶::SortedReferenceHeightField) =
    KernelFunctionOperation{Center, Center, Center}(minus_bz✶_ccc, z✶.grid, reference_buoyancy(z✶.operand), z✶)

"""
    $(SIGNATURES)

Return a `KernelFunctionOperation` computing the available potential energy per unit volume,

```
    Eₐ = Eₚ - E_b = -b (z - z✶)
```

the part of the potential energy that an adiabatic rearrangement can release into kinetic energy
([Winters et al., 1995](https://doi.org/10.1017/S002211209500125X), eq. 23). `z✶` is the reference
height computed by [`reference_height`](@ref); pass one explicitly to share a single sort with
[`BackgroundPotentialEnergy`](@ref), or pass `method` through to choose how it is built.

The decomposition `Eₚ = E_b + Eₐ` holds cell by cell, so up to roundoff `Integral(Eₐ)` is
`Integral(PotentialEnergy(model)) - Integral(BackgroundPotentialEnergy(model))`, and it vanishes for a
statically stable, horizontally uniform stratification. Only the volume integral is guaranteed
non-negative: cell by cell, `Eₐ` goes negative wherever a parcel sits below its reference height. It
also vanishes cell by cell in that uniform case only under [`HeavisideIntegral`](@ref), since
[`ThreeDimensionalSort`](@ref) spreads tied cells over a grid cell. The result lives at
`(Center, Center, Center)`, per unit mass (units `m² s⁻²`), and is defined for the same buoyancy
formulations as [`PotentialEnergy`](@ref). Under [`OneDimensionalSort`](@ref) it lands on the sorted
column, indexed by rank rather than by position in the flow.

```jldoctest
using Oceananigans, Oceanostics

grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1), topology=(Periodic, Periodic, Bounded))
model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)

AvailablePotentialEnergy(model)

# output

AvailablePotentialEnergy KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: minus_bδz✶_ccc (generic function with 2 methods)
└── arguments: ("Field", "Field")
└── computes: available potential energy per unit volume  Eₐ = -b(z - z✶)
```
"""
function AvailablePotentialEnergy(model; location = (Center, Center, Center), method = ThreeDimensionalSort(),
                                  geopotential_height = model_geopotential_height(model))

    validate_location(location, "AvailablePotentialEnergy")

    return AvailablePotentialEnergy(model, reference_height(model; method, geopotential_height))
end

AvailablePotentialEnergy(model, z✶::SortedReferenceHeightField) = available_potential_energy(z✶, reference_buoyancy(z✶.operand), sorted_height(z✶.operand))

# On the model grid the parcel's own height is the grid's; on a sorted column it has to be carried.
available_potential_energy(z✶, b, ::Nothing) = KernelFunctionOperation{Center, Center, Center}(minus_bδz✶_ccc, z✶.grid, b, z✶)

available_potential_energy(z✶, b, z) = KernelFunctionOperation{Center, Center, Center}(minus_bδz✶_ccc, z✶.grid, b, z, z✶)
#---

end # module
