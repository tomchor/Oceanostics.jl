module BackgroundPotentialEnergyEquation

using DocStringExtensions

export BackgroundPotentialEnergy, reference_height, reference_buoyancy, reference_buoyancy_at_height
export ThreeDimensionalSort, HeavisideIntegral, VerticalSort, ProfileLookup

using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Architectures: CPU, architecture, on_architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: Field, CenterField, FieldStatus, compute_at!, interior, set_status!
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid
using Oceananigans.Models: seawater_density, model_geopotential_height
using Oceananigans.Operators: Vᶜᶜᶜ
using Oceananigans.Grids: Center, Face, RectilinearGrid, Flat, znode, topology, XYZRegularRG
using Oceananigans.BuoyancyFormulations: buoyancy_perturbationᶜᶜᶜ, Zᶜᶜᶜ
using Oceanostics: validate_location, CustomKFO

# The buoyancy formulations this budget dispatches over are the same ones `PotentialEnergy` uses, and
# `PotentialEnergy` itself is imported so its docstring `@ref`s resolve in-module.
using ..PotentialEnergyEquation: PotentialEnergy, NoBuoyancyModel, BuoyancyTracerModel,
                                 BuoyancyLinearEOSModel, BuoyancyBoussinesqEOSModel,
                                 validate_gravity_unit_vector, validate_gravity_is_z_aligned

import Oceananigans.Fields: compute!

# The sort-and-stack reference state assumes uniform cells stacked under a horizontal cross-sectional
# area that does not vary with depth. Topography breaks both, so no method runs on an
# `ImmersedBoundaryGrid` yet; a stretched grid breaks only the first, so every method but
# `HeavisideIntegral` additionally needs uniform cell volumes. See `validate_grid_for_method` further
# down, once the method types exist.

#+++ Buoyancy as a single materialized `Field`
# The sorting below needs the buoyancy of every cell as plain data, and the `BackgroundPotentialEnergy`
# and `AvailablePotentialEnergy` kernels then read that same data, so `b` is materialized once here for
# all three buoyancy formulations. The sign convention matches `PotentialEnergyEquation`'s
# `minus_bz_ccc`: for a `BoussinesqEquationOfState` the buoyancy is `b = -gρ/ρ₀`, so the `eₚ = -bz` it
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
struct SortedReferenceState{M, B, V, P, W, S, A, FT}
    method :: M
    buoyancy :: B
    cell_volume :: V
    permutation :: P
    workspace :: W
    source_height :: S
    reference_potential :: A
    horizontal_area :: FT
    bottom_height :: FT
end

Base.summary(s::SortedReferenceState) =
    string("SortedReferenceState (", summary(s.method), ") of ", summary(s.buoyancy))

const SortedReferenceHeightField = Field{<:Any, <:Any, <:Any, <:SortedReferenceState}

#+++ Ways of building the sorted reference state
"""
    $(TYPEDEF)

Supertype of the four strategies [`reference_height`](@ref) offers for turning a buoyancy
field into a reference height: [`ThreeDimensionalSort`](@ref), [`HeavisideIntegral`](@ref),
[`ProfileLookup`](@ref) and [`VerticalSort`](@ref).
"""
abstract type AbstractReferenceHeightMethod end

"""
    $(TYPEDEF)

Give every cell the height of its own slot in the sorted column: rank the cells by buoyancy, stack
them from the bottom of the domain, and read off where each one lands. `z✶` comes back on the model
grid, which is what the name contrasts with [`VerticalSort`](@ref); note that
[`HeavisideIntegral`](@ref) also answers on the model grid and differs over ties instead. This is the
default.

Cells that share a buoyancy take consecutive slots rather than a shared height, so `z✶` spreads over
the depth the tied group fills, which is half a cell either side of `z` for a horizontally uniform
stratification. That spread is the volume-weighted mean of what [`HeavisideIntegral`](@ref) assigns,
so every volume integral agrees with it exactly, but it does make a cell-by-cell map of `z✶` noisy at
the grid scale wherever the buoyancy is uniform.
"""
struct ThreeDimensionalSort <: AbstractReferenceHeightMethod end

"""
    $(TYPEDEF)

Evaluate the reference height as [Winters et al. (1995)](https://doi.org/10.1017/S002211209500125X)
define it in their Eq. (11),

```
    z✶(x) = z_bottom + (1/A) ∫ H(ρ(x′) - ρ(x)) dV′ ,
```

where `H` is the Heaviside step function, taking the value `1/2` where the two densities are equal.
That half-weight makes every cell of a given buoyancy share one `z✶`, the mid-height of the layer
that buoyancy class occupies in the sorted column, so `z✶` is a function of buoyancy alone and is
constant on isopycnals. `z✶` comes back on the model grid.

Prefer this to [`ThreeDimensionalSort`](@ref) when you want a cell-by-cell map: a horizontally uniform,
statically stable stratification gives `z✶ = z` exactly here, hence `eₐ = 0` cell by cell rather than
only in the integral. It costs a couple of extra passes over the sorted cells to find the tied runs.
"""
struct HeavisideIntegral <: AbstractReferenceHeightMethod end

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
carried along, so [`AvailablePotentialEnergy`](@ref Oceanostics.AvailablePotentialEnergyEquation.AvailablePotentialEnergy) still works, but its result is indexed by rank
in the sorted column rather than by position in the flow. Requires every cell of the model grid to
hold the same volume, since otherwise the column's cell boundaries would move as the flow evolves.
"""
struct VerticalSort{B, H} <: AbstractReferenceHeightMethod
    reference_buoyancy :: B
    sorted_height :: H
end

VerticalSort() = VerticalSort(nothing, nothing)

"""
    $(TYPEDEF)

Give each cell the height of the slot whose buoyancy matches its own, found by a binary search into a
sorted reference profile. `z✶` comes back on the model grid.

A cell is matched to the profile through its buoyancy, never through where it came from, so the
profile does not have to be the one this field would produce by sorting itself. Three ways to supply
it:

  - `ProfileLookup()` sorts the field's own buoyancy, exactly as the other methods do. This
    reproduces the same `z✶` those methods assign, up to which slot of a tied run is picked.
  - `ProfileLookup(z✶_column)` takes the profile from a reference height built with
    [`VerticalSort`](@ref), and reads it back onto the model grid. The column is recomputed
    first, so the profile still tracks the flow.
  - `ProfileLookup(b✶, z✶)` takes any paired buoyancy and height, ordered from the densest fluid up,
    and does no sorting at all. Pass arrays to hold the reference state fixed while the flow evolves;
    they are moved to the model's architecture when the diagnostic is built, so a plain `Vector` works
    on a GPU. Pass `Field`s and they are recomputed on every `compute!` instead, so the profile tracks
    whatever they are built from rather than staying fixed.

Like [`HeavisideIntegral`](@ref) this makes `z✶` a function of buoyancy alone, so it is constant on
isopycnals: a tied run is placed at the mid-height of the band it fills, exactly as that method does. A
buoyancy that matches no entry of the profile (which only an external profile can produce, e.g. a
filtered field looked up in the full field's profile) is assigned to the nearest buoyancy class and
placed at that class's mid-height, so a value off the profile by roundoff lands exactly where an exact
match would.

!!! warning "Non-negativity needs a profile that resolves the field"
    `eₐ ≥ 0` rests on a parcel carrying `b = b✶(z✶)` exactly, which holds when the profile contains the
    buoyancies the field actually has. A profile sorted from the same field at the same time always
    does, so `ProfileLookup()` and `ProfileLookup(z✶_column)` are safe. For other profiles `eₐ` cannot
    be guaranteed to be non-negative.

All three forms describe the same reference state when the profile comes from the field being
diagnosed:

```jldoctest
using Oceananigans, Oceanostics
using Oceananigans.Fields: compute!, interior

grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1), topology=(Periodic, Periodic, Bounded))
model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)
set!(model, b = (x, y, z) -> z)

# sort the field's own buoyancy
z✶ = reference_height(model, method=ProfileLookup())

# borrow the column VerticalSort builds; it is recomputed alongside, so it tracks the flow
z✶_column = reference_height(model, method=VerticalSort())
z✶_borrowed = reference_height(model, method=ProfileLookup(z✶_column))

# any paired (b✶, z✶), here a snapshot of that column held fixed in time
compute!(z✶_column)
b✶ = vec(interior(reference_buoyancy(z✶_column)))
z✶_profile = vec(interior(z✶_column))
z✶_fixed = reference_height(model, method=ProfileLookup(b✶, z✶_profile))

interior(z✶) ≈ interior(z✶_borrowed) ≈ interior(z✶_fixed)

# output

true
```
"""
struct ProfileLookup{P} <: AbstractReferenceHeightMethod
    profile :: P
end

ProfileLookup() = ProfileLookup(nothing)
ProfileLookup(b✶, z✶) = ProfileLookup((b✶, z✶))

Base.summary(::ThreeDimensionalSort) = "ThreeDimensionalSort"
Base.summary(::HeavisideIntegral) = "HeavisideIntegral"
Base.summary(::VerticalSort) = "VerticalSort"
Base.summary(::ProfileLookup{Nothing}) = "ProfileLookup"
Base.summary(::ProfileLookup) = "ProfileLookup (external profile)"

"""
    $(TYPEDEF)

Operand of the `Field` [`reference_buoyancy`](@ref) returns for [`VerticalSort`](@ref). The column's
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

Under [`VerticalSort`](@ref) that is the sorted profile `b✶` living on the column alongside
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
reference_buoyancy(z✶, ::AbstractReferenceHeightMethod, buoyancy) = buoyancy

reference_buoyancy(z✶, ::VerticalSort, buoyancy) =
    Field{Center, Center, Center}(buoyancy.grid; data = buoyancy.data,
                                  operand = SortedBuoyancyState(z✶), status = FieldStatus())

# The buoyancy the diagnostics read, and the height they measure a parcel's displacement from. On the
# model grid both come straight from the flow (`nothing` height means "use the grid's own `Zᶜᶜᶜ`");
# on the sorted column both have to be carried in sorted order.
reference_buoyancy(s::SortedReferenceState) = reference_buoyancy(s.method, s.buoyancy)
reference_buoyancy(::AbstractReferenceHeightMethod, buoyancy) = buoyancy
reference_buoyancy(method::VerticalSort, buoyancy) = method.reference_buoyancy

sorted_height(s::SortedReferenceState) = sorted_height(s.method)
sorted_height(::AbstractReferenceHeightMethod) = nothing
sorted_height(method::VerticalSort) = method.sorted_height
#---

"""
    $(SIGNATURES)

Rank the cells of `s.buoyancy` by buoyancy, densest (lowest `b`) first, leaving the flattened
buoyancy in `s.workspace` and the ranking in `s.permutation`. Returns the cell volumes and the
buoyancies in that sorted order, which is what every [`AbstractReferenceHeightMethod`](@ref) then accumulates.
"""
function rank_by_buoyancy!(s::SortedReferenceState)

    reshape(s.workspace, size(s.buoyancy)) .= interior(s.buoyancy)
    sortperm!(s.permutation, s.workspace)

    return s.cell_volume[s.permutation], s.workspace[s.permutation]
end

"""
    $(SIGNATURES)

The centre height of each slot in the sorted column: cells stacked from the bottom of the domain by
cumulative volume, each placed at the midpoint of the slot it fills. `scratch` is overwritten with the
cumulative volumes, and the centres come back as a fresh array, so a caller may pass `s.workspace` and
still scatter the result through the permutation afterwards.
"""
function slot_centers!(scratch, s::SortedReferenceState, ΔV)

    cumsum!(scratch, ΔV)

    return @. s.bottom_height + (scratch - ΔV / 2) / s.horizontal_area
end

"""
    $(SIGNATURES)

The faces bounding those slots, from the bottom of the domain up: the same cumulative volumes divided by
the domain's horizontal area, with the bottom closing the stack.
"""
function slot_faces(s::SortedReferenceState, ΔV)

    N     = length(ΔV)
    Δz    = ΔV ./ s.horizontal_area                  # slot thickness, sorted order
    faces = similar(Δz, N + 1)
    @views faces[1:1] .= s.bottom_height
    cumsum!(view(faces, 2:N+1), Δz);  view(faces, 2:N+1) .+= s.bottom_height

    return faces
end

"""
    $(SIGNATURES)

Build `Ψ(z) = ∫ b✶ dz̃`, from the bottom of the domain up through a reference profile given by its slot
`faces` and buoyancies `b`, and return it as a callable. `b✶` is piecewise constant on the profile, so
`Ψ` is piecewise linear and evaluates exactly at any height. Taking the faces rather than the volumes
lets a caller that already has them (the external-profile path of [`ProfileLookup`](@ref)) skip
rebuilding them, and lets the profile have its own length rather than one slot per model cell.
"""
function reference_potential_function(faces, b)

    N     = length(b)
    Ψface = similar(faces)                           # Ψ evaluated at the slot faces
    @views Ψface[1:1] .= zero(eltype(faces))
    cumsum!(view(Ψface, 2:N+1), b .* diff(faces))

    ## Ψ inside slot k is Ψface[k] + b✶[k](z - faces[k]); `searchsortedlast` locates the slot
    return z -> (k = clamp(searchsortedlast(faces, z), 1, N); @inbounds Ψface[k] + b[k] * (z - faces[k]))
end

"""
    $(SIGNATURES)

Fill `s.reference_potential` with `Ψ(z) - Ψ(z✶)`, where `Ψ(z) = ∫ b✶ dz̃` runs from the bottom of the domain
up through the sorted profile. This is the part of the local available potential energy that only the
sort can supply; [`AvailablePotentialEnergy`](@ref Oceanostics.AvailablePotentialEnergyEquation.AvailablePotentialEnergy)'s kernel adds the remaining `-b(z - z✶)` pointwise.

`b✶` is piecewise constant on the sorted column, so `Ψ` is piecewise linear and can be evaluated
exactly at any height from the slot faces and the cumulative integral over them. Both `z` and `z✶` are
evaluated through that same `Ψ`, which is what makes the result non-negative: a parcel carries
`b = b✶(z✶)`, and `b✶` is non-decreasing, so the integrand `b✶(z̃) - b` has the same sign as `z̃ - z✶`
over the whole path.
"""
function fill_reference_potential!(s::SortedReferenceState, ΔV_sorted, b_sorted, z✶_sorted, sz, sorted_output)

    psi = reference_potential_function(slot_faces(s, ΔV_sorted), b_sorted)
    Ψ✶  = psi.(z✶_sorted)                            # sorted order

    if sorted_output   # the column: its cells are already in sorted order
        interior(s.reference_potential) .= reshape(psi.(s.source_height[s.permutation]) .- Ψ✶, sz)
    else               # the model grid: scatter Ψ(z✶) back to the original cell ordering
        work = s.workspace
        work[s.permutation] = Ψ✶
        interior(s.reference_potential) .= reshape(psi.(s.source_height) .- work, sz)
    end

    return nothing
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
assign_reference_height!(z✶_field, s::SortedReferenceState) = assign_reference_height!(z✶_field, s, s.method)

function assign_reference_height!(z✶_field, s::SortedReferenceState, ::ThreeDimensionalSort)
    sz   = size(z✶_field)
    work = s.workspace
    perm = s.permutation
    ΔV, b_sorted = rank_by_buoyancy!(s)

    z✶_sorted = slot_centers!(work, s, ΔV)   # cell-center z✶, in sorted order
    work[perm] = z✶_sorted                   # scatter to the original ordering

    interior(z✶_field) .= reshape(work, sz)
    fill_reference_potential!(s, ΔV, b_sorted, z✶_sorted, sz, false)

    return z✶_field
end

function assign_reference_height!(z✶_field, s::SortedReferenceState, ::HeavisideIntegral)
    sz   = size(z✶_field)
    work = s.workspace
    perm = s.permutation
    FT   = eltype(work)
    N    = length(work)
    ΔV, b = rank_by_buoyancy!(s)   # `b` is the sorted buoyancy, which is where the ties show up

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

    z✶_sorted = @. s.bottom_height + (denser + no_lighter) / (2 * s.horizontal_area)
    work[perm] = z✶_sorted

    interior(z✶_field) .= reshape(work, sz)
    fill_reference_potential!(s, ΔV, b, z✶_sorted, sz, false)

    return z✶_field
end

#+++ Reference profiles for `ProfileLookup`
"Flatten a profile given as a `Field` or as a plain vector into a vector."
profile_vector(p::AbstractVector) = p
profile_vector(p) = vec(interior(p))

"""
    $(SIGNATURES)

Return `(b✶, z✶)` for the profile a [`ProfileLookup`](@ref) was given: the sorted buoyancy and the
height paired with it, densest first. A reference height built with [`VerticalSort`](@ref)
carries both already, so it is unpacked; a pair is taken as it stands.
"""
profile_arrays(z✶::SortedReferenceHeightField) = (profile_vector(reference_buoyancy(z✶)), profile_vector(z✶))
profile_arrays(p::Tuple{Any, Any}) = (profile_vector(p[1]), profile_vector(p[2]))

# Anything else was never a profile. Without this the mismatch would surface as a `MethodError` from
# deep inside the first `compute!` rather than as a message naming the two forms that do work.
profile_arrays(p) =
    throw(ArgumentError("`ProfileLookup` takes either a reference height built with `VerticalSort()`, or a \
                         pair of buoyancies and heights given as `ProfileLookup(b✶, z✶)`. It was handed a \
                         $(typeof(p)), which is neither."))

"""
    $(SIGNATURES)

Recover the slot faces of an externally supplied profile, which gives only heights: the midpoints
between neighbouring `z✶`, closed off by the bottom and top of the domain, so the slots tile the depth
exactly however the profile is spaced. For a profile of equal-volume slots (what [`VerticalSort`](@ref)
produces) this recovers their boundaries exactly.
"""
function profile_slot_faces(bottom_height, top_height, z)

    M     = length(z)
    faces = similar(z, M + 1)
    @views faces[1:1]         .= bottom_height
    @views faces[M + 1:M + 1] .= top_height
    @views @. faces[2:M] = (z[1:M-1] + z[2:M]) / 2

    return faces
end

profile_slot_faces(s::SortedReferenceState, z) =
    profile_slot_faces(s.bottom_height, s.bottom_height + sum(s.cell_volume) / s.horizontal_area, z)

"""
    $(SIGNATURES)

Return `(b✶, z✶, faces)` for the profile a [`ProfileLookup`](@ref) will search, and leave the
model-ordered buoyancy in `s.workspace` either way. With no profile of its own the field is sorted,
exactly as [`ThreeDimensionalSort`](@ref) sorts it; with one supplied there is nothing to sort, and
the whole `O(N log N)` ranking is skipped.
"""
function lookup_profile(s::SortedReferenceState, method::ProfileLookup{Nothing})

    ΔV, b✶ = rank_by_buoyancy!(s)

    return b✶, slot_centers!(similar(ΔV), s, ΔV), slot_faces(s, ΔV)
end

# `issorted` walks its argument element by element, which is scalar indexing when the profile lives
# on a GPU. This is the same predicate written as a broadcast plus a reduction, so it runs on device.
@inline is_nondecreasing(v) = all(diff(v) .>= 0)

function lookup_profile(s::SortedReferenceState, method::ProfileLookup)

    reshape(s.workspace, size(s.buoyancy)) .= interior(s.buoyancy)   # what `rank_by_buoyancy!` would leave, unsorted
    b✶, z✶ = profile_arrays(method.profile)

    length(b✶) == length(z✶) ||
        throw(ArgumentError("`ProfileLookup` was given a profile whose buoyancy and height have different \
                             lengths ($(length(b✶)) and $(length(z✶)))."))
    is_nondecreasing(b✶) ||
        throw(ArgumentError("`ProfileLookup` needs a reference profile ordered from the densest fluid up, \
                             but the buoyancy it was given is not non-decreasing."))
    # `profile_slot_faces` reads the slot faces off the midpoints between neighbouring heights, so a
    # height that steps back down would leave them out of order and silently corrupt `Ψ`.
    is_nondecreasing(z✶) ||
        throw(ArgumentError("`ProfileLookup` needs the heights paired with `b✶` to rise with it, but the \
                             heights it was given are not non-decreasing. A reference profile runs from the \
                             densest fluid at the bottom to the lightest at the top."))

    return b✶, z✶, profile_slot_faces(s, z✶)
end
#---

function assign_reference_height!(z✶_field, s::SortedReferenceState, method::ProfileLookup)
    sz = size(z✶_field)
    b✶, _, faces = lookup_profile(s, method)   # the slot centres are not needed: every cell takes a run mid-height
    b = s.workspace   # the model-ordered buoyancy, which `lookup_profile` leaves in place
    M = length(b✶)

    # `b✶` is non-decreasing, so binary search brackets a cell's own buoyancy between two slots, and a
    # value that matches no slot — which only an externally supplied profile can produce, e.g. a filtered
    # buoyancy that is off the profile by roundoff — is assigned to the nearer of the two buoyancy
    # classes. A cell's reference height is then the mid-height of the band its class's run of slots
    # `[run_first, run_last]` fills, which is what `HeavisideIntegral` assigns and what leaves `∫e_b dV`
    # independent of the method; taking the run's first slot instead would drag every tied layer down to
    # its own floor. Placing near-matches by the same rule as exact ones is what keeps `z✶` a function of
    # buoyancy alone with no seams at roundoff: sending them to the nearest *slot* would put a filtered
    # buoyancy that is `b(1 + ε)` at the top or the bottom of its level's run depending on the sign of
    # `ε`, and `∇Υ` of the filtered field would pick up a ±half-cell comb where the full field's is zero.
    # A single-slot run reduces to that slot's own centre, so a field without ties is unaffected.
    lo  = clamp.(searchsortedfirst.(Ref(b✶), b), 1, M)   # first slot with b✶ ≥ b (b itself, on an exact match)
    loₗ = max.(lo .- 1, 1)                                 # the slot below it, the other bracket
    b_class = ifelse.(abs.(view(b✶, loₗ) .- b) .< abs.(view(b✶, lo) .- b), view(b✶, loₗ), view(b✶, lo))

    run_first = searchsortedfirst.(Ref(b✶), b_class)   # `b_class` is an entry of `b✶`, so both land in 1:M
    run_last  = searchsortedlast.(Ref(b✶), b_class)
    z✶ = (faces[run_first] .+ faces[run_last .+ 1]) ./ 2

    interior(z✶_field) .= reshape(z✶, sz)
    # `z✶` is already in the model's ordering, so `Ψ` is evaluated directly rather than scattered
    # through the permutation, which also lets the profile be a different length from the field.
    psi = reference_potential_function(faces, b✶)
    interior(s.reference_potential) .= reshape(psi.(s.source_height) .- psi.(z✶), sz)

    return z✶_field
end

function assign_reference_height!(z✶_field, s::SortedReferenceState, method::VerticalSort)
    sz   = size(z✶_field) # (1, 1, N): the sorted column, not the model grid
    work = s.workspace
    perm = s.permutation
    ΔV, b_sorted = rank_by_buoyancy!(s)

    interior(method.reference_buoyancy) .= reshape(b_sorted, sz)                 # buoyancy, densest first
    interior(method.sorted_height)      .= reshape(s.source_height[perm], sz)    # where each parcel came from

    z✶_sorted = slot_centers!(work, s, ΔV)
    interior(z✶_field) .= reshape(z✶_sorted, sz) # exactly the column's own cell centers
    fill_reference_potential!(s, ΔV, b_sorted, z✶_sorted, sz, true)

    return z✶_field
end

# Only a `ProfileLookup` can depend on anything beyond its own buoyancy, and only when the profile it
# was handed is itself computed. Refreshing it here is what keeps a borrowed profile tracking the flow.
refresh_profile!(::AbstractReferenceHeightMethod, time) = nothing
refresh_profile!(method::ProfileLookup, time) = refresh_profile_source!(method.profile, time)

refresh_profile_source!(::Nothing, time) = nothing
refresh_profile_source!(::AbstractVector, time) = nothing
refresh_profile_source!(p::Field, time) = compute_at!(p, time)
refresh_profile_source!(p::Tuple, time) = foreach(x -> refresh_profile_source!(x, time), p)

function compute!(z✶_field::SortedReferenceHeightField, time=nothing)
    s = z✶_field.operand
    compute_at!(s.buoyancy, time)
    refresh_profile!(s.method, time)
    assign_reference_height!(z✶_field, s)
    fill_halo_regions!(z✶_field)
    set_status!(z✶_field.status, time)

    return z✶_field
end

#+++ Reference buoyancy at a parcel's own height
"""
    $(SIGNATURES)

The reference profile the last sort left behind, as `(faces, b✶)`: the slot faces from the bottom of
the domain up, and the piecewise-constant buoyancy each slot holds. This is the same profile
`reference_potential_function` integrates into `Ψ`, so a buoyancy read off it and the `Ψ` the available
potential energy is built from describe one profile rather than two.

Every method that ranks the field leaves that ranking in `s.permutation`, so the profile is gathered
through it rather than sorted a second time. Only a [`ProfileLookup`](@ref) carrying a profile of its
own was never sorted here, and that one is read back the way `assign_reference_height!` reads it.
"""
reference_profile(s::SortedReferenceState) = reference_profile(s, s.method)

reference_profile(s::SortedReferenceState, ::AbstractReferenceHeightMethod) = sorted_profile(s)

# The profile-less lookup ranks the field itself, exactly as the stacking methods do, so its profile is
# in `s.permutation` too and rebuilding it through `lookup_profile` would repeat the whole `O(N log N)`
# sort that `compute!` has just done.
reference_profile(s::SortedReferenceState, ::ProfileLookup{Nothing}) = sorted_profile(s)

function reference_profile(s::SortedReferenceState, method::ProfileLookup)
    b✶, _, faces = lookup_profile(s, method)

    return faces, b✶
end

# `s.workspace` holds whatever the sort left there, which is scratch by now; every caller that needs it
# re-fills it first, as this does.
function sorted_profile(s::SortedReferenceState)
    work = s.workspace
    reshape(work, size(s.buoyancy)) .= interior(s.buoyancy)

    return slot_faces(s, s.cell_volume[s.permutation]), work[s.permutation]
end

"""
    $(SIGNATURES)

Build `b✶(z)`, the buoyancy the reference profile holds at a height `z`, from the profile's slot
`faces` and the buoyancies `b` it carries, and return it as a callable. `b✶` is piecewise constant on
the slots, which is exactly the profile `reference_potential_function` integrates, so this is the
derivative of that `Ψ` rather than an independent reconstruction of the same thing.
"""
function reference_buoyancy_function(faces, b)
    N = length(b)

    return z -> (k = clamp(searchsortedlast(faces, z), 1, N); @inbounds b[k])
end

"""
    $(TYPEDEF)

Operand of the `Field` [`reference_buoyancy_at_height`](@ref) returns. It carries the reference height
the profile is read off, so that `compute!` sorts that first and then samples it, the way
[`SortedBuoyancyState`](@ref) makes the sorted column's buoyancy depend on its parent.
"""
struct ReferenceBuoyancyAtHeightState{Z}
    reference_height :: Z
end

Base.summary(s::ReferenceBuoyancyAtHeightState) =
    string("ReferenceBuoyancyAtHeightState of ", summary(s.reference_height))

const ReferenceBuoyancyAtHeightField = Field{<:Any, <:Any, <:Any, <:ReferenceBuoyancyAtHeightState}

# Where each cell of the output sits: its own height on the model grid, and the height its parcel came
# from on a `VerticalSort` column, which is the same choice `fill_reference_potential!` makes.
profile_heights(s::SortedReferenceState, ::Nothing) = s.source_height
profile_heights(s::SortedReferenceState, sorted_height) = s.source_height[s.permutation]

function compute!(b✶ᶻ::ReferenceBuoyancyAtHeightField, time=nothing)
    z✶ = b✶ᶻ.operand.reference_height

    # Status-gated, so a `z✶` already sorted at this `time` is not sorted twice, and the profile below
    # is the one that sort produced.
    compute_at!(z✶, time)
    s = z✶.operand

    faces, b✶ = reference_profile(s)
    interior(b✶ᶻ) .= reshape(reference_buoyancy_function(faces, b✶).(profile_heights(s, sorted_height(s))), size(b✶ᶻ))

    fill_halo_regions!(b✶ᶻ)
    set_status!(b✶ᶻ.status, time)

    return b✶ᶻ
end

"""
    $(SIGNATURES)

Return a `Field` holding `b✶(z)`, the buoyancy the adiabatically sorted reference profile carries at
each cell's **own** height. It is what a parcel would have to be to sit where it does without any
available potential energy, so the difference from the buoyancy it actually carries is the anomaly
[`ReferenceBuoyancyAnomaly`](@ref Oceanostics.AvailablePotentialEnergyEquation.ReferenceBuoyancyAnomaly)
returns.

This is a different quantity from [`reference_buoyancy`](@ref), which pairs the profile with the
reference height `z✶` instead: `b✶(z✶)` is the parcel's own buoyancy, and on the model grid
`reference_buoyancy` hands back the buoyancy field itself.

The returned field recomputes itself, so it is safe to hand straight to an output writer: computing it
sorts the parent `z✶` if that has not already happened at this time, and then reads the profile off
that sort rather than repeating it. It answers on the grid `z✶` lives on, which for
[`VerticalSort`](@ref) is the sorted column rather than the model grid.

This method reads the profile a reference height already produced. The `(grid, profile)` method below
takes the profile itself instead, and reads it onto any grid, which is what a diagnostic measuring one
field against another field's reference state needs.
"""
reference_buoyancy_at_height(z✶::SortedReferenceHeightField) =
    compute!(Field{Center, Center, Center}(z✶.grid; operand = ReferenceBuoyancyAtHeightState(z✶),
                                           status = FieldStatus()))
#---

"""
    $(SIGNATURES)

Return a `Field` holding the reference height `z✶`: the height each parcel would occupy once the
buoyancy field is rearranged adiabatically into the state of minimum potential energy, following
[Winters et al. (1995)](https://doi.org/10.1017/S002211209500125X). It is the building block of
[`BackgroundPotentialEnergy`](@ref) and [`AvailablePotentialEnergy`](@ref Oceanostics.AvailablePotentialEnergyEquation.AvailablePotentialEnergy), and both accept a `z✶` you
built yourself so a pair of them can share one sort:

```jldoctest
using Oceananigans, Oceanostics

grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1), topology=(Periodic, Periodic, Bounded))
model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)
set!(model, b = (x, y, z) -> z)

z✶ = reference_height(model)                  # sort the domain once
e_b = BackgroundPotentialEnergy(model, z✶)    # and reuse it for both diagnostics
eₐ = AvailablePotentialEnergy(model, z✶)
e_b.grid === eₐ.grid === grid

# output

true
```

Unlike the pointwise diagnostics elsewhere in Oceanostics, `z✶` is defined by a sort of every cell in
the domain, and it is re-sorted on every `compute!`, so writing it (or anything built on it) out during
a simulation tracks the evolving flow, at a cost that grows like `N log N` in the number of cells. It
holds three `Nx*Ny*Nz` workspace arrays for its lifetime, and each sort allocates a handful more as
temporaries: measured on `65536` cells that is 1.5 such arrays for [`ThreeDimensionalSort`](@ref), 3.5
for [`VerticalSort`](@ref) and 5.8 for [`HeavisideIntegral`](@ref), which builds the most
intermediates. The `N log N` sort still dominates the runtime, so this shows up as allocation churn
rather than as wall-clock.

An `ImmersedBoundaryGrid` is rejected for every method: the sort weights each cell by its full volume,
so immersed cells would be stacked into the reference state as if they held fluid. Topography is not
supported yet. On a stretched grid (non-uniform cell volumes) only [`HeavisideIntegral`](@ref) runs,
since it builds `z✶` from a volume fraction; the three methods that stack cells into a column need
uniform cells and throw.

`method` picks how the sorted state is built. Sorting the field itself, all four give the same
`∫eₐ dV` in the continuous limit, but have different limitations due to the discretization:

  - [`ThreeDimensionalSort`](@ref) (the default) gives each cell the height of its own slot in the sorted
    column, on the model grid. Tied cells take consecutive slots, which spreads `z✶` over a grid cell
    wherever the buoyancy is uniform.
  - [`HeavisideIntegral`](@ref) is Eq. (11) of Winters et al. verbatim, also on the model grid. Tied
    cells share the mid-height of their layer, so `z✶` is a function of buoyancy alone and a
    cell-by-cell map is clean. Use this one for local fields.
  - [`ProfileLookup`](@ref) gives each cell the height of the slot whose buoyancy matches its own,
    found by binary search into the sorted profile, on the model grid. It is the column below read
    back onto the model grid, matched by value rather than by cell identity, so it is the one method
    that does not need the profile to have come from the field being diagnosed. Tied cells share the
    mid-height of their run, which makes `z✶` a function of buoyancy alone as above.
  - [`VerticalSort`](@ref) returns the sorted column itself, on a `1×1×N` grid of cells that span
    the domain's horizontal area, which is the form to use for a reference profile.

Where they differ is `z✶`, and so `e_b`: the placement of tied cells is the only freedom they have,
and `eₐ` is blind to it, since a cell's `z✶` always lands inside the run of slots its own buoyancy
fills and the reference profile is flat across that run.

```jldoctest
using Oceananigans, Oceanostics

grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1), topology=(Periodic, Periodic, Bounded))
model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)
set!(model, b = (x, y, z) -> z)

# with Eq. (11), a horizontally uniform stable stratification is exactly its own sorted state
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
buoyancy, which is what you want for a reference state built from a filtered buoyancy `filter(b)`.
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

    isnothing(model.buoyancy) ? nothing : validate_gravity_unit_vector("reference_height", model.buoyancy.gravity_unit_vector)

    return reference_height(buoyancy_field(model, model.buoyancy, geopotential_height); method)
end

function reference_height(b::Field; method = ThreeDimensionalSort())
    grid = b.grid

    arch = architecture(grid)
    FT   = eltype(grid)
    N    = prod(size(grid))

    cell_volume   = flat_grid_metric(grid, Vᶜᶜᶜ)
    validate_grid_for_method(method, grid, cell_volume)   # reject stretched/immersed for all but HeavisideIntegral

    source_height = flat_grid_metric(grid, Zᶜᶜᶜ)

    # A stretched grid keeps its coordinates on the device, so the bottom face is read off a CPU copy
    # of the grid rather than by indexing into GPU memory.
    z_bottom = convert(FT, znode(1, 1, 1, on_architecture(CPU(), grid), Center(), Center(), Face()))
    # Taking the horizontal area from the total volume (rather than from the surface area) makes the
    # sorted column fill the domain's depth exactly, which is what keeps ∫eₚ = ∫e_b for a sorted field.
    horizontal_area = convert(FT, sum(cell_volume) / grid.Lz)

    method = build_sorting_method(method, grid, cell_volume, horizontal_area, z_bottom)
    sgrid  = sorting_grid(method, grid)

    operand = SortedReferenceState(method, b, cell_volume,
                                   on_architecture(arch, zeros(Int, N)),
                                   on_architecture(arch, zeros(FT, N)),
                                   source_height, CenterField(sgrid),
                                   horizontal_area, z_bottom)

    return compute!(Field{Center, Center, Center}(sgrid; operand, status = FieldStatus()))
end

#+++ Per-method setup
# `XYZRegularRG` is Oceananigans' own name for a `RectilinearGrid` regular on all three axes, which is
# exactly the uniform-cell-volume case: a cell's volume is the product of the per-axis spacings, so it is
# uniform iff every axis is regular. Anything else (a stretched axis, or a curvilinear grid whose cells
# shrink with latitude) is stretched as far as the stacking methods are concerned.
stretched_grid(grid) = !(grid isa XYZRegularRG)

# No method handles topography yet, so every one of them rejects an `ImmersedBoundaryGrid`. Beyond that,
# `HeavisideIntegral` builds `z✶` from a volume fraction and tolerates a stretched grid, while the three
# methods that stack cells into a column need uniform cell volumes to reconstruct the reference height.
validate_grid_for_method(::HeavisideIntegral, grid, cell_volume) = validate_not_immersed(grid)

function validate_grid_for_method(method::AbstractReferenceHeightMethod, grid, cell_volume)
    validate_not_immersed(grid)

    if stretched_grid(grid)
        ΔV_min, ΔV_max = extrema(cell_volume)   # only for the message, so the scan stays off the happy path
        throw(ArgumentError("`$(summary(method))` needs a grid with uniform cell volumes (regular spacing in every \
                             direction), but this grid is stretched, with cell volumes ranging over [$ΔV_min, $ΔV_max]. \
                             Only `HeavisideIntegral()` supports stretched grids."))
    end

    return nothing
end

# The sort weights every cell by its full volume and maps a cumulative volume onto a height by dividing
# by one depth-independent area. Topography breaks both: immersed cells would be stacked into the
# reference state as though they held fluid, and the wet area varies with depth.
validate_not_immersed(grid) = nothing
validate_not_immersed(::ImmersedBoundaryGrid) =
    throw(ArgumentError("The sorted reference state is not available on an `ImmersedBoundaryGrid` yet: the sort \
                         counts every cell at its full volume, so immersed cells would be stacked into the \
                         reference state as if they held fluid, and the horizontal cross-sectional area is taken \
                         to be independent of depth. Supporting topography needs the dry cells masked out of the \
                         sort and a depth-dependent area."))

"Evaluate a grid metric (a `(i, j, k, grid)` operator) over every cell and return it as a flat vector."
function flat_grid_metric(grid, metric)

    flat = on_architecture(architecture(grid), zeros(eltype(grid), prod(size(grid))))
    reshape(flat, size(grid)) .= interior(Field(KernelFunctionOperation{Center, Center, Center}(metric, grid)))

    return flat
end

# `ThreeDimensionalSort` and `HeavisideIntegral` write `z✶` onto the model grid; `VerticalSort` writes it
# onto the sorted column it allocates below.
sorting_grid(::AbstractReferenceHeightMethod, grid) = grid
sorting_grid(method::VerticalSort, grid) = method.reference_buoyancy.grid

build_sorting_method(method::AbstractReferenceHeightMethod, grid, cell_volume, horizontal_area, z_bottom) = method

# A host `Vector` profile would be broadcast against on-architecture workspaces; move it once, not per `compute!`.
build_sorting_method(method::ProfileLookup, grid, cell_volume, horizontal_area, z_bottom) =
    ProfileLookup(on_architecture_profile(architecture(grid), method.profile))

# `Field`s are already on-architecture and must stay `Field`s for `refresh_profile!`; only arrays move.
on_architecture_profile(arch, profile) = profile
on_architecture_profile(arch, profile::AbstractVector) = on_architecture(arch, profile)
on_architecture_profile(arch, profile::Tuple) = map(p -> on_architecture_profile(arch, p), profile)

function build_sorting_method(::VerticalSort, grid, cell_volume, horizontal_area, z_bottom)

    # `validate_grid_for_method` has already rejected the non-uniform cell volumes this method cannot
    # handle, so the sorted column's cell boundaries (the cumulative volumes) are safe to bake into a grid.

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

    return VerticalSort(CenterField(column), CenterField(column))
end
#---
#---

#+++ The reference profile at a parcel's own height
"""
    $(TYPEDEF)

Operand of the `Field` [`reference_buoyancy_at_height`](@ref) returns. It carries the reference profile
to read, the height of every cell of the grid to read it at, and the domain's vertical bounds. Like
[`SortedReferenceState`](@ref) this hooks a whole-field computation into `compute!` — here a lookup
rather than a sort — so a profile that tracks the flow is re-read whenever the diagnostic is written out.
"""
struct ReferenceProfileAtHeight{P, S, FT}
    profile :: P
    source_height :: S
    bottom_height :: FT
    top_height :: FT
end

Base.summary(s::ReferenceProfileAtHeight) = string("ReferenceProfileAtHeight of ", summary(s.profile))

const ReferenceProfileAtHeightField = Field{<:Any, <:Any, <:Any, <:ReferenceProfileAtHeight}

function compute!(b✶z::ReferenceProfileAtHeightField, time=nothing)
    s = b✶z.operand
    refresh_profile_source!(s.profile, time)   # a borrowed column is re-sorted before it is read

    b✶, z✶ = profile_arrays(s.profile)
    faces  = profile_slot_faces(s.bottom_height, s.top_height, z✶)
    M      = length(b✶)

    # The profile is a stack of slots, each holding one buoyancy; a cell reads the slot its own height
    # falls in. `faces` are the midpoints between neighbouring `z✶`, so this is also the nearest slot.
    slot = clamp.(searchsortedlast.(Ref(faces), s.source_height), 1, M)
    interior(b✶z) .= reshape(view(b✶, slot), size(b✶z))

    fill_halo_regions!(b✶z)
    set_status!(b✶z.status, time)

    return b✶z
end

"""
    $(SIGNATURES)

Return a `Field` holding `b✶(z)`: the buoyancy the adiabatically resorted reference state carries at
each cell's **own height**, rather than at the reference height [`reference_height`](@ref) sends that
cell's buoyancy to. It is the inverse of that map — `b✶` is defined implicitly by `z✶(b✶(z)) = z`
([Wenegrat, Chor & Barkan, 2026](https://arxiv.org/abs/2605.15879), §2.1) — and it is what turns a
buoyancy into an anomaly against the reference state, `b_r = b - b✶(z)`.

`profile` is the reference profile to read, in any of the forms [`ProfileLookup`](@ref) accepts a
profile in: a reference height built with [`VerticalSort`](@ref), which is recomputed first so the
profile tracks the flow, or a `(b✶, z✶)` pair of arrays held fixed. Handed a reference height on the
model grid instead, use the single-argument method above, which reads the profile off that sort
directly and needs no slot geometry rebuilt from it. The profile is a stack of slots,
each holding one buoyancy class, and a cell reads the slot its own height falls in.

The result lives at `(Center, Center, Center)` on `grid`, in units of buoyancy (`m s⁻²`). Being a
lookup rather than a sort it is `O(N log M)` per `compute!`, cheaper than the reference height itself.
"""
function reference_buoyancy_at_height(grid, profile)

    validate_not_immersed(grid)

    FT       = eltype(grid)
    arch     = architecture(grid)
    z_bottom = convert(FT, znode(1, 1, 1, on_architecture(CPU(), grid), Center(), Center(), Face()))

    # A host `Vector` profile would be broadcast against on-architecture workspaces; move it once here,
    # exactly as `build_sorting_method` does for `ProfileLookup`.
    operand = ReferenceProfileAtHeight(on_architecture_profile(arch, profile), flat_grid_metric(grid, Zᶜᶜᶜ),
                                       z_bottom, convert(FT, z_bottom + grid.Lz))

    return compute!(Field{Center, Center, Center}(grid; operand, status = FieldStatus()))
end
#---

#+++ Background potential energy
@inline minus_bz✶_ccc(i, j, k, grid, b, z✶) = @inbounds -b[i, j, k] * z✶[i, j, k]

const BackgroundPotentialEnergy = CustomKFO{<:typeof(minus_bz✶_ccc)}

"""
    $(SIGNATURES)

Return a `KernelFunctionOperation` computing the background (or reference) potential energy per unit
volume,

```
    e_b = -b z✶ = (g/ρ₀) ρ z✶
```

the potential energy the fluid would retain if it were rearranged adiabatically into the state of
minimum potential energy ([Winters et al., 1995](https://doi.org/10.1017/S002211209500125X), Eq. 22).
`z✶` is the reference height that rearrangement assigns to each parcel, computed by
[`reference_height`](@ref); pass one explicitly to share a single sort with
[`AvailablePotentialEnergy`](@ref Oceanostics.AvailablePotentialEnergyEquation.AvailablePotentialEnergy), or pass `method` through to choose how it is built
([`ThreeDimensionalSort`](@ref), [`HeavisideIntegral`](@ref) or [`VerticalSort`](@ref); `Integral(e_b)` is
the same either way).

`e_b` responds only to irreversible changes in the buoyancy field, so in a closed domain the continuous
equations make `Integral(e_b)` grow monotonically at the diapycnal mixing rate. Numerically it also
picks up whatever spurious diapycnal transport the advection scheme introduces, in either direction,
which is what makes it a standard measure of a scheme's mixing. The remainder `eₚ - e_b` is the
[`AvailablePotentialEnergy`](@ref Oceanostics.AvailablePotentialEnergyEquation.AvailablePotentialEnergy), the part that can be converted to kinetic energy. The result lives
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
└── computes: background potential energy per unit volume  e_b = -bz✶
```
"""
function BackgroundPotentialEnergy(model; method = ThreeDimensionalSort(), geopotential_height = model_geopotential_height(model), location = (Center, Center, Center))
    validate_location(location, "BackgroundPotentialEnergy")
    return BackgroundPotentialEnergy(model, reference_height(model; method, geopotential_height))
end

function BackgroundPotentialEnergy(model, z✶::SortedReferenceHeightField)
    validate_gravity_is_z_aligned("BackgroundPotentialEnergy", model)
    return KernelFunctionOperation{Center, Center, Center}(minus_bz✶_ccc, z✶.grid, reference_buoyancy(z✶.operand), z✶)
end
#---

end # module
