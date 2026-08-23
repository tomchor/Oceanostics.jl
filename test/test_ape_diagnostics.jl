using Test
using CUDA: has_cuda_gpu

using Oceananigans
using Oceananigans.BuoyancyFormulations: Zᶜᶜᶜ
using Oceananigans.Fields: compute_at!
using Oceananigans.Grids: topology
using SeawaterPolynomials: RoquetEquationOfState, TEOS10EquationOfState

using Oceanostics
# `DisplacementPotential` is scoped to its own module, so it is imported rather than exported here
using Oceanostics.AvailablePotentialEnergyEquation: DisplacementPotential

# Include common test utilities
include("test_utils.jl")

# The geopotential height as a plain operation, used below to check that sorting only rearranges cells.
@inline z_ccc(i, j, k, grid) = Zᶜᶜᶜ(i, j, k, grid)
height_operation(grid) = KernelFunctionOperation{Center, Center, Center}(z_ccc, grid)

"""
The reference profile a sorting method describes, as `(z✶, b✶)` ordered from the bottom of the sorted
column up. `VerticalSort` already stores it that way; the two model-grid methods hold `z✶` and
`b` cell by cell, so pairing them and ordering by `z✶` recovers the same profile.
"""
function reference_profile(z✶)

    heights   = Array(vec(interior(z✶)))
    buoyancy  = Array(vec(interior(reference_buoyancy(z✶))))
    ascending = sortperm(heights)

    return heights[ascending], buoyancy[ascending]
end

#+++ Test functions
"Neither half of the split is defined without a buoyancy formulation to sort."
function test_available_potential_energy_errors(model)

    @test_throws ArgumentError AvailablePotentialEnergyEquation.BackgroundPotentialEnergy(model)
    @test_throws ArgumentError AvailablePotentialEnergyEquation.AvailablePotentialEnergy(model)
    @test_throws ArgumentError AvailablePotentialEnergyEquation.reference_height(model)

    return nothing
end

function test_background_and_available_pe(model; geopotential_height = nothing, method = ThreeDimensionalSort())

    kwargs = isnothing(geopotential_height) ? (;) : (; geopotential_height)

    z✶  = AvailablePotentialEnergyEquation.reference_height(model; method, kwargs...)
    Eₚ  = PotentialEnergyEquation.PotentialEnergy(model; kwargs...)
    E_b = AvailablePotentialEnergyEquation.BackgroundPotentialEnergy(model, z✶)
    Eₐ = AvailablePotentialEnergyEquation.AvailablePotentialEnergy(model, z✶)

    @test E_b isa AvailablePotentialEnergyEquation.BackgroundPotentialEnergy
    @test Eₐ isa AvailablePotentialEnergyEquation.AvailablePotentialEnergy

    Eₚ_field, E_b_field, Eₐ_field = Field(Eₚ), Field(E_b), Field(Eₐ)
    @test E_b_field isa Field
    @test Eₐ_field isa Field

    grid = model.grid
    FT = eltype(grid)

    # `Eₐ` is the *local* available potential energy of Holliday & McIntyre (1981), the work needed to
    # bring a parcel from its reference height to where it is. That is non-negative everywhere, which
    # `Eₚ - E_b` is not: the two agree in the volume integral, not cell by cell.
    @test minimum(interior(Eₐ_field)) > -sqrt(eps(FT)) * maximum(abs, interior(Eₚ_field))

    # Sorting only rearranges cells between heights, so it moves no volume and leaves the
    # volume-weighted mean height where it was. The comparison needs an absolute tolerance because
    # that mean is zero on a grid whose z is symmetric about the origin.
    @test volume_mean(z✶) ≈ volume_mean(height_operation(grid)) atol = sqrt(eps(FT)) * grid.Lz

    # The sorted state is the state of minimum potential energy, so ∫Eₐ = ∫Eₚ - ∫E_b ≥ 0
    ∫Eₚ, ∫E_b, ∫Eₐ = volume_integral(Eₚ), volume_integral(E_b), volume_integral(Eₐ)
    @test ∫Eₐ ≥ -sqrt(eps(FT)) * max(abs(∫Eₚ), abs(∫E_b))

    return nothing
end

"""
A two-cell column with the dense parcel sitting on top of the light one, where every term can be
worked out by hand. The column is 1 m deep with 0.5 m cells, so z = (-0.75, -0.25) and, once sorted,
z✶ = (-0.25, -0.75): the light parcel rises, the dense one sinks.
"""
function test_available_pe_analytic()

    grid = RectilinearGrid(arch, size=2, z=(-1, 0), topology=(Flat, Flat, Bounded))
    model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)
    set!(model, b = reshape([1.0, -1.0], 1, 1, 2)) # statically unstable: light below, dense above

    z✶ = AvailablePotentialEnergyEquation.reference_height(model)
    @test Array(vec(interior(z✶))) ≈ [-0.25, -0.75]

    @test volume_integral(PotentialEnergyEquation.PotentialEnergy(model))               ≈ +0.25
    @test volume_integral(AvailablePotentialEnergyEquation.BackgroundPotentialEnergy(model, z✶)) ≈ -0.25
    @test volume_integral(AvailablePotentialEnergyEquation.AvailablePotentialEnergy(model, z✶))  ≈ +0.50

    # Flipping the column makes it statically stable, hence already sorted, hence free of available PE
    set!(model, b = reshape([-1.0, 1.0], 1, 1, 2))
    z✶ = AvailablePotentialEnergyEquation.reference_height(model)
    @test Array(vec(interior(z✶))) ≈ [-0.75, -0.25]
    @test volume_integral(AvailablePotentialEnergyEquation.BackgroundPotentialEnergy(model, z✶)) ≈
          volume_integral(PotentialEnergyEquation.PotentialEnergy(model))
    @test volume_integral(AvailablePotentialEnergyEquation.AvailablePotentialEnergy(model, z✶)) ≈ 0 atol=1e-14

    return nothing
end

"A horizontally uniform, statically stable stratification is its own sorted state, so it holds no APE."
function test_available_pe_vanishes_when_sorted(grid; method = ThreeDimensionalSort())

    model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)
    set!(model, b = (x, y, z) -> 3z)

    ∫Eₐ = volume_integral(AvailablePotentialEnergyEquation.AvailablePotentialEnergy(model; method))
    ∫Eₚ  = volume_integral(PotentialEnergyEquation.PotentialEnergy(model))

    @test ∫Eₐ ≈ 0 atol=sqrt(eps(eltype(grid))) * abs(∫Eₚ)
    @test volume_integral(AvailablePotentialEnergyEquation.BackgroundPotentialEnergy(model; method)) ≈ ∫Eₚ

    return nothing
end

"""
The sorted reference state depends on every cell in the domain, so it has to be re-sorted on
every `compute!` rather than baked in when the diagnostic is constructed. Mixing the flow completely
(which conserves `∫b dV`) has to raise `∫E_b` to `∫Eₚ`, and a frozen reference state would not notice.
"""
function test_reference_state_is_recomputed(grid; method = ThreeDimensionalSort())

    model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)
    # A sorted state is a function of z alone, so the horizontal term is what makes this one unsorted
    # whatever the grid. A purely vertical wiggle is not enough: sampled on the stretched grid's six
    # levels, `z + 0.5sin(6z)` comes out monotonic, hence already sorted, hence free of available PE.
    set!(model, b = (x, y, z) -> z + 0.5 * sin(6z) + 0.2 * sin(7x))

    z✶   = AvailablePotentialEnergyEquation.reference_height(model; method)
    ∫E_b = ∫dV(AvailablePotentialEnergyEquation.BackgroundPotentialEnergy(model, z✶))
    ∫Eₚ  = ∫dV(PotentialEnergyEquation.PotentialEnergy(model))

    stirred_E_b = sum(∫E_b)
    @test stirred_E_b < sum(∫Eₚ) # an unsorted field holds available potential energy

    set!(model, b = sum(Field(Average(model.tracers.b)))) # mix completely, conserving ∫b dV
    compute!(∫E_b)
    compute!(∫Eₚ)

    @test sum(∫E_b) != stirred_E_b # the reference state was re-sorted rather than reused
    @test sum(∫E_b) > stirred_E_b  # mixing is irreversible: it can only raise the background PE
    # A uniform field is its own sorted state. Both energies are proportional to ∫z dV here, which
    # vanishes on a grid whose z is symmetric about the origin, so compare on the stirred scale.
    @test sum(∫E_b) ≈ sum(∫Eₚ) atol = sqrt(eps(eltype(grid))) * abs(stirred_E_b)

    return nothing
end

"""
The three sorting methods differ in how they place cells of equal buoyancy and in what grid they land
on, but they all describe the same reference state, so every volume integral built from `z✶` has to
agree between them.
"""
function test_sorting_methods_agree(grid)

    model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)
    set!(model, b = (x, y, z) -> z + 0.4 * sin(9x) * cos(7z))

    # Only called on a regular grid: cross-method agreement needs at least two methods to run, and a
    # stretched grid admits only `HeavisideIntegral`.
    reference = nothing
    for method in (ThreeDimensionalSort(), HeavisideIntegral(), VerticalSort())

        z✶  = AvailablePotentialEnergyEquation.reference_height(model; method)
        ∫E_b = volume_integral(AvailablePotentialEnergyEquation.BackgroundPotentialEnergy(model, z✶))
        ∫Eₐ = volume_integral(AvailablePotentialEnergyEquation.AvailablePotentialEnergy(model, z✶))

        if isnothing(reference)
            reference = (∫E_b, ∫Eₐ)
            @test ∫Eₐ > 0
            # `∫E_b + ∫Eₐ` only approaches `∫Eₚ` as the vertical grid refines — the local density
            # evaluates the reference profile at the model's cell centers rather than the sorted
            # column's. `test_local_ape_converges_to_the_winters_total` pins that convergence down;
            # on these small test grids the gap is a few percent, so all that is checked here is
            # that the three methods land on the same total as each other.
        else
            @test ∫E_b ≈ reference[1]
            @test ∫Eₐ ≈ reference[2]
        end
    end

    return nothing
end

"""
Under `HeavisideIntegral` the reference height is a function of buoyancy alone (eq. 11 weights equal
densities by 1/2), so a horizontally uniform stratification, whose cells are tied level by level, gets
`z✶ = z` and `Eₐ = 0` cell by cell. `ThreeDimensionalSort` gives the tied cells consecutive slots instead, which
spreads `z✶` over the cell they share without moving the integral.
"""
function test_heaviside_is_constant_on_isopycnals(grid)

    model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)
    set!(model, b = (x, y, z) -> 3z)

    heights = interior(Field(height_operation(grid)))
    Δz_max = maximum(zspacings(grid))

    z✶ = AvailablePotentialEnergyEquation.reference_height(model, method=HeavisideIntegral())
    @test interior(z✶) ≈ heights
    @test maximum(abs, interior(Field(AvailablePotentialEnergyEquation.AvailablePotentialEnergy(model, z✶)))) <
          sqrt(eps(eltype(grid)))

    # The ranked heights stay within half a cell of the parcel's own height, and no closer. Only
    # `HeavisideIntegral` runs on a stretched grid, so the contrast is drawn on uniform volumes alone.
    if minimum(zspacings(grid)) ≈ maximum(zspacings(grid))
        z✶_ranked = AvailablePotentialEnergyEquation.reference_height(model, method=ThreeDimensionalSort())
        @test maximum(abs, interior(z✶_ranked) .- heights) ≤ Δz_max / 2
    end

    return nothing
end

"""
`ProfileLookup` matches each cell to a sorted profile by buoyancy rather than by cell identity, and takes
that profile three ways: sorted from the field itself, borrowed from a `VerticalSort` column, or handed
over as a bare `(b✶, z✶)` pair. On a tie-free field every cell finds its own slot, so all three have to
reproduce exactly what `ThreeDimensionalSort` assigns. (With ties they legitimately place cells
differently, so the field here is built to have none.)

The second half pins the profile validation, which is what stands between a malformed pair and a silently
wrong `Eₐ`: the heights have to rise with the buoyancy, or the reconstructed slot volumes go negative.
"""
function test_profile_lookup_matches_the_ranked_sort()

    Nx, Ny, Nz = 3, 2, 4
    N = Nx * Ny * Nz
    grid = RectilinearGrid(arch, size=(Nx, Ny, Nz), x=(0, 1), y=(0, 1), z=(-1, 0))
    model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)

    # `N` distinct buoyancies scattered by a stride permutation (a bijection since gcd(7, N) = 1), so no
    # two cells are tied and the spatial arrangement bears no relation to the sorted one
    scrambled = zeros(N)
    for m in 1:N
        scrambled[1 + mod(7 * (m - 1), N)] = -0.5 + (m - 1) / (N - 1)
    end
    @test length(unique(scrambled)) == N
    set!(model, b = reshape(scrambled, Nx, Ny, Nz))

    ranked = AvailablePotentialEnergyEquation.reference_height(model, method=ThreeDimensionalSort())
    ∫E_b_expected = volume_integral(AvailablePotentialEnergyEquation.BackgroundPotentialEnergy(model, ranked))
    ∫Eₐ_expected = volume_integral(AvailablePotentialEnergyEquation.AvailablePotentialEnergy(model, ranked))

    # a column to borrow, and the same profile as bare vectors (kept on the model's architecture)
    column  = AvailablePotentialEnergyEquation.reference_height(model, method=VerticalSort())
    b✶      = vec(interior(reference_buoyancy(column)))
    z✶_prof = vec(interior(column))

    for method in (ProfileLookup(), ProfileLookup(column), ProfileLookup(b✶, z✶_prof))
        z✶ = AvailablePotentialEnergyEquation.reference_height(model; method)

        @test interior(z✶) ≈ interior(ranked)
        @test volume_integral(AvailablePotentialEnergyEquation.BackgroundPotentialEnergy(model, z✶)) ≈ ∫E_b_expected
        @test volume_integral(AvailablePotentialEnergyEquation.AvailablePotentialEnergy(model, z✶))  ≈ ∫Eₐ_expected

        Eₐ = interior(Field(AvailablePotentialEnergyEquation.AvailablePotentialEnergy(model, z✶)))
        @test minimum(Eₐ) > -sqrt(eps(eltype(grid))) * max(maximum(abs, Eₐ), eps(eltype(grid)))
    end

    # a well-formed profile has matched lengths, buoyancy running up, and heights rising with it. Each
    # assertion matches its own message, so a profile rejected for the wrong reason still fails.
    for (msg, bad) in ("buoyancy and height have different" => ProfileLookup(b✶[1:end-1], z✶_prof),
                       "ordered from the densest fluid up"  => ProfileLookup(reverse(b✶), z✶_prof),
                       "heights paired with"                => ProfileLookup(b✶, reverse(z✶_prof)))
        @test_throws msg AvailablePotentialEnergyEquation.reference_height(model; method=bad)
    end

    return nothing
end

"""
`VerticalSort` returns the sorted column on its own `1×1×N` grid: the same cells, reshaped to
span the domain's horizontal area, stacked in order of increasing buoyancy.
"""
function test_one_dimensional_sort_column(grid)

    model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)
    set!(model, b = (x, y, z) -> z + 0.4 * sin(9x) * cos(7z))

    z✶ = AvailablePotentialEnergyEquation.reference_height(model, method=VerticalSort())
    b✶ = reference_buoyancy(z✶)

    @test size(z✶) == (1, 1, prod(size(grid)))
    @test issorted(Array(vec(interior(b✶))))  # densest at the bottom
    @test issorted(Array(vec(interior(z✶))))
    # Reshaping the cells conserves volume, so the column integrates like the model grid does
    @test volume_integral(b✶) ≈ volume_integral(model.tracers.b)
    # The column keeps the model grid's topology rather than forcing one of its own
    @test topology(z✶.grid) == topology(grid)

    return nothing
end

"""
The column inherits the model grid's topology, so a `Flat` horizontal direction stays `Flat` rather
than becoming a spurious periodic axis. Exercised on grids the shared `grids` fixture does not cover:
a 2D grid `Flat` in one horizontal, and a single-column grid `Flat` in both.
"""
function test_one_dimensional_sort_matches_topology()

    grids_and_setters = ((RectilinearGrid(arch, size=(6, 8), x=(-1, 1), z=(-1, 0),
                                          topology=(Bounded, Flat, Bounded)),
                          (x, z) -> z + 0.3 * sin(7x)),
                         (RectilinearGrid(arch, size=10, z=(-1, 0), topology=(Flat, Flat, Bounded)),
                          z -> z + 0.2 * sin(6z)))

    for (grid, b₀) in grids_and_setters
        model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)
        set!(model, b = b₀)

        z✶ = AvailablePotentialEnergyEquation.reference_height(model, method=VerticalSort())

        @test topology(z✶.grid) == topology(grid)
        @test size(z✶) == (1, 1, prod(size(grid)))
        # The integral is unaffected by the topology fix, so it still matches the model-grid method
        z✶_ranked = AvailablePotentialEnergyEquation.reference_height(model, method=ThreeDimensionalSort())
        @test volume_integral(AvailablePotentialEnergyEquation.BackgroundPotentialEnergy(model, z✶)) ≈
              volume_integral(AvailablePotentialEnergyEquation.BackgroundPotentialEnergy(model, z✶_ranked))
    end

    return nothing
end

"""
The column's buoyancy is filled as a side effect of sorting `z✶`, not by a computation of its own, so
the field [`reference_buoyancy`](@ref) hands back has to know to trigger that sort. Otherwise an output
writer given it on its own would silently keep writing the previous output's profile, which looks
entirely plausible: it is a real sorted profile, just of the wrong step. Fetching it is what a writer
does, so that is what is exercised here.
"""
function test_reference_buoyancy_triggers_the_sort(grid)

    model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)
    set!(model, b = (x, y, z) -> z + 0.3 * sin(9x))

    z✶ = AvailablePotentialEnergyEquation.reference_height(model, method=VerticalSort())
    b✶ = reference_buoyancy(z✶)

    # Move the flow on and fetch `b✶` alone, never touching `z✶`, exactly as a writer would
    set!(model, b = (x, y, z) -> 2z - 0.4 * cos(7x))
    compute_at!(b✶, 1.0)

    sorted_b = sort(Array(vec(interior(b✶))))
    live_b   = sort(Array(vec(interior(model.tracers.b))))
    @test sorted_b ≈ live_b   # a permutation of the *current* buoyancy, not the previous one

    # The model-grid methods already hand back a field that recomputes itself, so it is passed through
    for method in (ThreeDimensionalSort(), HeavisideIntegral())
        @test reference_buoyancy(AvailablePotentialEnergyEquation.reference_height(model; method)) === model.tracers.b
    end

    return nothing
end

"""
Only `HeavisideIntegral` supports a stretched grid (non-uniform cell volumes). The other three stack
cells into a column, which needs uniform cells, so each throws an `ArgumentError`; `HeavisideIntegral`
runs and returns a finite reference height.
"""
function test_stretched_grid_restrictions(grid)

    model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)
    set!(model, b = grid_noise)   # a distinct buoyancy per cell, so the sort does real work

    # Match the message, not just the type: `reference_height` raises `ArgumentError` from several
    # places, so a bare type check cannot tell the grid restriction from an unrelated failure.
    for method in (ThreeDimensionalSort(), ProfileLookup(), VerticalSort())
        @test_throws "uniform cell volumes" AvailablePotentialEnergyEquation.reference_height(model; method)
    end

    # `HeavisideIntegral` runs, and has to land somewhere a reference state could actually be: inside the
    # domain, holding no negative APE, and leaving the volume-weighted mean height where it found it
    # (sorting moves cells between heights, not volume between them).
    z✶ = AvailablePotentialEnergyEquation.reference_height(model, method=HeavisideIntegral())
    z_bottom, z_top = extrema(znodes(grid, Face()))
    @test all(isfinite, interior(z✶))
    @test all(z -> z_bottom ≤ z ≤ z_top, interior(z✶))
    @test volume_mean(z✶) ≈ volume_mean(height_operation(grid)) atol = sqrt(eps(eltype(grid))) * grid.Lz
    @test volume_integral(AvailablePotentialEnergyEquation.AvailablePotentialEnergy(model, z✶)) ≥ 0

    return nothing
end

"""
A synthetic buoyancy field whose sorted state is known in closed form, used to check that the three
methods really do describe the same reference state rather than merely the same integrals.

The field is built backwards from the answer: a `tanh` profile is evaluated at the cell centers the
sorted column will have, then scattered over the domain by a stride permutation, so that no two cells
share a buoyancy and the spatial arrangement bears no relation to the sorted one. Every method then
has to recover that `tanh` and the background potential energy that goes with it.

The second half coarsens the same field into repeated buoyancy levels, where the methods genuinely
part ways: they place tied cells differently, so their profiles only have to agree to within the
thickness of a tied layer, while `∫E_b dV` still has to agree exactly.
"""
function test_sorting_methods_reproduce_a_known_profile()

    Nx, Ny, Nz = 4, 3, 5
    N = Nx * Ny * Nz
    Lx, Ly, Lz = 2, 3, 1
    grid = RectilinearGrid(arch, size=(Nx, Ny, Nz), x=(0, Lx), y=(0, Ly), z=(-Lz, 0))

    B₀, h, z_mid = 0.1, 0.2, -Lz/2
    Δz✶ = Lz / N                                   # the sorted column's cell thickness
    ΔV  = Lx * Ly * Lz / N                         # ... and the volume each of its cells holds
    z✶_expected = @. -Lz + ((1:N) - 0.5) * Δz✶     # the column's cell centers, densest at the bottom
    b✶_expected = @. B₀ * tanh((z✶_expected - z_mid) / h)

    # Send the m-th densest parcel to cell 1 + mod(7(m-1), N), a bijection since gcd(7, N) = 1
    scrambled = zeros(N)
    for m in 1:N
        scrambled[1 + mod(7 * (m - 1), N)] = b✶_expected[m]
    end

    model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)
    set!(model, b = reshape(scrambled, Nx, Ny, Nz))

    ∫E_b_expected = sum(@. -b✶_expected * z✶_expected * ΔV)

    for method in (ThreeDimensionalSort(), HeavisideIntegral(), VerticalSort())
        z✶ = AvailablePotentialEnergyEquation.reference_height(model; method)
        heights, buoyancies = reference_profile(z✶)

        @test heights ≈ z✶_expected
        @test buoyancies ≈ b✶_expected
        @test volume_integral(AvailablePotentialEnergyEquation.BackgroundPotentialEnergy(model, z✶)) ≈ ∫E_b_expected
    end

    # Now coarsen the distribution onto repeated levels so the methods have ties to disagree over
    levels = 12
    tied = @. B₀ * round(tanh((z✶_expected - z_mid) / h) * levels) / levels
    for m in 1:N
        scrambled[1 + mod(7 * (m - 1), N)] = tied[m]
    end
    set!(model, b = reshape(scrambled, Nx, Ny, Nz))

    tied_depth = maximum(v -> count(==(v), tied), unique(tied)) * Δz✶
    profiles, energies = [], Float64[]
    for method in (ThreeDimensionalSort(), HeavisideIntegral(), VerticalSort())
        z✶ = AvailablePotentialEnergyEquation.reference_height(model; method)
        push!(profiles, reference_profile(z✶))
        push!(energies, volume_integral(AvailablePotentialEnergyEquation.BackgroundPotentialEnergy(model, z✶)))
    end

    for (heights, buoyancies) in profiles[2:end]
        @test buoyancies ≈ profiles[1][2]                          # the same sequence of buoyancies
        @test maximum(abs, heights .- profiles[1][1]) ≤ tied_depth # placed within a tied layer of each other
    end
    @test all(energy -> energy ≈ energies[1], energies)             # and exactly the same ∫E_b dV

    return nothing
end

"""
The local available potential energy is the work needed to move a parcel from its reference height to
where it actually sits, `Eₐ = ∫_{z✶}^{z} [b✶(z̃) - b] dz̃`. A parcel carries `b = b✶(z✶)` and `b✶` is
non-decreasing, so the integrand has the same sign as `z̃ - z✶` over the whole path and the result is
non-negative wherever the parcel started. That is the property the Winters form `-b(z - z✶)` lacks,
and it has to hold for every cell, every buoyancy formulation and all three sorting methods.
"""
function test_local_ape_is_non_negative(grid)

    for buoyancy in (BuoyancyTracer(), SeawaterBuoyancy(), SeawaterBuoyancy(equation_of_state=TEOS10EquationOfState()))
        tracers = buoyancy isa BuoyancyTracer ? :b : (:S, :T)
        model = NonhydrostaticModel(grid; buoyancy, tracers)
        buoyancy isa BuoyancyTracer ? set!(model, b = grid_noise) :
                                      set!(model, S = grid_noise, T = grid_noise)

        # Only `HeavisideIntegral` runs on a stretched grid. The guard calls the source's own predicate
        # rather than re-deriving one, so the two cannot drift apart.
        uniform_volumes = !BackgroundPotentialEnergyEquation.stretched_grid(grid)

        for method in (ThreeDimensionalSort(), HeavisideIntegral())
            method isa HeavisideIntegral || uniform_volumes || continue
            z✶ = AvailablePotentialEnergyEquation.reference_height(model; method)
            Eₐ = interior(Field(AvailablePotentialEnergyEquation.AvailablePotentialEnergy(model, z✶)))
            scale = maximum(abs, Eₐ)
            # only roundoff may dip below zero, and it scales with the field, not with the grid
            @test minimum(Eₐ) > -sqrt(eps(eltype(grid))) * max(scale, eps(eltype(grid)))
            @test volume_integral(AvailablePotentialEnergyEquation.AvailablePotentialEnergy(model, z✶)) ≥ 0
        end
    end

    return nothing
end

"""
The local density integrates to the same total the Winters split gives, but only in the continuum
limit: `∫Eₐ dV` evaluates `Ψ` at the model's cell centers while `∫Eₚ - ∫E_b` effectively evaluates it
at the sorted column's, and the two midpoint quadratures differ at finite `Δz`. The gap is second
order, so refining the vertical must shrink it.
"""
function test_local_ape_converges_to_the_winters_total()

    gap(Nz) = begin
        grid  = RectilinearGrid(arch, size=(4, Nz), x=(0, 2), z=(-1, 0), topology=(Periodic, Flat, Bounded))
        model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)
        set!(model, b = (x, z) -> z + 0.4 * sin(9x) * cos(7z))
        z✶  = AvailablePotentialEnergyEquation.reference_height(model)
        ∫Eₐ = volume_integral(AvailablePotentialEnergyEquation.AvailablePotentialEnergy(model, z✶))
        ∫Eₚ = volume_integral(PotentialEnergyEquation.PotentialEnergy(model))
        ∫E_b = volume_integral(AvailablePotentialEnergyEquation.BackgroundPotentialEnergy(model, z✶))
        abs(∫Eₐ - (∫Eₚ - ∫E_b)) / abs(∫Eₚ - ∫E_b)
    end

    coarse, fine = gap(16), gap(64)
    @test fine < coarse / 4        # second order: 4x the resolution should give >4x the accuracy
    @test fine < 1e-2

    return nothing
end

"""
Topography is not supported yet. The sort weights every cell by its full volume, so immersed cells would
be stacked into the reference state as if they held fluid; every method therefore rejects an
`ImmersedBoundaryGrid`, `HeavisideIntegral` included, as do the `model`-level constructors.
"""
function test_sorting_rejects_immersed_boundaries(grid)

    immersed_grid = ImmersedBoundaryGrid(grid, GridFittedBottom((x, y) -> -0.5))
    model = NonhydrostaticModel(immersed_grid; buoyancy=BuoyancyTracer(), tracers=:b)
    set!(model, b = (x, y, z) -> z)

    # Match the message so the assertion pins the immersed check rather than any `ArgumentError`
    for method in (ThreeDimensionalSort(), HeavisideIntegral(), ProfileLookup(), VerticalSort())
        @test_throws "ImmersedBoundaryGrid" AvailablePotentialEnergyEquation.reference_height(model; method)
    end

    # the model-level constructors default to `ThreeDimensionalSort`, so they throw too
    @test_throws "ImmersedBoundaryGrid" AvailablePotentialEnergyEquation.reference_height(model)
    @test_throws "ImmersedBoundaryGrid" AvailablePotentialEnergyEquation.BackgroundPotentialEnergy(model)
    @test_throws "ImmersedBoundaryGrid" AvailablePotentialEnergyEquation.AvailablePotentialEnergy(model)

    return nothing
end

"""
`Υ = z✶ - z` is the displacement a parcel would have to undo to reach the resorted state, so it is
fixed by `z✶` and the grid alone. Checking it against that difference pins the sign, which is the one
thing about `Υ` a reader cannot verify from the result. Sorting moves cells between heights but no
volume, so `Υ` also has to average to zero over the domain.
"""
function test_displacement_potential(grid; method = HeavisideIntegral())

    model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)
    set!(model, b = grid_noise)

    z✶ = AvailablePotentialEnergyEquation.reference_height(model; method)
    Υ  = DisplacementPotential(model, z✶)
    @test Υ isa DisplacementPotential

    @test interior(Field(Υ)) ≈ interior(z✶) .- interior(Field(height_operation(grid)))
    @test volume_mean(Υ) ≈ 0 atol = sqrt(eps(eltype(grid))) * grid.Lz

    return nothing
end

"""
`bᵣ` is the buoyancy a parcel carries relative to the reference profile at its *own* height, so the
sharp check is against that profile rather than against the anomaly's own kernel: `b✶(z)` is the
buoyancy of whichever slot of the sorted column contains `z`, which for a horizontally uniform
stratification is the parcel's own level. Sorting cannot move fluid between levels there, so `bᵣ`
vanishes cell by cell — the same statement `Υ = 0` makes about the displacement, in buoyancy rather
than in height.
"""
function test_reference_buoyancy_anomaly(grid; method = HeavisideIntegral())

    model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)
    set!(model, b = grid_noise)

    z✶  = AvailablePotentialEnergyEquation.reference_height(model; method)
    b✶ᶻ = BackgroundPotentialEnergyEquation.reference_buoyancy_at_height(z✶)
    bᵣ  = ReferenceBuoyancyAnomaly(model, z✶)
    @test bᵣ isa ReferenceBuoyancyAnomaly

    @test interior(Field(bᵣ)) ≈ interior(model.tracers.b) .- interior(b✶ᶻ)

    # Every cell's `b✶(z)` is one of the buoyancies the field holds, since the profile is the sorted
    # field itself: reading it off any other profile would show up here. Both sides come to the host
    # first: `∈` over one array inside a reduction over another is a nested reduction, which a GPU
    # cannot run (`mapreduce cannot figure the output element type`), and a `Set` keeps it linear.
    buoyancies = Set(Array(interior(model.tracers.b)))
    @test all(b -> b ∈ buoyancies, Array(interior(b✶ᶻ)))

    sorted = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)
    set!(sorted, b = (x, y, z) -> 3z)
    @test maximum(abs, interior(Field(ReferenceBuoyancyAnomaly(sorted; method)))) == 0

    # A profile-less `ProfileLookup` ranks the field itself, so it describes the same profile the
    # stacking methods do and has to give the same anomaly. It reaches that profile through the ranking
    # the sort already left behind rather than repeating the sort, which is what `reference_profile`
    # specializes on. (`ProfileLookup` stacks cells, so it needs uniform cell volumes.)
    if !BackgroundPotentialEnergyEquation.stretched_grid(grid)
        z✶_lookup = AvailablePotentialEnergyEquation.reference_height(model, method = ProfileLookup())
        @test interior(Field(ReferenceBuoyancyAnomaly(model, z✶_lookup))) ≈ interior(Field(bᵣ))
    end

    return nothing
end

"""
The `eₐ` budget converts at `w bᵣ` while the `eₚ` budget converts at `wb`, and the two differ by
`w b✶(z)`. Both products are formed on the same face and interpolated the same way, so that difference
is exact cell by cell rather than approximate, which is what this pins down: the conversion is fed the
reference profile itself in place of the anomaly to compute `w b✶(z)`.

`w b✶(z)` is a flux divergence, so it integrates away and leaves `∫w bᵣ dV = ∫wb dV` — the identity the
lock release example's integrated budget rests on, and the reason `PotentialToKineticEnergyConversion`
closes it despite not being this quantity.
"""
function test_ape_to_ke_conversion(grid; method = HeavisideIntegral())

    model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)
    set!(model, b = (x, y, z) -> z + 0.3sin(6x)*cos(4y)*z, w = (x, y, z) -> 0.2sin(3x)*sin(2y)*sin(π*z/grid.Lz))

    z✶  = AvailablePotentialEnergyEquation.reference_height(model; method)
    b✶ᶻ = BackgroundPotentialEnergyEquation.reference_buoyancy_at_height(z✶)
    bᵣ  = Field(ReferenceBuoyancyAnomaly(model, z✶))

    wbᵣ = AvailablePotentialToKineticEnergyConversion(model, z✶; anomaly = bᵣ)
    wb✶ = AvailablePotentialToKineticEnergyConversion(model, z✶; anomaly = b✶ᶻ)
    wb  = PotentialToKineticEnergyConversion(model)
    @test wbᵣ isa AvailablePotentialToKineticEnergyConversion

    @test interior(Field(wb)) ≈ interior(Field(wbᵣ)) .+ interior(Field(wb✶))

    # The two conversions are different fields, so the identity above is a statement about the split
    # and not about them being the same thing.
    @test !(interior(Field(wbᵣ)) ≈ interior(Field(wb)))

    FT = eltype(grid)
    scale = maximum(abs, interior(Field(wb)))
    @test volume_mean(wb✶) ≈ 0 atol = sqrt(eps(FT)) * scale
    @test volume_mean(wbᵣ) ≈ volume_mean(wb)

    # Built without a prebuilt anomaly it has to give the same answer, sorting on its own.
    @test interior(Field(AvailablePotentialToKineticEnergyConversion(model; method))) ≈ interior(Field(wbᵣ))

    # A sorted field has no available energy to release, whatever the velocity does.
    sorted = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)
    set!(sorted, b = (x, y, z) -> 3z, w = (x, y, z) -> 0.2sin(3x)*sin(π*z/grid.Lz))
    @test maximum(abs, interior(Field(AvailablePotentialToKineticEnergyConversion(sorted; method)))) == 0

    return nothing
end

"""
Handed the buoyancy in place of `Υ`, the dissipation kernel computes `κ ∂ᵢb ∂ᵢb` in exactly the
conservative discretization [`TracerVarianceDissipationRate`](@ref) uses for `χ = 2 ∂ⱼb·Fⱼ`, so the two
have to agree to roundoff, `χ` carrying the extra factor of two. That pins the sign, the factor, the
grid metrics and the boundary treatment against a diagnostic tested independently — none of which the
approximate checks elsewhere could separate.
"""
function test_ape_dissipation_matches_tracer_variance_discretization(grid)

    model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b, closure=ScalarDiffusivity(ν=1e-6, κ=1e-3))
    set!(model, b = grid_noise)

    z✶ = AvailablePotentialEnergyEquation.reference_height(model, method=HeavisideIntegral())
    εₐ = AvailablePotentialEnergyDissipationRate(model, z✶; upsilon = model.tracers.b)
    @test εₐ isa AvailablePotentialEnergyDissipationRate

    χ = TracerVarianceEquation.TracerVarianceDissipationRate(model, :b)
    @test interior(Field(εₐ)) ≈ interior(Field(χ)) ./ 2

    return nothing
end

"""
A horizontally uniform, statically stable stratification is its own sorted state, so under
[`HeavisideIntegral`](@ref) it gets `z✶ = z` and holds no available energy to destroy: `Υ` and `εₐ`
both vanish cell by cell. `εₐ` is the sharper of the two, since the diapycnal mixing rate and the
diffusion of the reference state have to cancel rather than each being small — `κ|∇b|²`, which `εₐ` is
sometimes mistaken for, would be nowhere near zero here. `χ/2` is that non-cancelling scale, so it is
what the residual is measured against.
"""
function test_ape_dissipation_vanishes_when_sorted(grid)

    model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b, closure=ScalarDiffusivity(ν=1e-6, κ=1e-3))
    set!(model, b = (x, y, z) -> 3z)

    FT = eltype(grid)
    scale = maximum(abs, interior(Field(TracerVarianceEquation.TracerVarianceDissipationRate(model, :b)))) / 2
    @test scale > 0   # otherwise the assertion below would hold for any implementation

    @test maximum(abs, interior(Field(DisplacementPotential(model)))) < sqrt(eps(FT)) * grid.Lz
    @test maximum(abs, interior(Field(AvailablePotentialEnergyDissipationRate(model)))) < sqrt(eps(FT)) * scale

    return nothing
end

"""
`εₐ` holds `Υ`, which holds `z✶`, so a `compute!` has to refresh the whole chain rather than read a
stale displacement. A frozen `Υ` would not throw or look obviously wrong — it would just report the
previous step's dissipation — so this moves the flow on and checks the result against a diagnostic
built fresh from the new state.

The `upsilon` keyword, which exists so a pair of outputs can share one `Υ` and one sort, has to give
that same answer.
"""
function test_ape_dissipation_is_recomputed(grid)

    model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b, closure=ScalarDiffusivity(ν=1e-6, κ=1e-3))
    set!(model, b = (x, y, z) -> z + 0.3 * sin(9x))

    εₐ = Field(AvailablePotentialEnergyDissipationRate(model))
    stirred = maximum(abs, interior(εₐ))
    @test stirred > 0

    set!(model, b = (x, y, z) -> 2z + 0.5 * sin(7x) * cos(5z))
    compute!(εₐ)

    @test maximum(abs, interior(εₐ)) != stirred
    @test interior(εₐ) ≈ interior(Field(AvailablePotentialEnergyDissipationRate(model)))

    z✶ = AvailablePotentialEnergyEquation.reference_height(model, method=HeavisideIntegral())
    shared = Field(DisplacementPotential(model, z✶))
    @test interior(Field(AvailablePotentialEnergyDissipationRate(model, z✶; upsilon = shared))) ≈ interior(εₐ)

    return nothing
end

"""
`εₐ` is the diapycnal mixing rate less `Φ`, so adding the two back gives that mixing rate,
`κ ∇b·∇z✶`, the quantity mixing raises `E_b` by. `z✶` rises with `b`, so the sum has to be non-negative,
and that is the sharp check: neither `εₐ` nor `Φ` is sign-definite on its own, so getting either one's
sign, scaling or grid metrics wrong shows up here as a negative. Checked cell by cell, which is the
stronger statement, and in the volume integral, which is the one the budget rests on.
"""
function test_ape_dissipation_plus_diffusive_flux_is_the_mixing_rate(grid)

    model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b, closure=ScalarDiffusivity(ν=1e-6, κ=1e-3))
    set!(model, b = (x, y, z) -> 2z + 0.4 * sin(7x) * cos(5z))

    z✶  = AvailablePotentialEnergyEquation.reference_height(model, method=HeavisideIntegral())
    Υ   = Field(DisplacementPotential(model, z✶))
    εₐ = AvailablePotentialEnergyDissipationRate(model, z✶; upsilon = Υ)
    Φ   = PotentialEnergyDiffusiveVerticalBuoyancyFlux(model)

    mixing_rate = interior(Field(εₐ)) .+ interior(Field(Φ))
    scale = maximum(abs, interior(Field(TracerVarianceEquation.TracerVarianceDissipationRate(model, :b)))) / 2
    @test scale > 0
    @test minimum(mixing_rate) > -sqrt(eps(eltype(grid))) * scale
    @test maximum(mixing_rate) > 0.01 * scale   # and it is not uniformly zero, which would pass trivially
    @test volume_integral(εₐ) + volume_integral(Φ) > 0

    return nothing
end

"""
`εₐ` reads `κ∇b` off the closure's own diffusive flux, so the buoyancy has to be a tracer the closure
diffuses; and both diagnostics read the parcel's height off the grid `z✶` lives on, so a
[`VerticalSort`](@ref) column — where that height *is* `z✶`, and a horizontal gradient means nothing —
has to be rejected rather than silently returning zero.
"""
function test_upsilon_and_ape_dissipation_errors(grid)

    seawater = NonhydrostaticModel(grid; buoyancy=SeawaterBuoyancy(), tracers=(:S, :T), closure=ScalarDiffusivity(ν=1e-6, κ=1e-3))
    set!(seawater, S = grid_noise, T = grid_noise)
    @test_throws "BuoyancyTracer" AvailablePotentialEnergyDissipationRate(seawater)
    @test_throws "BuoyancyTracer" PotentialEnergyDiffusiveVerticalBuoyancyFlux(seawater)

    # `κ∇b` comes off the closure, so a model without one has to be refused at construction
    closureless = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)
    @test_throws "no closure" AvailablePotentialEnergyDissipationRate(closureless)

    # `VerticalSort` stacks cells into a column, so it needs uniform cell volumes and cannot be built
    # on a stretched grid at all — there is nothing to reject there.
    BackgroundPotentialEnergyEquation.stretched_grid(grid) && return nothing

    model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b, closure=ScalarDiffusivity(ν=1e-6, κ=1e-3))
    set!(model, b = grid_noise)
    column = AvailablePotentialEnergyEquation.reference_height(model, method=VerticalSort())

    @test_throws "model grid" DisplacementPotential(model, column)
    @test_throws "model grid" AvailablePotentialEnergyDissipationRate(model, column)
    @test_throws "model grid" ReferenceBuoyancyAnomaly(model, column)
    @test_throws "model grid" AvailablePotentialToKineticEnergyConversion(model, column)

    return nothing
end
#---

"""
The reference state is the arrangement of the fluid with the least potential energy, which means sorting
along gravity. These diagnostics sort along `z`, so under a tilted gravity the state they build is not
the minimum-energy one and everything derived from it inherits the error.

The `model` constructors already refuse it, since they route through `reference_height`. The
`(model, z✶)` methods are the ones worth pinning down: a `z✶` built from a bare `Field` carries no model
and so no gravity to validate, which is not a contrived path — it is how `lock_release.jl` builds one,
to hand the same reference height to several diagnostics.
"""
function test_tilted_gravity_is_rejected(grid)

    tilted = BuoyancyForce(BuoyancyTracer(); gravity_unit_vector = (0, sind(10), -cosd(10)))
    model = NonhydrostaticModel(grid; buoyancy = tilted, tracers = :b,
                                closure = ScalarDiffusivity(ν=1e-6, κ=1e-3))
    set!(model, b = grid_noise)

    @test_throws "NegativeZDirection" AvailablePotentialEnergyEquation.reference_height(model)

    # `HeavisideIntegral` so this runs on the stretched grid too, where it is the only method available
    z✶ = AvailablePotentialEnergyEquation.reference_height(model.tracers.b; method = HeavisideIntegral())
    @test_throws "`BackgroundPotentialEnergy`" BackgroundPotentialEnergy(model, z✶)
    @test_throws "`AvailablePotentialEnergy`" AvailablePotentialEnergy(model, z✶)
    @test_throws "`AvailablePotentialEnergyDisplacementPotential`" DisplacementPotential(model, z✶)
    @test_throws "`AvailablePotentialEnergyDissipationRate`" AvailablePotentialEnergyDissipationRate(model, z✶)
    @test_throws "`ReferenceBuoyancyAnomaly`" ReferenceBuoyancyAnomaly(model, z✶)
    @test_throws "`AvailablePotentialToKineticEnergyConversion`" AvailablePotentialToKineticEnergyConversion(model, z✶)

    return nothing
end
#---

@testset "Available potential energy diagnostics" begin
    @info "  Testing available and background potential energy"
    for (grid_class, grid) in zip(keys(grids), values(grids))
        @info "    with $grid_class"
        # On a stretched grid only `HeavisideIntegral` runs, so the general correctness tests, which
        # otherwise use the package default, sort with it there instead.
        grid_method = grid_class == "regular grid" ? ThreeDimensionalSort() : HeavisideIntegral()
        for model_type in model_types
            @info "      with $model_type"
            for buoyancy in extended_buoyancy_formulations

                tracers = buoyancy isa BuoyancyTracer ? :b : (:S, :T)
                model = model_type(grid; buoyancy, tracers)

                if isnothing(buoyancy)
                    @info "        Testing that the split errors when `buoyancy==nothing`"
                    test_available_potential_energy_errors(model)
                else
                    @info "        with $(summary(buoyancy))"
                    # A distinct buoyancy per cell, so the sorting is actually exercised
                    buoyancy isa BuoyancyTracer ? set!(model, b = grid_noise) :
                                                  set!(model, S = grid_noise, T = grid_noise)
                    test_background_and_available_pe(model; method = grid_method)
                    test_background_and_available_pe(model; method = grid_method, geopotential_height = 0)
                end
            end
        end

        @info "      Testing the adiabatically sorted reference state"
        test_available_pe_vanishes_when_sorted(grid; method = grid_method)
        test_reference_state_is_recomputed(grid; method = grid_method)
        test_sorting_rejects_immersed_boundaries(grid)

        @info "      Testing that the local available potential energy is non-negative"
        test_local_ape_is_non_negative(grid)

        @info "      Testing the displacement potential and the APE dissipation rate"
        test_displacement_potential(grid; method = grid_method)

        @info "      Testing the reference buoyancy anomaly and the APE to KE conversion"
        test_reference_buoyancy_anomaly(grid; method = grid_method)
        test_ape_to_ke_conversion(grid; method = grid_method)
        test_ape_dissipation_matches_tracer_variance_discretization(grid)
        test_ape_dissipation_vanishes_when_sorted(grid)
        test_ape_dissipation_is_recomputed(grid)
        test_upsilon_and_ape_dissipation_errors(grid)
        test_tilted_gravity_is_rejected(grid)

        # `Φ` itself is tested with the rest of the `eₚ` equation, in `test_pe_diagnostics.jl`; what
        # belongs here is the identity that defines `εₐ` in terms of it.
        @info "      Testing that εₐ + Φ is the diapycnal mixing rate"
        test_ape_dissipation_plus_diffusive_flux_is_the_mixing_rate(grid)

        @info "      Testing the `ThreeDimensionalSort`, `HeavisideIntegral` and `VerticalSort` methods"
        test_heaviside_is_constant_on_isopycnals(grid)
        if grid_class == "regular grid"
            test_sorting_methods_agree(grid)   # needs at least two methods to run, so regular grids only
            test_one_dimensional_sort_column(grid)
            test_reference_buoyancy_triggers_the_sort(grid)
        else
            @info "      Testing that only `HeavisideIntegral` runs on a stretched grid"
            test_stretched_grid_restrictions(grid)
        end
    end

    @info "  Testing `AvailablePotentialEnergy` against an analytic two-cell column"
    test_available_pe_analytic()

    @info "  Testing the sorting methods against a synthetic profile with a known sorted state"
    test_sorting_methods_reproduce_a_known_profile()

    @info "  Testing that `ProfileLookup` reproduces the ranked sort and validates its profile"
    test_profile_lookup_matches_the_ranked_sort()

    @info "  Testing that the local APE total converges to the Winters split"
    test_local_ape_converges_to_the_winters_total()

    @info "  Testing that `VerticalSort` inherits the model grid's topology"
    test_one_dimensional_sort_matches_topology()
end
