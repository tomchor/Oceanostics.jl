using Test
using CUDA: has_cuda_gpu, @allowscalar

using Oceananigans
using Oceananigans.BuoyancyFormulations: Zᶜᶜᶜ
using SeawaterPolynomials: RoquetEquationOfState, TEOS10EquationOfState

using Oceanostics

# Include common test utilities
include("test_utils.jl")

# The geopotential height as a plain operation, used below to check that sorting only rearranges cells.
@inline z_ccc(i, j, k, grid) = Zᶜᶜᶜ(i, j, k, grid)
height_operation(grid) = KernelFunctionOperation{Center, Center, Center}(z_ccc, grid)

volume_integral(op) = sum(Field(Integral(op))) # a scalar, without scalar indexing on GPUs
volume_mean(op)     = sum(Field(Average(op)))  # ditto, volume-weighted over the whole domain

"""
The reference profile a sorting method describes, as `(z✶, b✶)` ordered from the bottom of the sorted
column up. `OneDimensionalSort` already stores it that way; the two model-grid methods hold `z✶` and
`b` cell by cell, so pairing them and ordering by `z✶` recovers the same profile.
"""
function reference_profile(z✶)

    heights   = Array(vec(interior(z✶)))
    buoyancy  = Array(vec(interior(AvailablePotentialEnergyEquation.sorted_buoyancy(z✶.operand))))
    ascending = sortperm(heights)

    return heights[ascending], buoyancy[ascending]
end

#+++ Test functions
"Neither half of the split is defined without a buoyancy formulation to sort."
function test_available_potential_energy_errors(model)

    @test_throws ArgumentError AvailablePotentialEnergyEquation.BackgroundPotentialEnergy(model)
    @test_throws ArgumentError AvailablePotentialEnergyEquation.AvailablePotentialEnergy(model)
    @test_throws ArgumentError AvailablePotentialEnergyEquation.sorted_reference_height(model)

    return nothing
end

function test_background_and_available_pe(model; geopotential_height = nothing)

    kwargs = isnothing(geopotential_height) ? (;) : (; geopotential_height)

    z✶  = AvailablePotentialEnergyEquation.sorted_reference_height(model; kwargs...)
    Eₚ  = PotentialEnergyEquation.PotentialEnergy(model; kwargs...)
    E_b = AvailablePotentialEnergyEquation.BackgroundPotentialEnergy(model, z✶)
    E_a = AvailablePotentialEnergyEquation.AvailablePotentialEnergy(model, z✶)

    @test E_b isa AvailablePotentialEnergyEquation.BackgroundPotentialEnergy
    @test E_a isa AvailablePotentialEnergyEquation.AvailablePotentialEnergy

    Eₚ_field, E_b_field, E_a_field = Field(Eₚ), Field(E_b), Field(E_a)
    @test E_b_field isa Field
    @test E_a_field isa Field

    # The decomposition Eₚ = E_b + Eₐ holds cell by cell, since Eₐ is built from the same buoyancy
    @test interior(Eₚ_field) ≈ interior(E_b_field) .+ interior(E_a_field)

    # Sorting only rearranges cells between heights, so it moves no volume and leaves the
    # volume-weighted mean height where it was. The comparison needs an absolute tolerance because
    # that mean is zero on a grid whose z is symmetric about the origin.
    grid = model.grid
    FT = eltype(grid)
    @test volume_mean(z✶) ≈ volume_mean(height_operation(grid)) atol = sqrt(eps(FT)) * grid.Lz

    # The sorted state is the state of minimum potential energy, so ∫Eₐ = ∫Eₚ - ∫E_b ≥ 0
    ∫Eₚ, ∫E_b, ∫E_a = volume_integral(Eₚ), volume_integral(E_b), volume_integral(E_a)
    @test ∫E_a ≈ ∫Eₚ - ∫E_b
    @test ∫E_a ≥ -sqrt(eps(FT)) * max(abs(∫Eₚ), abs(∫E_b))

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

    z✶ = AvailablePotentialEnergyEquation.sorted_reference_height(model)
    @allowscalar @test interior(z✶)[1, 1, :] ≈ [-0.25, -0.75]

    @test volume_integral(PotentialEnergyEquation.PotentialEnergy(model))               ≈ +0.25
    @test volume_integral(AvailablePotentialEnergyEquation.BackgroundPotentialEnergy(model, z✶)) ≈ -0.25
    @test volume_integral(AvailablePotentialEnergyEquation.AvailablePotentialEnergy(model, z✶))  ≈ +0.50

    # Flipping the column makes it statically stable, hence already sorted, hence free of available PE
    set!(model, b = reshape([-1.0, 1.0], 1, 1, 2))
    z✶ = AvailablePotentialEnergyEquation.sorted_reference_height(model)
    @allowscalar @test interior(z✶)[1, 1, :] ≈ [-0.75, -0.25]
    @test volume_integral(AvailablePotentialEnergyEquation.BackgroundPotentialEnergy(model, z✶)) ≈
          volume_integral(PotentialEnergyEquation.PotentialEnergy(model))
    @test volume_integral(AvailablePotentialEnergyEquation.AvailablePotentialEnergy(model, z✶)) ≈ 0 atol=1e-14

    return nothing
end

"A horizontally uniform, statically stable stratification is its own sorted state, so it holds no APE."
function test_available_pe_vanishes_when_sorted(grid)

    model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)
    set!(model, b = (x, y, z) -> 3z)

    ∫E_a = volume_integral(AvailablePotentialEnergyEquation.AvailablePotentialEnergy(model))
    ∫Eₚ  = volume_integral(PotentialEnergyEquation.PotentialEnergy(model))

    @test ∫E_a ≈ 0 atol=sqrt(eps(eltype(grid))) * abs(∫Eₚ)
    @test volume_integral(AvailablePotentialEnergyEquation.BackgroundPotentialEnergy(model)) ≈ ∫Eₚ

    return nothing
end

"""
The sorted reference state depends on every cell in the domain, so it has to be re-sorted on
every `compute!` rather than baked in when the diagnostic is constructed. Mixing the flow completely
(which conserves `∫b dV`) has to raise `∫E_b` to `∫Eₚ`, and a frozen reference state would not notice.
"""
function test_reference_state_is_recomputed(grid)

    model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)
    set!(model, b = (x, y, z) -> z + 0.5 * sin(6z)) # statically unstable in places, so unsorted

    z✶   = AvailablePotentialEnergyEquation.sorted_reference_height(model)
    ∫E_b = Field(Integral(AvailablePotentialEnergyEquation.BackgroundPotentialEnergy(model, z✶)))
    ∫Eₚ  = Field(Integral(PotentialEnergyEquation.PotentialEnergy(model)))

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

    reference = nothing
    for method in (ThreeDimensionalSort(), HeavisideIntegral(), OneDimensionalSort())

        # `OneDimensionalSort` bakes the sorted column into a grid, so it only accepts uniform volumes
        uniform_volumes = minimum(zspacings(grid)) ≈ maximum(zspacings(grid))
        method isa OneDimensionalSort && !uniform_volumes && continue

        z✶  = AvailablePotentialEnergyEquation.sorted_reference_height(model; method)
        ∫E_b = volume_integral(AvailablePotentialEnergyEquation.BackgroundPotentialEnergy(model, z✶))
        ∫E_a = volume_integral(AvailablePotentialEnergyEquation.AvailablePotentialEnergy(model, z✶))

        if isnothing(reference)
            reference = (∫E_b, ∫E_a)
            @test ∫E_a > 0
            @test ∫E_b + ∫E_a ≈ volume_integral(PotentialEnergyEquation.PotentialEnergy(model))
        else
            @test ∫E_b ≈ reference[1]
            @test ∫E_a ≈ reference[2]
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

    z✶ = AvailablePotentialEnergyEquation.sorted_reference_height(model, method=HeavisideIntegral())
    @test interior(z✶) ≈ heights
    @test maximum(abs, interior(Field(AvailablePotentialEnergyEquation.AvailablePotentialEnergy(model, z✶)))) <
          sqrt(eps(eltype(grid)))

    # The ranked heights stay within half a cell of the parcel's own height, and no closer
    z✶_ranked = AvailablePotentialEnergyEquation.sorted_reference_height(model, method=ThreeDimensionalSort())
    @test maximum(abs, interior(z✶_ranked) .- heights) ≤ Δz_max / 2

    return nothing
end

"""
`OneDimensionalSort` returns the sorted column on its own `1×1×N` grid: the same cells, reshaped to
span the domain's horizontal area, stacked in order of increasing buoyancy.
"""
function test_one_dimensional_sort_column(grid)

    model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)
    set!(model, b = (x, y, z) -> z + 0.4 * sin(9x) * cos(7z))

    z✶ = AvailablePotentialEnergyEquation.sorted_reference_height(model, method=OneDimensionalSort())
    b✶ = z✶.operand.method.sorted_buoyancy

    @test size(z✶) == (1, 1, prod(size(grid)))
    @test issorted(Array(vec(interior(b✶))))  # densest at the bottom
    @test issorted(Array(vec(interior(z✶))))
    # Reshaping the cells conserves volume, so the column integrates like the model grid does
    @test volume_integral(b✶) ≈ volume_integral(model.tracers.b)

    return nothing
end

"`OneDimensionalSort` bakes the sorted column's cell boundaries into a grid, so volumes must be equal."
function test_one_dimensional_sort_rejects_stretched_grids(grid)

    model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)
    @test_throws ArgumentError AvailablePotentialEnergyEquation.sorted_reference_height(model, method=OneDimensionalSort())

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

    for method in (ThreeDimensionalSort(), HeavisideIntegral(), OneDimensionalSort())
        z✶ = AvailablePotentialEnergyEquation.sorted_reference_height(model; method)
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
    for method in (ThreeDimensionalSort(), HeavisideIntegral(), OneDimensionalSort())
        z✶ = AvailablePotentialEnergyEquation.sorted_reference_height(model; method)
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

"An `ImmersedBoundaryGrid` has a depth-dependent horizontal area, which the sorting does not support."
function test_sorting_rejects_immersed_boundaries(grid)

    immersed_grid = ImmersedBoundaryGrid(grid, GridFittedBottom((x, y) -> -0.5))
    model = NonhydrostaticModel(immersed_grid; buoyancy=BuoyancyTracer(), tracers=:b)

    @test_throws ArgumentError AvailablePotentialEnergyEquation.sorted_reference_height(model)
    @test_throws ArgumentError AvailablePotentialEnergyEquation.BackgroundPotentialEnergy(model)
    @test_throws ArgumentError AvailablePotentialEnergyEquation.AvailablePotentialEnergy(model)

    return nothing
end
#---

@testset "Available potential energy diagnostics" begin
    @info "  Testing available and background potential energy"
    for (grid_class, grid) in zip(keys(grids), values(grids))
        @info "    with $grid_class"
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
                    test_background_and_available_pe(model)
                    test_background_and_available_pe(model, geopotential_height = 0)
                end
            end
        end

        @info "      Testing the adiabatically sorted reference state"
        test_available_pe_vanishes_when_sorted(grid)
        test_reference_state_is_recomputed(grid)
        test_sorting_rejects_immersed_boundaries(grid)

        @info "      Testing the `ThreeDimensionalSort`, `HeavisideIntegral` and `OneDimensionalSort` methods"
        test_sorting_methods_agree(grid)
        test_heaviside_is_constant_on_isopycnals(grid)
        if grid_class == "regular grid"
            test_one_dimensional_sort_column(grid)
        else
            test_one_dimensional_sort_rejects_stretched_grids(grid)
        end
    end

    @info "  Testing `AvailablePotentialEnergy` against an analytic two-cell column"
    test_available_pe_analytic()

    @info "  Testing the sorting methods against a synthetic profile with a known sorted state"
    test_sorting_methods_reproduce_a_known_profile()
end
