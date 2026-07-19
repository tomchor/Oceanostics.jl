using Test
using CUDA: has_cuda_gpu, @allowscalar

using Oceananigans
using Oceananigans.BuoyancyFormulations: buoyancy_perturbationᶜᶜᶜ, Zᶜᶜᶜ
using Oceananigans.Models: seawater_density, model_geopotential_height
using Oceananigans.TurbulenceClosures.Smagorinskys: LagrangianAveraging
using SeawaterPolynomials: RoquetEquationOfState, TEOS10EquationOfState

using Oceanostics

# Include common test utilities
include("test_utils.jl")

# The geopotential height as a plain operation, used below to check that sorting only rearranges cells.
@inline z_ccc(i, j, k, grid) = Zᶜᶜᶜ(i, j, k, grid)
height_operation(grid) = KernelFunctionOperation{Center, Center, Center}(z_ccc, grid)

volume_integral(op) = sum(Field(Integral(op))) # a scalar, without scalar indexing on GPUs
volume_mean(op)     = sum(Field(Average(op)))  # ditto, volume-weighted over the whole domain

#+++ Test functions
function test_potential_energy_equation_terms_errors(model)

    @test_throws ArgumentError PotentialEnergyEquation.PotentialEnergy(model)
    @test_throws ArgumentError PotentialEnergyEquation.PotentialEnergy(model, geopotential_height = 0)
    @test_throws ArgumentError PotentialEnergyEquation.BackgroundPotentialEnergy(model)
    @test_throws ArgumentError PotentialEnergyEquation.AvailablePotentialEnergy(model)
    @test_throws ArgumentError PotentialEnergyEquation.sorted_reference_height(model)

    return nothing
end

function test_potential_energy_equation_terms(model; geopotential_height = nothing)

    Eₚ = isnothing(geopotential_height) ? PotentialEnergyEquation.PotentialEnergy(model) :
                                          PotentialEnergyEquation.PotentialEnergy(model; geopotential_height)

    Eₚ_field = Field(Eₚ)
    @test Eₚ isa PotentialEnergyEquation.PotentialEnergy
    @test Eₚ_field isa Field

    if model.buoyancy isa PotentialEnergyEquation.BuoyancyBoussinesqEOSModel
        ρ = isnothing(geopotential_height) ? Field(seawater_density(model)) :
                                             Field(seawater_density(model; geopotential_height))

        Z = Field(model_geopotential_height(model))
        ρ₀ = model.buoyancy.formulation.equation_of_state.reference_density
        g = model.buoyancy.formulation.gravitational_acceleration

        @allowscalar begin
            true_value = (g / ρ₀) .* ρ.data .* Z.data
            @test isequal(Eₚ_field.data, true_value)
        end
    end

    return nothing
end

function test_PEbuoyancytracer_equals_PElineareos(grid)

    model_buoyancytracer = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)
    model_lineareos = NonhydrostaticModel(grid; buoyancy=SeawaterBuoyancy(), tracers=(:S, :T))
    C_grad(x, y, z) = 0.01 * z
    set!(model_lineareos, S = C_grad, T = C_grad)
    linear_eos_buoyancy(grid, buoyancy, tracers) =
        KernelFunctionOperation{Center, Center, Center}(buoyancy_perturbationᶜᶜᶜ, grid, buoyancy, tracers)
    b_field = Field(linear_eos_buoyancy(model_lineareos.grid, model_lineareos.buoyancy.formulation, model_lineareos.tracers))
    set!(model_buoyancytracer, b = interior(b_field))
    pe_buoyancytracer = Field(PotentialEnergyEquation.PotentialEnergy(model_buoyancytracer))
    pe_lineareos = Field(PotentialEnergyEquation.PotentialEnergy(model_lineareos))

    @test all(interior(pe_buoyancytracer) .== interior(pe_lineareos))

    return nothing
end

#+++ Background and available potential energy
function test_background_and_available_pe(model; geopotential_height = nothing)

    kwargs = isnothing(geopotential_height) ? (;) : (; geopotential_height)

    z✶  = PotentialEnergyEquation.sorted_reference_height(model; kwargs...)
    Eₚ  = PotentialEnergyEquation.PotentialEnergy(model; kwargs...)
    E_b = PotentialEnergyEquation.BackgroundPotentialEnergy(model, z✶)
    E_a = PotentialEnergyEquation.AvailablePotentialEnergy(model, z✶)

    @test E_b isa PotentialEnergyEquation.BackgroundPotentialEnergy
    @test E_a isa PotentialEnergyEquation.AvailablePotentialEnergy

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

    z✶ = PotentialEnergyEquation.sorted_reference_height(model)
    @allowscalar @test interior(z✶)[1, 1, :] ≈ [-0.25, -0.75]

    @test volume_integral(PotentialEnergyEquation.PotentialEnergy(model))               ≈ +0.25
    @test volume_integral(PotentialEnergyEquation.BackgroundPotentialEnergy(model, z✶)) ≈ -0.25
    @test volume_integral(PotentialEnergyEquation.AvailablePotentialEnergy(model, z✶))  ≈ +0.50

    # Flipping the column makes it statically stable, hence already sorted, hence free of available PE
    set!(model, b = reshape([-1.0, 1.0], 1, 1, 2))
    z✶ = PotentialEnergyEquation.sorted_reference_height(model)
    @allowscalar @test interior(z✶)[1, 1, :] ≈ [-0.75, -0.25]
    @test volume_integral(PotentialEnergyEquation.BackgroundPotentialEnergy(model, z✶)) ≈
          volume_integral(PotentialEnergyEquation.PotentialEnergy(model))
    @test volume_integral(PotentialEnergyEquation.AvailablePotentialEnergy(model, z✶)) ≈ 0 atol=1e-14

    return nothing
end

"A horizontally uniform, statically stable stratification is its own sorted state, so it holds no APE."
function test_available_pe_vanishes_when_sorted(grid)

    model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)
    set!(model, b = (x, y, z) -> 3z)

    ∫E_a = volume_integral(PotentialEnergyEquation.AvailablePotentialEnergy(model))
    ∫Eₚ  = volume_integral(PotentialEnergyEquation.PotentialEnergy(model))

    @test ∫E_a ≈ 0 atol=sqrt(eps(eltype(grid))) * abs(∫Eₚ)
    @test volume_integral(PotentialEnergyEquation.BackgroundPotentialEnergy(model)) ≈ ∫Eₚ

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

    z✶   = PotentialEnergyEquation.sorted_reference_height(model)
    ∫E_b = Field(Integral(PotentialEnergyEquation.BackgroundPotentialEnergy(model, z✶)))
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
    for method in (CellRanking(), HeavisideIntegral(), OneDimensionalSort())

        # `OneDimensionalSort` bakes the sorted column into a grid, so it only accepts uniform volumes
        uniform_volumes = minimum(zspacings(grid)) ≈ maximum(zspacings(grid))
        method isa OneDimensionalSort && !uniform_volumes && continue

        z✶  = PotentialEnergyEquation.sorted_reference_height(model; method)
        ∫E_b = volume_integral(PotentialEnergyEquation.BackgroundPotentialEnergy(model, z✶))
        ∫E_a = volume_integral(PotentialEnergyEquation.AvailablePotentialEnergy(model, z✶))

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
`z✶ = z` and `Eₐ = 0` cell by cell. `CellRanking` gives the tied cells consecutive slots instead, which
spreads `z✶` over the cell they share without moving the integral.
"""
function test_heaviside_is_constant_on_isopycnals(grid)

    model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)
    set!(model, b = (x, y, z) -> 3z)

    heights = interior(Field(height_operation(grid)))
    Δz_max = maximum(zspacings(grid))

    z✶ = PotentialEnergyEquation.sorted_reference_height(model, method=HeavisideIntegral())
    @test interior(z✶) ≈ heights
    @test maximum(abs, interior(Field(PotentialEnergyEquation.AvailablePotentialEnergy(model, z✶)))) <
          sqrt(eps(eltype(grid)))

    # The ranked heights stay within half a cell of the parcel's own height, and no closer
    z✶_ranked = PotentialEnergyEquation.sorted_reference_height(model, method=CellRanking())
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

    z✶ = PotentialEnergyEquation.sorted_reference_height(model, method=OneDimensionalSort())
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
    @test_throws ArgumentError PotentialEnergyEquation.sorted_reference_height(model, method=OneDimensionalSort())

    return nothing
end

"An `ImmersedBoundaryGrid` has a depth-dependent horizontal area, which the sorting does not support."
function test_sorting_rejects_immersed_boundaries(grid)

    immersed_grid = ImmersedBoundaryGrid(grid, GridFittedBottom((x, y) -> -0.5))
    model = NonhydrostaticModel(immersed_grid; buoyancy=BuoyancyTracer(), tracers=:b)

    @test_throws ArgumentError PotentialEnergyEquation.sorted_reference_height(model)
    @test_throws ArgumentError PotentialEnergyEquation.BackgroundPotentialEnergy(model)
    @test_throws ArgumentError PotentialEnergyEquation.AvailablePotentialEnergy(model)

    return nothing
end
#---
#---

@testset "Diagnostics tests" begin
    @info "  Testing Diagnostics"
    for (grid_class, grid) in zip(keys(grids), values(grids))
        @info "    with $grid_class"
        for model_type in model_types
            @info "      with $model_type"
            for buoyancy in extended_buoyancy_formulations
                @info "        with $(summary(buoyancy))"

                tracers = buoyancy isa BuoyancyTracer ? :b : (:S, :T)
                model = model_type(grid; buoyancy, tracers)
                buoyancy isa BuoyancyTracer ? set!(model, b = 9.87) : set!(model, S = 34.7, T = 0.5)

                if isnothing(buoyancy)
                    @info "          Testing that potential energy equation terms throw error when `buoyancy==nothing`"
                    test_potential_energy_equation_terms_errors(model)
                else
                    @info "          Testing `PotentialEnergy`"
                    test_potential_energy_equation_terms(model)
                    test_potential_energy_equation_terms(model, geopotential_height = 0)

                    @info "          Testing `BackgroundPotentialEnergy` and `AvailablePotentialEnergy`"
                    # The shared setup sets a uniform buoyancy, whose sorted state is degenerate; give
                    # each cell a distinct value so the sorting is actually exercised.
                    buoyancy isa BuoyancyTracer ? set!(model, b = grid_noise) :
                                                  set!(model, S = grid_noise, T = grid_noise)
                    test_background_and_available_pe(model)
                    test_background_and_available_pe(model, geopotential_height = 0)
                end
            end
            test_PEbuoyancytracer_equals_PElineareos(grid)
        end

        @info "      Testing the adiabatically sorted reference state"
        test_available_pe_vanishes_when_sorted(grid)
        test_reference_state_is_recomputed(grid)
        test_sorting_rejects_immersed_boundaries(grid)

        @info "      Testing the `CellRanking`, `HeavisideIntegral` and `OneDimensionalSort` methods"
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
end
