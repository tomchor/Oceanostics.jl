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

#+++ Test functions
function test_potential_energy_equation_terms_errors(model)

    @test_throws ArgumentError PotentialEnergyEquation.PotentialEnergy(model)
    @test_throws ArgumentError PotentialEnergyEquation.PotentialEnergy(model, geopotential_height = 0)

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
#---

"""
`Φ = κ ∂b/∂z` is fixed by the buoyancy and the closure alone, so a linear profile pins its value, its
sign and the `κ` it picks up off the closure. The boundary treatment is the part worth guarding: no
buoyancy crosses a no-flux wall, which halves the cells against it and is exactly what makes the volume
integral telescope to `κ A [b(z_top) - b(z_bottom)]` however the interior is stirred. An interpolation
that mishandled the walls would still look right cell by cell in the interior and would break that.
"""
function test_reference_state_diffusion_rate(grid)

    κ = 1e-3
    model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b, closure=ScalarDiffusivity(; ν=1e-6, κ))

    Φ = ReferenceStateDiffusionRate(model)
    @test Φ isa ReferenceStateDiffusionRate

    set!(model, b = (x, y, z) -> 3z)
    Φ_column = Array(interior(Field(Φ)))[1, 1, :]
    @test all(Φ_column[2:end-1] .≈ 3κ)
    @test Φ_column[1] ≈ 3κ / 2 && Φ_column[end] ≈ 3κ / 2   # the no-flux walls halve the outermost cells

    set!(model, b = grid_noise)
    b = Array(interior(model.tracers.b))
    Δb = sum(b[:, :, end] .- b[:, :, 1]) / prod(size(b)[1:2])   # ⟨b(z_top) - b(z_bottom)⟩
    @test volume_integral(Φ) ≈ κ * grid.Lx * grid.Ly * Δb

    return nothing
end

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
                end
            end
            test_PEbuoyancytracer_equals_PElineareos(grid)
        end
        @info "      Testing the reference state diffusion rate"
        test_reference_state_diffusion_rate(grid)
    end
end
