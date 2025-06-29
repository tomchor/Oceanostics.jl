using Test
using CUDA: has_cuda_gpu, @allowscalar

using Oceananigans
using Oceananigans.BuoyancyFormulations: buoyancy_perturbationᶜᶜᶜ
using Oceananigans.Models: seawater_density, model_geopotential_height
using Oceananigans.TurbulenceClosures.Smagorinskys: LagrangianAveraging
using SeawaterPolynomials: RoquetEquationOfState, TEOS10EquationOfState

using Oceanostics

#+++ Default grids
arch = has_cuda_gpu() ? GPU() : CPU()

N = 6
regular_grid = RectilinearGrid(arch, size=(N, N, N), extent=(1, 1, 1))

S = .99 # Stretching factor. Positive number ∈ (0, 1]
f_asin(k) = -asin(S*(2k - N - 2) / N)/π + 1/2
F1 = f_asin(1); F2 = f_asin(N+1)
z_faces(k) = ((F1 + F2)/2 - f_asin(k)) / (F1 - F2)

stretched_grid = RectilinearGrid(arch, size=(N, N, N), x=(0, 1), y=(0, 1), z=z_faces)

grid_noise(x, y, z) = randn()
#---

#+++ Test options
closures = (ScalarDiffusivity(ν=1e-6, κ=1e-7),
            SmagorinskyLilly(),
            Smagorinsky(coefficient=DynamicCoefficient(averaging=(1, 2))),
            Smagorinsky(coefficient=DynamicCoefficient(averaging=LagrangianAveraging())),
            (ScalarDiffusivity(ν=1e-6, κ=1e-7), AnisotropicMinimumDissipation()),)

buoyancy_formulations = (nothing,
                         BuoyancyTracer(),
                         SeawaterBuoyancy(),
                         SeawaterBuoyancy(equation_of_state=TEOS10EquationOfState()),
                         SeawaterBuoyancy(equation_of_state=RoquetEquationOfState(:Linear)))

coriolis_formulations = (nothing,
                         FPlane(1e-4),
                         ConstantCartesianCoriolis(fx=1e-4, fy=1e-4, fz=1e-4))

grids = Dict("regular grid" => regular_grid,
             "stretched grid" => stretched_grid)

model_types = (NonhydrostaticModel,
               HydrostaticFreeSurfaceModel)
#---

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

    model_buoyancytracer = NonhydrostaticModel(; grid, buoyancy=BuoyancyTracer(), tracers=:b)
    model_lineareos = NonhydrostaticModel(; grid, buoyancy=SeawaterBuoyancy(), tracers=(:S, :T))
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

@testset "Diagnostics tests" begin
    @info "  Testing Diagnostics"
    for (grid_class, grid) in zip(keys(grids), values(grids))
        @info "    with $grid_class"
        for model_type in model_types
            @info "      with $model_type"
            for buoyancy in buoyancy_formulations
                @info "        with $(summary(buoyancy))"

                tracers = buoyancy isa BuoyancyTracer ? :b : (:S, :T)
                model = model_type(; grid, buoyancy, tracers)
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
    end
end
