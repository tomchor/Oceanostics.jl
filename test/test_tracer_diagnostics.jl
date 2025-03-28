using Test
using CUDA: has_cuda_gpu, @allowscalar

using Oceananigans
using Oceananigans.AbstractOperations: AbstractOperation
using Oceananigans.BuoyancyFormulations: buoyancy_perturbationᶜᶜᶜ
using Oceananigans.Fields: @compute
using Oceananigans.TurbulenceClosures.Smagorinskys: LagrangianAveraging
using Oceananigans.Models: seawater_density, model_geopotential_height
using Oceananigans.BuoyancyFormulations: buoyancy
using SeawaterPolynomials: RoquetEquationOfState, TEOS10EquationOfState

using Oceanostics
using Oceanostics: TKEBudgetTerms, TracerVarianceBudgetTerms, FlowDiagnostics, PressureRedistributionTerm, BuoyancyProductionTerm, AdvectionTerm
using Oceanostics.TKEBudgetTerms: AdvectionTerm
using Oceanostics: PotentialEnergy, PotentialEnergyEquationTerms.BuoyancyBoussinesqEOSModel
using Oceanostics: perturbation_fields, get_coriolis_frequency_components

const LinearBuoyancyForce = Union{BuoyancyTracer, SeawaterBuoyancy{<:Any, <:LinearEquationOfState}}

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
model_kwargs = (buoyancy = BuoyancyForce(BuoyancyTracer()),
                coriolis = FPlane(1e-4),
                tracers = :b)

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
function test_velocity_only_flow_diagnostics(model)
    op = RossbyNumber(model;)
    @test op isa AbstractOperation
    Ro = Field(op)
    @test all(interior(compute!(Ro)) .≈ 0)

    op = RossbyNumber(model, model.velocities..., model.coriolis)
    @test op isa AbstractOperation
    Ro = Field(op)
    @test all(interior(compute!(Ro)) .≈ 0)

    op = RossbyNumber(model; dUdy_bg=1, dVdx_bg=1)
    @test op isa AbstractOperation
    Ro = Field(op)
    @test all(interior(compute!(Ro)) .≈ 0)

    op = StrainRateTensorModulus(model)
    @test op isa AbstractOperation
    S = Field(op)
    @test all(interior(compute!(S)) .≈ 0)

    op = VorticityTensorModulus(model)
    @test op isa AbstractOperation
    Ω = Field(op)
    @test all(interior(compute!(Ω)) .≈ 0)

    op = QVelocityGradientTensorInvariant(model)
    @allowscalar @test op == Oceanostics.Q(model)
    @test op isa AbstractOperation
    q = Field(op)
    @test all(interior(compute!(q)) .≈ 0)

    return nothing
end

function test_buoyancy_diagnostics(model)
    u, v, w = model.velocities

    N² = 1e-5;
    S = 1e-3;
    shear(x, y, z) = S*z + S*y;
    set!(model, u=shear)

    if model.buoyancy != nothing && model.buoyancy.formulation isa SeawaterBuoyancy{<:Any, <:LinearEquationOfState}
        g = model.buoyancy.formulation.gravitational_acceleration
        α = model.buoyancy.formulation.equation_of_state.thermal_expansion
        stratification_T(x, y, z) = N² * z / (g * α)
        set!(model, T=stratification_T)

    else
        stratification_b(x, y, z) = N² * z
        set!(model, b=stratification_b)
    end

    fx, fy, fz = get_coriolis_frequency_components(model)
    if model.buoyancy != nothing && model.buoyancy.formulation isa LinearBuoyancyForce

        Ri = RichardsonNumber(model)
        @test Ri isa AbstractOperation
        @compute Ri_field = Field(Ri)
        @test Ri_field isa Field
        @test interior(Ri_field, 3, 3, 3)[1] ≈ N² / S^2

        b = buoyancy(model)
        Ri = RichardsonNumber(model, u, v, w, b)
        @test Ri isa AbstractOperation
        @compute Ri_field = Field(Ri)
        @test Ri_field isa Field
        @test interior(Ri_field, 3, 3, 3)[1] ≈ N² / S^2

    else
        b = model.tracers.b # b in this case is passive
    end

    if model.buoyancy != nothing && model.buoyancy.formulation isa SeawaterBuoyancy{<:Any, <:LinearEquationOfState}
        EPV = ErtelPotentialVorticity(model, tracer=:T)
        @test EPV isa AbstractOperation
        EPV_field = compute!(Field(EPV))
        @test EPV_field isa Field
        @test interior(EPV_field, 3, 3, 3)[1] ≈ N² * (fz - S) / (g * α)

    else
        EPV = ErtelPotentialVorticity(model)
        @test EPV isa AbstractOperation
        EPV_field = compute!(Field(EPV))
        @test EPV_field isa Field
        @test interior(EPV_field, 3, 3, 3)[1] ≈ N² * (fz - S)
    end

    EPV = ErtelPotentialVorticity(model, u, v, w, b, model.coriolis)
    @test EPV isa AbstractOperation
    EPV_field = compute!(Field(EPV))
    @test EPV_field isa Field
    @test interior(EPV_field, 3, 3, 3)[1] ≈ N² * (fz - S)

    PVtw = ThermalWindPotentialVorticity(model)
    @test PVtw isa AbstractOperation
    @test compute!(Field(PVtw)) isa Field

    PVtw = ThermalWindPotentialVorticity(model, u, v, b, FPlane(1e-4))
    @test PVtw isa AbstractOperation
    @test compute!(Field(PVtw)) isa Field

    DEPV = DirectionalErtelPotentialVorticity(model, (0, 0, 1))
    @test DEPV isa AbstractOperation
    @test compute!(Field(DEPV)) isa Field

    DEPV = DirectionalErtelPotentialVorticity(model, (0, 0, 1), u, v, w, b, model.coriolis)
    @test DEPV isa AbstractOperation
    @test compute!(Field(DEPV)) isa Field

    return nothing
end

function test_tracer_diagnostics(model)
    χ = TracerVarianceDissipationRate(model, :b)
    χ_field = compute!(Field(χ))
    @test χ isa AbstractOperation
    @test χ_field isa Field

    b̄ = Field(Average(model.tracers.b, dims=(1,2)))
    b′ = model.tracers.b - b̄
    χ = TracerVarianceDissipationRate(model, :b, tracer=b′)
    χ_field = compute!(Field(χ))
    @test χ isa AbstractOperation
    @test χ_field isa Field

    χ = TracerVarianceDiffusiveTerm(model, :b)
    χ_field = compute!(Field(χ))
    @test χ isa AbstractOperation
    @test χ_field isa Field

    set!(model, u = (x, y, z) -> z, v = grid_noise, w = grid_noise, b = grid_noise)
    @compute ε̄ₚ = Field(Average(TracerVarianceDissipationRate(model, :b)))
    @compute ε̄ₚ₂ = Field(Average(TracerVarianceDiffusiveTerm(model, :b)))
    @test ≈(Array(interior(ε̄ₚ, 1, 1, 1)), Array(interior(ε̄ₚ₂, 1, 1, 1)), rtol=1e-12, atol=2*eps())

    if model isa NonhydrostaticModel
        χ = TracerVarianceTendency(model, :b)
        χ_field = compute!(Field(χ))
        @test χ isa AbstractOperation
        @test χ_field isa Field

        @compute ∂ₜc² = Field(Average(TracerVarianceTendency(model, :b)))
    end

    return nothing
end

function test_potential_energy_equation_terms_errors(model)

    @test_throws ArgumentError PotentialEnergy(model)
    @test_throws ArgumentError PotentialEnergy(model, geopotential_height = 0)

    return nothing
end

function test_potential_energy_equation_terms(model; geopotential_height = nothing)

    Eₚ = isnothing(geopotential_height) ? PotentialEnergy(model) :
                                          PotentialEnergy(model; geopotential_height)

    Eₚ_field = Field(Eₚ)
    @test Eₚ isa AbstractOperation
    @test Eₚ_field isa Field
    compute!(Eₚ_field)

    if model.buoyancy isa BuoyancyBoussinesqEOSModel
        ρ = isnothing(geopotential_height) ? Field(seawater_density(model)) :
                                             Field(seawater_density(model; geopotential_height))

        compute!(ρ)
        Z = Field(model_geopotential_height(model))
        compute!(Z)
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
    compute!(b_field)
    set!(model_buoyancytracer, b = interior(b_field))
    pe_buoyancytracer = Field(PotentialEnergy(model_buoyancytracer))
    compute!(pe_buoyancytracer)
    pe_lineareos = Field(PotentialEnergy(model_lineareos))
    compute!(pe_lineareos)

    @test all(interior(pe_buoyancytracer) .== interior(pe_lineareos))

    return nothing

end
#---

#+++ Known-value function tests
function test_auxiliary_functions(model)
    set!(model, u=1, v=2)
    fields_without_means = perturbation_fields(model; u=1, v=2)
    compute!(fields_without_means)
    @test all(Array(interior(fields_without_means.u)) .== 0)
    @test all(Array(interior(fields_without_means.v)) .== 0)
    return
end
#---

@testset "Diagnostics tests" begin
    @info "  Testing Diagnostics"
    for (grid_class, grid) in zip(keys(grids), values(grids))
        @info "    with $grid_class"
        for model_type in model_types
            @info "      with $model_type"
            for closure in closures
                @info "        with closure $(summary(closure))"
                model = model_type(; grid, closure, model_kwargs...)

                @info "          Testing velocity-only diagnostics"
                test_velocity_only_flow_diagnostics(model)

                @info "          Testing auxiliary functions"
                test_auxiliary_functions(model)

                @info "          Testing tracer variance terms"
                test_tracer_diagnostics(model)
            end

            @info "        Testing diagnostics that use buoyancy"
            for buoyancy in buoyancy_formulations
                @info "        with $(summary(buoyancy))"

                tracers = buoyancy isa BuoyancyTracer ? :b : (:S, :T)
                model = model_type(; grid, buoyancy, tracers)
                buoyancy isa BuoyancyTracer ? set!(model, b = 9.87) : set!(model, S = 34.7, T = 0.5)

                if isnothing(buoyancy)
                    @info "            Testing that potential energy equation terms throw error when `buoyancy==nothing`"
                    test_potential_energy_equation_terms_errors(model)
                else
                    @info "            Testing `PotentialEnergy` with buoyancy " buoyancy
                    test_potential_energy_equation_terms(model)
                    test_potential_energy_equation_terms(model, geopotential_height = 0)
                end

                for coriolis in coriolis_formulations
                    tracers = buoyancy isa BuoyancyTracer ? :b : (:S, :T, :b)
                    model = model_type(; grid, buoyancy, tracers, coriolis)
                    buoyancy isa BuoyancyTracer ? set!(model, b = 9.87) : set!(model, S = 34.7, T = 0.5)

                    @info "          Testing buoyancy diagnostics"
                    test_buoyancy_diagnostics(model)
                end

            end
            test_PEbuoyancytracer_equals_PElineareos(grid)
        end
    end
end
