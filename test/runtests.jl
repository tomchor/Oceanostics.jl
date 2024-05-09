using Test
using CUDA

using Oceananigans
using Oceananigans.AbstractOperations: AbstractOperation
using Oceananigans.Fields: @compute
using Oceananigans.TurbulenceClosures: ThreeDimensionalFormulation
using Oceananigans.Models: seawater_density, model_geopotential_height
using SeawaterPolynomials: RoquetEquationOfState, TEOS10EquationOfState

using Oceanostics
using Oceanostics: TKEBudgetTerms, TracerVarianceBudgetTerms, FlowDiagnostics, PressureRedistributionTerm, BuoyancyProductionTerm, AdvectionTerm
using Oceanostics.TKEBudgetTerms: AdvectionTerm
using Oceanostics: PotentialEnergy, PotentialEnergyEquationTerms.BuoyancyBoussinesqEOSModel
using Oceanostics.ProgressMessengers
using Oceanostics: perturbation_fields

include("test_budgets.jl")

#+++ Default grids and functions
arch = has_cuda_gpu() ? GPU() : CPU()

N = 6
regular_grid = RectilinearGrid(arch, size=(N, N, N), extent=(1, 1, 1))

S = .99 # Stretching factor. Positive number ∈ (0, 1]
f_asin(k) = -asin(S*(2k - N - 2) / N)/π + 1/2
F1 = f_asin(1); F2 = f_asin(N+1)
z_faces(k) = ((F1 + F2)/2 - f_asin(k)) / (F1 - F2)

stretched_grid = RectilinearGrid(arch, size=(N, N, N), x=(0, 1), y=(0, 1), z=z_faces)

grid_noise(x, y, z) = randn()

is_LES(::SmagorinskyLilly) = true
is_LES(::AnisotropicMinimumDissipation) = true
is_LES(::Any) = false
is_LES(a::Tuple) = any(map(is_LES, a))
#---

#+++ Test functions
function test_progress_messenger(model, messenger)
    simulation = Simulation(model; Δt=1e-2, stop_iteration=10)
    simulation.callbacks[:progress] = Callback(messenger, IterationInterval(1))
    run!(simulation)
    return nothing
end

function test_vel_only_diagnostics(model)
    u, v, w = model.velocities
    U = Field(Average(u, dims=(2, 3)))
    V = Field(Average(v, dims=(2, 3)))
    W = Field(Average(w, dims=(2, 3)))

    ke_op = KineticEnergy(model)
    @test ke_op isa AbstractOperation
    ke_c = Field(ke_op)
    @test all(interior(compute!(ke_c)) .≈ 0)

    op = TurbulentKineticEnergy(model, U=U, V=V, W=W)
    @test op isa AbstractOperation
    tke_c = Field(op)
    @test all(interior(compute!(tke_c)) .≈ 0)

    op = XShearProductionRate(model, u, v, w, U, V, W)
    @test op isa AbstractOperation
    XSP = Field(op)
    @test all(interior(compute!(XSP)) .≈ 0)

    op = XShearProductionRate(model; U=U, V=V, W=W)
    @test op isa AbstractOperation
    XSP = Field(op)
    @test all(interior(compute!(XSP)) .≈ 0)


    U = Field(Average(u, dims=(1, 3)))
    V = Field(Average(v, dims=(1, 3)))
    W = Field(Average(w, dims=(1, 3)))

    op = YShearProductionRate(model; U=U, V=V, W=W)
    @test op isa AbstractOperation
    YSP = Field(op)
    @test all(interior(compute!(YSP)) .≈ 0)

    op = YShearProductionRate(model, u, v, w, U, V, W)
    @test op isa AbstractOperation
    YSP = Field(op)
    @test all(interior(compute!(YSP)) .≈ 0)


    U = Field(Average(u, dims=(1, 2)))
    V = Field(Average(v, dims=(1, 2)))
    W = Field(Average(w, dims=(1, 2)))

    op = ZShearProductionRate(model, u, v, w, U, V, W)
    @test op isa AbstractOperation
    ZSP = Field(op)
    @test all(interior(compute!(ZSP)) .≈ 0)

    op = ZShearProductionRate(model; U=U, V=V, W=W)
    @test op isa AbstractOperation
    ZSP = Field(op)
    @test all(interior(compute!(ZSP)) .≈ 0)

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
    CUDA.@allowscalar @test op == Oceanostics.Q(model)
    @test op isa AbstractOperation
    q = Field(op)
    @test all(interior(compute!(q)) .≈ 0)

    return nothing
end

function test_buoyancy_diagnostics(model)
    u, v, w = model.velocities
    b = model.tracers.b

    Ri = RichardsonNumber(model)
    @test Ri isa AbstractOperation
    @test compute!(Field(Ri)) isa Field

    Ri = RichardsonNumber(model, u, v, w, b)
    @test Ri isa AbstractOperation
    @test compute!(Field(Ri)) isa Field

    PVe = ErtelPotentialVorticity(model)
    @test PVe isa AbstractOperation
    @test compute!(Field(PVe)) isa Field

    PVe = ErtelPotentialVorticity(model, u, v, w, b, model.coriolis)
    @test PVe isa AbstractOperation
    @test compute!(Field(PVe)) isa Field

    PVtw = ThermalWindPotentialVorticity(model)
    @test PVtw isa AbstractOperation
    @test compute!(Field(PVtw)) isa Field

    PVtw = ThermalWindPotentialVorticity(model, f=1e-4)
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


function test_pressure_term(model)
    u⃗∇p = PressureRedistributionTerm(model)
    @test u⃗∇p isa AbstractOperation
    @test compute!(Field(u⃗∇p)) isa Field

    u⃗∇pNHS = PressureRedistributionTerm(model, pressure=model.pressures.pNHS)
    @test u⃗∇pNHS isa AbstractOperation
    @test compute!(Field(u⃗∇pNHS)) isa Field

    # Test calculation with a hydrostatic pressure separation
    model2 = NonhydrostaticModel(grid=model.grid, hydrostatic_pressure_anomaly=CenterField(model.grid))
    u⃗∇p_from_model2 = PressureRedistributionTerm(model2)
    @test u⃗∇p_from_model2 isa AbstractOperation
    @test compute!(Field(u⃗∇p_from_model2)) isa Field

    return nothing
end

function test_momentum_advection_term(grid; model_type=NonhydrostaticModel)
    model = model_type(; grid)
    C₁ = 2; C₂ = 3
    set!(model, u=(x, y, z) -> C₁*y, v=C₂)

    ADV = AdvectionTerm(model)
    @compute ADV_field = Field(ADV)
    @test ADV isa AbstractOperation
    @test ADV_field isa Field

    # Test excluding the grid boundaries
    @test Array(interior(ADV_field, 1, 2:grid.Ny-1, 1)) ≈ collect(C₁^2*C₂*grid.yᵃᶜᵃ[2:grid.Ny-1])

    return nothing
end

function test_ke_dissipation_rate_terms(grid; model_type=NonhydrostaticModel, closure=ScalarDiffusivity(ν=1))
    model = model_type(; grid, closure, buoyancy=BuoyancyTracer(), tracers=:b)

    if !(model.closure isa Tuple) || all(isa.(model.closure, ScalarDiffusivity{ThreeDimensionalFormulation}))
        ε_iso = IsotropicKineticEnergyDissipationRate(model; U=0, V=0, W=0)
        ε_iso_field = compute!(Field(ε_iso))
        @test ε_iso isa AbstractOperation
        @test ε_iso_field isa Field
    end

    dudz = 2
    set!(model, u=(x, y, z) -> dudz*z)

    ε = KineticEnergyDissipationRate(model)
    ε_field = compute!(Field(ε))
    @test ε isa AbstractOperation
    @test ε_field isa Field

    εp = KineticEnergyDissipationRate(model; U=Field(Average(model.velocities.u, dims=(1,2))))
    εp_field = compute!(Field(εp))
    @test εp isa AbstractOperation
    @test εp_field isa Field

    idxs = (model.grid.Nx÷2, model.grid.Ny÷2, model.grid.Nz÷2)

    if closure isa Tuple
        @compute ν_field = Field(sum(viscosity(closure, model.diffusivity_fields)))
    else
        ν_field = viscosity(closure, model.diffusivity_fields)
    end

    rtol = zspacings(grid, Center()) isa Number ? 1e-12 : 0.06 # less accurate for stretched grid

    CUDA.@allowscalar begin
        true_ε = (ν_field isa Field ? getindex(ν_field, idxs...) : ν_field) * dudz^2
        @test isapprox(getindex(ε_field,  idxs...), true_ε, rtol=rtol, atol=eps())
        @test isapprox(getindex(εp_field, idxs...), 0.0,    rtol=rtol, atol=eps())
    end

    ε = KineticEnergyStressTerm(model)
    ε_field = compute!(Field(ε))
    @test ε isa AbstractOperation
    @test ε_field isa Field

    set!(model, u=grid_noise, v=grid_noise, w=grid_noise, b=grid_noise)
    @compute ε̄ₖ = Field(Average(KineticEnergyDissipationRate(model)))
    @compute ε̄ₖ₂= Field(Average(KineticEnergyStressTerm(model)))


    if model isa NonhydrostaticModel
        @test ≈(Array(interior(ε̄ₖ, 1, 1, 1)), Array(interior(ε̄ₖ₂, 1, 1, 1)), rtol=1e-12, atol=eps())

        ε = KineticEnergyTendency(model)
        @compute ε_field = Field(ε)
        @test ε isa AbstractOperation
        @test ε_field isa Field

        @compute ∂ₜKE = Field(Average(TracerVarianceTendency(model, :b)))
    end

    return nothing
end

function test_ke_forcing_term(grid; model_type=NonhydrostaticModel)
    Fᵘ_func(x, y, z, t, u) = -u
    Fᵛ_func(x, y, z, t, v) = -v
    Fʷ_func(x, y, z, t, w) = -w

    Fᵘ = Forcing(Fᵘ_func, field_dependencies = :u)
    Fᵛ = Forcing(Fᵛ_func, field_dependencies = :v)
    Fʷ = Forcing(Fʷ_func, field_dependencies = :w)

    model = model_type(; grid, forcing = (u=Fᵘ, v=Fᵛ, w=Fʷ))
    set!(model, u=grid_noise, v=grid_noise, w=grid_noise)

    ε = KineticEnergyForcingTerm(model)
    @compute ε_field = Field(ε)
    @test ε isa AbstractOperation
    @test ε_field isa Field

    @compute ε_truth = Field(@at (Center, Center, Center) (-model.velocities.u^2 -model.velocities.v^2 -model.velocities.w^2))

    @test isapprox(Array(interior(ε_field, 1, 1, 1)), Array(interior(ε_truth, 1, 1, 1)), rtol=1e-12, atol=eps())

    return nothing
end

function test_buoyancy_production_term(grid; model_type=NonhydrostaticModel)
    model = model_type(grid=grid, buoyancy=BuoyancyTracer(), tracers=:b)
    w₀ = 2; b₀ = 3
    set!(model, w=w₀, b=b₀, enforce_incompressibility=false)

    wb = BuoyancyProductionTerm(model)
    @compute wb_field = Field(wb)
    @test wb isa AbstractOperation
    @test wb_field isa Field
    @test Array(interior(wb_field, 1, 1, 2)) .== w₀ * b₀

    w′ = Field(model.velocities.w - Field(Average(model.velocities.w)))
    b′ = Field(model.tracers.b - Field(Average(model.tracers.b)))
    w′b′ = BuoyancyProductionTerm(model, velocities=(u=model.velocities.u, v=model.velocities.v, w=w′), tracers=(b=b′,))
    @compute w′b′_field = Field(w′b′)
    @test w′b′ isa AbstractOperation
    @test w′b′_field isa Field
    @test .≈(Array(interior(w′b′_field, 1, 1, 2)), 0, rtol=1e-12, atol=1e-13) # less accurate for stretched grid

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

    # Some of the models have LES closure, which means they don't have dissipation if u=v=w=0
    set!(model, u=grid_noise, v=grid_noise, w=grid_noise, b=grid_noise)
    @compute ε̄ₚ = Field(Average(TracerVarianceDissipationRate(model, :b)))
    @compute ε̄ₚ₂ = Field(Average(TracerVarianceDiffusiveTerm(model, :b)))
    @test ≈(Array(interior(ε̄ₚ, 1, 1, 1)), Array(interior(ε̄ₚ₂, 1, 1, 1)), rtol=1e-12, atol=eps())

    if model isa NonhydrostaticModel
        χ = TracerVarianceTendency(model, :b)
        χ_field = compute!(Field(χ))
        @test χ isa AbstractOperation
        @test χ_field isa Field

        @compute ∂ₜc² = Field(Average(TracerVarianceTendency(model, :b)))
        @test ≈(Array(interior(ε̄ₚ, 1, 1, 1)), -Array(interior(∂ₜc², 1, 1, 1)), rtol=1e-10, atol=eps())
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
        ρ₀ = model.buoyancy.model.equation_of_state.reference_density
        g = model.buoyancy.model.gravitational_acceleration

        CUDA.@allowscalar begin
            true_value = (g / ρ₀) .* ρ.data .* Z.data
            @test isequal(Eₚ_field.data, true_value)
        end
    end

    return nothing
end
#---

#+++ Known-value function tests
function test_uniform_strain_flow(grid; model_type=NonhydrostaticModel, closure=ScalarDiffusivity(ν=1), α=1)
    model = model_type(; grid, closure)
    u₀(x, y, z) = +α*x
    v₀(x, y, z) = -α*y
    set!(model, u=u₀, v=v₀, w=0, enforce_incompressibility=false)

    u, v, w = model.velocities

    @compute ε = Field(KineticEnergyDissipationRate(model))
    @compute S = Field(StrainRateTensorModulus(model))
    @compute Ω = Field(VorticityTensorModulus(model))
    @compute q = Field(QVelocityGradientTensorInvariant(model))

    idxs = (model.grid.Nx÷2, model.grid.Ny÷2, model.grid.Nz÷2) # Get a value far from boundaries

    if model.closure isa Tuple
        @compute ν_field = Field(sum(viscosity(model.closure, model.diffusivity_fields)))
    else
        ν_field = viscosity(model.closure, model.diffusivity_fields)
    end

    CUDA.@allowscalar begin
        ν = ν_field isa Number ? ν_field : getindex(ν_field, idxs...)

        @test getindex(S, idxs...) ≈ √2*α
        @test getindex(Ω, idxs...) ≈ 0
        @test getindex(q, idxs...) ≈ (getindex(Ω, idxs...)^2 - getindex(S, idxs...)^2)/2 ≈ -α^2
        @test getindex(ε, idxs...) ≈ 2 * ν * getindex(S, idxs...)^2
    end

    return nothing
end

function test_solid_body_rotation_flow(grid; model_type=NonhydrostaticModel, closure=ScalarDiffusivity(ν=1), ζ=1)
    model = model_type(; grid, closure)
    u₀(x, y, z) = +ζ*y / 2
    v₀(x, y, z) = -ζ*x / 2
    set!(model, u=u₀, v=v₀, w=0, enforce_incompressibility=false)

    u, v, w = model.velocities

    @compute ε = Field(KineticEnergyDissipationRate(model))
    @compute S = Field(StrainRateTensorModulus(model))
    @compute Ω = Field(VorticityTensorModulus(model))
    @compute q = Field(QVelocityGradientTensorInvariant(model))

    idxs = (model.grid.Nx÷2, model.grid.Ny÷2, model.grid.Nz÷2) # Get a value far from boundaries

    if model.closure isa Tuple
        @compute ν_field = Field(sum(viscosity(model.closure, model.diffusivity_fields)))
    else
        ν_field = viscosity(model.closure, model.diffusivity_fields)
    end

    CUDA.@allowscalar begin
        ν = ν_field isa Number ? ν_field : getindex(ν_field, idxs...)

        @test getindex(S, idxs...) ≈ 0
        @test getindex(Ω, idxs...) ≈ ζ/√2
        @test getindex(q, idxs...) ≈ (getindex(Ω, idxs...)^2 - getindex(S, idxs...)^2)/2 ≈ ζ^2/4
        @test getindex(ε, idxs...) ≈ 0
    end
end

function test_uniform_shear_flow(grid; model_type=NonhydrostaticModel, closure=ScalarDiffusivity(ν=1), σ=1)
    model = model_type(; grid, closure)
    u₀(x, y, z) = +σ * y
    set!(model, u=u₀, v=0, w=0, enforce_incompressibility=false)

    u, v, w = model.velocities

    @compute ε = Field(KineticEnergyDissipationRate(model))
    @compute S = Field(StrainRateTensorModulus(model))
    @compute Ω = Field(VorticityTensorModulus(model))
    @compute q = Field(QVelocityGradientTensorInvariant(model))

    idxs = (model.grid.Nx÷2, model.grid.Ny÷2, model.grid.Nz÷2) # Get a value far from boundaries

    if model.closure isa Tuple
        @compute ν_field = Field(sum(viscosity(model.closure, model.diffusivity_fields)))
    else
        ν_field = viscosity(model.closure, model.diffusivity_fields)
    end

    CUDA.@allowscalar begin
        ν = ν_field isa Number ? ν_field : getindex(ν_field, idxs...)

        @test getindex(S, idxs...) ≈ σ/√2
        @test getindex(Ω, idxs...) ≈ σ/√2
        @test ≈(getindex(q, idxs...), (getindex(Ω, idxs...)^2 - getindex(S, idxs...)^2)/2, atol=eps())
        @test ≈(getindex(q, idxs...), 0, atol=eps())
        @test getindex(ε, idxs...) ≈ 2 * ν * getindex(S, idxs...)^2
    end
end

function test_auxiliary_functions(model)
    set!(model, u=1, v=2)
    fields_without_means = perturbation_fields(model; u=1, v=2)
    compute!(fields_without_means)
    @test all(Array(interior(fields_without_means.u)) .== 0)
    @test all(Array(interior(fields_without_means.v)) .== 0)
    return
end
#---

model_kwargs = (buoyancy = Buoyancy(model=BuoyancyTracer()),
                coriolis = FPlane(1e-4),
                tracers = :b)

closures = (ScalarDiffusivity(ν=1e-6, κ=1e-7),
            SmagorinskyLilly(),
            (ScalarDiffusivity(ν=1e-6, κ=1e-7), AnisotropicMinimumDissipation()),)

buoyancy_models = (nothing, BuoyancyTracer(), SeawaterBuoyancy(),
                   SeawaterBuoyancy(equation_of_state=TEOS10EquationOfState()),
                   SeawaterBuoyancy(equation_of_state=RoquetEquationOfState(:Linear)))

grids = (regular_grid, stretched_grid)

model_types = (NonhydrostaticModel, HydrostaticFreeSurfaceModel)

@testset "Oceanostics" begin
    for grid in grids
        for model_type in model_types
            for closure in closures
                @info "Testing $model_type on grid and with closure" grid closure
                model = model_type(; grid, closure, model_kwargs...)

                @info "Testing velocity-only diagnostics"
                test_vel_only_diagnostics(model)

                @info "Testing buoyancy diagnostics"
                test_buoyancy_diagnostics(model)

                if model isa NonhydrostaticModel
                    @info "Testing pressure terms"
                    test_pressure_term(model)

                    @info "Testing buoyancy production term"
                    test_buoyancy_production_term(grid; model_type)
                end

                @info "Testing auxiliary functions"
                test_auxiliary_functions(model)

                @info "Testing energy dissipation rate terms"
                test_ke_dissipation_rate_terms(grid; model_type, closure)


                if model_type == NonhydrostaticModel
                    @info "Testing advection terms"
                    test_momentum_advection_term(grid; model_type)

                    @info "Testing forcing terms"
                    test_ke_forcing_term(grid; model_type)

                    @info "Testing uniform strain flow"
                    test_uniform_strain_flow(grid; model_type, closure, α=3)

                    @info "Testing solid body rotation flow"
                    test_solid_body_rotation_flow(grid; model_type, closure, ζ=3)

                    @info "Testing uniform shear flow"
                    test_uniform_shear_flow(grid; model_type, closure, σ=3)
                end

                @info "Testing tracer variance terms"
                model = model_type(; grid, closure, model_kwargs...)
                test_tracer_diagnostics(model)

            end

            @info "Testing `PotentialEnergy`"
            for buoyancy in buoyancy_models

                tracers = buoyancy isa BuoyancyTracer ? :b : (:S, :T)
                model = model_type(; grid, buoyancy, tracers)
                buoyancy isa BuoyancyTracer ? set!(model, b = 9.87) : set!(model, S = 34.7, T = 0.5)
                if isnothing(buoyancy)
                    test_potential_energy_equation_terms_errors(model)
                else
                    test_potential_energy_equation_terms(model)
                    test_potential_energy_equation_terms(model, geopotential_height = 0)
                end

            end

        end

        @info "Testing input validation for dissipation rates"
        invalid_closures = [HorizontalScalarDiffusivity(ν=1e-6, κ=1e-7),
                            VerticalScalarDiffusivity(ν=1e-6, κ=1e-7),
                            (ScalarDiffusivity(ν=1e-6, κ=1e-7), HorizontalScalarDiffusivity(ν=1e-6, κ=1e-7))]

        for closure in invalid_closures
            model = NonhydrostaticModel(grid = regular_grid; model_kwargs..., closure)
            @test_throws ErrorException IsotropicKineticEnergyDissipationRate(model; U=0, V=0, W=0)
        end

    end


    for closure in closures
        LES = is_LES(closure)
        model = NonhydrostaticModel(grid = regular_grid;
                                    buoyancy = Buoyancy(model=BuoyancyTracer()),
                                    coriolis = FPlane(1e-4),
                                    tracers = :b,
                                    closure = closure)

        @info "Testing BasicMessenger with closure" closure
        model.clock.iteration = 0
        test_progress_messenger(model, BasicMessenger())

        @info "Testing SingleLineMessenger with closure" closure
        model.clock.iteration = 0
        test_progress_messenger(model, SingleLineMessenger())

        # Test that SingleLineMessenger is indeed a single line
        simulation = Simulation(model; Δt=1e-2, stop_iteration=1)
        msg = SingleLineMessenger(print=false)(simulation)
        @test countlines(IOBuffer(msg)) == 1

        @info "Testing TimedMessenger with closure" closure
        model.clock.iteration = 0
        test_progress_messenger(model, TimedMessenger())

        @info "Testing custom progress messenger with closure" closure
        model.clock.iteration = 0
        step_duration = StepDuration()
        progress(simulation) = @info (PercentageProgress(with_prefix=false, with_units=false)
                                      + SimulationTime() + TimeStep() + MaxVelocities()
                                      + AdvectiveCFLNumber() + step_duration)(simulation)
        test_progress_messenger(model, progress)
    end

    rtol = 0.02; N = 80
    @info "Testing tracer variance budget on and a regular grid with N=$N and tolerance $rtol"
    test_tracer_variance_budget(N=N, rtol=rtol, regular_grid=true)

end
