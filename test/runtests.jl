using Test
using Oceananigans
using Oceananigans.AbstractOperations: AbstractOperation
using Oceananigans.Fields: compute_at!
using Oceanostics
using Oceanostics.TKEBudgetTerms
using Oceanostics.TKEBudgetTerms: turbulent_kinetic_energy_ccc
using Oceanostics.FlowDiagnostics
using Oceanostics: SimpleProgressMessenger, SingleLineProgressMessenger, make_message

# Default grid for all tests
grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1))

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

    op = RossbyNumber(model; dUdy_bg=1, dVdx_bg=1)
    @test op isa AbstractOperation
    Ro = Field(op)
    @test all(interior(compute!(Ro)) .≈ 0)

    return nothing
end

function test_buoyancy_diagnostics(model)
    u, v, w = model.velocities
    b = model.tracers.b
    κ = model.closure.κ.b
    N²₀ = 1e-6

    Ri = RichardsonNumber(model)
    @test Ri isa AbstractOperation
    @test compute!(Field(Ri)) isa Field

    PVe = ErtelPotentialVorticity(model)
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

    return nothing
end

function test_pressure_terms(model)
    ∂x_up = XPressureRedistribution(model)
    @test ∂x_up isa AbstractOperation
    @test compute!(Field(∂x_up)) isa Field

    ∂y_vp = XPressureRedistribution(model)
    @test ∂y_vp isa AbstractOperation
    @test compute!(Field(∂y_vp)) isa Field

    ∂z_wp = XPressureRedistribution(model)
    @test ∂z_wp isa AbstractOperation
    @test compute!(Field(∂z_wp)) isa Field

    return nothing
end

function test_ke_dissipation_rate_terms(model)
    u, v, w = model.velocities
    b = model.tracers.b

    ε_iso = IsotropicViscousDissipationRate(model; U=0, V=0, W=0)
    ε_iso_field = compute!(Field(ε_iso))
    @test ε_iso isa AbstractOperation
    @test ε_iso_field isa Field

    ε_iso = IsotropicPseudoViscousDissipationRate(model; U=0, V=0, W=0)
    ε_iso_field = compute!(Field(ε_iso))
    @test ε_iso isa AbstractOperation
    @test ε_iso_field isa Field

    return nothing
end

function test_tracer_diagnostics(model)
    u, v, w = model.velocities
    b = model.tracers.b

    χ_iso = IsotropicTracerVarianceDissipationRate(model, :b)
    χ_iso_field = compute!(Field(χ_iso))
    @test χ_iso isa AbstractOperation
    @test χ_iso_field isa Field

    return nothing
end


scalar_diff = ScalarDiffusivity(ν=1e-6, κ=1e-7)

@testset "Oceanostics" begin
    model_kwargs = (buoyancy = Buoyancy(model=BuoyancyTracer()), 
                    coriolis = FPlane(1e-4),
                    tracers = :b)

    models = (NonhydrostaticModel(; grid, model_kwargs...,
                                  closure = scalar_diff),
              HydrostaticFreeSurfaceModel(; grid, model_kwargs...,
                                          closure = scalar_diff))

    for model in models
        model_type = split(summary(model), "{")[1]
        @info "Testing velocity-only diagnostics in $model_type"
        test_vel_only_diagnostics(model)

        @info "Testing buoyancy diagnostics in $model_type"
        test_buoyancy_diagnostics(model)

        if model isa NonhydrostaticModel
            @info "Testing pressure terms in $model_type"
            test_pressure_terms(model)
        end
    end

    @info "Testing input validation for dissipation rates"
    invalid_closures = [HorizontalScalarDiffusivity(ν=1e-6, κ=1e-7),
                        VerticalScalarDiffusivity(ν=1e-6, κ=1e-7),
                        (ScalarDiffusivity(ν=1e-6, κ=1e-7), HorizontalScalarDiffusivity(ν=1e-6, κ=1e-7))]
        
    for closure in invalid_closures
        model = NonhydrostaticModel(; grid, model_kwargs..., closure)
        @test_throws ErrorException IsotropicViscousDissipationRate(model; U=0, V=0, W=0)
        @test_throws ErrorException IsotropicPseudoViscousDissipationRate(model; U=0, V=0, W=0)
    end

    closures = [
        ScalarDiffusivity(ν=1e-6, κ=1e-7),
        SmagorinskyLilly(),
        AnisotropicMinimumDissipation(),
        (ScalarDiffusivity(ν=1e-6, κ=1e-7), SmagorinskyLilly())
    ]
        
    LESs = [false, true, true, true]
    messengers = (SingleLineProgressMessenger, TimedProgressMessenger)
    
    for (LES, closure) in zip(LESs, closures)
        model = NonhydrostaticModel(; grid,
                                    buoyancy = Buoyancy(model=BuoyancyTracer()), 
                                    coriolis = FPlane(1e-4),
                                    tracers = :b,
                                    closure = closure)

        @info "Testing energy dissipation rate terms with closure" closure
        test_ke_dissipation_rate_terms(model)

        @info "Testing tracer variance terms wth closure" closure
        test_tracer_diagnostics(model)

        @info "Testing SimpleProgressMessenger with closure" closure
        model.clock.iteration = 0
        time_now = time_ns()*1e-9
        test_progress_messenger(model, SimpleProgressMessenger(initial_wall_time_seconds=1e-9*time_ns()))

        @info "Testing SingleLineProgressMessenger with closure" closure
        model.clock.iteration = 0
        time_now = time_ns()*1e-9
        test_progress_messenger(model, SingleLineProgressMessenger(initial_wall_time_seconds=1e-9*time_ns()))

        simulation = Simulation(model; Δt=1e-2, stop_iteration=1)
        msg = make_message(simulation, true)
        @test count(s -> s === '\n', msg) == 1

        @info "Testing TimedProgressMessenger with closure" closure
        model.clock.iteration = 0
        time_now = time_ns()*1e-9
        test_progress_messenger(model, TimedProgressMessenger(; LES=LES))
    end
end
