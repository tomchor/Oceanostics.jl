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

function computed_operation(op)
    op_cf = Field(op)
    compute!(op_cf)
    return op_cf
end

function test_vel_only_diagnostics(; model_kwargs...)
    model = NonhydrostaticModel(; grid, model_kwargs...)
    test_vel_only_diagnostics(model)
end

function test_vel_only_diagnostics(model)
    u, v, w = model.velocities
    U = Field(Average(u, dims=(1, 2)))
    V = Field(Average(v, dims=(1, 2)))
    W = Field(Average(w, dims=(1, 2)))

    @test begin
        tke_op = KineticEnergy(model)
        ke_c = Field(tke_op)
        compute!(ke_c)
        tke_op isa AbstractOperation
    end

    @test begin
        op = TurbulentKineticEnergy(model, U=U, V=V, W=W)
        tke_c = Field(op)
        compute!(tke_c)
        op isa AbstractOperation
    end

    @test begin
        op = XShearProduction(model, u, v, w, U, V, W)
        XSP = Field(op)
        compute!(XSP)
        op isa AbstractOperation
    end

    @test begin
        op = YShearProduction(model, u, v, w, U, V, W)
        YSP = Field(op)
        compute!(YSP)
        op isa AbstractOperation
    end

    @test begin
        op = ZShearProduction(model, u, v, w, U, V, W)
        ZSP = Field(op)
        compute!(ZSP)
        op isa AbstractOperation
    end

    @test begin
        op = RossbyNumber(model;)
        Ro = Field(op)
        compute!(Ro)
        op isa AbstractOperation
    end

    @test begin
        op = RossbyNumber(model; dUdy_bg=1, dVdx_bg=1, f=1e-4)
        Ro = Field(op)
        compute!(Ro)
        op isa AbstractOperation
    end

    return nothing
end

function test_buoyancy_diagnostics(; model_kwargs...)
    model = NonhydrostaticModel(; grid, model_kwargs...)
    test_buoyancy_diagnostics(model)
end

function test_buoyancy_diagnostics(model)
    u, v, w = model.velocities
    b = model.tracers.b
    κ = model.closure.κ.b
    N²₀ = 1e-6

    Ri = RichardsonNumber(model)
    @test Ri isa AbstractOperation

    Ri = RichardsonNumber(model; N²_bg=1, dUdz_bg=1, dVdz_bg=1)
    @test Ri isa AbstractOperation

    PVe = ErtelPotentialVorticity(model)
    @test PVe isa AbstractOperation

    PVtw = ThermalWindPotentialVorticity(model)
    @test PVtw isa AbstractOperation

    PVtw = ThermalWindPotentialVorticity(model, f=1e-4)
    @test PVtw isa AbstractOperation

    return nothing
end

function test_buoyancy_diagnostics(; model_kwargs...)
    model = NonhydrostaticModel(; grid, model_kwargs...)
    test_buoyancy_diagnostics(model)
end

function test_pressure_terms(model)
    ∂x_up = XPressureRedistribution(model)
    @test ∂x_up isa AbstractOperation

    ∂y_vp = XPressureRedistribution(model)
    @test ∂y_vp isa AbstractOperation

    ∂z_wp = XPressureRedistribution(model)
    @test ∂z_wp isa AbstractOperation

    return nothing
end

function test_ke_dissipation_rate_terms(; model_kwargs...)
    model = NonhydrostaticModel(; grid, model_kwargs...)
    test_ke_dissipation_rate_terms(model)
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

function test_tracer_diagnostics(; model_kwargs...)
    model = NonhydrostaticModel(; grid, model_kwargs...)
    test_tracer_diagnostics(model)
end

function test_tracer_diagnostics(model)
    u, v, w = model.velocities
    b = model.tracers.b

    χiso = IsotropicTracerVarianceDissipationRate(model, :b)
    @test χiso isa AbstractOperation

    return nothing
end

@testset "Oceanostics" begin
    model_kwargs = (buoyancy = Buoyancy(model=BuoyancyTracer()), 
                    coriolis = FPlane(1e-4),
                    tracers = :b)

    model = NonhydrostaticModel(; grid, model_kwargs...,
                                closure = ScalarDiffusivity(ν=1e-6, κ=1e-7))

    @info "Testing velocity-only diagnostics"
    test_vel_only_diagnostics(model)

    @info "Testing buoyancy diagnostics"
    test_buoyancy_diagnostics(model)

    @info "Testing pressure terms"
    test_pressure_terms(model)

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
        (ScalarDiffusivity(ν=1e-6, κ=1e-7), SmagorinskyLilly())
    ]
        
    LESs = [false, true, true]
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
