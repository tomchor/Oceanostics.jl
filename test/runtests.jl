using Test
using Oceananigans
using Oceananigans.AbstractOperations: AbstractOperation
using Oceananigans.Fields: compute_at!
using Oceanostics
using Oceanostics.TKEBudgetTerms
using Oceanostics.TKEBudgetTerms: turbulent_kinetic_energy_ccc
using Oceanostics.FlowDiagnostics
using Oceanostics: SimpleProgressMessenger, SingleLineProgressMessenger


function create_model(; kwargs...)
    model = NonhydrostaticModel(grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1)); kwargs...)
    return model
end


function viscosity(model)
    try
        return model.diffusivity_fields.νₑ
    catch e
        return model.closure.ν
    end
end




include("test_progress_messengers.jl")

function computed_operation(op)
    op_cf = ComputedField(op)
    compute!(op_cf)
    return op_cf
end


function test_vel_only_diagnostics(; model_kwargs...)
    model = create_model(; model_kwargs...)
    test_vel_only_diagnostics(model)
end

function test_vel_only_diagnostics(model)
    u, v, w = model.velocities
    U = AveragedField(u, dims=(1, 2))
    V = AveragedField(v, dims=(1, 2))
    W = AveragedField(w, dims=(1, 2))


    @test begin
        tke_op = KineticEnergy(model)
        ke_c = ComputedField(tke_op)
        compute!(ke_c)
        tke_op isa AbstractOperation
    end


    @test begin
        op = TurbulentKineticEnergy(model, U=U, V=V, W=W)
        tke_c = ComputedField(op)
        compute!(tke_c)
        op isa AbstractOperation
    end

    @test begin
        op = XShearProduction(model, u, v, w, U, V, W)
        XSP = ComputedField(op)
        compute!(XSP)
        op isa AbstractOperation
    end

    @test begin
        op = YShearProduction(model, u, v, w, U, V, W)
        YSP = ComputedField(op)
        compute!(YSP)
        op isa AbstractOperation
    end

    @test begin
        op = ZShearProduction(model, u, v, w, U, V, W)
        ZSP = ComputedField(op)
        compute!(ZSP)
        op isa AbstractOperation
    end


    @test begin
        op = RossbyNumber(model;)
        Ro = ComputedField(op)
        compute!(Ro)
        op isa AbstractOperation
    end

    @test begin
        op = RossbyNumber(model; dUdy_bg=1, dVdx_bg=1, f=1e-4)
        Ro = ComputedField(op)
        compute!(Ro)
        op isa AbstractOperation
    end

    return nothing
end




function test_buoyancy_diagnostics(; model_kwargs...)
    model = create_model(; model_kwargs...)
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
    model = create_model(; model_kwargs...)
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
    model = create_model(; model_kwargs...)
    test_ke_dissipation_rate_terms(model)
end

function test_ke_dissipation_rate_terms(model)
    u, v, w = model.velocities
    b = model.tracers.b
    #ν = viscosity(model)

    if model.closure isa AnisotropicDiffusivity
        ε_ani = AnisotropicPseudoViscousDissipationRate(model; U=0, V=0, W=0)
        @test ε_ani isa AbstractOperation

    else
        ε_iso = IsotropicViscousDissipationRate(model; U=0, V=0, W=0)
        @test ε_iso isa AbstractOperation

        ε_iso = IsotropicPseudoViscousDissipationRate(model; U=0, V=0, W=0)
        @test ε_iso isa AbstractOperation
    end

    return nothing
end




function test_tracer_diagnostics(; model_kwargs...)
    model = create_model(; model_kwargs...)
    test_tracer_diagnostics(model)
end

function test_tracer_diagnostics(model)
    u, v, w = model.velocities
    b = model.tracers.b

    if model.closure isa AnisotropicDiffusivity
        κx = model.closure.κx.b
        κy = model.closure.κy.b
        κz = model.closure.κz.b
        χani = AnisotropicTracerVarianceDissipationRate(model, b, κx, κy, κz,)
        @test χani isa AbstractOperation

    else
        κ = model.closure.κ.b
        χiso = IsotropicTracerVarianceDissipationRate(model, b, κ)
        @test χiso isa AbstractOperation
    end

    return nothing
end





@testset "Oceanostics" begin
    model = create_model(; buoyancy=Buoyancy(model=BuoyancyTracer()), 
                         coriolis=FPlane(1e-4),
                         tracers=:b,
                         closure=IsotropicDiffusivity(ν=1e-6, κ=1e-7),
                        )

    @info "Testing velocity-only diagnostics"
    test_vel_only_diagnostics(model)

    @info "Testing buoyancy diagnostics"
    test_buoyancy_diagnostics(model)

    @info "Testing pressure terms"
    test_pressure_terms(model)


    closures = (IsotropicDiffusivity(ν=1e-6, κ=1e-7),
                AnisotropicDiffusivity(νz=1e-5, νh=3e-4, κz=1e-6, κh=1e-5),
                SmagorinskyLilly(ν=1e-6, κ=1e-7))
    LESs = (false, false, true)
    messengers = (SingleLineProgressMessenger, TimedProgressMessenger)

    for (LES, closure) in zip(LESs, closures)
        model = create_model(; buoyancy=Buoyancy(model=BuoyancyTracer()), 
                             coriolis=FPlane(1e-4),
                             tracers=:b,
                             closure=closure,
                            )

        @info "Testing energy dissipation rate terms with closure" closure
        test_ke_dissipation_rate_terms(model)

        @info "Testing tracer variance terms wth closure" closure
        test_tracer_diagnostics(model)

        @info "Testing SimpleProgressMessenger with closure" closure
        model.clock.iteration = 0
        time_now = time_ns()*1e-9
        test_progress_messenger(model, SimpleProgressMessenger(initial_wall_time_seconds=1e-9*time_ns(), LES=LES))

        @info "Testing SingleLineProgressMessenger with closure" closure
        model.clock.iteration = 0
        time_now = time_ns()*1e-9
        test_progress_messenger(model, SingleLineProgressMessenger(initial_wall_time_seconds=1e-9*time_ns(), LES=LES))

        @info "Testing TimedProgressMessenger with closure" closure
        model.clock.iteration = 0
        time_now = time_ns()*1e-9
        test_progress_messenger(model, TimedProgressMessenger(; LES=LES))
    end

end


