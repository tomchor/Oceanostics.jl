using Test
using Oceananigans
using Oceananigans.AbstractOperations: AbstractOperation
using Oceanostics
using Oceanostics.TKEBudgetTerms
using Oceanostics.FlowDiagnostics


function create_model(; kwargs...)
    model = NonhydrostaticModel(grid = RegularRectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));
                                kwargs...
                                )
    return model
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

    ke = KineticEnergy(model)
    @test ke isa AbstractOperation

    tke = TurbulentKineticEnergy(model, U=U, V=V, W=W)
    @test tke isa AbstractOperation

    SPx = XShearProduction(model, u, v, w, U, V, W)
    @test SPx isa AbstractOperation

    SPy = YShearProduction(model, u, v, w, U, V, W)
    @test SPy isa AbstractOperation

    SPz = ZShearProduction(model, u, v, w, U, V, W)
    @test SPz isa AbstractOperation


    Ro = RossbyNumber(model;)
    @test Ro isa AbstractOperation

    Ro = RossbyNumber(model; dUdy_bg=1, dVdx_bg=1, f=1e-4)
    @test Ro isa AbstractOperation

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

    χiso = IsotropicBuoyancyMixingRate(model, b, κ, N²₀)
    @test χiso isa AbstractOperation

    χani = AnisotropicBuoyancyMixingRate(model, b, κ, κ, κ, N²₀)
    @test χani isa AbstractOperation


    Ri = RichardsonNumber(model)
    @test Ri isa AbstractOperation

    Ri = RichardsonNumber(model; N²_bg=1, dUdz_bg=1, dVdz_bg=1)
    @test Ri isa AbstractOperation


    PVe = ErtelPotentialVorticityᶠᶠᶠ(model)
    @test PVe isa AbstractOperation

    PVe = ErtelPotentialVorticityᶠᶠᶠ(model, f=1e-4)
    @test PVe isa AbstractOperation

    PVtw = ThermalWindPotentialVorticityᶠᶠᶠ(model)
    @test PVtw isa AbstractOperation

    PVtw = ThermalWindPotentialVorticityᶠᶠᶠ(model, f=1e-4)
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
    ν = model.closure.ν

    ε_iso = IsotropicViscousDissipationRate(model, u, v, w, ν)
    @test ε_iso isa AbstractOperation

    ε_iso = IsotropicPseudoViscousDissipationRate(model, u, v, w, ν)
    @test ε_iso isa AbstractOperation

    ε_ani = AnisotropicPseudoViscousDissipationRate(model, u, v, w, ν, ν, ν)
    @test ε_ani isa AbstractOperation

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

    @info "Testing energy dissipation rate terms"
    test_ke_dissipation_rate_terms(model)

end


