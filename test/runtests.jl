using Test
using Oceananigans
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
    @test ke isa KernelComputedField

    tke = TurbulentKineticEnergy(model, U=U, V=V, W=W)
    @test tke isa KernelComputedField

    SPx = XShearProduction(model, u, v, w, U, V, W)
    @test SPx isa KernelComputedField

    SPy = YShearProduction(model, u, v, w, U, V, W)
    @test SPy isa KernelComputedField

    SPz = ZShearProduction(model, u, v, w, U, V, W)
    @test SPz isa KernelComputedField

    return nothing
end




function test_buoyancy_diagnostics(; model_kwargs...)
    model = create_model(; model_kwargs...)
    test_buoyancy_diagnostics(model)
end

function test_buoyancy_diagnostics(model)
    u, v, w = model.velocities
    b = model.tracers.b
    κ = model.closure.κ
    N²₀ = 1e-6

    χiso = IsotropicBuoyancyMixingRate(model, b, κ, N²₀)
    @test χiso isa KernelComputedField

    χani = AnisotropicBuoyancyMixingRate(model, b, κ, κ, κ, N²₀)
    @test χani isa KernelComputedField

end


@testset "Oceanostics" begin
    model = create_model(; buoyancy=Buoyancy(model=BuoyancyTracer()), 
                         tracers=:b,
                         closure=IsotropicDiffusivity(ν=1e-6, κ=1e-7),
                        )

    test_vel_only_diagnostics(model)

    test_buoyancy_diagnostics(model)

end


