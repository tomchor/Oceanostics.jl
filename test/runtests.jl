using Test
using Oceananigans
using Oceanostics


function create_model(; kwargs...)
    model = NonhydrostaticModel(grid = RegularRectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));
                                kwargs...
                                )
    return model
end

function test_velocity_diagnostics(; model_kwargs...)
    model = create_model(; model_kwargs...)
    u, v, w = model.velocities
    U = AveragedField(u, dims=(1, 2))
    V = AveragedField(v, dims=(1, 2))
    W = AveragedField(w, dims=(1, 2))

    ke = KineticEnergy(model)
    @test ke isa KernelComputedField

    tke = TurbulentKineticEnergy(model, U=U, V=V, W=W)
    @test tke isa KernelComputedField

    return nothing
end


@testset "Oceanostics" begin
    model = create_model()

    test_velocity_diagnostics()
end


