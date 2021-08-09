using Test
using Oceananigans
using Oceanostics

function create_model(; kwargs...)

    model = NonhydrostaticModel(grid = RegularRectilinearGrid(size=(2, 2, 2), extent=(1, 1, 1));
                                kwargs...
                                )
    return model
end


function test_velocity_diagnostics(; model_kwargs...)
    model = create_model(; model_kwargs...)
    u, v, w = model.velocities

    ke = KineticEnergy(model)
    compute!(ke)
    @test all(interior(ke) .== 0)

    tke = TurbulentKineticEnergy(model, U=0, V=0)
    compute!(tke)
    @test all(interior(tke) .== 0)
    return nothing
end

@testset "Oceanostics" begin
    topo = (Periodic, Periodic, Bounded)
    grid = RegularRectilinearGrid(topology=topo, size=(4, 5, 6), extent=(1, 1, 1))
    model = NonhydrostaticModel(grid=grid)

    u, v, w = model.velocities

    U = AveragedField(u, dims=(1, 2))
    V = AveragedField(u, dims=(1, 2))

    ke = KineticEnergy(model)
    @test ke isa KernelComputedField

    tke = TurbulentKineticEnergy(model, U=U, V=V)
    @test tke isa KernelComputedField

    test_velocity_diagnostics()
end

