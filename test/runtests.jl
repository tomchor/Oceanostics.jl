using Test
using Oceananigans
using Oceanostics

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
end
