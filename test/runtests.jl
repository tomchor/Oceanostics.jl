using Test
using Oceananigans
using Oceanostics
using Oceanostics.FlowDiagnostics

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
    
    model = NonhydrostaticModel(grid=grid, coriolis=FPlane(1e-4), buoyancy=BuoyancyTracer(), tracers=:b)

    Ro = RossbyNumber(model; dUdy_bg=1, dVdx_bg=1)
    @test Ro isa Oceananigans.AbstractOperations.AbstractOperation

    Ri = RichardsonNumber(model; N²_bg=1, dUdz_bg=1, dVdz_bg=1)
    @test Ri isa Oceananigans.AbstractOperations.AbstractOperation

    PVe = ErtelPotentialVorticityᶠᶠᶠ(model)
    @test PVe isa Oceananigans.AbstractOperations.AbstractOperation

    PVtw = ThermalWindPotentialVorticityᶠᶠᶠ(model)
    @test PVtw isa Oceananigans.AbstractOperations.AbstractOperation
end

