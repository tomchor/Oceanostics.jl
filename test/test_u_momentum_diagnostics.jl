using Test
using CUDA: has_cuda_gpu

using Oceananigans
using Oceananigans.TurbulenceClosures.Smagorinskys: LagrangianAveraging

using Oceanostics

#+++ Default grids
arch = has_cuda_gpu() ? GPU() : CPU()

N = 6
underlying_regular_grid = RectilinearGrid(arch, size=(N, N, N), extent=(1, 1, 1))

S = .99 # Stretching factor. Positive number ∈ (0, 1]
f_asin(k) = -asin(S*(2k - N - 2) / N)/π + 1/2
F1 = f_asin(1); F2 = f_asin(N+1)
z_faces(k) = ((F1 + F2)/2 - f_asin(k)) / (F1 - F2)

underlying_stretched_grid = RectilinearGrid(arch, size=(N, N, N), x=(0, 1), y=(0, 1), z=z_faces)

bottom(x, y) = -1/2
regular_grid   = ImmersedBoundaryGrid(underlying_regular_grid, GridFittedBottom(bottom))
stretched_grid = ImmersedBoundaryGrid(underlying_stretched_grid, GridFittedBottom(bottom))
#---

#+++ Model arguments
tracers = :b

forcing_function(x, y, z, t) = sin(t)
forcing = (; u = Forcing(forcing_function), v = Forcing(forcing_function), w = Forcing(forcing_function))

bc_function(x, y, z, t) = sin(t)
immersed_bc = FluxBoundaryCondition(bc_function)
u_boundary_conditions = FieldBoundaryConditions(immersed=immersed_bc)
boundary_conditions = (; u = u_boundary_conditions)

model_kwargs = (; tracers, forcing, boundary_conditions, buoyancy=BuoyancyTracer(), coriolis=FPlane(1e-4))
#---

#+++ Test options
grids = Dict("regular grid" => regular_grid,
             "stretched grid" => stretched_grid)

model_types = (NonhydrostaticModel,
               HydrostaticFreeSurfaceModel)
#---

#+++ Test functions
function test_u_momentum_terms(model)
    # Test Advection
    ADV = UMomentumEquation.Advection(model, model.velocities..., model.advection)
    ADV_field = Field(ADV)
    @test ADV isa UMomentumEquation.Advection
    @test ADV isa UAdvection
    @test ADV_field isa Field

    ADV = UMomentumEquation.Advection(model)
    @test ADV isa UMomentumEquation.Advection
    ADV_field = Field(ADV)
    @test ADV_field isa Field

    # Test BuoyancyAcceleration
    BUOY = UMomentumEquation.BuoyancyAcceleration(model, model.buoyancy, model.tracers)
    BUOY_field = Field(BUOY)
    @test BUOY isa UMomentumEquation.BuoyancyAcceleration
    @test BUOY isa UBuoyancyAcceleration
    @test BUOY_field isa Field

    BUOY = UMomentumEquation.BuoyancyAcceleration(model)
    BUOY_field = Field(BUOY)
    @test BUOY isa UMomentumEquation.BuoyancyAcceleration
    @test BUOY isa UBuoyancyAcceleration
    @test BUOY_field isa Field

    # Test CoriolisAcceleration
    COR = UMomentumEquation.CoriolisAcceleration(model, model.coriolis, model.velocities)
    COR_field = Field(COR)
    @test COR isa UMomentumEquation.CoriolisAcceleration
    @test COR isa UCoriolisAcceleration
    @test COR_field isa Field

    COR = UMomentumEquation.CoriolisAcceleration(model)
    COR_field = Field(COR)
    @test COR isa UMomentumEquation.CoriolisAcceleration
    @test COR isa UCoriolisAcceleration
    @test COR_field isa Field

    # Test PressureGradient
    hydrostatic_pressure = hasfield(typeof(model), :free_surface) ? model.free_surface : nothing
    PRES = UMomentumEquation.PressureGradient(model, hydrostatic_pressure)
    PRES_field = Field(PRES)
    @test PRES isa UMomentumEquation.PressureGradient
    @test PRES isa UPressureGradient
    @test PRES_field isa Field

    PRES = UMomentumEquation.PressureGradient(model)
    PRES_field = Field(PRES)
    @test PRES isa UMomentumEquation.PressureGradient
    @test PRES isa UPressureGradient
    @test PRES_field isa Field

    # Test ViscousDissipation
    VISC = UMomentumEquation.ViscousDissipation(model, model.closure, model.diffusivity_fields, model.clock, fields(model), model.buoyancy)
    VISC_field = Field(VISC)
    @test VISC isa UMomentumEquation.ViscousDissipation
    @test VISC isa UViscousDissipation
    @test VISC_field isa Field

    VISC = UMomentumEquation.ViscousDissipation(model)
    VISC_field = Field(VISC)
    @test VISC isa UMomentumEquation.ViscousDissipation
    @test VISC isa UViscousDissipation
    @test VISC_field isa Field

    # Test ImmersedViscousDissipation
    u_immersed_bc = model.velocities.u.boundary_conditions.immersed
    IVISC = UMomentumEquation.ImmersedViscousDissipation(model, model.velocities, u_immersed_bc, model.closure, model.diffusivity_fields, model.clock, fields(model))
    IVISC_field = Field(IVISC)
    @test IVISC isa UMomentumEquation.ImmersedViscousDissipation
    @test IVISC isa UImmersedViscousDissipation
    @test IVISC_field isa Field

    IVISC = UMomentumEquation.ImmersedViscousDissipation(model)
    IVISC_field = Field(IVISC)
    @test IVISC isa UMomentumEquation.ImmersedViscousDissipation
    @test IVISC isa UImmersedViscousDissipation
    @test IVISC_field isa Field

    # Test TotalViscousDissipation
    TVISC = UMomentumEquation.TotalViscousDissipation(model, model.velocities, u_immersed_bc, model.closure, model.diffusivity_fields, model.clock, fields(model), model.buoyancy)
    TVISC_field = Field(TVISC)
    @test TVISC isa UMomentumEquation.TotalViscousDissipation
    @test TVISC isa UTotalViscousDissipation
    @test TVISC_field isa Field

    TVISC = UMomentumEquation.TotalViscousDissipation(model)
    TVISC_field = Field(TVISC)
    @test TVISC isa UMomentumEquation.TotalViscousDissipation
    @test TVISC isa UTotalViscousDissipation
    @test TVISC_field isa Field

    # Test StokesShear
    SSTOKES = UMomentumEquation.StokesShear(model, model.stokes_drift, model.velocities, model.clock.time)
    SSTOKES_field = Field(SSTOKES)
    @test SSTOKES isa UMomentumEquation.StokesShear
    @test SSTOKES isa UStokesShear
    @test SSTOKES_field isa Field

    SSTOKES = UMomentumEquation.StokesShear(model)
    SSTOKES_field = Field(SSTOKES)
    @test SSTOKES isa UMomentumEquation.StokesShear
    @test SSTOKES isa UStokesShear
    @test SSTOKES_field isa Field

    # Test StokesTendency
    TSTOKES = UMomentumEquation.StokesTendency(model, model.stokes_drift, model.clock.time)
    TSTOKES_field = Field(TSTOKES)
    @test TSTOKES isa UMomentumEquation.StokesTendency
    @test TSTOKES isa UStokesTendency
    @test TSTOKES_field isa Field

    TSTOKES = UMomentumEquation.StokesTendency(model)
    TSTOKES_field = Field(TSTOKES)
    @test TSTOKES isa UMomentumEquation.StokesTendency
    @test TSTOKES isa UStokesTendency
    @test TSTOKES_field isa Field

    # Test Forcing
    FORC = UMomentumEquation.Forcing(model, model.forcing.u, model.clock, fields(model), Val(:u))
    FORC_field = Field(FORC)
    @test FORC isa UMomentumEquation.Forcing
    @test FORC isa UForcing
    @test FORC_field isa Field

    FORC = UMomentumEquation.Forcing(model)
    FORC_field = Field(FORC)
    @test FORC isa UMomentumEquation.Forcing
    @test FORC isa UForcing
    @test FORC_field isa Field

    # Test TotalTendency
    if model isa HydrostaticFreeSurfaceModel
        TEND = UMomentumEquation.TotalTendency(model, model.advection.momentum, model.coriolis, model.stokes_drift, model.closure, u_immersed_bc, model.buoyancy, model.background_fields, model.velocities, model.tracers, model.auxiliary_fields, model.diffusivity_fields, model.free_surface, model.clock, model.forcing.u)
    else
        TEND = UMomentumEquation.TotalTendency(model, model.advection, model.coriolis, model.stokes_drift, model.closure, u_immersed_bc, model.buoyancy, model.background_fields, model.velocities, model.tracers, model.auxiliary_fields, model.diffusivity_fields, nothing, model.clock, model.forcing.u)
    end
    TEND_field = Field(TEND)
    @test TEND isa UMomentumEquation.TotalTendency
    @test TEND isa UTotalTendency
    @test TEND_field isa Field

    TEND = UMomentumEquation.TotalTendency(model)
    TEND_field = Field(TEND)
    @test TEND isa UMomentumEquation.TotalTendency
    @test TEND isa UTotalTendency
    @test TEND_field isa Field

    return nothing
end

function test_u_momentum_field_locations(model)
    # All UMomentumEquation functions should return operations at (Face, Center, Center)
    ADV = UMomentumEquation.Advection(model)
    @test location(ADV) == (Face, Center, Center)

    BUOY = UMomentumEquation.BuoyancyAcceleration(model)
    @test location(BUOY) == (Face, Center, Center)

    COR = UMomentumEquation.CoriolisAcceleration(model)
    @test location(COR) == (Face, Center, Center)

    PRES = UMomentumEquation.PressureGradient(model)
    @test location(PRES) == (Face, Center, Center)

    VISC = UMomentumEquation.ViscousDissipation(model)
    @test location(VISC) == (Face, Center, Center)

    IVISC = UMomentumEquation.ImmersedViscousDissipation(model)
    @test location(IVISC) == (Face, Center, Center)

    TVISC = UMomentumEquation.TotalViscousDissipation(model)
    @test location(TVISC) == (Face, Center, Center)

    SSTOKES = UMomentumEquation.StokesShear(model)
    @test location(SSTOKES) == (Face, Center, Center)

    TSTOKES = UMomentumEquation.StokesTendency(model)
    @test location(TSTOKES) == (Face, Center, Center)

    FORC = UMomentumEquation.Forcing(model)
    @test location(FORC) == (Face, Center, Center)

    TEND = UMomentumEquation.TotalTendency(model)
    @test location(TEND) == (Face, Center, Center)

    return nothing
end

function test_u_momentum_location_validation(model)
    # Test that invalid locations throw errors
    @test_throws ArgumentError UMomentumEquation.Advection(model; location = (Center, Center, Center))
    @test_throws ArgumentError UMomentumEquation.BuoyancyAcceleration(model; location = (Center, Face, Center))
    @test_throws ArgumentError UMomentumEquation.CoriolisAcceleration(model; location = (Center, Center, Face))
    @test_throws ArgumentError UMomentumEquation.PressureGradient(model; location = (Center, Center, Center))
    @test_throws ArgumentError UMomentumEquation.ViscousDissipation(model; location = (Center, Center, Center))
    @test_throws ArgumentError UMomentumEquation.ImmersedViscousDissipation(model; location = (Center, Center, Center))
    @test_throws ArgumentError UMomentumEquation.TotalViscousDissipation(model; location = (Center, Center, Center))
    @test_throws ArgumentError UMomentumEquation.StokesShear(model; location = (Center, Center, Center))
    @test_throws ArgumentError UMomentumEquation.StokesTendency(model; location = (Center, Center, Center))
    @test_throws ArgumentError UMomentumEquation.Forcing(model; location = (Center, Center, Center))
    @test_throws ArgumentError UMomentumEquation.TotalTendency(model; location = (Center, Center, Center))

    return nothing
end
#---

@testset "U-momentum equation diagnostics tests" begin
    @info "  Testing u-momentum diagnostics"
    for (grid_class, grid) in zip(keys(grids), values(grids))
        @info "    with $grid_class"
        for model_type in model_types
            @info "      with $model_type"
            model = model_type(; grid, model_kwargs...)

            @info "        Testing u-momentum terms"
            test_u_momentum_terms(model)

            @info "        Testing u-momentum field locations"
            test_u_momentum_field_locations(model)

            @info "        Testing u-momentum location validation"
            test_u_momentum_location_validation(model)
        end
    end
end
