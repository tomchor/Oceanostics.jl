using Test
using CUDA: has_cuda_gpu

using Oceananigans
using Oceananigans.Fields: location
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
w_boundary_conditions = FieldBoundaryConditions(immersed=immersed_bc)
boundary_conditions = (; w = w_boundary_conditions)

model_kwargs = (; tracers, forcing, boundary_conditions, buoyancy=BuoyancyTracer(), coriolis=FPlane(f=1e-4))
#---

#+++ Test options
grids = Dict("regular grid" => regular_grid,
             "stretched grid" => stretched_grid)

# Only NonhydrostaticModel evolves w prognostically; HydrostaticFreeSurfaceModel
# diagnoses w from continuity and has no w-momentum equation.
model_types = (NonhydrostaticModel,)
#---

#+++ Test functions
function test_w_momentum_terms(model)
    # Test Advection
    ADV = WMomentumEquation.Advection(model, model.velocities..., model.advection)
    ADV_field = Field(ADV)
    @test ADV isa WMomentumEquation.Advection
    @test ADV isa WAdvection
    @test ADV_field isa Field

    ADV = WMomentumEquation.Advection(model)
    @test ADV isa WMomentumEquation.Advection
    ADV_field = Field(ADV)
    @test ADV_field isa Field

    # Test BuoyancyAcceleration
    BUOY = WMomentumEquation.BuoyancyAcceleration(model, model.buoyancy, model.tracers)
    BUOY_field = Field(BUOY)
    @test BUOY isa WMomentumEquation.BuoyancyAcceleration
    @test BUOY isa WBuoyancyAcceleration
    @test BUOY_field isa Field

    BUOY = WMomentumEquation.BuoyancyAcceleration(model)
    BUOY_field = Field(BUOY)
    @test BUOY isa WMomentumEquation.BuoyancyAcceleration
    @test BUOY isa WBuoyancyAcceleration
    @test BUOY_field isa Field

    # Test CoriolisAcceleration
    COR = WMomentumEquation.CoriolisAcceleration(model, model.coriolis, model.velocities)
    COR_field = Field(COR)
    @test COR isa WMomentumEquation.CoriolisAcceleration
    @test COR isa WCoriolisAcceleration
    @test COR_field isa Field

    COR = WMomentumEquation.CoriolisAcceleration(model)
    COR_field = Field(COR)
    @test COR isa WMomentumEquation.CoriolisAcceleration
    @test COR isa WCoriolisAcceleration
    @test COR_field isa Field

    # Test ViscousDissipation
    VISC = WMomentumEquation.ViscousDissipation(model, model.closure, model.closure_fields, model.clock, fields(model), model.buoyancy)
    VISC_field = Field(VISC)
    @test VISC isa WMomentumEquation.ViscousDissipation
    @test VISC isa WViscousDissipation
    @test VISC_field isa Field

    VISC = WMomentumEquation.ViscousDissipation(model)
    VISC_field = Field(VISC)
    @test VISC isa WMomentumEquation.ViscousDissipation
    @test VISC isa WViscousDissipation
    @test VISC_field isa Field

    # Test ImmersedViscousDissipation
    w_immersed_bc = model.velocities.w.boundary_conditions.immersed
    IVISC = WMomentumEquation.ImmersedViscousDissipation(model, model.velocities, w_immersed_bc, model.closure, model.closure_fields, model.clock, fields(model))
    IVISC_field = Field(IVISC)
    @test IVISC isa WMomentumEquation.ImmersedViscousDissipation
    @test IVISC isa WImmersedViscousDissipation
    @test IVISC_field isa Field

    IVISC = WMomentumEquation.ImmersedViscousDissipation(model)
    IVISC_field = Field(IVISC)
    @test IVISC isa WMomentumEquation.ImmersedViscousDissipation
    @test IVISC isa WImmersedViscousDissipation
    @test IVISC_field isa Field

    # Test TotalViscousDissipation
    TVISC = WMomentumEquation.TotalViscousDissipation(model, model.velocities, w_immersed_bc, model.closure, model.closure_fields, model.clock, fields(model), model.buoyancy)
    TVISC_field = Field(TVISC)
    @test TVISC isa WMomentumEquation.TotalViscousDissipation
    @test TVISC isa WTotalViscousDissipation
    @test TVISC_field isa Field

    TVISC = WMomentumEquation.TotalViscousDissipation(model)
    TVISC_field = Field(TVISC)
    @test TVISC isa WMomentumEquation.TotalViscousDissipation
    @test TVISC isa WTotalViscousDissipation
    @test TVISC_field isa Field

    # Test StokesShear
    SSTOKES = WMomentumEquation.StokesShear(model, model.stokes_drift, model.velocities, model.clock.time)
    SSTOKES_field = Field(SSTOKES)
    @test SSTOKES isa WMomentumEquation.StokesShear
    @test SSTOKES isa WStokesShear
    @test SSTOKES_field isa Field

    SSTOKES = WMomentumEquation.StokesShear(model)
    SSTOKES_field = Field(SSTOKES)
    @test SSTOKES isa WMomentumEquation.StokesShear
    @test SSTOKES isa WStokesShear
    @test SSTOKES_field isa Field

    # Test StokesTendency
    TSTOKES = WMomentumEquation.StokesTendency(model, model.stokes_drift, model.clock.time)
    TSTOKES_field = Field(TSTOKES)
    @test TSTOKES isa WMomentumEquation.StokesTendency
    @test TSTOKES isa WStokesTendency
    @test TSTOKES_field isa Field

    TSTOKES = WMomentumEquation.StokesTendency(model)
    TSTOKES_field = Field(TSTOKES)
    @test TSTOKES isa WMomentumEquation.StokesTendency
    @test TSTOKES isa WStokesTendency
    @test TSTOKES_field isa Field

    # Test Forcing
    FORC = WMomentumEquation.Forcing(model, model.forcing.w, model.clock, fields(model), Val(:w))
    FORC_field = Field(FORC)
    @test FORC isa WMomentumEquation.Forcing
    @test FORC isa WForcing
    @test FORC_field isa Field

    FORC = WMomentumEquation.Forcing(model, Val(:w))
    FORC_field = Field(FORC)
    @test FORC isa WMomentumEquation.Forcing
    @test FORC isa WForcing
    @test FORC_field isa Field

    # Test TotalTendency (NonhydrostaticModel only)
    TEND = WMomentumEquation.TotalTendency(model, model.advection, model.coriolis, model.stokes_drift, model.closure, w_immersed_bc, model.buoyancy, model.background_fields, model.velocities, model.tracers, model.auxiliary_fields, model.closure_fields, model.pressures.pHY′, model.clock, model.forcing.w)
    TEND_field = Field(TEND)
    @test TEND isa WMomentumEquation.TotalTendency
    @test TEND isa WTotalTendency
    @test TEND_field isa Field

    TEND = WMomentumEquation.TotalTendency(model)
    TEND_field = Field(TEND)
    @test TEND isa WMomentumEquation.TotalTendency
    @test TEND isa WTotalTendency
    @test TEND_field isa Field

    return nothing
end

function test_w_momentum_field_locations(model)
    # All WMomentumEquation functions should return operations at (Center, Center, Face)
    ADV = WMomentumEquation.Advection(model)
    @test location(ADV) == (Center, Center, Face)

    BUOY = WMomentumEquation.BuoyancyAcceleration(model)
    @test location(BUOY) == (Center, Center, Face)

    COR = WMomentumEquation.CoriolisAcceleration(model)
    @test location(COR) == (Center, Center, Face)

    VISC = WMomentumEquation.ViscousDissipation(model)
    @test location(VISC) == (Center, Center, Face)

    IVISC = WMomentumEquation.ImmersedViscousDissipation(model)
    @test location(IVISC) == (Center, Center, Face)

    TVISC = WMomentumEquation.TotalViscousDissipation(model)
    @test location(TVISC) == (Center, Center, Face)

    SSTOKES = WMomentumEquation.StokesShear(model)
    @test location(SSTOKES) == (Center, Center, Face)

    TSTOKES = WMomentumEquation.StokesTendency(model)
    @test location(TSTOKES) == (Center, Center, Face)

    FORC = WMomentumEquation.Forcing(model, Val(:w))
    @test location(FORC) == (Center, Center, Face)

    TEND = WMomentumEquation.TotalTendency(model)
    @test location(TEND) == (Center, Center, Face)

    return nothing
end

function test_w_momentum_location_validation(model)
    # Test that invalid locations throw errors
    @test_throws ArgumentError WMomentumEquation.Advection(model; location = (Center, Center, Center))
    @test_throws ArgumentError WMomentumEquation.BuoyancyAcceleration(model; location = (Face, Center, Center))
    @test_throws ArgumentError WMomentumEquation.CoriolisAcceleration(model; location = (Center, Face, Center))
    @test_throws ArgumentError WMomentumEquation.ViscousDissipation(model; location = (Center, Center, Center))
    @test_throws ArgumentError WMomentumEquation.ImmersedViscousDissipation(model; location = (Center, Center, Center))
    @test_throws ArgumentError WMomentumEquation.TotalViscousDissipation(model; location = (Center, Center, Center))
    @test_throws ArgumentError WMomentumEquation.StokesShear(model; location = (Center, Center, Center))
    @test_throws ArgumentError WMomentumEquation.StokesTendency(model; location = (Center, Center, Center))
    @test_throws ArgumentError WMomentumEquation.TotalTendency(model; location = (Center, Center, Center))

    return nothing
end

function test_w_momentum_hfs_unsupported()
    grid = first(values(grids))
    hfs_model = HydrostaticFreeSurfaceModel(grid; tracers, buoyancy=BuoyancyTracer())
    @test_throws ArgumentError WMomentumEquation.TotalTendency(hfs_model)
    return nothing
end
#---

@testset "W-momentum equation diagnostics tests" begin
    @info "  Testing w-momentum diagnostics"
    for (grid_class, grid) in zip(keys(grids), values(grids))
        @info "    with $grid_class"
        for model_type in model_types
            @info "      with $model_type"
            model = model_type(grid; model_kwargs...)

            @info "        Testing w-momentum terms"
            test_w_momentum_terms(model)

            @info "        Testing w-momentum field locations"
            test_w_momentum_field_locations(model)

            @info "        Testing w-momentum location validation"
            test_w_momentum_location_validation(model)
        end
    end

    @info "    Testing that TotalTendency is unsupported on HydrostaticFreeSurfaceModel"
    test_w_momentum_hfs_unsupported()
end
