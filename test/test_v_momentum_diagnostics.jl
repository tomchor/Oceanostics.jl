using Test
using CUDA: has_cuda_gpu

using Oceananigans
using Oceananigans.Fields: location
using Oceananigans.TimeSteppers: update_state!
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
forcing = (; u = Forcing(forcing_function), v = Forcing(forcing_function))

bc_function(x, y, z, t) = sin(t)
immersed_bc = FluxBoundaryCondition(bc_function)
v_boundary_conditions = FieldBoundaryConditions(immersed=immersed_bc)
boundary_conditions = (; v = v_boundary_conditions)

stokes_drift = UniformStokesDrift(∂z_uˢ = (z, t) -> exp(z) * cos(t),
                                  ∂z_vˢ = (z, t) -> exp(z) * sin(t),
                                  ∂t_uˢ = (z, t) -> -exp(z) * sin(t),
                                  ∂t_vˢ = (z, t) ->  exp(z) * cos(t))

model_kwargs    = (; tracers, forcing, boundary_conditions, buoyancy=BuoyancyTracer(), coriolis=FPlane(f=1e-4))
nh_model_kwargs = (; model_kwargs..., stokes_drift) # stokes_drift only applies to NonhydrostaticModel
#---

#+++ Test options
grids = Dict("regular grid" => regular_grid,
             "stretched grid" => stretched_grid)

model_types = (NonhydrostaticModel,
               HydrostaticFreeSurfaceModel)
#---

#+++ Test functions
function test_v_momentum_terms(model)
    # Test Advection
    advection_scheme = model isa HydrostaticFreeSurfaceModel ? model.advection.momentum : model.advection
    ADV = VMomentumEquation.Advection(model, model.velocities..., advection_scheme)
    ADV_field = Field(ADV)
    @test ADV isa VMomentumEquation.Advection
    @test ADV isa VAdvection
    @test ADV_field isa Field

    ADV = VMomentumEquation.Advection(model)
    @test ADV isa VMomentumEquation.Advection
    ADV_field = Field(ADV)
    @test ADV_field isa Field

    # Test BuoyancyAcceleration
    BUOY = VMomentumEquation.BuoyancyAcceleration(model, model.buoyancy, model.tracers)
    BUOY_field = Field(BUOY)
    @test BUOY isa VMomentumEquation.BuoyancyAcceleration
    @test BUOY isa VBuoyancyAcceleration
    @test BUOY_field isa Field

    BUOY = VMomentumEquation.BuoyancyAcceleration(model)
    BUOY_field = Field(BUOY)
    @test BUOY isa VMomentumEquation.BuoyancyAcceleration
    @test BUOY isa VBuoyancyAcceleration
    @test BUOY_field isa Field

    # Test CoriolisAcceleration
    COR = VMomentumEquation.CoriolisAcceleration(model, model.coriolis, model.velocities)
    COR_field = Field(COR)
    @test COR isa VMomentumEquation.CoriolisAcceleration
    @test COR isa VCoriolisAcceleration
    @test COR_field isa Field

    COR = VMomentumEquation.CoriolisAcceleration(model)
    COR_field = Field(COR)
    @test COR isa VMomentumEquation.CoriolisAcceleration
    @test COR isa VCoriolisAcceleration
    @test COR_field isa Field

    # Test PressureGradient
    hydrostatic_pressure = model isa HydrostaticFreeSurfaceModel ? model.pressure.pHY′ : model.pressures.pHY′
    PRES = VMomentumEquation.PressureGradient(model, hydrostatic_pressure)
    PRES_field = Field(PRES)
    @test PRES isa VMomentumEquation.PressureGradient
    @test PRES isa VPressureGradient
    @test PRES_field isa Field

    PRES = VMomentumEquation.PressureGradient(model)
    PRES_field = Field(PRES)
    @test PRES isa VMomentumEquation.PressureGradient
    @test PRES isa VPressureGradient
    @test PRES_field isa Field

    # Test ViscousDissipation
    VISC = VMomentumEquation.ViscousDissipation(model, model.closure, model.closure_fields, model.clock, fields(model), model.buoyancy)
    VISC_field = Field(VISC)
    @test VISC isa VMomentumEquation.ViscousDissipation
    @test VISC isa VViscousDissipation
    @test VISC_field isa Field

    VISC = VMomentumEquation.ViscousDissipation(model)
    VISC_field = Field(VISC)
    @test VISC isa VMomentumEquation.ViscousDissipation
    @test VISC isa VViscousDissipation
    @test VISC_field isa Field

    # Test ImmersedViscousDissipation
    v_immersed_bc = model.velocities.v.boundary_conditions.immersed
    IVISC = VMomentumEquation.ImmersedViscousDissipation(model, model.velocities, v_immersed_bc, model.closure, model.closure_fields, model.clock, fields(model))
    IVISC_field = Field(IVISC)
    @test IVISC isa VMomentumEquation.ImmersedViscousDissipation
    @test IVISC isa VImmersedViscousDissipation
    @test IVISC_field isa Field

    IVISC = VMomentumEquation.ImmersedViscousDissipation(model)
    IVISC_field = Field(IVISC)
    @test IVISC isa VMomentumEquation.ImmersedViscousDissipation
    @test IVISC isa VImmersedViscousDissipation
    @test IVISC_field isa Field

    # Test TotalViscousDissipation
    TVISC = VMomentumEquation.TotalViscousDissipation(model, model.velocities, v_immersed_bc, model.closure, model.closure_fields, model.clock, fields(model), model.buoyancy)
    TVISC_field = Field(TVISC)
    @test TVISC isa VMomentumEquation.TotalViscousDissipation
    @test TVISC isa VTotalViscousDissipation
    @test TVISC_field isa Field

    TVISC = VMomentumEquation.TotalViscousDissipation(model)
    TVISC_field = Field(TVISC)
    @test TVISC isa VMomentumEquation.TotalViscousDissipation
    @test TVISC isa VTotalViscousDissipation
    @test TVISC_field isa Field

    # Stokes terms only apply to NonhydrostaticModel (HFS has no stokes_drift field)
    if !(model isa HydrostaticFreeSurfaceModel)
        # Test StokesShear
        SSTOKES = VMomentumEquation.StokesShear(model, model.stokes_drift, model.velocities, model.clock.time)
        SSTOKES_field = Field(SSTOKES)
        @test SSTOKES isa VMomentumEquation.StokesShear
        @test SSTOKES isa VStokesShear
        @test SSTOKES_field isa Field

        SSTOKES = VMomentumEquation.StokesShear(model)
        SSTOKES_field = Field(SSTOKES)
        @test SSTOKES isa VMomentumEquation.StokesShear
        @test SSTOKES isa VStokesShear
        @test SSTOKES_field isa Field

        # Test StokesTendency
        TSTOKES = VMomentumEquation.StokesTendency(model, model.stokes_drift, model.clock.time)
        TSTOKES_field = Field(TSTOKES)
        @test TSTOKES isa VMomentumEquation.StokesTendency
        @test TSTOKES isa VStokesTendency
        @test TSTOKES_field isa Field

        TSTOKES = VMomentumEquation.StokesTendency(model)
        TSTOKES_field = Field(TSTOKES)
        @test TSTOKES isa VMomentumEquation.StokesTendency
        @test TSTOKES isa VStokesTendency
        @test TSTOKES_field isa Field
    else
        @test_throws ArgumentError VMomentumEquation.StokesShear(model)
        @test_throws ArgumentError VMomentumEquation.StokesTendency(model)
    end

    # Test Forcing
    FORC = VMomentumEquation.Forcing(model, model.forcing.v, model.clock, fields(model), Val(:v))
    FORC_field = Field(FORC)
    @test FORC isa VMomentumEquation.Forcing
    @test FORC isa VForcing
    @test FORC_field isa Field

    FORC = VMomentumEquation.Forcing(model, Val(:v))
    FORC_field = Field(FORC)
    @test FORC isa VMomentumEquation.Forcing
    @test FORC isa VForcing
    @test FORC_field isa Field

    # Test Tendency
    if model isa HydrostaticFreeSurfaceModel
        TEND = VMomentumEquation.Tendency(model, model.advection.momentum, model.coriolis, model.closure, v_immersed_bc, model.velocities, model.free_surface, model.tracers, model.buoyancy, model.closure_fields, model.pressure.pHY′, model.auxiliary_fields, model.vertical_coordinate, model.clock, model.forcing.v)
    else
        TEND = VMomentumEquation.Tendency(model, model.advection, model.coriolis, model.stokes_drift, model.closure, v_immersed_bc, model.buoyancy, model.background_fields, model.velocities, model.tracers, model.auxiliary_fields, model.closure_fields, model.pressures.pHY′, model.clock, model.forcing.v)
    end
    TEND_field = Field(TEND)
    @test TEND isa VMomentumEquation.Tendency
    @test TEND isa VTendency
    @test TEND_field isa Field

    TEND = VMomentumEquation.Tendency(model)
    TEND_field = Field(TEND)
    @test TEND isa VMomentumEquation.Tendency
    @test TEND isa VTendency
    @test TEND_field isa Field

    return nothing
end

function test_v_momentum_field_locations(model)
    # All VMomentumEquation functions should return operations at (Center, Face, Center)
    ADV = VMomentumEquation.Advection(model)
    @test location(ADV) == (Center, Face, Center)

    BUOY = VMomentumEquation.BuoyancyAcceleration(model)
    @test location(BUOY) == (Center, Face, Center)

    COR = VMomentumEquation.CoriolisAcceleration(model)
    @test location(COR) == (Center, Face, Center)

    PRES = VMomentumEquation.PressureGradient(model)
    @test location(PRES) == (Center, Face, Center)

    VISC = VMomentumEquation.ViscousDissipation(model)
    @test location(VISC) == (Center, Face, Center)

    IVISC = VMomentumEquation.ImmersedViscousDissipation(model)
    @test location(IVISC) == (Center, Face, Center)

    TVISC = VMomentumEquation.TotalViscousDissipation(model)
    @test location(TVISC) == (Center, Face, Center)

    if !(model isa HydrostaticFreeSurfaceModel)
        SSTOKES = VMomentumEquation.StokesShear(model)
        @test location(SSTOKES) == (Center, Face, Center)

        TSTOKES = VMomentumEquation.StokesTendency(model)
        @test location(TSTOKES) == (Center, Face, Center)
    end

    FORC = VMomentumEquation.Forcing(model, Val(:v))
    @test location(FORC) == (Center, Face, Center)

    TEND = VMomentumEquation.Tendency(model)
    @test location(TEND) == (Center, Face, Center)

    return nothing
end

function test_v_momentum_budget_closure(grid)
    # Build a NH model with every term active so the budget exercises the full RHS of v_velocity_tendency.
    sd = UniformStokesDrift(∂z_uˢ = (z, t) -> exp(z) * cos(t),
                            ∂z_vˢ = (z, t) -> exp(z) * sin(t),
                            ∂t_uˢ = (z, t) -> -exp(z) * sin(t),
                            ∂t_vˢ = (z, t) ->  exp(z) * cos(t))
    model = NonhydrostaticModel(grid; tracers = :b,
                                      buoyancy = BuoyancyTracer(),
                                      coriolis = FPlane(f = 1e-4),
                                      stokes_drift = sd,
                                      closure = ScalarDiffusivity(ν = 1e-4, κ = 1e-4),
                                      forcing = (; v = Forcing((x, y, z, t) -> cos(t))))
    set!(model, u = (x, y, z) -> sin(2π*x) * cos(2π*y) * exp(z),
                v = (x, y, z) -> cos(2π*x) * sin(2π*y) * exp(z),
                w = (x, y, z) -> sin(2π*x) * sin(2π*z),
                b = (x, y, z) -> sin(2π*z))
    update_state!(model) # populates model.pressures.pHY′ from b

    # Reconstruct G_v from individual diagnostics, matching Oceananigans' v_velocity_tendency sign convention:
    #   G_v = -ADV + BUOY - COR - PRES - TVISC + STOKES_SHEAR + STOKES_TENDENCY + FORCING
    ADV   = VMomentumEquation.Advection(model)
    BUOY  = VMomentumEquation.BuoyancyAcceleration(model)
    COR   = VMomentumEquation.CoriolisAcceleration(model)
    PRES  = VMomentumEquation.PressureGradient(model)
    TVISC = VMomentumEquation.TotalViscousDissipation(model)
    SS    = VMomentumEquation.StokesShear(model)
    ST    = VMomentumEquation.StokesTendency(model)
    FORC  = VMomentumEquation.Forcing(model, Val(:v))
    TEND  = VMomentumEquation.Tendency(model)

    budget = Field(-ADV + BUOY - COR - PRES - TVISC + SS + ST + FORC)
    tend   = Field(TEND)
    @test interior(budget) ≈ interior(tend)
    return nothing
end

function test_v_momentum_hfs_budget_closure(grid)
    # Build an HFS model with every term active so the budget exercises the full RHS of
    # hydrostatic_free_surface_v_velocity_tendency. `momentum_advection = Centered()` is required
    # so the diagnostic Advection (which wraps U_dot_∇v with the model's scheme) matches the
    # tendency's advection for any non-VectorInvariant scheme.
    model = HydrostaticFreeSurfaceModel(grid; tracers = :b,
                                              buoyancy = BuoyancyTracer(),
                                              coriolis = FPlane(f = 1e-4),
                                              momentum_advection = Centered(),
                                              closure = ScalarDiffusivity(ν = 1e-4, κ = 1e-4),
                                              forcing = (; v = Forcing((x, y, z, t) -> cos(t))))
    set!(model, u = (x, y, z) -> sin(2π*x) * cos(2π*y) * exp(z),
                v = (x, y, z) -> cos(2π*x) * sin(2π*y) * exp(z),
                b = (x, y, z) -> sin(2π*z))
    update_state!(model) # populates model.pressure.pHY′ from b

    # Reconstruct G_v for HFS:
    #   G_v = -ADV - BARO - COR - PRES - TVISC + FORCING
    # HFS has no Stokes terms and no BuoyancyAcceleration term (buoyancy is absorbed into pHY′).
    # BARO is the explicit barotropic free-surface gradient; on ImplicitFreeSurface it returns
    # zero (the contribution is handled inside the pressure solve) so the budget closes
    # without it, but including it keeps the formula correct for ExplicitFreeSurface too.
    ADV   = VMomentumEquation.Advection(model)
    COR   = VMomentumEquation.CoriolisAcceleration(model)
    PRES  = VMomentumEquation.PressureGradient(model)
    BARO  = VMomentumEquation.BarotropicPressureGradient(model)
    TVISC = VMomentumEquation.TotalViscousDissipation(model)
    FORC  = VMomentumEquation.Forcing(model, Val(:v))
    TEND  = VMomentumEquation.Tendency(model)

    budget = Field(-ADV - BARO - COR - PRES - TVISC + FORC)
    tend   = Field(TEND)
    @test interior(budget) ≈ interior(tend)
    return nothing
end

function test_v_momentum_location_validation(model)
    # Test that invalid locations throw errors
    @test_throws ArgumentError VMomentumEquation.Advection(model; location = (Center, Center, Center))
    @test_throws ArgumentError VMomentumEquation.BuoyancyAcceleration(model; location = (Face, Center, Center))
    @test_throws ArgumentError VMomentumEquation.CoriolisAcceleration(model; location = (Center, Center, Face))
    @test_throws ArgumentError VMomentumEquation.PressureGradient(model; location = (Center, Center, Center))
    @test_throws ArgumentError VMomentumEquation.ViscousDissipation(model; location = (Center, Center, Center))
    @test_throws ArgumentError VMomentumEquation.ImmersedViscousDissipation(model; location = (Center, Center, Center))
    @test_throws ArgumentError VMomentumEquation.TotalViscousDissipation(model; location = (Center, Center, Center))
    @test_throws ArgumentError VMomentumEquation.StokesShear(model; location = (Center, Center, Center))
    @test_throws ArgumentError VMomentumEquation.StokesTendency(model; location = (Center, Center, Center))
    @test_throws ArgumentError VMomentumEquation.Tendency(model; location = (Center, Center, Center))

    return nothing
end
#---

@testset "V-momentum equation diagnostics tests" begin
    @info "  Testing v-momentum diagnostics"
    for (grid_class, grid) in zip(keys(grids), values(grids))
        @info "    with $grid_class"
        for model_type in model_types
            @info "      with $model_type"
            # HFS defaults to VectorInvariant momentum advection, which uses U_dot_∇u
            # rather than div_𝐯v. Force the flux-form scheme so the Advection diagnostic
            # (which wraps div_𝐯v) is well-defined. HFS has no stokes_drift field, so we
            # pass the Stokes-drift only to the NH model.
            model = model_type === HydrostaticFreeSurfaceModel ?
                model_type(grid; model_kwargs..., momentum_advection=Centered()) :
                model_type(grid; nh_model_kwargs...)

            @info "        Testing v-momentum terms"
            test_v_momentum_terms(model)

            @info "        Testing v-momentum field locations"
            test_v_momentum_field_locations(model)

            @info "        Testing v-momentum location validation"
            test_v_momentum_location_validation(model)
        end
    end

    @info "    Testing v-momentum budget closure on NonhydrostaticModel"
    test_v_momentum_budget_closure(underlying_regular_grid)

    @info "    Testing v-momentum budget closure on HydrostaticFreeSurfaceModel"
    test_v_momentum_hfs_budget_closure(underlying_regular_grid)
end
