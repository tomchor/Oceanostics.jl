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
u_boundary_conditions = FieldBoundaryConditions(immersed=immersed_bc)
boundary_conditions = (; u = u_boundary_conditions)

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
function test_u_momentum_terms(model)
    # Test Advection
    advection_scheme = model isa HydrostaticFreeSurfaceModel ? model.advection.momentum : model.advection
    ADV = UMomentumEquation.Advection(model, model.velocities..., advection_scheme)
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
    hydrostatic_pressure = model isa HydrostaticFreeSurfaceModel ? model.pressure.pHY′ : model.pressures.pHY′
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

    # Test BarotropicPressureGradient
    free_surface = hasfield(typeof(model), :free_surface) ? model.free_surface : nothing
    BARO = UMomentumEquation.BarotropicPressureGradient(model, free_surface)
    BARO_field = Field(BARO)
    @test BARO isa UMomentumEquation.BarotropicPressureGradient
    @test BARO isa UBarotropicPressureGradient
    @test BARO_field isa Field

    BARO = UMomentumEquation.BarotropicPressureGradient(model)
    BARO_field = Field(BARO)
    @test BARO isa UMomentumEquation.BarotropicPressureGradient
    @test BARO isa UBarotropicPressureGradient
    @test BARO_field isa Field

    # Test ViscousDissipation
    VISC = UMomentumEquation.ViscousDissipation(model, model.closure, model.closure_fields, model.clock, fields(model), model.buoyancy)
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
    IVISC = UMomentumEquation.ImmersedViscousDissipation(model, model.velocities, u_immersed_bc, model.closure, model.closure_fields, model.clock, fields(model))
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
    TVISC = UMomentumEquation.TotalViscousDissipation(model, model.velocities, u_immersed_bc, model.closure, model.closure_fields, model.clock, fields(model), model.buoyancy)
    TVISC_field = Field(TVISC)
    @test TVISC isa UMomentumEquation.TotalViscousDissipation
    @test TVISC isa UTotalViscousDissipation
    @test TVISC_field isa Field

    TVISC = UMomentumEquation.TotalViscousDissipation(model)
    TVISC_field = Field(TVISC)
    @test TVISC isa UMomentumEquation.TotalViscousDissipation
    @test TVISC isa UTotalViscousDissipation
    @test TVISC_field isa Field

    # Stokes terms only apply to NonhydrostaticModel (HFS has no stokes_drift field)
    if !(model isa HydrostaticFreeSurfaceModel)
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
    else
        @test_throws ArgumentError UMomentumEquation.StokesShear(model)
        @test_throws ArgumentError UMomentumEquation.StokesTendency(model)
    end

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

    # Test Tendency
    if model isa HydrostaticFreeSurfaceModel
        TEND = UMomentumEquation.Tendency(model, model.advection.momentum, model.coriolis, model.closure, u_immersed_bc, model.velocities, model.free_surface, model.tracers, model.buoyancy, model.closure_fields, model.pressure.pHY′, model.auxiliary_fields, model.vertical_coordinate, model.clock, model.forcing.u)
    else
        TEND = UMomentumEquation.Tendency(model, model.advection, model.coriolis, model.stokes_drift, model.closure, u_immersed_bc, model.buoyancy, model.background_fields, model.velocities, model.tracers, model.auxiliary_fields, model.closure_fields, model.pressures.pHY′, model.clock, model.forcing.u)
    end
    TEND_field = Field(TEND)
    @test TEND isa UMomentumEquation.Tendency
    @test TEND isa UTendency
    @test TEND_field isa Field

    TEND = UMomentumEquation.Tendency(model)
    TEND_field = Field(TEND)
    @test TEND isa UMomentumEquation.Tendency
    @test TEND isa UTendency
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

    BARO = UMomentumEquation.BarotropicPressureGradient(model)
    @test location(BARO) == (Face, Center, Center)

    VISC = UMomentumEquation.ViscousDissipation(model)
    @test location(VISC) == (Face, Center, Center)

    IVISC = UMomentumEquation.ImmersedViscousDissipation(model)
    @test location(IVISC) == (Face, Center, Center)

    TVISC = UMomentumEquation.TotalViscousDissipation(model)
    @test location(TVISC) == (Face, Center, Center)

    if !(model isa HydrostaticFreeSurfaceModel)
        SSTOKES = UMomentumEquation.StokesShear(model)
        @test location(SSTOKES) == (Face, Center, Center)

        TSTOKES = UMomentumEquation.StokesTendency(model)
        @test location(TSTOKES) == (Face, Center, Center)
    end

    FORC = UMomentumEquation.Forcing(model)
    @test location(FORC) == (Face, Center, Center)

    TEND = UMomentumEquation.Tendency(model)
    @test location(TEND) == (Face, Center, Center)

    return nothing
end

function test_u_momentum_budget_closure(grid)
    # Build a NH model with every term active so the budget exercises the full RHS of u_velocity_tendency.
    sd = UniformStokesDrift(∂z_uˢ = (z, t) -> exp(z) * cos(t),
                            ∂z_vˢ = (z, t) -> exp(z) * sin(t),
                            ∂t_uˢ = (z, t) -> -exp(z) * sin(t),
                            ∂t_vˢ = (z, t) ->  exp(z) * cos(t))
    model = NonhydrostaticModel(grid; tracers = :b,
                                      buoyancy = BuoyancyTracer(),
                                      coriolis = FPlane(f = 1e-4),
                                      stokes_drift = sd,
                                      closure = ScalarDiffusivity(ν = 1e-4, κ = 1e-4),
                                      forcing = (; u = Forcing((x, y, z, t) -> cos(t))))
    set!(model, u = (x, y, z) -> sin(2π*x) * cos(2π*y) * exp(z),
                v = (x, y, z) -> cos(2π*x) * sin(2π*y) * exp(z),
                w = (x, y, z) -> sin(2π*x) * sin(2π*z),
                b = (x, y, z) -> sin(2π*z))
    update_state!(model) # populates model.pressures.pHY′ from b

    # Reconstruct G_u from individual diagnostics, matching Oceananigans' u_velocity_tendency sign convention:
    #   G_u = -ADV + BUOY - COR - PRES - TVISC + STOKES_SHEAR + STOKES_TENDENCY + FORCING
    ADV   = UMomentumEquation.Advection(model)
    BUOY  = UMomentumEquation.BuoyancyAcceleration(model)
    COR   = UMomentumEquation.CoriolisAcceleration(model)
    PRES  = UMomentumEquation.PressureGradient(model)
    TVISC = UMomentumEquation.TotalViscousDissipation(model)
    SS    = UMomentumEquation.StokesShear(model)
    ST    = UMomentumEquation.StokesTendency(model)
    FORC  = UMomentumEquation.Forcing(model)
    TEND  = UMomentumEquation.Tendency(model)

    budget = Field(-ADV + BUOY - COR - PRES - TVISC + SS + ST + FORC)
    tend   = Field(TEND)
    @test interior(budget) ≈ interior(tend)
    return nothing
end

function test_u_momentum_hfs_budget_closure(grid)
    # Build an HFS model with every term active so the budget exercises the full RHS of
    # hydrostatic_free_surface_u_velocity_tendency. The default `momentum_advection`
    # (`VectorInvariant`) is used — the diagnostic `Advection` dispatches on the model type
    # to wrap `U_dot_∇u` with whatever scheme the model carries.
    model = HydrostaticFreeSurfaceModel(grid; tracers = :b,
                                              buoyancy = BuoyancyTracer(),
                                              coriolis = FPlane(f = 1e-4),
                                              closure = ScalarDiffusivity(ν = 1e-4, κ = 1e-4),
                                              forcing = (; u = Forcing((x, y, z, t) -> cos(t))))
    set!(model, u = (x, y, z) -> sin(2π*x) * cos(2π*y) * exp(z),
                v = (x, y, z) -> cos(2π*x) * sin(2π*y) * exp(z),
                b = (x, y, z) -> sin(2π*z))
    update_state!(model) # populates model.pressure.pHY′ from b

    # Reconstruct G_u for HFS:
    #   G_u = -ADV - BARO - COR - PRES - TVISC + FORCING
    # HFS has no Stokes terms and no BuoyancyAcceleration term (buoyancy is absorbed into pHY′).
    # BARO is the explicit barotropic free-surface gradient; on ImplicitFreeSurface it returns
    # zero (the contribution is handled inside the pressure solve) so the budget closes
    # without it, but including it keeps the formula correct for ExplicitFreeSurface too.
    ADV   = UMomentumEquation.Advection(model)
    COR   = UMomentumEquation.CoriolisAcceleration(model)
    PRES  = UMomentumEquation.PressureGradient(model)
    BARO  = UMomentumEquation.BarotropicPressureGradient(model)
    TVISC = UMomentumEquation.TotalViscousDissipation(model)
    FORC  = UMomentumEquation.Forcing(model)
    TEND  = UMomentumEquation.Tendency(model)

    budget = Field(-ADV - BARO - COR - PRES - TVISC + FORC)
    tend   = Field(TEND)
    @test interior(budget) ≈ interior(tend)
    return nothing
end

function test_u_momentum_location_validation(model)
    # Test that invalid locations throw errors
    @test_throws ArgumentError UMomentumEquation.Advection(model; location = (Center, Center, Center))
    @test_throws ArgumentError UMomentumEquation.BuoyancyAcceleration(model; location = (Center, Face, Center))
    @test_throws ArgumentError UMomentumEquation.CoriolisAcceleration(model; location = (Center, Center, Face))
    @test_throws ArgumentError UMomentumEquation.PressureGradient(model; location = (Center, Center, Center))
    @test_throws ArgumentError UMomentumEquation.BarotropicPressureGradient(model; location = (Center, Center, Center))
    @test_throws ArgumentError UMomentumEquation.ViscousDissipation(model; location = (Center, Center, Center))
    @test_throws ArgumentError UMomentumEquation.ImmersedViscousDissipation(model; location = (Center, Center, Center))
    @test_throws ArgumentError UMomentumEquation.TotalViscousDissipation(model; location = (Center, Center, Center))
    @test_throws ArgumentError UMomentumEquation.StokesShear(model; location = (Center, Center, Center))
    @test_throws ArgumentError UMomentumEquation.StokesTendency(model; location = (Center, Center, Center))
    @test_throws ArgumentError UMomentumEquation.Forcing(model; location = (Center, Center, Center))
    @test_throws ArgumentError UMomentumEquation.Tendency(model; location = (Center, Center, Center))

    return nothing
end
#---

@testset "Momentum equation type-alias orthogonality" begin
    # Each kernel-specific type alias is parameterised on its underlying Oceananigans
    # kernel (div_𝐯u vs div_𝐯v vs div_𝐯w, …) and must be distinct across U/V/W so that
    # `isa` discriminates the momentum component being computed.
    @test UAdvection !== VAdvection && UAdvection !== WAdvection && VAdvection !== WAdvection
    @test UBuoyancyAcceleration !== VBuoyancyAcceleration && UBuoyancyAcceleration !== WBuoyancyAcceleration
    @test UCoriolisAcceleration !== VCoriolisAcceleration && UCoriolisAcceleration !== WCoriolisAcceleration
    @test UPressureGradient !== VPressureGradient # W has no PressureGradient
    @test UBarotropicPressureGradient !== VBarotropicPressureGradient # W has no BarotropicPressureGradient
    @test UViscousDissipation !== VViscousDissipation && UViscousDissipation !== WViscousDissipation
    @test UImmersedViscousDissipation !== VImmersedViscousDissipation && UImmersedViscousDissipation !== WImmersedViscousDissipation
    @test UTotalViscousDissipation !== VTotalViscousDissipation && UTotalViscousDissipation !== WTotalViscousDissipation
    @test UStokesShear !== VStokesShear && UStokesShear !== WStokesShear
    @test UStokesTendency !== VStokesTendency && UStokesTendency !== WStokesTendency
    @test UTendency !== VTendency && UTendency !== WTendency

    # Forcing is intentionally aliased to the generic KernelFunctionOperation in every
    # module (no kernel narrowing), so the prefixed Forcing aliases are the same type.
    @test UForcing === VForcing === WForcing
end

@testset "U-momentum equation diagnostics tests" begin
    @info "  Testing u-momentum diagnostics"
    for (grid_class, grid) in zip(keys(grids), values(grids))
        @info "    with $grid_class"
        for model_type in model_types
            @info "      with $model_type"
            # HFS uses the default `VectorInvariant` momentum advection. The diagnostic
            # `Advection` dispatches on the model type to wrap `U_dot_∇u` with whatever
            # scheme the model carries, so no override is needed. HFS has no
            # `stokes_drift` field, so we pass the Stokes-drift only to the NH model.
            model = model_type === HydrostaticFreeSurfaceModel ?
                model_type(grid; model_kwargs...) :
                model_type(grid; nh_model_kwargs...)

            @info "        Testing u-momentum terms"
            test_u_momentum_terms(model)

            @info "        Testing u-momentum field locations"
            test_u_momentum_field_locations(model)

            @info "        Testing u-momentum location validation"
            test_u_momentum_location_validation(model)
        end
    end

    @info "    Testing u-momentum budget closure on NonhydrostaticModel"
    test_u_momentum_budget_closure(underlying_regular_grid)

    @info "    Testing u-momentum budget closure on HydrostaticFreeSurfaceModel"
    test_u_momentum_hfs_budget_closure(underlying_regular_grid)
end
