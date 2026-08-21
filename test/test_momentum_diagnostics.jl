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
bc_function(x, y, z, t) = sin(t)
immersed_bc = FluxBoundaryCondition(bc_function)
component_bcs = FieldBoundaryConditions(immersed=immersed_bc)

# HydrostaticFreeSurfaceModel has no prognostic w -- it diagnoses w from continuity -- so it
# rejects a forcing or boundary condition keyed on w, while the w-momentum tests need both.
uv_forcing  = (; u = Forcing(forcing_function), v = Forcing(forcing_function))
uvw_forcing = (; uv_forcing..., w = Forcing(forcing_function))
uv_bcs      = (; u = component_bcs, v = component_bcs)
uvw_bcs     = (; uv_bcs..., w = component_bcs)

stokes_drift = UniformStokesDrift(∂z_uˢ = (z, t) -> exp(z) * cos(t),
                                  ∂z_vˢ = (z, t) -> exp(z) * sin(t),
                                  ∂t_uˢ = (z, t) -> -exp(z) * sin(t),
                                  ∂t_vˢ = (z, t) ->  exp(z) * cos(t))

common_kwargs    = (; tracers, buoyancy=BuoyancyTracer(), coriolis=FPlane(f=1e-4))
hfs_model_kwargs = (; common_kwargs..., forcing = uv_forcing,  boundary_conditions = uv_bcs)
# stokes_drift only applies to NonhydrostaticModel
nh_model_kwargs  = (; common_kwargs..., forcing = uvw_forcing, boundary_conditions = uvw_bcs, stokes_drift)
#---

#+++ Test options
grids = Dict("regular grid" => regular_grid,
             "stretched grid" => stretched_grid)
#---

#+++ Direction specs
# One entry per momentum component. Everything that differs between the u-, v- and w-momentum
# equations is captured here, so each test function below is written once and run against all
# three specs rather than being copied per component.
const DIRECTIONS = (
    (; component = :u, prefix = :U, eq = UMomentumEquation, loc = (Face, Center, Center),
       model_types = (NonhydrostaticModel, HydrostaticFreeSurfaceModel)),

    (; component = :v, prefix = :V, eq = VMomentumEquation, loc = (Center, Face, Center),
       model_types = (NonhydrostaticModel, HydrostaticFreeSurfaceModel)),

    # Only NonhydrostaticModel evolves w prognostically; HydrostaticFreeSurfaceModel diagnoses w
    # from continuity and has no w-momentum equation.
    (; component = :w, prefix = :W, eq = WMomentumEquation, loc = (Center, Center, Face),
       model_types = (NonhydrostaticModel,)),
)

# u and v carry the two pressure-gradient terms; w carries neither, because Oceananigans treats
# the vertical hydrostatic balance as a property of the pressure-projection step rather than a
# term in `w_velocity_tendency`.
has_pressure_gradient(dir) = dir.component !== :w

const BASE_TERMS     = (:Advection, :BuoyancyAcceleration, :CoriolisAcceleration,
                        :ViscousDissipation, :ImmersedViscousDissipation, :TotalViscousDissipation)
const PRESSURE_TERMS = (:PressureGradient, :BarotropicPressureGradient)
const STOKES_TERMS   = (:StokesShear, :StokesTendency)  # NonhydrostaticModel only: HFS has no stokes_drift
const TAIL_TERMS     = (:Forcing, :Tendency)

terms(dir) = (BASE_TERMS...,
              (has_pressure_gradient(dir) ? PRESSURE_TERMS : ())...,
              STOKES_TERMS..., TAIL_TERMS...)

# `UMomentumEquation.Advection` and friends are type aliases, so the module member *is* the type to
# construct with and to test membership against. The exported prefixed aliases (`UAdvection`,
# `VTendency`, …) follow the component prefix exactly, so they are looked up rather than tabulated.
term_type(dir, term) = getfield(dir.eq, term)
alias(dir, term)     = getfield(Oceanostics, Symbol(dir.prefix, term))
velocity(dir, model) = getfield(model.velocities, dir.component)

# Every location that is not this component's own must be rejected at construction.
const ALL_LOCATIONS = ((Face, Center, Center), (Center, Face, Center),
                       (Center, Center, Face), (Center, Center, Center))
wrong_locations(dir) = filter(!=(dir.loc), ALL_LOCATIONS)
#---

#+++ Explicit-argument forms
# The argument list each term accepts after `model`, i.e. the explicit form
# `Term(model, args...)` as opposed to the bare `Term(model)`. This is the one genuinely
# per-term part of the suite; it is shared across components apart from the velocity component
# picked out for the immersed boundary condition and the forcing.
function long_form_args(dir, model, term)
    hfs         = model isa HydrostaticFreeSurfaceModel
    immersed_bc = velocity(dir, model).boundary_conditions.immersed
    component_forcing = getfield(model.forcing, dir.component)

    term === :Advection               && return (model.velocities, hfs ? model.advection.momentum : model.advection)
    term === :BuoyancyAcceleration    && return (model.buoyancy, model.tracers)
    term === :CoriolisAcceleration    && return (model.coriolis, model.velocities)
    term === :PressureGradient        && return (hfs ? model.pressure.pHY′ : model.pressures.pHY′,)
    term === :BarotropicPressureGradient && return (hasfield(typeof(model), :free_surface) ? model.free_surface : nothing,)
    term === :ViscousDissipation      && return (model.closure, model.closure_fields, model.clock, fields(model), model.buoyancy)
    term === :ImmersedViscousDissipation && return (model.velocities, immersed_bc, model.closure, model.closure_fields, model.clock, fields(model))
    term === :TotalViscousDissipation && return (model.velocities, immersed_bc, model.closure, model.closure_fields, model.clock, fields(model), model.buoyancy)
    term === :StokesShear             && return (model.stokes_drift, model.velocities, model.clock.time)
    term === :StokesTendency          && return (model.stokes_drift, model.clock.time)
    term === :Forcing                 && return (component_forcing, model.clock, fields(model))

    term === :Tendency && return hfs ?
        (model.advection.momentum, model.coriolis, model.closure, immersed_bc, model.velocities,
         model.free_surface, model.tracers, model.buoyancy, model.closure_fields, model.pressure.pHY′,
         model.auxiliary_fields, model.vertical_coordinate, model.clock, component_forcing) :
        (model.advection, model.coriolis, model.stokes_drift, model.closure, immersed_bc, model.buoyancy,
         model.background_fields, model.velocities, model.tracers, model.auxiliary_fields,
         model.closure_fields, model.pressures.pHY′, model.clock, component_forcing)

    error("no explicit-argument form recorded for $term")
end
#---

#+++ Test functions (shared across components)
function test_momentum_terms(dir, model)
    for term in terms(dir)
        T = term_type(dir, term)

        # HFS carries no `stokes_drift`, so the Stokes terms have to refuse it rather than build.
        if term in STOKES_TERMS && model isa HydrostaticFreeSurfaceModel
            @test_throws ArgumentError T(model)
            continue
        end

        for op in (T(model, long_form_args(dir, model, term)...), T(model))
            @test op isa T
            @test op isa alias(dir, term)
            @test Field(op) isa Field
        end
    end

    # Forcing is narrowed on a per-module wrapper kernel (forcing_fcc/cfc/ccf) rather than left as
    # the generic KFO, so this component's Forcing must not satisfy another component's alias.
    FORC = term_type(dir, :Forcing)(model)
    for other in DIRECTIONS
        other.component === dir.component && continue
        @test !(FORC isa alias(other, :Forcing))
    end

    return nothing
end

function test_momentum_field_locations(dir, model)
    for term in terms(dir)
        term in STOKES_TERMS && model isa HydrostaticFreeSurfaceModel && continue
        @test location(term_type(dir, term)(model)) == dir.loc
    end
    return nothing
end

function test_momentum_location_validation(dir, model)
    for term in terms(dir), loc in wrong_locations(dir)
        @test_throws ArgumentError term_type(dir, term)(model; location = loc)
    end
    return nothing
end
#---

#+++ Budget closure
# Every NH budget model is identical apart from which component is forced, so the setup is shared
# and each component supplies only the sign combination reproducing its own `*_velocity_tendency`.
function nh_budget_model(grid, component; kw...)
    sd = UniformStokesDrift(∂z_uˢ = (z, t) -> exp(z) * cos(t),
                            ∂z_vˢ = (z, t) -> exp(z) * sin(t),
                            ∂t_uˢ = (z, t) -> -exp(z) * sin(t),
                            ∂t_vˢ = (z, t) ->  exp(z) * cos(t))
    model = NonhydrostaticModel(grid; tracers = :b,
                                      buoyancy = BuoyancyTracer(),
                                      coriolis = FPlane(f = 1e-4),
                                      stokes_drift = sd,
                                      closure = ScalarDiffusivity(ν = 1e-4, κ = 1e-4),
                                      forcing = NamedTuple{(component,)}((Forcing((x, y, z, t) -> cos(t)),)),
                                      kw...)
    set!(model, u = (x, y, z) -> sin(2π*x) * cos(2π*y) * exp(z),
                v = (x, y, z) -> cos(2π*x) * sin(2π*y) * exp(z),
                w = (x, y, z) -> sin(2π*x) * sin(2π*z),
                b = (x, y, z) -> sin(2π*z))
    update_state!(model) # populates model.pressures.pHY′ from b (or leaves it nothing)
    return model
end

# `pressure_splitting=true` keeps NH's default hydrostatic pressure anomaly field; `false` disables
# the split (`hydrostatic_pressure_anomaly = nothing`), which only w is sensitive to.
function test_momentum_budget_closure(dir, grid; pressure_splitting = true)
    model = pressure_splitting ? nh_budget_model(grid, dir.component) :
                                 nh_budget_model(grid, dir.component; hydrostatic_pressure_anomaly = nothing)
    term(t) = term_type(dir, t)(model)

    # Matching Oceananigans' `*_velocity_tendency` sign convention:
    #   G_u = -ADV + BUOY - COR - PRES - TVISC + STOKES_SHEAR + STOKES_TENDENCY + FORCING
    # Unlike u/v, `w_velocity_tendency` has no `-∂z(pHY′)` term, and whether buoyancy appears at all
    # depends on the split (via `maybe_z_dot_g_bᶜᶜᶠ`): with splitting it is assumed absorbed by pHY′.
    budget = if dir.component === :w
        pressure_splitting ?
            Field(-term(:Advection) - term(:CoriolisAcceleration) - term(:TotalViscousDissipation)
                  + term(:StokesShear) + term(:StokesTendency) + term(:Forcing)) :
            Field(-term(:Advection) + term(:BuoyancyAcceleration) - term(:CoriolisAcceleration)
                  - term(:TotalViscousDissipation) + term(:StokesShear) + term(:StokesTendency) + term(:Forcing))
    else
        Field(-term(:Advection) + term(:BuoyancyAcceleration) - term(:CoriolisAcceleration)
              - term(:PressureGradient) - term(:TotalViscousDissipation)
              + term(:StokesShear) + term(:StokesTendency) + term(:Forcing))
    end

    @test interior(budget) ≈ interior(Field(term(:Tendency)))
    return nothing
end

function test_momentum_hfs_budget_closure(dir, grid)
    # The default `momentum_advection` (`VectorInvariant`) is used — the diagnostic `Advection`
    # dispatches on the model type to wrap `U_dot_∇u` with whatever scheme the model carries.
    model = HydrostaticFreeSurfaceModel(grid; tracers = :b,
                                              buoyancy = BuoyancyTracer(),
                                              coriolis = FPlane(f = 1e-4),
                                              closure = ScalarDiffusivity(ν = 1e-4, κ = 1e-4),
                                              forcing = NamedTuple{(dir.component,)}((Forcing((x, y, z, t) -> cos(t)),)))
    set!(model, u = (x, y, z) -> sin(2π*x) * cos(2π*y) * exp(z),
                v = (x, y, z) -> cos(2π*x) * sin(2π*y) * exp(z),
                b = (x, y, z) -> sin(2π*z))
    update_state!(model) # populates model.pressure.pHY′ from b
    term(t) = term_type(dir, t)(model)

    # G_u = -ADV - BARO - COR - PRES - TVISC + FORCING. HFS has no Stokes terms and no
    # BuoyancyAcceleration term (buoyancy is absorbed into pHY′). BARO is the explicit barotropic
    # free-surface gradient; on ImplicitFreeSurface it returns zero (the contribution is handled
    # inside the pressure solve) so the budget closes without it, but including it keeps the
    # formula correct for ExplicitFreeSurface too.
    budget = Field(-term(:Advection) - term(:BarotropicPressureGradient) - term(:CoriolisAcceleration)
                   - term(:PressureGradient) - term(:TotalViscousDissipation) + term(:Forcing))

    @test interior(budget) ≈ interior(Field(term(:Tendency)))
    return nothing
end

function test_w_momentum_hfs_unsupported()
    grid = first(values(grids))
    hfs_model = HydrostaticFreeSurfaceModel(grid; tracers, buoyancy=BuoyancyTracer())
    @test_throws ArgumentError WMomentumEquation.Tendency(hfs_model)
    @test_throws ArgumentError WMomentumEquation.Forcing(hfs_model)
    @test_throws ArgumentError WMomentumEquation.StokesShear(hfs_model)
    @test_throws ArgumentError WMomentumEquation.StokesTendency(hfs_model)
    return nothing
end
#---

@testset "Momentum equation type-alias orthogonality" begin
    # Each kernel-specific type alias is parameterised on its underlying Oceananigans kernel
    # (div_𝐯u vs div_𝐯v vs div_𝐯w, …) and must be distinct across U/V/W so that `isa`
    # discriminates the momentum component being computed. `PressureGradient` and
    # `BarotropicPressureGradient` exist for u and v only.
    for term in (BASE_TERMS..., STOKES_TERMS..., TAIL_TERMS...)
        aliases = [alias(dir, term) for dir in DIRECTIONS]
        @test allunique(aliases)
    end
    for term in PRESSURE_TERMS
        @test alias(DIRECTIONS[1], term) !== alias(DIRECTIONS[2], term) # W has no $term
    end

    # None of the per-module Forcing aliases is the generic KernelFunctionOperation.
    for dir in DIRECTIONS
        @test alias(dir, :Forcing) !== KernelFunctionOperation
    end
    @test TracerForcing !== KernelFunctionOperation
end

@testset "Momentum equation diagnostics tests" begin
    for dir in DIRECTIONS
        @info "  Testing $(dir.component)-momentum diagnostics"
        for (grid_class, grid) in zip(keys(grids), values(grids))
            @info "    with $grid_class"
            for model_type in dir.model_types
                @info "      with $model_type"
                # HFS uses the default `VectorInvariant` momentum advection, and has no
                # `stokes_drift` field, so the Stokes drift goes only to the NH model.
                model = model_type === HydrostaticFreeSurfaceModel ?
                    model_type(grid; hfs_model_kwargs...) :
                    model_type(grid; nh_model_kwargs...)

                @info "        Testing $(dir.component)-momentum terms"
                test_momentum_terms(dir, model)

                @info "        Testing $(dir.component)-momentum field locations"
                test_momentum_field_locations(dir, model)

                @info "        Testing $(dir.component)-momentum location validation"
                test_momentum_location_validation(dir, model)
            end
        end

        @info "    Testing $(dir.component)-momentum budget closure on NonhydrostaticModel"
        test_momentum_budget_closure(dir, underlying_regular_grid)

        if HydrostaticFreeSurfaceModel in dir.model_types
            @info "    Testing $(dir.component)-momentum budget closure on HydrostaticFreeSurfaceModel"
            test_momentum_hfs_budget_closure(dir, underlying_regular_grid)
        end
    end

    @info "    Testing that w-momentum Tendency is unsupported on HydrostaticFreeSurfaceModel"
    test_w_momentum_hfs_unsupported()

    @info "    Testing w-momentum budget closure on NonhydrostaticModel (without pressure splitting)"
    test_momentum_budget_closure(DIRECTIONS[3], underlying_regular_grid; pressure_splitting = false)
end
