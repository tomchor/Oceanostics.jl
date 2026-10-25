using Test
using Oceananigans: fill_halo_regions!
using Oceananigans.Fields: ZeroField
using Oceananigans.TurbulenceClosures: AbstractScalarDiffusivity, ThreeDimensionalFormulation

using Oceanostics
using Oceanostics.KineticEnergyEquation: KineticEnergyTendency, KineticEnergyAdvection, KineticEnergyStress,
                                         KineticEnergyForcing, KineticEnergyPressureRedistribution

# Include common test utilities
include("test_utils.jl")

#+++ Helpers
# Face values averaged onto the cell centres the way the `ℑ` operators do it, on the test grids:
# periodic in `x` and `y`, where `interior` holds `N` faces, and bounded in `z`, where it holds `N + 1`
ℑx_to_centers(a) = (a .+ circshift(a, (-1, 0, 0))) ./ 2
ℑy_to_centers(a) = (a .+ circshift(a, (0, -1, 0))) ./ 2
ℑz_to_centers(a) = (a[:, :, 1:end-1] .+ a[:, :, 2:end]) ./ 2

# A hydrostatic model steps `u` and `v` and diagnoses `w`, so its kinetic energy and the terms of its
# budget have no `w` row
has_w_equation(model) = model isa NonhydrostaticModel

velocity_arrays(model) = map(cpu_interior, model.velocities)
#---

#+++ Test functions
"""
`eₖ = ½uᵢuᵢ` with each square formed on the face its velocity lives on and averaged onto the centre.
A hydrostatic model's `eₖ` is `½(u² + v²)`; the flow there has a nonzero diagnosed `w`, so that leaving
it out is what gets checked.
"""
function test_kinetic_energy(grid; model_type)
    model = model_type(grid)
    u₀(x, y, z) = sin(2π * x) + z * cos(2π * y)
    v₀(x, y, z) = cos(2π * x) * sin(2π * y)

    if has_w_equation(model)
        set!(model, u=u₀, v=v₀, w=(x, y, z) -> sin(2π * y), enforce_incompressibility=false)
    else
        set!(model, u=u₀, v=v₀)
    end

    eₖ = KineticEnergyEquation.KineticEnergy(model)
    @test eₖ isa KineticEnergy

    u, v, w = velocity_arrays(model)
    w² = has_w_equation(model) ? ℑz_to_centers(w .^ 2) : 0
    @test cpu_interior(eₖ) ≈ (ℑx_to_centers(u .^ 2) .+ ℑy_to_centers(v .^ 2) .+ w²) ./ 2
    @test maximum(abs, w) > 0

    return nothing
end

"""
For `u = C₁y`, `v = C₂` and `w = 0`, `uᵢ∂ⱼ(uⱼuᵢ) = uv∂y u = C₁²C₂y`. A hydrostatic model's default
`VectorInvariant` scheme writes the same advection as `-ζv + ∂ₓ(½u⃗²)` in the `u` equation and
`ζu + ∂y(½u⃗²)` in the `v` one, and comes out the same. The rows next to the periodic seam in `y`,
where `u` jumps, are left out.
"""
function test_kinetic_energy_advection(grid; model_type)
    model = model_type(grid)
    C₁ = 2; C₂ = 3
    set!(model, u=(x, y, z) -> C₁ * y, v=C₂)

    ADV = KineticEnergyAdvection(model)
    @test ADV isa KineticEnergyAdvection

    yᶜ = Array(ynodes(grid, Center()))
    rows = 2:grid.Ny-1
    @test cpu_interior(ADV)[:, rows, :] ≈ C₁^2 * C₂ .* repeat(reshape(yᶜ[rows], 1, :, 1), grid.Nx, 1, grid.Nz)

    return nothing
end

"""
`u = U₀ sin(2πy)` under a constant viscosity `ν` is a pure `τ₁₂ = -ν∂y u` flow, and the second
difference of a sine on a regular grid is known exactly, so `∂ⱼτ₁ⱼ = 4ν sin²(πΔy)/Δy² u` and
`uᵢ∂ⱼτᵢⱼ = 4ν sin²(πΔy)/Δy² u²` in every cell, on both test grids (they are regular in `y`).
"""
function test_kinetic_energy_stress(grid; model_type)
    ν = 1e-2; U₀ = 2
    model = model_type(grid; closure=ScalarDiffusivity(; ν))
    set!(model, u=(x, y, z) -> U₀ * sin(2π * y))

    DIFF = KineticEnergyStress(model)
    @test DIFF isa KineticEnergyStress

    Δy = grid.Ly / grid.Ny
    u = cpu_interior(model.velocities.u)   # independent of x, so its face and centre values agree
    @test cpu_interior(DIFF) ≈ 4ν * sin(π * Δy)^2 / Δy^2 .* u .^ 2

    return nothing
end

"""
A linear drag `Fᵤᵢ = -uᵢ` on every momentum equation makes `uᵢFᵤᵢ = -uᵢuᵢ`, each product formed on its
own face. A hydrostatic model has no `w` equation to force, and the `w` it diagnoses from this random
flow stays out of the sum.
"""
function test_kinetic_energy_forcing(grid; model_type)
    drag(x, y, z, t, ψ) = -ψ
    forcing = (u = Forcing(drag, field_dependencies=:u),
               v = Forcing(drag, field_dependencies=:v))

    if model_type == NonhydrostaticModel
        model = NonhydrostaticModel(grid; forcing=merge(forcing, (; w = Forcing(drag, field_dependencies=:w))))
        set!(model, u=grid_noise, v=grid_noise, w=grid_noise)
    else
        model = HydrostaticFreeSurfaceModel(grid; forcing)
        set!(model, u=grid_noise, v=grid_noise)
    end

    FORC = KineticEnergyForcing(model)
    @test FORC isa KineticEnergyForcing

    u, v, w = velocity_arrays(model)
    w² = has_w_equation(model) ? ℑz_to_centers(w .^ 2) : 0
    @test cpu_interior(FORC) ≈ .-(ℑx_to_centers(u .^ 2) .+ ℑy_to_centers(v .^ 2) .+ w²)
    @test maximum(abs, w) > 0

    return nothing
end

"""
With `p = P₀ sin(2πx) + P₁z` prescribed, `u = U₀ sin(2πx)` and, on a nonhydrostatic model, `w = W₀`,
`uᵢ∂ᵢp` is the `x` and `z` rows formed on their faces. `∂z p` is `P₁` on every interior face and drops
out on the walls, where the pressure's no-flux halo zeroes it. A hydrostatic model leaves the `z` row
out, although the `w` it diagnoses from `u` is nonzero.

On a nonhydrostatic model the default pressure is the total one, whose work closes the full budget
`∂ₜeₖ = -ADV - DIFF + FORC + wb - uᵢ∂ᵢp`: the nonhydrostatic pressure plus the hydrostatic anomaly when
the model carries one, and the nonhydrostatic pressure alone otherwise. On a hydrostatic model it is
the anomaly, whose place in the budget `test_kinetic_energy_budget` checks.
"""
function test_pressure_redistribution(grid; model_type)
    U₀ = 2; W₀ = 3; P₀ = 5; P₁ = 7
    model = model_type(grid; buoyancy=BuoyancyTracer(), tracers=:b)

    if has_w_equation(model)
        set!(model, u=(x, y, z) -> U₀ * sin(2π * x), w=W₀, enforce_incompressibility=false)
    else
        set!(model, u=(x, y, z) -> U₀ * sin(2π * x))
    end

    p = CenterField(grid)
    set!(p, (x, y, z) -> P₀ * sin(2π * x) + P₁ * z)
    fill_halo_regions!(p)

    PR = KineticEnergyPressureRedistribution(model; pressure=p)
    @test PR isa KineticEnergyPressureRedistribution

    Δx = grid.Lx / grid.Nx
    zᶜ = Array(znodes(grid, Center()))
    pᶜ = cpu_interior(p)
    u, v, w = velocity_arrays(model)

    ∂x_p = (pᶜ .- circshift(pᶜ, (1, 0, 0))) ./ Δx                         # on the u faces
    ∂z_p = zeros(size(w))                                                  # on the w faces, zero on the walls
    ∂z_p[:, :, 2:end-1] .= diff(pᶜ, dims=3) ./ reshape(diff(zᶜ), 1, 1, :)

    w∂z_p = has_w_equation(model) ? ℑz_to_centers(w .* ∂z_p) : 0
    @test cpu_interior(PR) ≈ ℑx_to_centers(u .* ∂x_p) .+ w∂z_p
    @test maximum(abs, w) > 0

    if has_w_equation(model)
        pNHS, pHY′ = model.pressures.pNHS, model.pressures.pHY′
        set!(pNHS, (x, y, z) -> sin(2π * y) * z)
        set!(pHY′, (x, y, z) -> cos(2π * x) + z^2)
        fill_halo_regions!(pNHS); fill_halo_regions!(pHY′)
        @test cpu_interior(KineticEnergyPressureRedistribution(model)) ≈
              cpu_interior(KineticEnergyPressureRedistribution(model; pressure=pNHS)) .+
              cpu_interior(KineticEnergyPressureRedistribution(model; pressure=pHY′))

        unbuoyant = NonhydrostaticModel(grid)
        set!(unbuoyant, u=(x, y, z) -> U₀ * sin(2π * x))
        set!(unbuoyant.pressures.pNHS, (x, y, z) -> sin(2π * y) * z)
        fill_halo_regions!(unbuoyant.pressures.pNHS)
        @test isnothing(unbuoyant.pressures.pHY′)
        @test cpu_interior(KineticEnergyPressureRedistribution(unbuoyant)) ≈
              cpu_interior(KineticEnergyPressureRedistribution(unbuoyant; pressure=unbuoyant.pressures.pNHS))
    end

    return nothing
end

"""
`uᵢbᵢ` is `w` times the buoyancy on the `w` faces, averaged onto the centre, so a uniform `b₀` gives
`b₀ ℑz(w)`. A nonhydrostatic model takes a prescribed `w`; a hydrostatic one diagnoses it from
continuity, here from `u = U₀ sin(2πx)`. The turbulent conversion `w′b′`, formed from fluctuations about
the domain mean, vanishes in both cases since `b` has none, up to the roundoff the mean of `b` carries on
a stretched grid, which the large `w′` of the hydrostatic flow multiplies; hence the relative bound.
"""
function test_potential_to_kinetic_energy_conversion(grid; model_type)
    w₀ = 2; b₀ = 3; U₀ = 2
    model = model_type(grid; buoyancy=BuoyancyTracer(), tracers=:b)

    if has_w_equation(model)
        set!(model, w=w₀, b=b₀, enforce_incompressibility=false)
    else
        set!(model, u=(x, y, z) -> U₀ * sin(2π * x), b=b₀)
    end

    wb = KineticEnergyEquation.PotentialEnergyConversion(model)
    @test wb isa KineticEnergyEquation.PotentialEnergyConversion

    w = cpu_interior(model.velocities.w)
    wb_values = cpu_interior(wb)
    @test wb_values ≈ b₀ .* ℑz_to_centers(w)
    @test maximum(abs, w) > 0

    w′ = Field(model.velocities.w - Field(Average(model.velocities.w)))
    b′ = Field(model.tracers.b - Field(Average(model.tracers.b)))
    w′b′ = KineticEnergyEquation.PotentialEnergyConversion(model, tracers=(b=b′,),
                                                           velocities=(u=model.velocities.u, v=model.velocities.v, w=w′))
    @test w′b′ isa KineticEnergyEquation.PotentialEnergyConversion
    @test maximum(abs, cpu_interior(w′b′)) < 1e-12 * maximum(abs, wb_values)

    return nothing
end

"""
`u = (∂u/∂z) z` under a constant viscosity `ν` dissipates `εₖ = ν(∂u/∂z)²` away from the walls, where the
free-slip condition zeroes the shear on the boundary face. The isotropic form gets exactly that from the
face shears. The general form `-∂ⱼuᵢτᵢⱼ` weights the two faces of a cell by the spacing between the
centres either side of them, which a stretched grid makes unequal to the cell height, so its interior
value is `ν(∂u/∂z)² (Δzᶠₖ + Δzᶠₖ₊₁) / 2Δzᶜₖ`, which is `ν(∂u/∂z)²` again on a regular grid.
"""
function test_kinetic_energy_dissipation_rates(grid; model_type)
    ν = 1e-2; dudz = 2
    model = model_type(grid; closure=ScalarDiffusivity(; ν))
    set!(model, u=(x, y, z) -> dudz * z)

    ε = KineticEnergyEquation.KineticEnergyDissipationRate(model)
    ε_iso = KineticEnergyEquation.KineticEnergyIsotropicDissipationRate(model)
    @test ε isa KineticEnergyEquation.KineticEnergyDissipationRate
    @test ε_iso isa KineticEnergyEquation.KineticEnergyIsotropicDissipationRate

    zᶜ = Array(znodes(grid, Center()))
    zᶠ = Array(znodes(grid, Face()))
    Δzᶜ = diff(zᶠ)   # cell heights
    Δzᶠ = diff(zᶜ)   # spacing between centres, on the interior faces 2:Nz
    k = 2:grid.Nz-1
    face_weights = reshape((Δzᶠ[k .- 1] .+ Δzᶠ[k]) ./ (2 .* Δzᶜ[k]), 1, 1, :)

    @test cpu_interior(ε_iso)[:, :, k] ≈ fill(ν * dudz^2, grid.Nx, grid.Ny, length(k))
    @test cpu_interior(ε)[:, :, k] ≈ ν * dudz^2 .* repeat(face_weights, grid.Nx, grid.Ny, 1)

    return nothing
end

"""
`KineticEnergyTendency` is `uᵢGᵢ` off the model's own momentum tendency kernels, so the other terms have
to rebuild it cell by cell, to machine precision,

    uᵢGᵢ = -ADV - DIFF + FORC + wb - uᵢ∂ᵢp′ ,

where `p′` is the hydrostatic pressure anomaly, when the model carries one, and with no Coriolis, Stokes
drift or background flow in play. A nonhydrostatic model with an anomaly moves the buoyancy out of the
`w` equation and into `p′`, whose `z` row `w∂z p′` is exactly `wb`, since Oceananigans integrates `p′`
from the same face buoyancy; so the split holds with an anomaly and without one, and both are checked.

A hydrostatic model's `uₕGₕ` has no `w` row, so there `-ADV - DIFF + FORC - uₕ⋅∇ₕp′` is the whole of it
(its free surface treats the barotropic pressure gradient implicitly, so that is not in `uₕGₕ` either).
`wb` is then the `z` row of the pressure work, which is checked separately. That needs the diagnosed `w`
to vanish at the surface as well as at the bottom, so the flow's depth mean is removed first.

Every term has to be at least 1% of the tendency, or the identity would be checking nothing.
"""
function test_kinetic_energy_budget(grid; model_type)
    drag(x, y, z, t, ψ) = -ψ
    flow(x, y, z) = 1e-1 * sin(2π * x) * cos(2π * y) * sin(π * z) + 5e-2 * cos(2π * z)

    forcing = (u = Forcing(drag, field_dependencies=:u),
               v = Forcing(drag, field_dependencies=:v))
    kwargs = (; buoyancy = BuoyancyTracer(), tracers = :b, closure = ScalarDiffusivity(ν=1e-2, κ=1e-2))

    if model_type == NonhydrostaticModel
        forcing = merge(forcing, (; w = Forcing(drag, field_dependencies=:w)))

        for hydrostatic_pressure_anomaly in (CenterField(grid), nothing)
            model = NonhydrostaticModel(grid; forcing, hydrostatic_pressure_anomaly, kwargs...)
            set!(model, u=flow, v=flow, w=flow, b=flow)

            ADV, DIFF, FORC, wb = map(cpu_interior, (KineticEnergyAdvection(model), KineticEnergyStress(model),
                                                     KineticEnergyForcing(model), PotentialToKineticEnergyConversion(model)))
            p′ = model.pressures.pHY′
            PR = isnothing(p′) ? zero(ADV) : cpu_interior(KineticEnergyPressureRedistribution(model; pressure=p′))

            tendency = cpu_interior(KineticEnergyTendency(model))
            @test maximum(abs, tendency .- (.-ADV .- DIFF .+ FORC .+ wb .- PR)) < 1e-12 * maximum(abs, tendency)

            for term in (ADV, DIFF, FORC, wb, (isnothing(p′) ? () : (PR,))...)
                @test maximum(abs, term) > 1e-2 * maximum(abs, tendency)
            end
        end
    else
        model = HydrostaticFreeSurfaceModel(grid; forcing, kwargs...)
        set!(model, u=flow, v=flow, b=flow)
        remove_barotropic_flow!(model)

        ADV, DIFF, FORC, PR, wb = map(cpu_interior, (KineticEnergyAdvection(model), KineticEnergyStress(model),
                                                     KineticEnergyForcing(model), KineticEnergyPressureRedistribution(model),
                                                     PotentialToKineticEnergyConversion(model)))

        tendency = cpu_interior(KineticEnergyTendency(model))
        @test maximum(abs, tendency .- (.-ADV .- DIFF .+ FORC .- PR)) < 1e-12 * maximum(abs, tendency)

        for term in (ADV, DIFF, FORC, PR, wb)
            @test maximum(abs, term) > 1e-2 * maximum(abs, tendency)
        end

        # The `z` row of the work done by the hydrostatic pressure anomaly is the conversion `wb`
        u⃗∇p′ = cpu_interior(KineticEnergyPressureRedistribution(model; velocities=model.velocities))
        @test maximum(abs, u⃗∇p′ .- PR .- wb) < 1e-12 * maximum(abs, wb)
    end

    return nothing
end

# Random velocities without a depth mean, so the depth-integrated transport vanishes and with it the
# diagnostic w of a HydrostaticFreeSurfaceModel at the top, where the flux of u would otherwise be -ν∂ₓw
function set_velocities_without_depth_mean!(model)
    set!(model, u=grid_noise, v=grid_noise)
    remove_barotropic_flow!(model)
    return nothing
end

# Each velocity with a prognostic equation times the viscous term of that equation, from the momentum
# modules, whose budgets are checked against the model's own tendencies
function velocity_times_viscous_term(model, model_fields=fields(model))
    u, v, w = model.velocities
    args = (model.closure, model.closure_fields, model.clock, model_fields, model.buoyancy)
    u∂ⱼτ₁ⱼ = Field(u * UMomentumEquation.ViscousDissipation(model, args...))
    v∂ⱼτ₂ⱼ = Field(v * VMomentumEquation.ViscousDissipation(model, args...))
    model isa HydrostaticFreeSurfaceModel && return Field(@at (Center, Center, Center) u∂ⱼτ₁ⱼ + v∂ⱼτ₂ⱼ)
    w∂ⱼτ₃ⱼ = Field(w * WMomentumEquation.ViscousDissipation(model, args...))
    return Field(@at (Center, Center, Center) u∂ⱼτ₁ⱼ + v∂ⱼτ₂ⱼ + w∂ⱼτ₃ⱼ)
end

function test_ke_stress_and_dissipation_rate_values(grid; model_type=NonhydrostaticModel, closure=ScalarDiffusivity(ν=1))
    model = model_type(grid; closure, buoyancy=BuoyancyTracer(), tracers=:b)
    set_velocities_without_depth_mean!(model)

    DIFF = Field(KineticEnergyEquation.KineticEnergyStress(model))
    @test interior(DIFF) ≈ interior(velocity_times_viscous_term(model))

    # No momentum crosses the boundaries, so the transport divergence integrates to zero and the stress
    # term and the dissipation rate have the same volume integral
    εₖ = KineticEnergyEquation.KineticEnergyDissipationRate(model)
    @test volume_integral(DIFF) ≈ volume_integral(εₖ)

    return nothing
end

# With ThreeDimensionalFormulation the fluxes τ₁₃ and τ₂₃ contain ∂ₓw and ∂ᵧw, so on a
# HydrostaticFreeSurfaceModel both diagnostics change if w is dropped from the fluxes as well as from
# the sum over i
function test_hydrostatic_fluxes_keep_w(grid)
    model = HydrostaticFreeSurfaceModel(grid; closure=ScalarDiffusivity(ν=1), buoyancy=BuoyancyTracer(), tracers=:b)
    set_velocities_without_depth_mean!(model)

    # What both volume integrals would be with w zeroed in the fluxes
    fields_without_w = merge(fields(model), (; w=ZeroField()))
    ∫DIFF_without_w = volume_integral(velocity_times_viscous_term(model, fields_without_w))

    @test !(volume_integral(KineticEnergyEquation.KineticEnergyStress(model)) ≈ ∫DIFF_without_w)
    @test !(volume_integral(KineticEnergyEquation.KineticEnergyDissipationRate(model)) ≈ ∫DIFF_without_w)

    return nothing
end

"""
Every closure the other test groups use has to build the closure-dependent terms on either model, and
the isotropic dissipation rate has to agree between its two signatures.
"""
function test_closure_dependent_terms(grid; model_type, closure)
    model = model_type(grid; closure, buoyancy=BuoyancyTracer(), tracers=:b)
    set!(model, u=grid_noise, v=grid_noise, b=grid_noise)

    for term in (KineticEnergyTendency(model), KineticEnergyStress(model),
                 KineticEnergyEquation.KineticEnergyDissipationRate(model))
        @test all(isfinite, cpu_interior(term))
    end

    if model.closure isa AbstractScalarDiffusivity{<:Any, <:ThreeDimensionalFormulation}
        # The convenience signature, KineticEnergyIsotropicDissipationRate(model; location)
        ε_iso1 = KineticEnergyEquation.KineticEnergyIsotropicDissipationRate(model)
        @test ε_iso1 isa KineticEnergyEquation.KineticEnergyIsotropicDissipationRate

        # The full signature, KineticEnergyIsotropicDissipationRate(u, v, w, closure, closure_fields,
        # fields(model), clock; location)
        u, v, w = model.velocities
        ε_iso2 = KineticEnergyEquation.KineticEnergyIsotropicDissipationRate(u, v, w, model.closure, model.closure_fields,
                                                                             fields(model), model.clock)
        @test ε_iso2 isa KineticEnergyEquation.KineticEnergyIsotropicDissipationRate
        @test cpu_interior(ε_iso1) ≈ cpu_interior(ε_iso2)

        # ... and with the location given explicitly
        ε_iso3 = KineticEnergyEquation.IsotropicDissipationRate(model; location = (Center, Center, Center))
        @test ε_iso3 isa KineticEnergyEquation.IsotropicDissipationRate
        @test cpu_interior(ε_iso3) ≈ cpu_interior(ε_iso1)
    end

    return nothing
end
#---

@testset "Kinetic Energy Equation tests" begin
    @info "  Testing Kinetic Energy Equation"
    for (grid_class, grid) in zip(keys(grids), values(grids))
        @info "    with $grid_class"
        for model_type in model_types
            @info "      with $model_type"

            @info "        Testing the value of every term"
            test_kinetic_energy(grid; model_type)
            test_kinetic_energy_advection(grid; model_type)
            test_kinetic_energy_stress(grid; model_type)
            test_kinetic_energy_forcing(grid; model_type)
            test_pressure_redistribution(grid; model_type)
            test_potential_to_kinetic_energy_conversion(grid; model_type)
            test_kinetic_energy_dissipation_rates(grid; model_type)

            @info "        Testing that the terms add up to the tendency"
            test_kinetic_energy_budget(grid; model_type)

            for closure in closures
                @info "        Testing the closure-dependent terms with $(summary(closure))"
                test_closure_dependent_terms(grid; model_type, closure)

                @info "          Testing stress term and dissipation rate values"
                test_ke_stress_and_dissipation_rate_values(grid; model_type, closure)
            end
        end

        @info "      Testing that the HydrostaticFreeSurfaceModel stress term and dissipation rate keep w in the fluxes"
        test_hydrostatic_fluxes_keep_w(grid)
    end
end
