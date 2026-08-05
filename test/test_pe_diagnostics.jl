using Test
using CUDA: has_cuda_gpu, @allowscalar

using Oceananigans
using Oceananigans.BuoyancyFormulations: buoyancy_perturbationᶜᶜᶜ, Zᶜᶜᶜ
using Oceananigans.Models: seawater_density, model_geopotential_height
using Oceananigans.TurbulenceClosures.Smagorinskys: LagrangianAveraging
using SeawaterPolynomials: RoquetEquationOfState, TEOS10EquationOfState

using Oceanostics

# Include common test utilities
include("test_utils.jl")

#+++ Test functions
function test_potential_energy_equation_terms_errors(model)

    @test_throws ArgumentError PotentialEnergyEquation.PotentialEnergy(model)
    @test_throws ArgumentError PotentialEnergyEquation.PotentialEnergy(model, geopotential_height = 0)

    return nothing
end

function test_potential_energy_equation_terms(model; geopotential_height = nothing)

    Eₚ = isnothing(geopotential_height) ? PotentialEnergyEquation.PotentialEnergy(model) :
                                          PotentialEnergyEquation.PotentialEnergy(model; geopotential_height)

    Eₚ_field = Field(Eₚ)
    @test Eₚ isa PotentialEnergyEquation.PotentialEnergy
    @test Eₚ_field isa Field

    if model.buoyancy isa PotentialEnergyEquation.BuoyancyBoussinesqEOSModel
        ρ = isnothing(geopotential_height) ? Field(seawater_density(model)) :
                                             Field(seawater_density(model; geopotential_height))

        Z = Field(model_geopotential_height(model))
        ρ₀ = model.buoyancy.formulation.equation_of_state.reference_density
        g = model.buoyancy.formulation.gravitational_acceleration

        @allowscalar begin
            true_value = (g / ρ₀) .* ρ.data .* Z.data
            @test isequal(Eₚ_field.data, true_value)
        end
    end

    return nothing
end

function test_PEbuoyancytracer_equals_PElineareos(grid)

    model_buoyancytracer = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b)
    model_lineareos = NonhydrostaticModel(grid; buoyancy=SeawaterBuoyancy(), tracers=(:S, :T))
    C_grad(x, y, z) = 0.01 * z
    set!(model_lineareos, S = C_grad, T = C_grad)
    linear_eos_buoyancy(grid, buoyancy, tracers) =
        KernelFunctionOperation{Center, Center, Center}(buoyancy_perturbationᶜᶜᶜ, grid, buoyancy, tracers)
    b_field = Field(linear_eos_buoyancy(model_lineareos.grid, model_lineareos.buoyancy.formulation, model_lineareos.tracers))
    set!(model_buoyancytracer, b = interior(b_field))
    pe_buoyancytracer = Field(PotentialEnergyEquation.PotentialEnergy(model_buoyancytracer))
    pe_lineareos = Field(PotentialEnergyEquation.PotentialEnergy(model_lineareos))

    @test all(interior(pe_buoyancytracer) .== interior(pe_lineareos))

    return nothing
end
#---

"""
`Φ = κ ∂b/∂z` is fixed by the buoyancy and the closure alone, so a linear profile pins its value, its
sign and the `κ` it picks up off the closure. The boundary treatment is the part worth guarding: no
buoyancy crosses a no-flux wall, which halves the cells against it and is exactly what makes the volume
integral telescope to `κ A [b(z_top) - b(z_bottom)]` however the interior is stirred. An interpolation
that mishandled the walls would still look right cell by cell in the interior and would break that.

That telescoping needs `κ` to come out of the sum, so a constant-`κ` test cannot see whether `Φ` is the
flux the closure actually returns or something reconstructed from the boundary values. The depth-varying
case below is what separates them, and it is the case the docstring warns not to apply the boundary form
to.
"""
function test_diffusive_buoyancy_flux(grid)

    κ = 1e-3
    model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b, closure=ScalarDiffusivity(; ν=1e-6, κ))

    Φ = PotentialEnergyDiffusiveVerticalBuoyancyFlux(model)
    @test Φ isa PotentialEnergyDiffusiveVerticalBuoyancyFlux

    set!(model, b = (x, y, z) -> 3z)
    Φ_column = Array(interior(Field(Φ)))[1, 1, :]
    @test all(Φ_column[2:end-1] .≈ 3κ)
    @test Φ_column[1] ≈ 3κ / 2 && Φ_column[end] ≈ 3κ / 2   # the no-flux walls halve the outermost cells

    set!(model, b = grid_noise)
    b = Array(interior(model.tracers.b))
    Δb = sum(b[:, :, end] .- b[:, :, 1]) / prod(size(b)[1:2])   # ⟨b(z_top) - b(z_bottom)⟩
    @test volume_integral(Φ) ≈ κ * grid.Lx * grid.Ly * Δb

    # With `κ` a function of depth, `b = 3z` still gives `3κ` on every face, so `Φ` is that averaged
    # onto the centre with the wall faces zeroed. The boundary form no longer applies.
    κ_of_z(x, y, z, t) = 1e-3 * (1 + 5z)
    varying = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b,
                                  closure=ScalarDiffusivity(ν=1e-6, κ=κ_of_z))
    set!(varying, b = (x, y, z) -> 3z)

    κ_face = [κ_of_z(0, 0, z, 0) for z in znodes(grid, Face())]
    κ_face[1] = κ_face[end] = 0                                # no buoyancy crosses the walls
    @test Array(interior(Field(PotentialEnergyDiffusiveVerticalBuoyancyFlux(varying))))[1, 1, :] ≈
          3 .* (κ_face[1:end-1] .+ κ_face[2:end]) ./ 2

    # ... and the constant-κ collapse is not merely inexact here, it is the wrong answer
    @test !isapprox(volume_integral(PotentialEnergyDiffusiveVerticalBuoyancyFlux(varying)),
                    1e-3 * grid.Lx * grid.Ly * 3 * grid.Lz; rtol = 0.1)

    # A model without a closure is legal and has no flux to read, so it has to be refused at
    # construction rather than at `compute!`, where the failure is a `MethodError` from inside the
    # kernel that names none of the caller's code.
    @test_throws "no closure" PotentialEnergyDiffusiveVerticalBuoyancyFlux(
        NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b))

    # Closures that compute their own κ do define a flux and must keep working.
    for closure in (SmagorinskyLilly(), AnisotropicMinimumDissipation())
        eddy = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b, closure)
        set!(eddy, b = grid_noise)
        @test volume_integral(PotentialEnergyDiffusiveVerticalBuoyancyFlux(eddy)) isa Number
    end

    return nothing
end

"""
The terms of the `eₚ = -bz` equation are `-z` times the terms of the equation the model steps for `b`,
each built on Oceananigans' own kernel for that term. Their sum is therefore the model's own buoyancy
tendency taken apart and put back together, so it has to match `PotentialEnergyTendency` to machine
precision — on any grid, with a background velocity and forcing in play, whatever the flow is doing.
That exactness is the whole point of keeping the `-z ×` form, and it is what the rearranged
`∂ⱼ(uⱼeₚ)` form would give up, so it is checked to `atol` rather than `rtol`.

The buoyancy deliberately has no `BackgroundField` here. Such a field puts `-∂ⱼ(uⱼB)` into the equation
the model steps, which `Tendency` picks up (it comes off the model's own kernel) but which no other
diagnostic in this module accounts for yet, so the split would not close. The background *velocity*
stays, since that one enters through `BuoyancyAdvection`'s total velocity and is covered.

The two integral identities are the other half of the arrangement: pulling `z` inside the derivative
splits `BuoyancyAdvection` into `-Advection - wb` and `BuoyancyDiffusion` into `Diffusion + Φ`, so over
a periodic domain `∫BuoyancyAdvection = -∫wb` and `∫BuoyancyDiffusion = ∫Φ`. Those come from the
continuum product rule and need `∂ⱼuⱼ = 0`, which is why the model is stepped first: the pressure
projection is what makes the velocity field divergence-free, and a random one straight out of `set!` is
not.

`Advection` and `Diffusion` are the transports themselves, built as genuine flux divergences rather
than by rearranging their `Buoyancy*` partners, so each telescopes and its volume integral vanishes to
roundoff rather than to truncation error. That is checked separately below.
"""
function test_potential_energy_budget_terms(grid)

    U(x, y, z, t) = 1e-2 * z                        # background velocity
    Fᵇ(x, y, z, t) = 1e-4 * sin(2π * z)             # sized to be a real term in the budget, see below

    model = NonhydrostaticModel(grid; buoyancy = BuoyancyTracer(), tracers = :b,
                                advection = Centered(order=4),
                                closure = ScalarDiffusivity(ν=1e-4, κ=1e-4),
                                background_fields = (; u = BackgroundField(U)),
                                forcing = (; b = Forcing(Fᵇ)))

    smooth(x, y, z) = 1e-2 * sin(2π * x) * cos(2π * y) * sin(π * z)
    set!(model, u = smooth, v = smooth, b = smooth)
    for _ in 1:3
        time_step!(model, 1e-2)   # the pressure projection makes u⃗ divergence-free
    end

    TEND  = PotentialEnergyTendency(model)
    BADV  = PotentialEnergyBuoyancyAdvection(model)
    BDIFF = PotentialEnergyBuoyancyDiffusion(model)
    FORC  = PotentialEnergyForcing(model)
    ADV   = PotentialEnergyAdvection(model)
    DIFF  = PotentialEnergyDiffusion(model)

    @test TEND  isa PotentialEnergyTendency
    @test BADV  isa PotentialEnergyBuoyancyAdvection
    @test BDIFF isa PotentialEnergyBuoyancyDiffusion
    @test FORC  isa PotentialEnergyForcing
    @test ADV   isa PotentialEnergyAdvection
    @test DIFF  isa PotentialEnergyDiffusion

    tendency = Array(interior(Field(TEND)))
    terms = sum(Array(interior(Field(term))) for term in (BADV, BDIFF, FORC))
    @test maximum(abs, tendency .- terms) < 1e-12 * maximum(abs, tendency)

    # Every term has to be doing something, or the sum above would be checking nothing. A term worth 1%
    # of the tendency is still pinned to about `1e-10` of its own size by that bound, which is what makes
    # 1% a meaningful floor. `Fᵇ`'s amplitude is set so the forcing clears it by more than a factor of
    # ten on both grids, as the other two terms do; a forcing weak enough to be a rounding correction
    # would satisfy the sum above no matter what `PotentialEnergyForcing` returned.
    for term in (BADV, BDIFF, FORC)
        @test maximum(abs, Array(interior(Field(term)))) > 1e-2 * maximum(abs, tendency)
    end

    # `Advection` and `Diffusion` are flux divergences, so each telescopes to the boundary and vanishes
    # over this periodic-in-x-and-y, no-flux-in-z domain. What survives is roundoff, not truncation
    # error, so the bound goes with the term's own magnitude rather than the tendency's. A volume
    # integral carries that magnitude times a volume, so the domain volume is what belongs here and a
    # cell count would leave the bound with the wrong units. Both terms land below a single ulp of that
    # product, so the factor of `1e3` is slack.
    V = grid.Lx * grid.Ly * grid.Lz
    for transport in (ADV, DIFF)
        scale = maximum(abs, Array(interior(Field(transport))))
        @test abs(volume_integral(transport)) < 1e3 * eps(eltype(grid)) * scale * V
    end

    # `∫Diffusion = ∫Φ` telescopes and comes out to a dozen digits, but `∫Advection = -∫wb` is second
    # order in the grid spacing and the shared test grids are six cells wide, so it is only good to a
    # few percent here. The baroclinic adjustment example, on a grid that resolves something, is where
    # it is checked in earnest.
    @test volume_integral(BADV)  ≈ -volume_integral(PotentialToKineticEnergyConversion(model)) rtol=0.1
    @test volume_integral(BDIFF) ≈  volume_integral(PotentialEnergyDiffusiveVerticalBuoyancyFlux(model)) rtol=1e-6

    # These are terms of the `b` equation weighted by `-z`, so they need `b` to be a tracer
    plain = NonhydrostaticModel(grid; buoyancy = BuoyancyTracer(), tracers = :b)
    seawater = NonhydrostaticModel(grid; buoyancy = SeawaterBuoyancy(), tracers = (:T, :S),
                                   closure = ScalarDiffusivity(κ=1e-4))
    for diagnostic in (PotentialEnergyTendency, PotentialEnergyAdvection, PotentialEnergyBuoyancyAdvection,
                       PotentialEnergyDiffusion, PotentialEnergyBuoyancyDiffusion, PotentialEnergyForcing)
        @test_throws ArgumentError diagnostic(seawater)
    end
    # ... and the two diffusive terms need a closure too
    @test_throws "no closure" PotentialEnergyDiffusion(plain)
    @test_throws "no closure" PotentialEnergyBuoyancyDiffusion(plain)

    return nothing
end

"""
Every term of the `eₚ = -bz` equation weights buoyancy by `z`, which is the height that works against
gravity only when gravity points along `-z`. Tilt it and the height becomes `z cosθ - y sinθ`, so `-bz`
is wrong by a factor and by a cross-slope term: not a term computed approximately, a different quantity.
Nothing about the tilt is visible in the output, so it has to be refused at construction.

Two diagnostics are exempt on purpose. `DiffusiveVerticalBuoyancyFlux` never touches `z`; it returns the
vertical component of the closure's diffusive flux, which is what its name promises whatever gravity is
doing. `PotentialToKineticEnergyConversion` is the full `u⃗·(bĝ)` contraction over Oceananigans' own
gravity-projected components, so it is correct under any tilt and only reads as `wb` when `ĝ = -ẑ`.
"""
function test_gravity_direction_validation(grid)

    tilted = BuoyancyForce(BuoyancyTracer(); gravity_unit_vector = (0, sind(10), -cosd(10)))
    model = NonhydrostaticModel(grid; buoyancy = tilted, tracers = :b,
                                closure = ScalarDiffusivity(ν=1e-6, κ=1e-3))
    set!(model, b = grid_noise)

    for diagnostic in (PotentialEnergy, PotentialEnergyTendency, PotentialEnergyAdvection,
                       PotentialEnergyBuoyancyAdvection, PotentialEnergyDiffusion,
                       PotentialEnergyBuoyancyDiffusion, PotentialEnergyForcing)
        @test_throws "NegativeZDirection" diagnostic(model)
    end

    # The message has to name the diagnostic the caller asked for, not whichever one happens to hold the
    # validator, or it sends them looking in the wrong place
    @test_throws "`PotentialEnergyBuoyancyDiffusion`" PotentialEnergyBuoyancyDiffusion(model)

    # Exempt, per the docstring above; asserted so that neither becomes collateral damage of a later
    # sweep that adds the check everywhere
    @test PotentialEnergyDiffusiveVerticalBuoyancyFlux(model) isa PotentialEnergyDiffusiveVerticalBuoyancyFlux
    @test PotentialToKineticEnergyConversion(model) isa PotentialToKineticEnergyConversion

    # An upright model of course still builds every one of them
    upright = NonhydrostaticModel(grid; buoyancy = BuoyancyTracer(), tracers = :b,
                                  closure = ScalarDiffusivity(ν=1e-6, κ=1e-3))
    for diagnostic in (PotentialEnergy, PotentialEnergyTendency, PotentialEnergyAdvection,
                       PotentialEnergyBuoyancyAdvection, PotentialEnergyDiffusion,
                       PotentialEnergyBuoyancyDiffusion, PotentialEnergyForcing)
        @test diagnostic(upright) isa KernelFunctionOperation
    end

    return nothing
end

"""
`wb` is the one term the kinetic and potential energy budgets share, so it is defined once in
`KineticEnergyEquation` and re-exported by the two potential energy modules, where `KineticEnergyConversion`
names the exchange rather than the side it feeds. The alias is deliberately *not* exported from
`Oceanostics`, since unprefixed it says nothing about which budget it belongs to. Nothing about that
arrangement is enforced by the code, so it is asserted here: a stray `export` in `Oceanostics.jl`, or a
re-export dropped from either module, would otherwise pass unnoticed.
"""
function test_kinetic_energy_conversion_alias()

    PE, APE = Oceanostics.PotentialEnergyEquation, Oceanostics.AvailablePotentialEnergyEquation

    @test PE.KineticEnergyConversion === Oceanostics.KineticEnergyEquation.PotentialToKineticEnergyConversion
    @test APE.KineticEnergyConversion === PE.KineticEnergyConversion

    for M in (PE, APE)
        @test :KineticEnergyConversion in names(M)
        @test :PotentialToKineticEnergyConversion in names(M)
    end
    @test !(:KineticEnergyConversion in names(Oceanostics))

    return nothing
end

@testset "Diagnostics tests" begin
    @info "  Testing Diagnostics"
    for (grid_class, grid) in zip(keys(grids), values(grids))
        @info "    with $grid_class"
        for model_type in model_types
            @info "      with $model_type"
            for buoyancy in extended_buoyancy_formulations
                @info "        with $(summary(buoyancy))"

                tracers = buoyancy isa BuoyancyTracer ? :b : (:S, :T)
                model = model_type(grid; buoyancy, tracers)
                buoyancy isa BuoyancyTracer ? set!(model, b = 9.87) : set!(model, S = 34.7, T = 0.5)

                if isnothing(buoyancy)
                    @info "          Testing that potential energy equation terms throw error when `buoyancy==nothing`"
                    test_potential_energy_equation_terms_errors(model)
                else
                    @info "          Testing `PotentialEnergy`"
                    test_potential_energy_equation_terms(model)
                    test_potential_energy_equation_terms(model, geopotential_height = 0)
                end
            end
            test_PEbuoyancytracer_equals_PElineareos(grid)
        end
        @info "      Testing the diffusive buoyancy flux"
        test_diffusive_buoyancy_flux(grid)

        @info "      Testing the terms of the potential energy equation"
        test_potential_energy_budget_terms(grid)

        @info "      Testing that a tilted gravity is rejected"
        test_gravity_direction_validation(grid)
    end

    @info "  Testing the `KineticEnergyConversion` alias and its export scope"
    test_kinetic_energy_conversion_alias()
end
