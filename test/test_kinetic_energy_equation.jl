using Test
using CUDA: @allowscalar
using Oceananigans.TurbulenceClosures: AbstractScalarDiffusivity, ThreeDimensionalFormulation

using Oceanostics
using Oceanostics.TracerVarianceEquation: TracerVarianceTendency

# Include common test utilities
include("test_utils.jl")

#+++ Test functions
function test_ke_calculation(model)
    u, v, w = model.velocities

    ke_op = KineticEnergyEquation.KineticEnergy(model)
    @test ke_op isa KineticEnergy
    ke_c = Field(ke_op)
    @test all(interior(Field(ke_c)) .≈ 0)

    return nothing
end

function test_pressure_term(model)
    u⃗∇p = KineticEnergyEquation.PressureRedistribution(model)
    @test u⃗∇p isa KineticEnergyEquation.KineticEnergyPressureRedistribution
    @test Field(u⃗∇p) isa Field

    u⃗∇pNHS = KineticEnergyEquation.PressureRedistribution(model, pressure=model.pressures.pNHS)
    @test u⃗∇pNHS isa KineticEnergyEquation.KineticEnergyPressureRedistribution
    @test Field(u⃗∇pNHS) isa Field

    # Test calculation with a hydrostatic pressure separation
    model2 = NonhydrostaticModel(grid=model.grid, hydrostatic_pressure_anomaly=CenterField(model.grid))
    u⃗∇p_from_model2 = KineticEnergyEquation.PressureRedistribution(model2)
    @test u⃗∇p_from_model2 isa KineticEnergyEquation.KineticEnergyPressureRedistribution
    @test Field(u⃗∇p_from_model2) isa Field

    return nothing
end

function test_momentum_advection_term(grid; model_type=NonhydrostaticModel)
    model = model_type(; grid)
    C₁ = 2; C₂ = 3
    set!(model, u=(x, y, z) -> C₁*y, v=C₂)

    ADV = KineticEnergyEquation.KineticEnergyAdvection(model)
    ADV_field = Field(ADV)
    @test ADV isa KineticEnergyEquation.KineticEnergyAdvection
    @test ADV_field isa Field

    # Test excluding the grid boundaries
    @test Array(interior(ADV_field, 1, 2:grid.Ny-1, 1)) ≈ collect(C₁^2 * C₂ * grid.yᵃᶜᵃ[2:grid.Ny-1])

    return nothing
end

function test_ke_dissipation_rate_terms(grid; model_type=NonhydrostaticModel, closure=ScalarDiffusivity(ν=1))
    model = model_type(; grid, closure, buoyancy=BuoyancyTracer(), tracers=:b)

    dudz = 2
    set!(model, u=(x, y, z) -> dudz*z)

    ε = KineticEnergyEquation.KineticEnergyStress(model)
    ε_field = Field(ε)
    @test ε isa KineticEnergyEquation.KineticEnergyStress
    @test ε_field isa Field

    # Test both signatures of KineticEnergyIsotropicDissipationRate
    if model.closure isa AbstractScalarDiffusivity{<:Any, <:ThreeDimensionalFormulation}
        # Test the convenience signature: KineticEnergyIsotropicDissipationRate(model; location)
        ε_iso1 = KineticEnergyEquation.KineticEnergyIsotropicDissipationRate(model)
        ε_iso1_field = Field(ε_iso1)
        @test ε_iso1 isa KineticEnergyEquation.KineticEnergyIsotropicDissipationRate

        # Test the full signature: KineticEnergyIsotropicDissipationRate(u, v, w, closure, diffusivity_fields, clock; location)
        u, v, w = model.velocities
        ε_iso2 = KineticEnergyEquation.KineticEnergyIsotropicDissipationRate(u, v, w, model.closure, model.diffusivity_fields, model.clock)
        ε_iso2_field = Field(ε_iso2)
        @test ε_iso2 isa KineticEnergyEquation.KineticEnergyIsotropicDissipationRate

        # Test that both signatures produce the same result
        @test all(interior(ε_iso1_field) .≈ interior(ε_iso2_field))

        # Test with different location parameters
        ε_iso1_ccc = KineticEnergyEquation.IsotropicDissipationRate(model; location = (Center, Center, Center))
        ε_iso2_ccc = KineticEnergyEquation.IsotropicDissipationRate(u, v, w, model.closure, model.diffusivity_fields, model.clock; location = (Center, Center, Center))
        @test ε_iso1_ccc isa KineticEnergyEquation.IsotropicDissipationRate
        @test ε_iso2_ccc isa KineticEnergyEquation.IsotropicDissipationRate
        @test all(interior(Field(ε_iso1_ccc)) .≈ interior(Field(ε_iso2_ccc)))
    end

    set!(model, u=grid_noise, v=grid_noise, w=grid_noise, b=grid_noise)
    ε̄ₖ₂= Field(Average(KineticEnergyEquation.KineticEnergyStress(model)))

    if model isa NonhydrostaticModel
        ε = KineticEnergyEquation.KineticEnergyTendency(model)
        ε_field = Field(ε)
        @test ε isa KineticEnergyEquation.KineticEnergyTendency
        @test ε_field isa Field

        ∂ₜKE = Field(Average(TracerVarianceEquation.TracerVarianceTendency(model, :b)))
    end

    return nothing
end

function test_ke_forcing_term(grid; model_type=NonhydrostaticModel)
    Fᵘ_func(x, y, z, t, u) = -u
    Fᵛ_func(x, y, z, t, v) = -v
    Fʷ_func(x, y, z, t, w) = -w

    Fᵘ = Forcing(Fᵘ_func, field_dependencies = :u)
    Fᵛ = Forcing(Fᵛ_func, field_dependencies = :v)
    Fʷ = Forcing(Fʷ_func, field_dependencies = :w)

    model = model_type(; grid, forcing = (u=Fᵘ, v=Fᵛ, w=Fʷ))
    set!(model, u=grid_noise, v=grid_noise, w=grid_noise)

    ε = KineticEnergyEquation.KineticEnergyForcing(model)
    ε_field = Field(ε)
    @test ε isa KineticEnergyEquation.KineticEnergyForcing
    @test ε_field isa Field

    ε_truth = Field(@at (Center, Center, Center) (-model.velocities.u^2 -model.velocities.v^2 -model.velocities.w^2))

    @test isapprox(Array(interior(ε_field, 1, 1, 1)), Array(interior(ε_truth, 1, 1, 1)), rtol=1e-12, atol=eps())

    return nothing
end

function test_buoyancy_production_term(grid; model_type=NonhydrostaticModel)
    model = model_type(grid=grid, buoyancy=BuoyancyTracer(), tracers=:b)
    w₀ = 2; b₀ = 3
    set!(model, w=w₀, b=b₀, enforce_incompressibility=false)

    wb = KineticEnergyEquation.BuoyancyProduction(model)
    wb_field = Field(wb)
    @test wb isa KineticEnergyEquation.BuoyancyProduction
    @test wb_field isa Field
    @test Array(interior(wb_field, 1, 1, 2)) .== w₀ * b₀

    w′ = Field(model.velocities.w - Field(Average(model.velocities.w)))
    b′ = Field(model.tracers.b - Field(Average(model.tracers.b)))
    w′b′ = KineticEnergyEquation.BuoyancyProduction(model, velocities=(u=model.velocities.u, v=model.velocities.v, w=w′), tracers=(b=b′,))
    w′b′_field = Field(w′b′)
    @test w′b′ isa KineticEnergyEquation.BuoyancyProduction
    @test w′b′_field isa Field
    @test .≈(Array(interior(w′b′_field, 1, 1, 2)), 0, rtol=1e-12, atol=1e-13) # less accurate for stretched grid

    return nothing
end
#---

@testset "Kinetic Energy Equation tests" begin
    @info "  Testing Kinetic Energy Equation"
    for (grid_class, grid) in zip(keys(grids), values(grids))
        @info "    with $grid_class"
        for model_type in model_types
            @info "      with $model_type"
            for closure in closures
                @info "        with closure $(summary(closure))"
                model = model_type(; grid, closure, model_kwargs...)

                if model isa NonhydrostaticModel
                    @info "          Testing pressure terms"
                    test_pressure_term(model)

                    @info "          Testing buoyancy production term"
                    test_buoyancy_production_term(grid; model_type)
                end

                @info "          Testing energy dissipation rate terms"
                test_ke_dissipation_rate_terms(grid; model_type, closure)

                if model_type == NonhydrostaticModel
                    @info "          Testing advection terms"
                    test_momentum_advection_term(grid; model_type)

                    @info "          Testing forcing terms"
                    test_ke_forcing_term(grid; model_type)
                end
            end
        end
    end
end 