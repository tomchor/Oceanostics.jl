using Test
using CUDA: @allowscalar
using Oceananigans.AbstractOperations: AbstractOperation
using Oceananigans.TurbulenceClosures: ThreeDimensionalFormulation

using Oceanostics

# Include common test utilities
include("test_utils.jl")

#+++ Test functions
function test_tke_calculation(model)
    u, v, w = model.velocities
    U = Field(Average(u, dims=(2, 3)))
    V = Field(Average(v, dims=(2, 3)))
    W = Field(Average(w, dims=(2, 3)))

    op = TurbulentKineticEnergyEquation.TurbulentKineticEnergy(model, U=U, V=V, W=W)
    @test op isa TurbulentKineticEnergyEquation.TurbulentKineticEnergy
    tke_c = Field(op)
    @test all(interior(Field(tke_c)) .≈ 0)

    return nothing
end

function test_shear_production_terms(model)
    u, v, w = model.velocities
    U = Field(Average(u, dims=(2, 3)))
    V = Field(Average(v, dims=(2, 3)))
    W = Field(Average(w, dims=(2, 3)))

    op = TurbulentKineticEnergyEquation.XShearProductionRate(model, u, v, w, U, V, W)
    @test op isa TurbulentKineticEnergyEquation.XShearProductionRate
    XSP = Field(op)
    @test all(interior(Field(XSP)) .≈ 0)

    op = TurbulentKineticEnergyEquation.XShearProductionRate(model; U=U, V=V, W=W)
    @test op isa TurbulentKineticEnergyEquation.XShearProductionRate
    XSP = Field(op)
    @test all(interior(Field(XSP)) .≈ 0)

    U = Field(Average(u, dims=(1, 3)))
    V = Field(Average(v, dims=(1, 3)))
    W = Field(Average(w, dims=(1, 3)))

    op = TurbulentKineticEnergyEquation.YShearProductionRate(model; U=U, V=V, W=W)
    @test op isa TurbulentKineticEnergyEquation.YShearProductionRate
    YSP = Field(op)
    @test all(interior(Field(YSP)) .≈ 0)

    op = TurbulentKineticEnergyEquation.YShearProductionRate(model, u, v, w, U, V, W)
    @test op isa TurbulentKineticEnergyEquation.YShearProductionRate
    YSP = Field(op)
    @test all(interior(Field(YSP)) .≈ 0)

    U = Field(Average(u, dims=(1, 2)))
    V = Field(Average(v, dims=(1, 2)))
    W = Field(Average(w, dims=(1, 2)))

    op = TurbulentKineticEnergyEquation.ZShearProductionRate(model, u, v, w, U, V, W)
    @test op isa TurbulentKineticEnergyEquation.ZShearProductionRate
    ZSP = Field(op)
    @test all(interior(Field(ZSP)) .≈ 0)

    op = TurbulentKineticEnergyEquation.ZShearProductionRate(model; U=U, V=V, W=W)
    @test op isa TurbulentKineticEnergyEquation.ZShearProductionRate
    ZSP = Field(op)
    @test all(interior(Field(ZSP)) .≈ 0)

end

function test_ke_dissipation_rate_terms(grid; model_type=NonhydrostaticModel, closure=ScalarDiffusivity(ν=1))
    model = model_type(; grid, closure, buoyancy=BuoyancyTracer(), tracers=:b)

    if !(model.closure isa Tuple) || all(isa.(model.closure, ScalarDiffusivity{ThreeDimensionalFormulation}))
        ε_iso = TurbulentKineticEnergyEquation.TurbulentKineticEnergyIsotropicDissipationRate(model; U=0, V=0, W=0)
        ε_iso_field = Field(ε_iso)
        @test ε_iso isa TurbulentKineticEnergyEquation.TurbulentKineticEnergyIsotropicDissipationRate
        @test ε_iso_field isa Field
    end

    dudz = 2
    set!(model, u=(x, y, z) -> dudz*z)

    ε = KineticEnergyEquation.KineticEnergyDissipationRate(model)
    ε_field = Field(ε)
    @test ε isa KineticEnergyEquation.KineticEnergyDissipationRate
    @test ε_field isa Field

    εp = KineticEnergyEquation.KineticEnergyDissipationRate(model; U=Field(Average(model.velocities.u, dims=(1,2))))
    εp_field = Field(εp)
    @test εp isa KineticEnergyEquation.KineticEnergyDissipationRate
    @test εp_field isa Field

    idxs = (model.grid.Nx÷2, model.grid.Ny÷2, model.grid.Nz÷2)

    if closure isa Tuple
        ν_field = Field(sum(viscosity(closure, model.diffusivity_fields)))
    else
        ν_field = viscosity(closure, model.diffusivity_fields)
    end

    rtol = zspacings(grid, Center()) isa Number ? 1e-12 : 0.06 # less accurate for stretched grid

    @allowscalar begin
        true_ε = (ν_field isa Field ? getindex(ν_field, idxs...) : ν_field) * dudz^2
        @test isapprox(getindex(ε_field,  idxs...), true_ε, rtol=rtol, atol=eps())
        @test isapprox(getindex(εp_field, idxs...), 0.0,    rtol=rtol, atol=eps())
    end

    set!(model, u=grid_noise, v=grid_noise, w=grid_noise, b=grid_noise)
    ε̄ₖ = Field(Average(KineticEnergyEquation.KineticEnergyDissipationRate(model)))

    return nothing
end
#---

@testset "Turbulent Kinetic Energy Equation tests" begin
    @info "  Testing Turbulent Kinetic Energy Equation"
    for (grid_class, grid) in zip(keys(grids), values(grids))
        @info "    with $grid_class"
        for model_type in model_types
            @info "      with $model_type"
            for closure in closures
                @info "        with closure $(summary(closure))"
                model = model_type(; grid, closure, model_kwargs...)

                @info "          Testing energy dissipation rate terms"
                test_ke_dissipation_rate_terms(grid; model_type, closure)

                if model_type == NonhydrostaticModel
                    @info "          Testing velocity-only diagnostics"
                    test_shear_production_terms(model)
                end
            end
        end

        @info "        Testing input validation for dissipation rates"
        invalid_closures = [HorizontalScalarDiffusivity(ν=1e-6, κ=1e-7),
                            VerticalScalarDiffusivity(ν=1e-6, κ=1e-7),
                            (ScalarDiffusivity(ν=1e-6, κ=1e-7), HorizontalScalarDiffusivity(ν=1e-6, κ=1e-7))]

        for closure in invalid_closures
            model = NonhydrostaticModel(grid = regular_grid; model_kwargs..., closure)
            @test_throws ErrorException TurbulentKineticEnergyEquation.TurbulentKineticEnergyIsotropicDissipationRate(model; U=0, V=0, W=0)
        end

    end
end 