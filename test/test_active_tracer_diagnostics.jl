using Test
using CUDA: @allowscalar
using Oceananigans: fill_halo_regions!
using Oceananigans.AbstractOperations: AbstractOperation
using Oceananigans.Grids: znode
using Oceananigans.Models: buoyancy_operation
using SeawaterPolynomials: BoussinesqEquationOfState
using SeawaterPolynomials.SecondOrderSeawaterPolynomials: LinearRoquetSeawaterPolynomial

using Oceanostics
using Oceanostics: get_coriolis_frequency_components

# Include common test utilities
include("test_utils.jl")

LinearBuoyancyForce = Union{BuoyancyTracer, SeawaterBuoyancy{<:Any, <:LinearEquationOfState}}

# Use extended buoyancy formulations for this test
buoyancy_formulations = extended_buoyancy_formulations

#+++ Test functions
function test_buoyancy_diagnostics(model)
    u, v, w = model.velocities

    N² = 1e-5;
    S = 1e-3;
    shear(x, y, z) = S*z + S*y;
    set!(model, u=shear)

    if model.buoyancy != nothing && model.buoyancy.formulation isa SeawaterBuoyancy{<:Any, <:LinearEquationOfState}
        g = model.buoyancy.formulation.gravitational_acceleration
        α = model.buoyancy.formulation.equation_of_state.thermal_expansion
        stratification_T(x, y, z) = N² * z / (g * α)
        set!(model, T=stratification_T)

    else
        stratification_b(x, y, z) = N² * z
        set!(model, b=stratification_b)
    end

    fx, fy, fz = get_coriolis_frequency_components(model)
    if model.buoyancy != nothing && model.buoyancy.formulation isa LinearBuoyancyForce

        Ri = GradientRichardsonNumber(model)
        @test Ri isa GradientRichardsonNumber
        Ri_field = Field(Ri)
        @test Ri_field isa Field
        @test Array(interior(Ri_field))[3, 3, 3] ≈ N² / S^2

        b = buoyancy_operation(model)
        Ri = GradientRichardsonNumber(model, u, v, w, b)
        @test Ri isa GradientRichardsonNumber
        Ri_field = Field(Ri)
        @test Ri_field isa Field
        @test Array(interior(Ri_field))[3, 3, 3] ≈ N² / S^2

    else
        b = model.tracers.b # b in this case is passive
    end

    if model.buoyancy != nothing && model.buoyancy.formulation isa SeawaterBuoyancy{<:Any, <:LinearEquationOfState}
        EPV = ErtelPotentialVorticity(model, tracer_name = :T)
        @test EPV isa ErtelPotentialVorticity
        EPV_field = Field(EPV)
        @test EPV_field isa Field
        @test Array(interior(EPV_field))[3, 3, 3] ≈ N² * (fz - S) / (g * α)

    else
        EPV = ErtelPotentialVorticity(model)
        @test EPV isa ErtelPotentialVorticity
        EPV_field = Field(EPV)
        @test EPV_field isa Field
        @test Array(interior(EPV_field))[3, 3, 3] ≈ N² * (fz - S)
    end

    EPV = ErtelPotentialVorticity(model, u, v, w, b, model.coriolis)
    @test EPV isa ErtelPotentialVorticity
    EPV_field = Field(EPV)
    @test EPV_field isa Field
    @test Array(interior(EPV_field))[3, 3, 3] ≈ N² * (fz - S)

    PVtw = ErtelPotentialVorticity(model; thermal_wind = true)
    @test PVtw isa ThermalWindPotentialVorticity
    @test PVtw isa ErtelPotentialVorticity
    @test Field(PVtw) isa Field

    PVtw = ErtelPotentialVorticity(model, u, v, b, FPlane(f=1e-4))
    @test PVtw isa ThermalWindPotentialVorticity
    @test PVtw isa ErtelPotentialVorticity
    @test Field(PVtw) isa Field

    @test_throws ArgumentError ErtelPotentialVorticity(model, u, v, b, FPlane(f=1e-4); thermal_wind = false)

    DEPV = DirectionalErtelPotentialVorticity(model, (0, 0, 1))
    @test DEPV isa DirectionalErtelPotentialVorticity
    @test Field(DEPV) isa Field

    DEPV = DirectionalErtelPotentialVorticity(model, (0, 0, 1), u, v, w, b, model.coriolis)
    @test DEPV isa DirectionalErtelPotentialVorticity
    @test Field(DEPV) isa Field

    #+++ Veering shear (issue #304)
    # These change the velocities, so they must come after every test above. With `u` constant and
    # `v` linear in z the shear *vector* turns with height, so its squared norm is exactly S² on any
    # grid, while the squared shear of the *speed* is S⁴z²/(U₀² + S²z²) instead: zero at z = 0 and
    # below S² everywhere. Only the former gives the gradient Richardson number.
    if model.buoyancy != nothing && model.buoyancy.formulation isa LinearBuoyancyForce
        U₀ = 1e-2
        set!(model, u = (x, y, z) -> U₀, v = (x, y, z) -> S * z)

        Ri_field = Field(GradientRichardsonNumber(model))
        @test Array(interior(Ri_field))[3, 3, 3] ≈ N² / S^2

        # Same check with gravity along -x, which exercises the tilted branch. `x` is then the true
        # vertical, so `u` is the vertical velocity component and its shear must cancel out of the
        # result, leaving only `v`'s. A zero `w` is passed explicitly so the horizontal shear comes
        # from `v` alone regardless of what continuity leaves in the model's own `w`.
        if model.buoyancy.formulation isa BuoyancyTracer
            A = 3S
            set!(model, u = (x, y, z) -> A * x, v = (x, y, z) -> S * x, b = (x, y, z) -> N² * x)

            Ri_tilted = GradientRichardsonNumber(model, u, v, ZFaceField(model.grid), model.tracers.b, (1, 0, 0))
            @test Array(interior(Field(Ri_tilted)))[3, 3, 3] ≈ N² / S^2 # not N² / (A² + S²)
        end
    end
    #---

    return nothing
end

function test_tracer_diagnostics(model)
    χ = TracerVarianceEquation.TracerVarianceDissipationRate(model, :b)
    χ_field = Field(χ)
    @test χ isa TracerVarianceEquation.DissipationRate
    @test χ isa TracerVarianceDissipationRate
    @test χ_field isa Field

    b̄ = Field(Average(model.tracers.b, dims=(1,2)))
    b′ = model.tracers.b - b̄
    χ = TracerVarianceEquation.TracerVarianceDissipationRate(model, :b, tracer=b′)
    χ_field = Field(χ)
    @test χ isa TracerVarianceEquation.DissipationRate
    @test χ isa TracerVarianceDissipationRate
    @test χ_field isa Field

    χ = TracerVarianceEquation.TracerVarianceDiffusion(model, :b)
    χ_field = Field(χ)
    @test χ isa TracerVarianceEquation.Diffusion
    @test χ isa TracerVarianceDiffusion
    @test χ_field isa Field

    set!(model, u = (x, y, z) -> z, v = grid_noise, w = grid_noise, b = grid_noise)
    ε̄ₚ = Field(Average(TracerVarianceEquation.TracerVarianceDissipationRate(model, :b)))
    ε̄ₚ₂ = Field(Average(TracerVarianceEquation.TracerVarianceDiffusion(model, :b)))
    @test ≈(Array(interior(ε̄ₚ, 1, 1, 1)), Array(interior(ε̄ₚ₂, 1, 1, 1)), rtol=1e-12, atol=2*eps())

    if model isa NonhydrostaticModel
        χ = TracerVarianceEquation.TracerVarianceTendency(model, :b)
        χ_field = Field(χ)
        @test χ isa TracerVarianceEquation.Tendency
        @test χ isa TracerVarianceTendency
        @test χ_field isa Field

        ∂ₜc² = Field(Average(TracerVarianceEquation.TracerVarianceTendency(model, :b)))
    end

    return nothing
end

function test_mixed_layer_depth(grid, buoyancy; zₘₓₗ = 0.5, δb = -1e-4 * Oceananigans.defaults.gravitational_acceleration, naive_thermal_expansion=0.000167)
    density_is_defined = (!(buoyancy isa BuoyancyTracer)) && (buoyancy.equation_of_state isa BoussinesqEquationOfState)
    ∂z_b = - δb / zₘₓₗ

    if buoyancy isa BuoyancyTracer
        boundary_conditions = FieldBoundaryConditions(grid, (Center(), Center(), Center()); top = GradientBoundaryCondition(∂z_b))
        C = (; b = CenterField(grid; boundary_conditions))

    else
        g = buoyancy.gravitational_acceleration
        ∂z_T = ∂z_b / (g * naive_thermal_expansion)

        boundary_conditions = FieldBoundaryConditions(grid, (Center(), Center(), Center()); top = GradientBoundaryCondition(∂z_T))
        C = (; T = CenterField(grid; boundary_conditions), S = CenterField(grid))
    end

    # Missing criterion arguments must throw at construction time rather than return a broken operation
    @test_throws ArgumentError MixedLayerDepth(grid; criterion = BuoyancyAnomalyCriterion(δb))
    @test_throws ArgumentError MixedLayerDepth(grid; criterion = DensityAnomalyCriterion())

    mld_b = MixedLayerDepth(grid, buoyancy, C; criterion = BuoyancyAnomalyCriterion(δb))

    if density_is_defined
        ρᵣ = buoyancy.equation_of_state.reference_density
        δρ = - δb * ρᵣ / g

        criterion = DensityAnomalyCriterion(buoyancy; threshold = convert(eltype(grid), δρ))
        mld_ρ = MixedLayerDepth(grid, buoyancy, C; criterion)
    end 

    @allowscalar @test isinf(mld_b[1, 1])
    density_is_defined && (@allowscalar @test isinf(mld_ρ[1, 1]) | (mld_ρ[1, 1] < znode(1, 1, 1, grid, Center(), Center(), Face()))) # for TEOS10 we don't get -Inf just a really deep depth

    if buoyancy isa BuoyancyTracer
        set!(C.b, (x, y, z) -> z * ∂z_b)
    else
        set!(C.T, (x, y, z) -> z * ∂z_T + 10)
        set!(C.S, 35) # TEOS10SeawaterPolynomial doesn't seem to like it when this is zero
    end

    fill_halo_regions!(C)

    @allowscalar @test isapprox(mld_b[1, 1], -zₘₓₗ + znode(1, 1, grid.Nz+1, grid, Center(), Center(), Face()), atol=0.02) # high tollerance from the approximation in ∂z_T
    density_is_defined && (@allowscalar @test isapprox(mld_ρ[1, 1], -zₘₓₗ + znode(1, 1, grid.Nz+1, grid, Center(), Center(), Face()), atol=0.02)) # high tollerance from the approximation in ∂z_T
end
#---

@testset "Tracer active diagnostics tests" begin
    @info "  Testing tracer diagnostics"
    for (grid_class, grid) in zip(keys(grids), values(grids))
        @info "    with $grid_class"
        for model_type in model_types
            @info "        with $model_type"
            for buoyancy in buoyancy_formulations
                @info "        with $(summary(buoyancy))"

                for coriolis in coriolis_formulations
                    @info "          with $(summary(coriolis))"
                    tracers = buoyancy isa BuoyancyTracer ? :b : (:S, :T, :b)
                    model = model_type(grid; buoyancy, tracers, coriolis)
                    buoyancy isa BuoyancyTracer ? set!(model, b = 9.87) : set!(model, S = 34.7, T = 0.5)

                    @info "            Testing buoyancy diagnostics"
                    test_buoyancy_diagnostics(model)
                end

                @info "            Testing mixed layer depth diagnostic"
                if !isnothing(buoyancy)
                    test_mixed_layer_depth(grid, buoyancy)
                else
                    @test_throws ErrorException test_mixed_layer_depth(grid, buoyancy)
                end
            end
        end
    end
end
