using Test
using CUDA: has_cuda_gpu, @allowscalar

using Oceananigans
using Oceananigans: fill_halo_regions!
using Oceananigans.AbstractOperations: AbstractOperation
using Oceananigans.Fields: @compute, ConstantField
using Oceananigans.Grids: znode
using Oceananigans.TurbulenceClosures.Smagorinskys: LagrangianAveraging
using Oceananigans.BuoyancyFormulations: buoyancy
using SeawaterPolynomials: RoquetEquationOfState, TEOS10EquationOfState, BoussinesqEquationOfState
using SeawaterPolynomials.SecondOrderSeawaterPolynomials: LinearRoquetSeawaterPolynomial

using Oceanostics
using Oceanostics: get_coriolis_frequency_components

LinearBuoyancyForce = Union{BuoyancyTracer, SeawaterBuoyancy{<:Any, <:LinearEquationOfState}}

#+++ Default grids
arch = has_cuda_gpu() ? GPU() : CPU()

N = 6
regular_grid = RectilinearGrid(arch, size=(N, N, N), extent=(1, 1, 1))

S = .99 # Stretching factor. Positive number ∈ (0, 1]
f_asin(k) = -asin(S*(2k - N - 2) / N)/π + 1/2
F1 = f_asin(1); F2 = f_asin(N+1)
z_faces(k) = ((F1 + F2)/2 - f_asin(k)) / (F1 - F2)

stretched_grid = RectilinearGrid(arch, size=(N, N, N), x=(0, 1), y=(0, 1), z=z_faces)

grid_noise(x, y, z) = randn()
#---

#+++ Test options
model_kwargs = (buoyancy = BuoyancyForce(BuoyancyTracer()),
                coriolis = FPlane(1e-4),
                tracers = :b)

closures = (ScalarDiffusivity(ν=1e-6, κ=1e-7),
            SmagorinskyLilly(),
            Smagorinsky(coefficient=DynamicCoefficient(averaging=(1, 2))),
            Smagorinsky(coefficient=DynamicCoefficient(averaging=LagrangianAveraging())),
            (ScalarDiffusivity(ν=1e-6, κ=1e-7), AnisotropicMinimumDissipation()),)

buoyancy_formulations = (nothing,
                         BuoyancyTracer(),
                         SeawaterBuoyancy(),
                         SeawaterBuoyancy(equation_of_state=TEOS10EquationOfState()),
                         SeawaterBuoyancy(equation_of_state=RoquetEquationOfState(:Linear)))

coriolis_formulations = (nothing,
                         FPlane(1e-4),
                         ConstantCartesianCoriolis(fx=1e-4, fy=1e-4, fz=1e-4))

grids = Dict("regular grid" => regular_grid,
             "stretched grid" => stretched_grid)

model_types = (NonhydrostaticModel,
               HydrostaticFreeSurfaceModel)
#---

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

        Ri = RichardsonNumber(model)
        @test Ri isa AbstractOperation
        @compute Ri_field = Field(Ri)
        @test Ri_field isa Field
        @test interior(Ri_field, 3, 3, 3)[1] ≈ N² / S^2

        b = buoyancy(model)
        Ri = RichardsonNumber(model, u, v, w, b)
        @test Ri isa AbstractOperation
        @compute Ri_field = Field(Ri)
        @test Ri_field isa Field
        @test interior(Ri_field, 3, 3, 3)[1] ≈ N² / S^2

    else
        b = model.tracers.b # b in this case is passive
    end

    if model.buoyancy != nothing && model.buoyancy.formulation isa SeawaterBuoyancy{<:Any, <:LinearEquationOfState}
        EPV = ErtelPotentialVorticity(model, tracer=:T)
        @test EPV isa AbstractOperation
        EPV_field = compute!(Field(EPV))
        @test EPV_field isa Field
        @test interior(EPV_field, 3, 3, 3)[1] ≈ N² * (fz - S) / (g * α)

    else
        EPV = ErtelPotentialVorticity(model)
        @test EPV isa AbstractOperation
        EPV_field = compute!(Field(EPV))
        @test EPV_field isa Field
        @test interior(EPV_field, 3, 3, 3)[1] ≈ N² * (fz - S)
    end

    EPV = ErtelPotentialVorticity(model, u, v, w, b, model.coriolis)
    @test EPV isa AbstractOperation
    EPV_field = compute!(Field(EPV))
    @test EPV_field isa Field
    @test interior(EPV_field, 3, 3, 3)[1] ≈ N² * (fz - S)

    PVtw = ThermalWindPotentialVorticity(model)
    @test PVtw isa AbstractOperation
    @test compute!(Field(PVtw)) isa Field

    PVtw = ThermalWindPotentialVorticity(model, u, v, b, FPlane(1e-4))
    @test PVtw isa AbstractOperation
    @test compute!(Field(PVtw)) isa Field

    DEPV = DirectionalErtelPotentialVorticity(model, (0, 0, 1))
    @test DEPV isa AbstractOperation
    @test compute!(Field(DEPV)) isa Field

    DEPV = DirectionalErtelPotentialVorticity(model, (0, 0, 1), u, v, w, b, model.coriolis)
    @test DEPV isa AbstractOperation
    @test compute!(Field(DEPV)) isa Field

    return nothing
end

function test_tracer_diagnostics(model)
    χ = TracerVarianceDissipationRate(model, :b)
    χ_field = compute!(Field(χ))
    @test χ isa AbstractOperation
    @test χ_field isa Field

    b̄ = Field(Average(model.tracers.b, dims=(1,2)))
    b′ = model.tracers.b - b̄
    χ = TracerVarianceDissipationRate(model, :b, tracer=b′)
    χ_field = compute!(Field(χ))
    @test χ isa AbstractOperation
    @test χ_field isa Field

    χ = TracerVarianceDiffusiveTerm(model, :b)
    χ_field = compute!(Field(χ))
    @test χ isa AbstractOperation
    @test χ_field isa Field

    set!(model, u = (x, y, z) -> z, v = grid_noise, w = grid_noise, b = grid_noise)
    @compute ε̄ₚ = Field(Average(TracerVarianceDissipationRate(model, :b)))
    @compute ε̄ₚ₂ = Field(Average(TracerVarianceDiffusiveTerm(model, :b)))
    @test ≈(Array(interior(ε̄ₚ, 1, 1, 1)), Array(interior(ε̄ₚ₂, 1, 1, 1)), rtol=1e-12, atol=2*eps())

    if model isa NonhydrostaticModel
        χ = TracerVarianceTendency(model, :b)
        χ_field = compute!(Field(χ))
        @test χ isa AbstractOperation
        @test χ_field isa Field

        @compute ∂ₜc² = Field(Average(TracerVarianceTendency(model, :b)))
    end

    return nothing
end

function test_mixed_layer_depth(grid, buoyancy; zₘₓₗ = 0.5, δb = -1e-4 * 9.81, naive_thermal_expansion=0.000167)
    density_is_defined = (!(buoyancy isa BuoyancyTracer)) && (buoyancy.equation_of_state isa BoussinesqEquationOfState)

    ∂z_b = - δb / zₘₓₗ

    if buoyancy isa BuoyancyTracer
        boundary_conditions = FieldBoundaryConditions(grid, (Center, Center, Center); top = GradientBoundaryCondition(∂z_b))

        C = (; b = CenterField(grid; boundary_conditions))
    else
        g = buoyancy.gravitational_acceleration

        ∂z_T = ∂z_b / (g * naive_thermal_expansion)

        boundary_conditions = FieldBoundaryConditions(grid, (Center, Center, Center); top = GradientBoundaryCondition(∂z_T))
        
        C = (; T = CenterField(grid; boundary_conditions), S = CenterField(grid))
    end
    
    mld_b = MixedLayerDepth(grid, buoyancy, C; criterion = BuoyancyAnomalyCriterion(δb))

    if density_is_defined
        ρᵣ = buoyancy.equation_of_state.reference_density

        δρ = - δb * ρᵣ / g

        criterion = DensityAnomalyCriterion(buoyancy; threshold = convert(eltype(grid), δρ))

        mld_ρ = MixedLayerDepth(grid, buoyancy, C; criterion)
    end 

    @test isinf(mld_b[1, 1])
    density_is_defined && (@test isinf(mld_ρ[1, 1]) | (mld_ρ[1, 1] < znode(1, 1, 1, grid, Center(), Center(), Face()))) # for TEOS10 we don't get -Inf just a really deep depth

    if buoyancy isa BuoyancyTracer
        set!(C.b, (x, y, z) -> z * ∂z_b)
    else
        set!(C.T, (x, y, z) -> z * ∂z_T + 10)
        set!(C.S, 35) # TEOS10SeawaterPolynomial doesn't seem to like it when this is zero
    end

    fill_halo_regions!(C)

    @test isapprox(mld_b[1, 1], -zₘₓₗ + znode(1, 1, grid.Nz+1, grid, Center(), Center(), Face()), atol=0.02) # high tollerance from the approximation in ∂z_T
    density_is_defined && (@test isapprox(mld_ρ[1, 1], -zₘₓₗ + znode(1, 1, grid.Nz+1, grid, Center(), Center(), Face()), atol=0.02)) # high tollerance from the approximation in ∂z_T
end
#---

@testset "Tracer diagnostics tests" begin
    @info "  Testing tracer diagnostics"
    for (grid_class, grid) in zip(keys(grids), values(grids))
        @info "    with $grid_class"
        for model_type in model_types
            @info "      with $model_type"
            for closure in closures
                @info "        with closure $(summary(closure))"
                model = model_type(; grid, closure, model_kwargs...)

                @info "          Testing tracer variance terms"
                test_tracer_diagnostics(model)
            end

            @info "        Testing diagnostics that use buoyancy"
            for buoyancy in buoyancy_formulations
                @info "        with $(summary(buoyancy))"

                for coriolis in coriolis_formulations
                    @info "          with $(summary(coriolis))"
                    tracers = buoyancy isa BuoyancyTracer ? :b : (:S, :T, :b)
                    model = model_type(; grid, buoyancy, tracers, coriolis)
                    buoyancy isa BuoyancyTracer ? set!(model, b = 9.87) : set!(model, S = 34.7, T = 0.5)

                    @info "            Testing buoyancy diagnostics"
                    test_buoyancy_diagnostics(model)
                end

                if !isnothing(buoyancy)
                    @info "            Testing mixed layer depth diagnostic"
                    test_mixed_layer_depth(grid, buoyancy)
                end
            end
        end
    end
end
