using Test
using CUDA: has_cuda_gpu

using Oceananigans
using Oceananigans.AbstractOperations: AbstractOperation
using Oceananigans.TurbulenceClosures.Smagorinskys: LagrangianAveraging
using Oceananigans.Fields: @compute

using Oceanostics

#+++ Default grids
arch = has_cuda_gpu() ? GPU() : CPU()

N = 6
regular_grid = RectilinearGrid(arch, size=(N, N, N), extent=(1, 1, 1))

S = .99 # Stretching factor. Positive number ∈ (0, 1]
f_asin(k) = -asin(S*(2k - N - 2) / N)/π + 1/2
F1 = f_asin(1); F2 = f_asin(N+1)
z_faces(k) = ((F1 + F2)/2 - f_asin(k)) / (F1 - F2)

stretched_grid = RectilinearGrid(arch, size=(N, N, N), x=(0, 1), y=(0, 1), z=z_faces)
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

grids = Dict("regular grid" => regular_grid,
             "stretched grid" => stretched_grid)

model_types = (NonhydrostaticModel,
               HydrostaticFreeSurfaceModel)
#---

#+++ Test functions
function test_tracer_variance_terms(model)
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

    grid_noise(x, y, z) = randn()
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
#---

@testset "Tracer variance diagnostics tests" begin
    @info "  Testing tracer diagnostics"
    for (grid_class, grid) in zip(keys(grids), values(grids))
        @info "    with $grid_class"
        for model_type in model_types
            @info "      with $model_type"
            for closure in closures
                @info "        with closure $(summary(closure))"
                model = model_type(; grid, closure, model_kwargs...)

                @info "          Testing tracer variance terms"
                test_tracer_variance_terms(model)
            end
        end
    end
end
