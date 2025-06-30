using Test

using Oceanostics
using Oceanostics.TracerVarianceEquation: TracerVarianceDissipationRate, TracerVarianceDiffusiveTerm, TracerVarianceTendency

# Include common test utilities
include("test_utils.jl")

#+++ Test functions
function test_tracer_variance_terms(model)
    χ = TracerVarianceEquation.TracerVarianceDissipationRate(model, :b)
    χ_field = Field(χ)
    @test χ isa TracerVarianceEquation.TracerVarianceDissipationRate
    @test χ_field isa Field

    b̄ = Field(Average(model.tracers.b, dims=(1,2)))
    b′ = model.tracers.b - b̄
    χ = TracerVarianceEquation.TracerVarianceDissipationRate(model, :b, tracer=b′)
    χ_field = Field(χ)
    @test χ isa TracerVarianceEquation.TracerVarianceDissipationRate
    @test χ_field isa Field

    χ = TracerVarianceEquation.TracerVarianceDiffusiveTerm(model, :b)
    χ_field = Field(χ)
    @test χ isa TracerVarianceEquation.TracerVarianceDiffusiveTerm
    @test χ_field isa Field

    set!(model, u = (x, y, z) -> z, v = grid_noise, w = grid_noise, b = grid_noise)
    ε̄ₚ = Field(Average(TracerVarianceEquation.TracerVarianceDissipationRate(model, :b)))
    ε̄ₚ₂ = Field(Average(TracerVarianceEquation.TracerVarianceDiffusiveTerm(model, :b)))
    @test ≈(Array(interior(ε̄ₚ, 1, 1, 1)), Array(interior(ε̄ₚ₂, 1, 1, 1)), rtol=1e-12, atol=2*eps())

    if model isa NonhydrostaticModel
        χ = TracerVarianceEquation.TracerVarianceTendency(model, :b)
        χ_field = Field(χ)
        @test χ isa TracerVarianceEquation.TracerVarianceTendency
        @test χ_field isa Field

        ∂ₜc² = Field(Average(TracerVarianceEquation.TracerVarianceTendency(model, :b)))
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
