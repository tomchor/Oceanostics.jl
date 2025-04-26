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

#+++ Model arguments
tracers = :a

forcing_function(x, y, z, t) = sin(t)
forcing = (; a = Forcing(forcing_function))

model_kwargs = (; tracers, forcing)
#---

#+++ Test options
grids = Dict("regular grid" => regular_grid,
             "stretched grid" => stretched_grid)

model_types = (NonhydrostaticModel,
               HydrostaticFreeSurfaceModel)
#---

#+++ Test functions
function test_tracer_terms(model)
    if model isa NonhydrostaticModel
        ADV = TracerAdvection(model, model.velocities..., model.tracers.a, model.advection)
    elseif model isa HydrostaticFreeSurfaceModel
        ADV = TracerAdvection(model, model.velocities..., model.tracers.a, model.advection.a)
    end
    @compute ADV_field = Field(ADV)
    @test ADV isa AbstractOperation
    @test ADV_field isa Field

    ADV = TracerAdvection(model, :a)
    @compute ADV_field = Field(ADV)
    @test ADV isa AbstractOperation
    @test ADV_field isa Field

    DIFF = TracerDiffusion(model, :a, model.tracers.a, model.closure, model.diffusivity_fields)
    @compute DIFF_field = Field(DIFF)
    @test DIFF isa AbstractOperation
    @test DIFF_field isa Field

    DIFF = TracerDiffusion(model, :a)
    @compute DIFF_field = Field(DIFF)
    @test DIFF isa AbstractOperation
    @test DIFF_field isa Field

    FORC = TracerForcing(model, model.forcing.a, model.clock, fields(model))
    @compute FORC_field = Field(FORC)
    @test FORC isa AbstractOperation
    @test FORC_field isa Field

    FORC = TracerForcing(model, :a)
    @compute FORC_field = Field(FORC)
    @test FORC isa AbstractOperation
    @test FORC_field isa Field

    return nothing
end
#---

@testset "Tracer variance diagnostics tests" begin
    @info "  Testing tracer diagnostics"
    for (grid_class, grid) in zip(keys(grids), values(grids))
        @info "    with $grid_class"
        for model_type in model_types
            @info "      with $model_type"
            model = model_type(; grid, model_kwargs...)

            @info "        Testing tracer terms"
            test_tracer_terms(model)
        end
    end
end
