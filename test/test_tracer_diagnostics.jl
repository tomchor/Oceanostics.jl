using Test
using CUDA: has_cuda_gpu

using Oceananigans
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
tracers = :a

forcing_function(x, y, z, t) = cos(t)
forcing = (; a = Forcing(forcing_function))

bc_function(x, y, z, t) = cos(t)
immersed_bc = FluxBoundaryCondition(bc_function)
boundary_conditions = (; a = FieldBoundaryConditions(immersed=immersed_bc))

model_kwargs = (; tracers, forcing, boundary_conditions)
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
        ADV = TracerEquation.Advection(model, model.velocities..., model.tracers.a, model.advection)
    elseif model isa HydrostaticFreeSurfaceModel
        ADV = TracerEquation.Advection(model, model.velocities..., model.tracers.a, model.advection.a)
    end
    ADV_field = Field(ADV)
    @test ADV isa TracerEquation.Advection
    @test ADV isa TracerAdvection
    @test ADV_field isa Field

    ADV = TracerEquation.Advection(model, :a)
    @test ADV isa TracerEquation.Advection
    ADV_field = Field(ADV)
    @test ADV_field isa Field

    DIFF = TracerEquation.TracerDiffusion(model, :a, model.tracers.a, model.closure, model.diffusivity_fields, model.clock, fields(model), model.buoyancy)
    DIFF_field = Field(DIFF)
    @test DIFF isa TracerEquation.Diffusion
    @test DIFF isa TracerDiffusion
    @test DIFF_field isa Field

    DIFF = TracerEquation.TracerDiffusion(model, :a)
    DIFF_field = Field(DIFF)
    @test DIFF isa TracerEquation.Diffusion
    @test DIFF isa TracerDiffusion
    @test DIFF_field isa Field

    DIFF = TracerEquation.ImmersedDiffusion(model, model.tracers.a, model.tracers.a.boundary_conditions.immersed,
                                   model.closure, model.diffusivity_fields, Val(:a), model.clock, fields(model))
    DIFF_field = Field(DIFF)
    @test DIFF isa TracerEquation.ImmersedDiffusion
    @test DIFF isa TracerImmersedDiffusion
    @test DIFF_field isa Field

    DIFF = TracerEquation.ImmersedDiffusion(model, :a)
    DIFF_field = Field(DIFF)
    @test DIFF isa TracerEquation.ImmersedDiffusion
    @test DIFF isa TracerImmersedDiffusion
    @test DIFF_field isa Field

    DIFF = TracerEquation.TotalDiffusion(model, model.tracers.a, model.tracers.a.boundary_conditions.immersed,
                                model.closure, model.diffusivity_fields, Val(:a), model.clock, fields(model), model.buoyancy)
    DIFF_field = Field(DIFF)
    @test DIFF isa TracerEquation.TotalDiffusion
    @test DIFF isa TracerTotalDiffusion
    @test DIFF_field isa Field

    DIFF = TracerEquation.TotalDiffusion(model, :a)
    DIFF_field = Field(DIFF)
    @test DIFF isa TracerEquation.TotalDiffusion
    @test DIFF isa TracerTotalDiffusion
    @test DIFF_field isa Field

    FORC = TracerEquation.Forcing(model, model.forcing.a, model.clock, fields(model))
    FORC_field = Field(FORC)
    @test FORC isa TracerEquation.Forcing
    @test FORC isa TracerForcing
    @test FORC_field isa Field

    FORC = TracerEquation.Forcing(model, :a)
    FORC_field = Field(FORC)
    @test FORC isa TracerEquation.Forcing
    @test FORC isa TracerForcing
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
