using Test
using CUDA: has_cuda_gpu, @allowscalar

using Oceananigans
using Oceananigans.AbstractOperations: AbstractOperation

using Oceanostics
using Oceanostics: perturbation_fields

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
model_kwargs = (coriolis = FPlane(1e-4),)

coriolis_formulations = (nothing,
                         FPlane(1e-4),
                         ConstantCartesianCoriolis(fx=1e-4, fy=1e-4, fz=1e-4))

grids = Dict("regular grid" => regular_grid,
             "stretched grid" => stretched_grid)

model_types = (NonhydrostaticModel,
               HydrostaticFreeSurfaceModel)
#---

#+++ Test functions
function test_velocity_only_flow_diagnostics(model)
    op = RossbyNumber(model;)
    @test op isa AbstractOperation
    Ro = Field(op)
    @test all(interior(compute!(Ro)) .≈ 0)

    op = RossbyNumber(model, model.velocities..., model.coriolis)
    @test op isa AbstractOperation
    Ro = Field(op)
    @test all(interior(compute!(Ro)) .≈ 0)

    op = RossbyNumber(model; dUdy_bg=1, dVdx_bg=1)
    @test op isa AbstractOperation
    Ro = Field(op)
    @test all(interior(compute!(Ro)) .≈ 0)

    op = StrainRateTensorModulus(model)
    @test op isa AbstractOperation
    S = Field(op)
    @test all(interior(compute!(S)) .≈ 0)

    op = VorticityTensorModulus(model)
    @test op isa AbstractOperation
    Ω = Field(op)
    @test all(interior(compute!(Ω)) .≈ 0)

    op = QVelocityGradientTensorInvariant(model)
    @allowscalar @test op == Oceanostics.Q(model)
    @test op isa AbstractOperation
    q = Field(op)
    @test all(interior(compute!(q)) .≈ 0)

    return nothing
end

function test_auxiliary_functions(model)
    set!(model, u=1, v=2)
    fields_without_means = perturbation_fields(model; u=1, v=2)
    compute!(fields_without_means)
    @test all(Array(interior(fields_without_means.u)) .== 0)
    @test all(Array(interior(fields_without_means.v)) .== 0)
    return
end
#---

@testset "Flow diagnostics tests" begin
    @info "  Testing flow diagnostics"
    for (grid_class, grid) in zip(keys(grids), values(grids))
        @info "    with $grid_class"
        for model_type in model_types
            @info "      with $model_type"
            model = model_type(; grid, model_kwargs...)

            @info "          Testing auxiliary functions"
            test_auxiliary_functions(model)

            @info "          Testing velocity-only diagnostics"
            test_velocity_only_flow_diagnostics(model)
        end
    end
end
