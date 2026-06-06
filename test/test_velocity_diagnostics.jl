using Test
using CUDA: has_cuda_gpu, @allowscalar

using Oceananigans
using Oceananigans.Fields: location

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
model_kwargs = (coriolis = FPlane(f=1e-4),)

coriolis_formulations = (nothing,
                         FPlane(f=1e-4),
                         ConstantCartesianCoriolis(fx=1e-4, fy=1e-4, fz=1e-4))

grids = Dict("regular grid" => regular_grid,
             "stretched grid" => stretched_grid)

model_types = (NonhydrostaticModel,
               HydrostaticFreeSurfaceModel)
#---

#+++ Test functions
function test_velocity_only_flow_diagnostics(model)
    op = RossbyNumber(model;)
    @test op isa RossbyNumber
    Ro = Field(op)
    @test all(interior(Ro) .≈ 0)

    op = RossbyNumber(model, model.velocities..., model.coriolis)
    @test op isa RossbyNumber
    Ro = Field(op)
    @test all(interior(Ro) .≈ 0)

    op = RossbyNumber(model; dUdy_bg=1, dVdx_bg=1)
    @test op isa RossbyNumber
    Ro = Field(op)
    @test all(interior(Ro) .≈ 0)

    op = StrainRateTensorModulus(model)
    @test op isa StrainRateTensorModulus
    S = Field(op)
    @test all(interior(S) .≈ 0)

    Sij = StrainRateTensor(model)
    @test keys(Sij) == (:S₁₁, :S₂₂, :S₃₃, :S₁₂, :S₁₃, :S₂₃)
    @test location(Sij.S₁₁) == (Center, Center, Center)
    @test location(Sij.S₁₂) == (Face, Face, Center)
    @test location(Sij.S₁₃) == (Face, Center, Face)
    @test location(Sij.S₂₃) == (Center, Face, Face)
    @test Sij == StrainRateTensor(model.grid, model.velocities...) # field-based constructor agrees
    for Sᵢⱼ in Sij
        @test all(interior(Field(Sᵢⱼ)) .≈ 0)
    end

    # `dims` selects sub-dimensional strain tensors: Sᵢⱼ is kept only if both i and j are in `dims`
    @test keys(StrainRateTensor(model; dims=(1, 2, 3))) == keys(Sij)
    @test keys(StrainRateTensor(model; dims=(1, 2))) == (:S₁₁, :S₂₂, :S₁₂)
    @test keys(StrainRateTensor(model; dims=(1, 3))) == (:S₁₁, :S₃₃, :S₁₃)
    @test keys(StrainRateTensor(model; dims=(2, 3))) == (:S₂₂, :S₃₃, :S₂₃)
    @test keys(StrainRateTensor(model; dims=(1,)))   == (:S₁₁,)
    @test keys(StrainRateTensor(model; dims=(2,)))   == (:S₂₂,)
    @test keys(StrainRateTensor(model; dims=(3,)))   == (:S₃₃,)
    @test keys(StrainRateTensor(model; dims=(3, 1))) == (:S₁₁, :S₃₃, :S₁₃) # order of `dims` doesn't matter

    # selected components are the very same KFOs as in the full tensor, and `dims` is forwarded
    Sxz = StrainRateTensor(model; dims=(1, 3))
    @test (Sxz.S₁₁, Sxz.S₃₃, Sxz.S₁₃) == (Sij.S₁₁, Sij.S₃₃, Sij.S₁₃)
    @test Sxz == StrainRateTensor(model.grid, model.velocities...; dims=(1, 3))

    # invalid `dims` are rejected
    @test_throws ArgumentError StrainRateTensor(model; dims=(1, 4))
    @test_throws ArgumentError StrainRateTensor(model; dims=(1, 1))
    @test_throws ArgumentError StrainRateTensor(model; dims=())
    @test_throws ArgumentError StrainRateTensor(model; dims=1)

    op = VorticityTensorModulus(model)
    @test op isa VorticityTensorModulus
    Ω = Field(op)
    @test all(interior(Ω) .≈ 0)

    Ωij = VorticityTensor(model)
    @test keys(Ωij) == (:Ω₁₂, :Ω₁₃, :Ω₂₃) # antisymmetric tensor: only off-diagonals, no diagonals
    @test location(Ωij.Ω₁₂) == (Face, Face, Center)
    @test location(Ωij.Ω₁₃) == (Face, Center, Face)
    @test location(Ωij.Ω₂₃) == (Center, Face, Face)
    @test Ωij == VorticityTensor(model.grid, model.velocities...) # field-based constructor agrees
    for Ωᵢⱼ in Ωij
        @test all(interior(Field(Ωᵢⱼ)) .≈ 0)
    end

    # `dims` selects sub-dimensional vorticity tensors: Ωᵢⱼ is kept only if both i and j are in `dims`
    @test keys(VorticityTensor(model; dims=(1, 2, 3))) == keys(Ωij)
    @test keys(VorticityTensor(model; dims=(1, 2))) == (:Ω₁₂,)
    @test keys(VorticityTensor(model; dims=(1, 3))) == (:Ω₁₃,)
    @test keys(VorticityTensor(model; dims=(2, 3))) == (:Ω₂₃,)
    @test keys(VorticityTensor(model; dims=(3, 1))) == (:Ω₁₃,) # order of `dims` doesn't matter
    # every component couples two distinct directions, so a single-direction `dims` is empty
    @test keys(VorticityTensor(model; dims=(1,))) == ()
    @test keys(VorticityTensor(model; dims=(2,))) == ()
    @test keys(VorticityTensor(model; dims=(3,))) == ()

    # selected components are the very same KFOs as in the full tensor, and `dims` is forwarded
    Ωxz = VorticityTensor(model; dims=(1, 3))
    @test Ωxz.Ω₁₃ == Ωij.Ω₁₃
    @test Ωxz == VorticityTensor(model.grid, model.velocities...; dims=(1, 3))

    # invalid `dims` are rejected
    @test_throws ArgumentError VorticityTensor(model; dims=(1, 4))
    @test_throws ArgumentError VorticityTensor(model; dims=(1, 1))
    @test_throws ArgumentError VorticityTensor(model; dims=())
    @test_throws ArgumentError VorticityTensor(model; dims=1)

    op = QVelocityGradientTensorInvariant(model)
    @allowscalar @test op == Oceanostics.Q(model)
    @test op isa QVelocityGradientTensorInvariant
    q = Field(op)
    @test all(interior(q) .≈ 0)

    return nothing
end

function test_auxiliary_functions(model)
    set!(model, u=1, v=2)
    fields_without_means = perturbation_fields(model; u=1, v=2)
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
            model = model_type(grid; model_kwargs...)

            @info "          Testing auxiliary functions"
            test_auxiliary_functions(model)

            @info "          Testing velocity-only diagnostics"
            test_velocity_only_flow_diagnostics(model)
        end
    end
end
