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

    τij = StressTensor(model)
    @test keys(τij) == (:τ₁₁, :τ₂₂, :τ₃₃, :τ₁₂, :τ₁₃, :τ₂₃)
    @test location(τij.τ₁₁) == (Center, Center, Center)
    @test location(τij.τ₂₂) == (Center, Center, Center)
    @test location(τij.τ₃₃) == (Center, Center, Center)
    @test location(τij.τ₁₂) == (Face, Face, Center)
    @test location(τij.τ₁₃) == (Face, Center, Face)
    @test location(τij.τ₂₃) == (Center, Face, Face)
    @test τij == StressTensor(model.grid, model.velocities...) # field-based constructor agrees
    for τᵢⱼ in τij
        @test Field(τᵢⱼ) isa Field # every component is computable
    end

    # `dims` selects sub-dimensional stress tensors: τᵢⱼ is kept only if both i and j are in `dims`
    @test keys(StressTensor(model; dims=(1, 2, 3))) == keys(τij)
    @test keys(StressTensor(model; dims=(1, 2))) == (:τ₁₁, :τ₂₂, :τ₁₂)
    @test keys(StressTensor(model; dims=(1, 3))) == (:τ₁₁, :τ₃₃, :τ₁₃)
    @test keys(StressTensor(model; dims=(2, 3))) == (:τ₂₂, :τ₃₃, :τ₂₃)
    @test keys(StressTensor(model; dims=(1,)))   == (:τ₁₁,)
    @test keys(StressTensor(model; dims=(2,)))   == (:τ₂₂,)
    @test keys(StressTensor(model; dims=(3,)))   == (:τ₃₃,)
    @test keys(StressTensor(model; dims=(3, 1))) == (:τ₁₁, :τ₃₃, :τ₁₃) # order of `dims` doesn't matter

    # selected components are the very same KFOs as in the full tensor, and `dims` is forwarded
    τxz = StressTensor(model; dims=(1, 3))
    @test (τxz.τ₁₁, τxz.τ₃₃, τxz.τ₁₃) == (τij.τ₁₁, τij.τ₃₃, τij.τ₁₃)
    @test τxz == StressTensor(model.grid, model.velocities...; dims=(1, 3))

    # invalid `dims` are rejected
    @test_throws ArgumentError StressTensor(model; dims=(1, 4))
    @test_throws ArgumentError StressTensor(model; dims=(1, 1))
    @test_throws ArgumentError StressTensor(model; dims=())
    @test_throws ArgumentError StressTensor(model; dims=1)

    op = VorticityTensorModulus(model)
    @test op isa VorticityTensorModulus
    Ω = Field(op)
    @test all(interior(Ω) .≈ 0)

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
