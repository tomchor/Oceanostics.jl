using Test
using CUDA: has_cuda_gpu, @allowscalar

using Oceananigans
using Oceananigans.TurbulenceClosures.Smagorinskys: LagrangianAveraging, DynamicCoefficient

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

grid_noise(x, y, z) = randn()
#---

#+++ Test options
closures = (ScalarDiffusivity(ν=1e-6, κ=1e-7),
            SmagorinskyLilly(),
            DynamicSmagorinsky(averaging=(1, 2)),
            DynamicSmagorinsky(averaging=LagrangianAveraging()),
            (ScalarDiffusivity(ν=1e-6, κ=1e-7), AnisotropicMinimumDissipation()),)

grids = Dict("regular grid" => regular_grid,
             "stretched grid" => stretched_grid)

model_types = (NonhydrostaticModel,
               HydrostaticFreeSurfaceModel)
#---

#+++ Test functions
"""
Test that KineticEnergyDissipationRate, StrainRateTensorModulus, VorticityTensorModulus
and QVelocityGradientTensorInvariant have the right values for a uniform strain flow.
"""
function test_uniform_strain_flow(grid; model_type=NonhydrostaticModel, closure=ScalarDiffusivity(ν=1), α=1)
    model = model_type(grid; closure)
    u₀(x, y, z) = +α*x
    v₀(x, y, z) = -α*y
    if model isa NonhydrostaticModel
        set!(model, u=u₀, v=v₀, w=0, enforce_incompressibility=false)
    else
        set!(model, u=u₀, v=v₀)
    end

    u, v, w = model.velocities

    ε = Field(KineticEnergyEquation.KineticEnergyDissipationRate(model))
    S = Field(StrainRateTensorModulus(model))
    Ω = Field(VorticityTensorModulus(model))
    q = Field(QVelocityGradientTensorInvariant(model))

    Sij = StrainRateTensor(model)
    S₁₁ = Field(Sij.S₁₁); S₂₂ = Field(Sij.S₂₂); S₃₃ = Field(Sij.S₃₃)
    S₁₂ = Field(Sij.S₁₂); S₁₃ = Field(Sij.S₁₃); S₂₃ = Field(Sij.S₂₃)

    idxs = (model.grid.Nx÷2, model.grid.Ny÷2, model.grid.Nz÷2) # Get a value far from boundaries

    if model.closure isa Tuple
        ν_field = Field(sum(viscosity(model.closure, model.closure_fields)))
    else
        ν_field = viscosity(model.closure, model.closure_fields)
    end

    @allowscalar begin
        ν = ν_field isa Number ? ν_field : getindex(ν_field, idxs...)

        @test getindex(S, idxs...) ≈ √2*α
        @test ≈(getindex(Ω, idxs...), 0, atol=10eps())
        @test getindex(q, idxs...) ≈ (getindex(Ω, idxs...)^2 - getindex(S, idxs...)^2)/2 ≈ -α^2
        @test getindex(ε, idxs...) ≈ 2 * ν * getindex(S, idxs...)^2

        # Strain rate tensor for uⱼ = (αx, -αy, 0) is S = diag(α, -α, 0), off-diagonals zero
        @test getindex(S₁₁, idxs...) ≈ +α
        @test getindex(S₂₂, idxs...) ≈ -α
        @test ≈(getindex(S₃₃, idxs...), 0, atol=10eps())
        @test ≈(getindex(S₁₂, idxs...), 0, atol=10eps())
        @test ≈(getindex(S₁₃, idxs...), 0, atol=10eps())
        @test ≈(getindex(S₂₃, idxs...), 0, atol=10eps())

        # The modulus is recovered from the components: ‖S‖ = √(Sᵢⱼ Sᵢⱼ)
        SᵢⱼSᵢⱼ = getindex(S₁₁, idxs...)^2 + getindex(S₂₂, idxs...)^2 + getindex(S₃₃, idxs...)^2 +
                 2 * (getindex(S₁₂, idxs...)^2 + getindex(S₁₃, idxs...)^2 + getindex(S₂₃, idxs...)^2)
        @test √(SᵢⱼSᵢⱼ) ≈ getindex(S, idxs...)
    end

    return nothing
end

"""
Test that KineticEnergyDissipationRate, StrainRateTensorModulus, VorticityTensorModulus
and QVelocityGradientTensorInvariant have the right values for a solid body rotation flow.
"""
function test_solid_body_rotation_flow(grid; model_type=NonhydrostaticModel, closure=ScalarDiffusivity(ν=1), ζ=1)
    model = model_type(grid; closure)
    u₀(x, y, z) = +ζ*y / 2
    v₀(x, y, z) = -ζ*x / 2
    if model isa NonhydrostaticModel
        set!(model, u=u₀, v=v₀, w=0, enforce_incompressibility=false)
    else
        set!(model, u=u₀, v=v₀)
    end

    u, v, w = model.velocities

    ε = Field(KineticEnergyEquation.KineticEnergyDissipationRate(model))
    S = Field(StrainRateTensorModulus(model))
    Ω = Field(VorticityTensorModulus(model))
    q = Field(QVelocityGradientTensorInvariant(model))

    Ωij = VorticityTensor(model)
    Ω₁₂ = Field(Ωij.Ω₁₂); Ω₁₃ = Field(Ωij.Ω₁₃); Ω₂₃ = Field(Ωij.Ω₂₃)

    idxs = (model.grid.Nx÷2, model.grid.Ny÷2, model.grid.Nz÷2) # Get a value far from boundaries

    if model.closure isa Tuple
        ν_field = Field(sum(viscosity(model.closure, model.closure_fields)))
    else
        ν_field = viscosity(model.closure, model.closure_fields)
    end

    @allowscalar begin
        ν = ν_field isa Number ? ν_field : getindex(ν_field, idxs...)

        @test getindex(S, idxs...) ≈ 0
        @test getindex(Ω, idxs...) ≈ ζ/√2
        @test getindex(q, idxs...) ≈ (getindex(Ω, idxs...)^2 - getindex(S, idxs...)^2)/2 ≈ ζ^2/4
        @test getindex(ε, idxs...) ≈ 0

        # Vorticity tensor for uⱼ = (ζy/2, -ζx/2, 0) has only Ω₁₂ = ½(∂u/∂y - ∂v/∂x) = ζ/2 nonzero
        @test getindex(Ω₁₂, idxs...) ≈ +ζ/2
        @test ≈(getindex(Ω₁₃, idxs...), 0, atol=10eps())
        @test ≈(getindex(Ω₂₃, idxs...), 0, atol=10eps())

        # The modulus is recovered from the components: ‖Ω‖ = √(Ωᵢⱼ Ωᵢⱼ) = √(2(Ω₁₂² + Ω₁₃² + Ω₂₃²))
        ΩᵢⱼΩᵢⱼ = 2 * (getindex(Ω₁₂, idxs...)^2 + getindex(Ω₁₃, idxs...)^2 + getindex(Ω₂₃, idxs...)^2)
        @test √(ΩᵢⱼΩᵢⱼ) ≈ getindex(Ω, idxs...)
    end
end
"""
Test that KineticEnergyDissipationRate, StrainRateTensorModulus, VorticityTensorModulus
and QVelocityGradientTensorInvariant have the right values for a uniform shear flow.
"""

function test_uniform_shear_flow(grid; model_type=NonhydrostaticModel, closure=ScalarDiffusivity(ν=1), σ=1)
    model = model_type(grid; closure)
    u₀(x, y, z) = σ * y
    if model isa NonhydrostaticModel
        set!(model, u=u₀, v=0, w=0, enforce_incompressibility=false)
    else
        set!(model, u=u₀)
    end

    u, v, w = model.velocities

    ε = Field(KineticEnergyEquation.KineticEnergyDissipationRate(model))
    S = Field(StrainRateTensorModulus(model))
    Ω = Field(VorticityTensorModulus(model))
    q = Field(QVelocityGradientTensorInvariant(model))

    idxs = (model.grid.Nx÷2, model.grid.Ny÷2, model.grid.Nz÷2) # Get a value far from boundaries

    if model.closure isa Tuple
        ν_field = Field(sum(viscosity(model.closure, model.closure_fields)))
    else
        ν_field = viscosity(model.closure, model.closure_fields)
    end

    @allowscalar begin
        ν = ν_field isa Number ? ν_field : getindex(ν_field, idxs...)

        @test getindex(S, idxs...) ≈ σ/√2
        @test getindex(Ω, idxs...) ≈ σ/√2
        @test ≈(getindex(q, idxs...), (getindex(Ω, idxs...)^2 - getindex(S, idxs...)^2)/2, atol=eps())
        @test ≈(getindex(q, idxs...), 0, atol=eps())
        @test getindex(ε, idxs...) ≈ 2 * ν * getindex(S, idxs...)^2
    end
end
#---

@testset "Known flows" begin
    @info "  Testing known flows"
    for (grid_class, grid) in zip(keys(grids), values(grids))
        @info "    with $grid_class"
        for model_type in model_types
            @info "      with $model_type"
            for closure in closures
                @info "        with $(summary(closure))"
                @info "          Testing uniform strain flow"
                test_uniform_strain_flow(grid; model_type, closure, α=3)

                @info "          Testing solid body rotation flow"
                test_solid_body_rotation_flow(grid; model_type, closure, ζ=3)

                @info "          Testing uniform shear flow"
                test_uniform_shear_flow(grid; model_type, closure, σ=3)
            end
        end
    end
end
