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
    set!(model, u=u₀, v=v₀, w=0, enforce_incompressibility=false)

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

        @test getindex(S, idxs...) ≈ √2*α
        @test getindex(Ω, idxs...) ≈ 0
        @test getindex(q, idxs...) ≈ (getindex(Ω, idxs...)^2 - getindex(S, idxs...)^2)/2 ≈ -α^2
        @test getindex(ε, idxs...) ≈ 2 * ν * getindex(S, idxs...)^2
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
    set!(model, u=u₀, v=v₀, w=0, enforce_incompressibility=false)

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

        @test getindex(S, idxs...) ≈ 0
        @test getindex(Ω, idxs...) ≈ ζ/√2
        @test getindex(q, idxs...) ≈ (getindex(Ω, idxs...)^2 - getindex(S, idxs...)^2)/2 ≈ ζ^2/4
        @test getindex(ε, idxs...) ≈ 0
    end
end
"""
Test that KineticEnergyDissipationRate, StrainRateTensorModulus, VorticityTensorModulus
and QVelocityGradientTensorInvariant have the right values for a uniform shear flow.
"""

function test_uniform_shear_flow(grid; model_type=NonhydrostaticModel, closure=ScalarDiffusivity(ν=1), σ=1)
    model = model_type(grid; closure)
    u₀(x, y, z) = σ * y
    set!(model, u=u₀, v=0, w=0, enforce_incompressibility=false)

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
        for closure in closures
            @info "        with $(summary(closure))"
            @info "          Testing uniform strain flow"
            test_uniform_strain_flow(grid; closure, α=3)

            @info "          Testing solid body rotation flow"
            test_solid_body_rotation_flow(grid; closure, ζ=3)

            @info "          Testing uniform shear flow"
            test_uniform_shear_flow(grid; closure, σ=3)
        end
    end
end
