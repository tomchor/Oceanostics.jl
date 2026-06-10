using Test
using CUDA: has_cuda_gpu
using Random
using Oceananigans
using Oceananigans: fill_halo_regions!
using Oceananigans.AbstractOperations: @at
using Oceananigans.Grids: Center, Face

using Oceanostics
using Oceanostics: SubfilterFlux, GaussianFilter

arch = has_cuda_gpu() ? GPU() : CPU()

# A small, fully periodic model so the filter has clean periodic boundaries everywhere.
function test_model(; Nx=8, Ny=8, Nz=8)
    grid = RectilinearGrid(arch, size=(Nx, Ny, Nz), x=(0, 1), y=(0, 1), z=(0, 1),
                           topology=(Periodic, Periodic, Periodic))
    model = NonhydrostaticModel(grid; tracers=:c)
    return model
end

# `filt` filters in all three directions with a modest Gaussian width.
filt(ψ) = GaussianFilter(ψ; dims=(1, 2, 3), σ=0.1)

@testset "SubfilterFlux" begin
    Random.seed!(1234)

    model = test_model()
    u, v, w = model.velocities
    c = model.tracers.c

    set!(model, u=(x, y, z) -> randn(), v=(x, y, z) -> randn(),
                w=(x, y, z) -> randn(), c=(x, y, z) -> randn())
    fill_halo_regions!(model.velocities)
    fill_halo_regions!(c)

    loc = (Center, Center, Center)

    #+++ Matches the by-hand formula for co-located operands (guards wiring/sign)
    # Use two already-co-located Center fields so the @at is a no-op and the only
    # thing under test is the covariance assembly itself.
    a = c
    b = Field(@at loc u)     # interpolate u to centers once, treat as a plain field
    fill_halo_regions!(b)

    τ_diag = Field(SubfilterFlux(a, b, filt; loc=loc))

    a_loc = Field(@at loc a)
    b_loc = Field(@at loc b)
    τ_hand = Field(Field(filt(Field(a_loc * b_loc))) - Field(filt(a_loc)) * Field(filt(b_loc)))

    @test all(interior(τ_diag) .≈ interior(τ_hand))
    #---

    #+++ Subfilter tracer flux special case reproduces the hand-rolled construction
    # This mirrors docs/examples/spatial_filtering.jl ("Subfilter tracer flux"):
    #   uᶜ = Field(@at ccc u);  ū = filt(uᶜ);  c̄ = filt(c)
    #   ūc̄ = filt(Field(uᶜ * c));  τx = ūc̄ - ū * c̄
    uᶜ = Field(@at loc u)
    ū  = Field(filt(uᶜ))
    c̄  = Field(filt(c))
    ūc̄ = Field(filt(Field(uᶜ * c)))
    τx_hand = Field(ūc̄ - ū * c̄)

    τx_diag = Field(SubfilterFlux(u, c, filt; loc=loc))

    @test all(interior(τx_diag) .≈ interior(τx_hand))
    #---

    #+++ Uniform fields give exactly zero subfilter flux (to machine precision)
    # filter(const) = const, so filter(ab) = filter(a)filter(b) for constant a, b.
    uniform_model = test_model()
    set!(uniform_model, u=(x, y, z) -> 2.0, c=(x, y, z) -> -3.0)
    fill_halo_regions!(uniform_model.velocities)
    fill_halo_regions!(uniform_model.tracers.c)

    τ_uniform = Field(SubfilterFlux(uniform_model.velocities.u, uniform_model.tracers.c, filt; loc=loc))
    @test all(abs.(interior(τ_uniform)) .< 1e-12)
    #---

    #+++ Momentum-stress special case is also assembled correctly
    τxy_diag = Field(SubfilterFlux(u, v, filt; loc=loc))
    uᶜ = Field(@at loc u)
    vᶜ = Field(@at loc v)
    τxy_hand = Field(Field(filt(Field(uᶜ * vᶜ))) - Field(filt(uᶜ)) * Field(filt(vᶜ)))
    @test all(interior(τxy_diag) .≈ interior(τxy_hand))
    #---
end
