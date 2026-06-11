using Test
using CUDA: has_cuda_gpu, @allowscalar
using Random
using Oceananigans
using Oceananigans: fill_halo_regions!
using Oceananigans.AbstractOperations: volume, @at
using Oceananigans.Grids: Center, Face
using Oceanostics
using Oceanostics: SubfilterCovariance, GaussianFilter

arch = has_cuda_gpu() ? GPU() : CPU()

@testset "General flow diagnostics on $(typeof(arch).name.wrapper)" begin
    @testset "BottomCellValue" begin
        # Test case 1: Flat bottom at z=0
        @testset "Flat bottom" begin
            Lx, Lz = 1, 1
            Nx = Nz = 5

            grid_base = RectilinearGrid(arch, topology = (Bounded, Flat, Bounded),
                                        size = (Nx, Nz),
                                        x = (0, Lx),
                                        z = (0, Lz))
            
            flat_bottom(x) = x
            grid = ImmersedBoundaryGrid(grid_base, GridFittedBottom(flat_bottom))

            c = CenterField(grid)
            set!(c, (x, z) -> z)

            Δv = volume(1, 1, 1, grid, Center(), Center(), Center())
            bottom_mass_truth = sum(znodes(grid, Center())[2:end]) * Δv

            cᵇ = BottomCellValue(c)

            bottom_mass = Field(Integral(cᵇ))
            @allowscalar @test bottom_mass[] ≈ bottom_mass_truth
        end

        # Test case 2: Sloped immersed bottom
        @testset "Sloped immersed bottom" begin
            Lx, Lz = 1, 1
            Nx = Nz = 5

            grid_base = RectilinearGrid(arch, topology = (Bounded, Flat, Bounded),
                                        size = (Nx, Nz),
                                        x = (0, Lx),
                                        z = (0, Lz))
            
            flat_bottom(x) = x - 1/2
            grid = ImmersedBoundaryGrid(grid_base, GridFittedBottom(flat_bottom))

            c = CenterField(grid)
            set!(c, (x, z) -> z)

            x = xnodes(grid, Center())
            z = znodes(grid, Center())
            Δv = volume(1, 1, 1, grid, Center(), Center(), Center())
            domain_bottom_mass = length(x[x .>= 0.5]) * z[1] * Δv
            immersed_bottom_mass = sum(z[2 : grid.Nz÷2 + 1]) * Δv
            bottom_mass_truth = domain_bottom_mass + immersed_bottom_mass

            cᵇ = BottomCellValue(c)

            bottom_mass = Field(Integral(cᵇ))
            @allowscalar @test bottom_mass[] ≈ bottom_mass_truth
        end
    end

    @testset "SubfilterCovariance" begin
        Random.seed!(1234)
        sf_grid = RectilinearGrid(arch, size=(8, 8, 8), x=(0, 1), y=(0, 1), z=(0, 1),
                                  topology=(Periodic, Periodic, Periodic))
        model = NonhydrostaticModel(sf_grid; tracers=:c)
        u, v, w = model.velocities
        c = model.tracers.c
        set!(model, u=(x, y, z) -> randn(), v=(x, y, z) -> randn(),
                    w=(x, y, z) -> randn(), c=(x, y, z) -> randn())
        fill_halo_regions!(model.velocities)
        fill_halo_regions!(c)

        filt(ψ) = GaussianFilter(ψ; dims=(1, 2, 3), σ=0.1)
        loc = (Center, Center, Center)

        # matches the by-hand covariance formula for co-located operands (guards wiring/sign)
        b = Field(@at loc u); fill_halo_regions!(b)
        a_loc = Field(@at loc c); b_loc = Field(@at loc b)
        τ_hand = Field(Field(filt(Field(a_loc * b_loc))) - Field(filt(a_loc)) * Field(filt(b_loc)))
        @test all(interior(Field(SubfilterCovariance(c, b, filt; loc))) .≈ interior(τ_hand))

        # subfilter tracer flux special case reproduces the hand-rolled construction (cf. spatial_filtering.jl)
        uᶜ = Field(@at loc u); ū = Field(filt(uᶜ)); c̄ = Field(filt(c)); ūc̄ = Field(filt(Field(uᶜ * c)))
        τx_hand = Field(ūc̄ - ū * c̄)
        @test all(interior(Field(SubfilterCovariance(u, c, filt; loc))) .≈ interior(τx_hand))

        # subfilter momentum-stress special case
        vᶜ = Field(@at loc v)
        τxy_hand = Field(Field(filt(Field(uᶜ * vᶜ))) - Field(filt(uᶜ)) * Field(filt(vᶜ)))
        @test all(interior(Field(SubfilterCovariance(u, v, filt; loc))) .≈ interior(τxy_hand))

        # uniform fields ⇒ subfilter flux ≈ 0 (a normalized filter preserves constants)
        uniform_model = NonhydrostaticModel(sf_grid; tracers=:c)
        set!(uniform_model, u=(x, y, z) -> 2.0, c=(x, y, z) -> -3.0)
        fill_halo_regions!(uniform_model.velocities); fill_halo_regions!(uniform_model.tracers.c)
        τ_uniform = Field(SubfilterCovariance(uniform_model.velocities.u, uniform_model.tracers.c, filt; loc))
        @test all(abs.(interior(τ_uniform)) .< 1e-12)
    end
end
