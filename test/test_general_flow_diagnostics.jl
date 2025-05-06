using Test
using Oceananigans
using Oceananigans.Fields: @compute
using Oceananigans.AbstractOperations: volume
using Oceanostics

@testset "General flow diagnostics" begin
    @testset "BottomCellValue" begin
        # Test case 1: Flat bottom at z=0
        @testset "Flat bottom" begin
            Lx, Lz = 1, 1
            Nx = Nz = 5

            grid_base = RectilinearGrid(topology = (Bounded, Flat, Bounded),
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

            @compute bottom_mass = Field(Integral(cᵇ))
            @test bottom_mass[] ≈ bottom_mass_truth
        end

        # Test case 2: Sloped immersed bottom
        @testset "Sloped immersed bottom" begin
            Lx, Lz = 1, 1
            Nx = Nz = 5

            grid_base = RectilinearGrid(topology = (Bounded, Flat, Bounded),
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

            @compute bottom_mass = Field(Integral(cᵇ))
            @test bottom_mass[] ≈ bottom_mass_truth
        end
    end
end 
