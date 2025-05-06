using Oceananigans
using CairoMakie
using Oceananigans.Grids: inactive_node
using Oceananigans.Fields: @compute
using Oceananigans.AbstractOperations: volume
using Oceanostics
using Test

Lx, Lz = 1, 1
Nx = Nz = 5

grid_base = RectilinearGrid(topology = (Bounded, Flat, Bounded), size = (Nx, Nz), x = (0, Lx), z = (0, Lz))
flat_bottom(x) = x
grid = ImmersedBoundaryGrid(grid_base, GridFittedBottom(flat_bottom))

c = CenterField(grid)
set!(c, (x, z) -> z)

fig1 = Figure()
axis1 = Axis(fig1[1, 1])
hm1 = heatmap!(axis1, c, colorrange=(0, 1))
Colorbar(fig1[1, 2], hm1; label="c")

Δv = volume(1, 1, 1, grid, Center(), Center(), Center())
bottom_mass_truth = sum(znodes(grid, Center())[2:end]) * Δv

@inline bottom_adjacent_node(i, j, k, grid, LX, LY, LZ) = !inactive_node(i, j, k, grid, LX, LY, LZ) && inactive_node(i, j, k - 1, grid, LX, LY, LZ)
ban = KernelFunctionOperation{Center, Center, Center}(bottom_adjacent_node, grid, Center(), Center(), Center())

axis2 = Axis(fig1[2, 1])
hm2 = heatmap!(axis2, ban, colorrange=(0, 1))
Colorbar(fig1[2, 2], hm2; label="mask")

cᵇ = ban * c
cᵇ = BottomCellValue(c)

axis3 = Axis(fig1[3, 1])
hm3 = heatmap!(axis3, cᵇ, colorrange=(0, 1))
Colorbar(fig1[3, 2], hm3; label="c * mask")

@compute bottom_mass = Field(Integral(cᵇ))
@test bottom_mass[] ≈ bottom_mass_truth


flat_bottom(x) = x - 1/2
grid = ImmersedBoundaryGrid(grid_base, GridFittedBottom(flat_bottom))

c = CenterField(grid)
set!(c, (x, z) -> z)

fig2 = Figure()
axis1 = Axis(fig2[1, 1])
hm1 = heatmap!(axis1, c, colorrange=(0, 1))
Colorbar(fig2[1, 2], hm1; label="c")

x = xnodes(grid, Center())
z = znodes(grid, Center())
domain_bottom_mass = length(x[ x.>= 0.5 ]) * z[1] * Δv
immersed_bottom_mass = sum(z[2 : grid.Nz÷2 + 1]) * Δv
bottom_mass_truth = domain_bottom_mass + immersed_bottom_mass

ban = KernelFunctionOperation{Center, Center, Center}(bottom_adjacent_node, grid, Center(), Center(), Center())

axis2 = Axis(fig2[2, 1])
hm2 = heatmap!(axis2, ban, colorrange=(0, 1))
Colorbar(fig2[2, 2], hm2; label="mask")

cᵇ = ban * c
cᵇ = BottomCellValue(c)

axis3 = Axis(fig2[3, 1])
hm3 = heatmap!(axis3, cᵇ, colorrange=(0, 1))
Colorbar(fig2[3, 2], hm3; label="c * mask")

@compute bottom_mass = Field(Integral(cᵇ))
@test bottom_mass[] ≈ bottom_mass_truth

