using Oceananigans
using Oceanostics

grid = RectilinearGrid(size=(4,4,4), extent=(1,1,1))
model = IncompressibleModel(grid=grid)
u, v, w = model.velocities

ke = Oceanostics.KineticEnergy(model, u, v, w)

SPx = Oceanostics.XShearProduction(model, u, v, w, 0, 0, 0)
SPy = Oceanostics.YShearProduction(model, u, v, w, 0, 0, 0)
SPz = Oceanostics.ZShearProduction(model, u, v, w, 0, 0, 0)

#ε_iso = Oceanostics.IsotropicViscousDissipation(model, ν, u, v, w)
#ε_ani = Oceanostics.AnisotropicViscousDissipation(model, ν, ν, ν, u, v, w)
