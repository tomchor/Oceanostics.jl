using Oceananigans
using Oceanostics

grid = RegularRectilinearGrid(size=(4,4,4), extent=(1,1,1))
model = IncompressibleModel(grid=grid)
u, v, w = model.velocities
ν = model.closure.ν

ke = Oceanostics.TurbulentKineticEnergyTerms.KineticEnergy(model, u, v, w)

SPx = Oceanostics.TurbulentKineticEnergyTerms.ShearProduction_x(model, u, v, w, 0, 0, 0)
SPy = Oceanostics.TurbulentKineticEnergyTerms.ShearProduction_y(model, u, v, w, 0, 0, 0)
SPz = Oceanostics.TurbulentKineticEnergyTerms.ShearProduction_z(model, u, v, w, 0, 0, 0)

ε_iso = Oceanostics.TurbulentKineticEnergyTerms.IsotropicViscousDissipation(model, ν, u, v, w)
ε_ani = Oceanostics.TurbulentKineticEnergyTerms.AnisotropicViscousDissipation(model, ν, ν, ν, u, v, w)
