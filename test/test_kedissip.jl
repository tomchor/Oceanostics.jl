using Oceananigans
using Oceanostics
using Oceananigans.TurbulenceClosures: HorizontalFormulation, VerticalFormulation
using Oceananigans.TurbulenceClosures: diffusive_flux_x, diffusive_flux_y, diffusive_flux_z
import Oceananigans.TurbulenceClosures: viscous_flux_ux, viscous_flux_uy, viscous_flux_uz, 
                                        viscous_flux_vx, viscous_flux_vy, viscous_flux_vz,
                                        viscous_flux_wx, viscous_flux_wy, viscous_flux_wz

N = 4
ν = 2
#closure = (ScalarDiffusivity(HorizontalFormulation(), κ=2κ),
#           ScalarDiffusivity(VerticalFormulation(), κ=.5κ))
closure = (ScalarDiffusivity(ν=ν))

grid = RectilinearGrid(topology=(Periodic, Periodic, Periodic),
                       size=(N,N,N), extent=(1,1,1))
model = NonhydrostaticModel(grid=grid, closure=closure)

# A kind of convoluted way to create x-periodic, resolved initial noise
σx = 2grid.Δxᶜᵃᵃ # x length scale of the noise
σy = 2grid.Δyᵃᶜᵃ # x length scale of the noise
σz = 2grid.Δzᵃᵃᶜ # z length scale of the noise

N = 2^4 # How many Gaussians do we want sprinkled throughout the domain?
x₀ = grid.Lx * rand(N); y₀ = grid.Ly * rand(N); z₀ = -grid.Lz * rand(N) # Locations of the Gaussians

xₚ = x₀ .+ (grid.Lx .* [-2;;-1;;0;;1;;2]) # Make that noise periodic by "infinite" horizontal reflection
yₚ = y₀ .+ (grid.Ly .* [-2;;-1;;0;;1;;2]) # Make that noise periodic by "infinite" horizontal reflection
zₚ = z₀ .+ (grid.Lz .* [-2;;-1;;0;;1;;2]) # Make that noise periodic by "infinite" horizontal reflection

resolved_noise(x, y, z) = sum(@. exp(-(x-xₚ)^2/σx^2 -(y-yₚ)^2/σy^2 -(z-zₚ)^2/σz^2))
set!(model, u=resolved_noise)
u, v, w = model.velocities

Δt = 0.5grid.Δxᶜᵃᵃ/maximum(u)
simulation = Simulation(model; Δt=Δt, stop_iteration=ceil(Int64, 300Δt))

ε = Oceanostics.TKEBudgetTerms.IsotropicPseudoViscousDissipationRate(model)
e = Oceanostics.TKEBudgetTerms.KineticEnergy(model)

ddx² = Field(∂x(u)^2 + ∂x(v)^2 + ∂x(w)^2)
ddy² = Field(∂y(u)^2 + ∂y(v)^2 + ∂y(w)^2)
ddz² = Field(∂z(u)^2 + ∂z(v)^2 + ∂z(w)^2)
ε2 = Field(ν * (ddx² + ddy² + ddz²))


#compute!(Field(ε))

∫εdV   = Integral(ε)
∫ε2dV   = Integral(ε2)
∫edV   = Integral(e)

outputs = (; ε, ∫εdV,
           ε2, ∫ε2dV,
           e, ∫edV,
           )

dt = 1e-4
simulation.output_writers[:tracer] = NetCDFOutputWriter(model, outputs;
                                                        filename = "ke_dissip.nc",
                                                        schedule = TimeInterval(dt),
                                                        overwrite_existing = true)
run!(simulation)


using NCDatasets, GLMakie

ds = NCDataset(simulation.output_writers[:tracer].filepath, "r")

lines(ds["time"], ds["∫εdV"], label="∫εdV (new conservative form)", linestyle=:dashdot)
lines!(ds["time"], ds["∫ε2dV"], label="∫ε2dV (old non-conservative form)", linestyle=:dot)

∂ₜ∫edV = -diff(ds["∫edV"]) / dt
lines!(ds["time"][2:end], ∂ₜ∫edV, label="∂(∫edV)/∂ₜ")
axislegend()

#∫∫εdVdt_final = ds["∫edV"][1] .- (cumsum(ds["∫εdV"]) * simulation.Δt)[end]
#∫edV_final   = ds["∫edV"][end]
#@show ∫∫εdVdt_final ∫edV_final

@show ds["∫εdV"][2:end] ./ ∂ₜ∫edV

close(ds)

