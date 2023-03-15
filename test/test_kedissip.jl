using Oceananigans
using Oceanostics
using Oceananigans.TurbulenceClosures: HorizontalFormulation, VerticalFormulation
using Oceananigans.Fields: @compute

N = 16
ν = 1
#closure = (ScalarDiffusivity(HorizontalFormulation(), κ=2κ),
#           ScalarDiffusivity(VerticalFormulation(), κ=.5κ))
closure = (ScalarDiffusivity(ν=ν))

grid = RectilinearGrid(topology=(Periodic, Periodic, Periodic),
                       size=(N,N,N), extent=(1,1,1))
model = NonhydrostaticModel(grid=grid, advection=WENO(order=9), closure=closure,
                           auxiliary_fields=(; ∫∫εdVdt=0.0))

# A kind of convoluted way to create x-periodic, resolved initial noise
σx = 4grid.Δxᶜᵃᵃ # x length scale of the noise
σy = 4grid.Δyᵃᶜᵃ # x length scale of the noise
σz = 4grid.Δzᵃᵃᶜ # z length scale of the noise

N_gaussians = 16 # How many Gaussians do we want sprinkled throughout the domain?
x₀ = grid.Lx * rand(N_gaussians); y₀ = grid.Ly * rand(N_gaussians); z₀ = -grid.Lz * rand(N_gaussians) # Locations of the Gaussians

xₚ = x₀ .+ (grid.Lx .* [-2;;-1;;0;;1;;2]) # Make that noise periodic by "infinite" horizontal reflection
yₚ = y₀ .+ (grid.Ly .* [-2;;-1;;0;;1;;2]) # Make that noise periodic by "infinite" horizontal reflection
zₚ = z₀ .+ (grid.Lz .* [-2;;-1;;0;;1;;2]) # Make that noise periodic by "infinite" horizontal reflection

resolved_noise(x, y, z) = sum(@. exp(-(x-xₚ)^2/σx^2 -(y-yₚ)^2/σy^2 -(z-zₚ)^2/σz^2))
set!(model, u=resolved_noise)
u, v, w = model.velocities

using Statistics
u.data.parent .-= mean(u)
v.data.parent .-= mean(v)
w.data.parent .-= mean(w)

Δt = 0.01grid.Δxᶜᵃᵃ/maximum(u)
simulation = Simulation(model; Δt=Δt, stop_time=0.01)

wizard = TimeStepWizard(cfl=0.01, diffusive_cfl=0.01)
simulation.callbacks[:wizard] = Callback(wizard)

@compute ε = Field(Oceanostics.TKEBudgetTerms.IsotropicPseudoViscousDissipationRate(model))
@compute e = Field(Oceanostics.TKEBudgetTerms.KineticEnergy(model))

ddx² = Field(∂x(u)^2 + ∂x(v)^2 + ∂x(w)^2)
ddy² = Field(∂y(u)^2 + ∂y(v)^2 + ∂y(w)^2)
ddz² = Field(∂z(u)^2 + ∂z(v)^2 + ∂z(w)^2)
@compute ε2 = Field(ν * (ddx² + ddy² + ddz²))


@compute ∫εdV  = Field(Integral(ε))
@compute ∫ε2dV = Field(Integral(ε2))
@compute ∫edV  = Field(Integral(e))
@compute speed = Field(√(u^2 + v^2 + w^2))

using Printf
function progress(sim)
    compute!(∫edV)
    compute!(speed)
    @printf("Time: %s, Δt: %s,  e: %f, max(speed): %f \n", prettytime(sim), prettytime(sim.Δt), interior(∫edV)[1,1,1], maximum(speed))
end
simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))


∫edV_t⁰ = parent(∫edV)[1,1,1]
function accumulate_ε(sim)
    compute!(∫εdV)
    increment = sim.Δt * parent(∫εdV)[1,1,1]
    model.auxiliary_fields = (; ∫∫εdVdt = model.auxiliary_fields.∫∫εdVdt + increment)
    return nothing
end
simulation.callbacks[:integrate_ε] = Callback(accumulate_ε)

get_∫∫εdVdt(model) = model.auxiliary_fields.∫∫εdVdt

outputs = (; u, v, w, 
           ε, ∫εdV,
           ε2, ∫ε2dV,
           e, ∫edV,
           ∫∫εdVdt = get_∫∫εdVdt,
           )

dt = simulation.Δt
filename = "ke_dissip"
simulation.output_writers[:tracer] = NetCDFOutputWriter(model, outputs;
                                                        filename = filename,
                                                        schedule = TimeInterval(dt),
                                                        dimensions = (; ∫∫εdVdt = ()),
                                                        overwrite_existing = true)
run!(simulation)


compute!(∫edV)
∫edV_tᶠ = parent(∫edV)[1,1,1]
∫∫εdVdt_tᶠ = ∫edV_t⁰- model.auxiliary_fields.∫∫εdVdt
abs_error = (abs(∫∫εdVdt_tᶠ - ∫edV_tᶠ)/∫edV_tᶠ)




using NCDatasets, GLMakie

ds = NCDataset(simulation.output_writers[:tracer].filepath, "r")

∂ₜ∫edV = -diff(ds["∫edV"]) / dt

∫edV = ds["∫edV"]

∫∫εdVdt = cumsum(ds["∫εdV"])[1:end-1] * dt
pushfirst!(∫∫εdVdt, 0)
∫∫εdVdt = ∫edV[1] .- ∫∫εdVdt

∫∫ε2dVdt = cumsum(ds["∫ε2dV"])[1:end-1] * dt
pushfirst!(∫∫ε2dVdt, 0)
∫∫ε2dVdt = ∫edV[1] .- ∫∫ε2dVdt


∫∫εdVdt_final = ∫∫εdVdt[end]
∫edV_final   = ∫edV[end]

fig = Figure(resolution = (800, 800))

ax1 = Axis(fig[1, 1]; title = "KE")
ax2 = Axis(fig[2, 1]; title = "KE dissip rate")

lines!(ax1, ∫∫εdVdt, label="∫∫εdVdt (conservative form)", linestyle=:dashdot)
lines!(ax1, ∫∫ε2dVdt, label="∫∫ε2dVdt (non-conservative form)", linestyle=:dot)
lines!(ax1, ∫edV, label="∫edV")
axislegend(ax1)

lines!(ax2, ds["time"], ds["∫εdV"], label="∫εdV (conservative form)", linestyle=:dashdot)
lines!(ax2, ds["time"], ds["∫ε2dV"], label="∫ε2dV (non-conservative form)", linestyle=:dot)
lines!(ax2, ds["time"][2:end], ∂ₜ∫edV, label="∂(∫edV)/∂ₜ")
axislegend(ax2)

close(ds)
