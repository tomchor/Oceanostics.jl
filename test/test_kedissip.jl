using Oceananigans
using Oceanostics
using Oceananigans.TurbulenceClosures: HorizontalFormulation, VerticalFormulation
using Oceananigans.Fields: @compute
using Random

function periodic_locations(N_locations, flip_z=true)
    Random.seed!(772)
    reflections = [-2;;-1;;0;;1;;2]

    x₀ =  rand(N_locations) .+ reflections
    y₀ =  rand(N_locations) .+ reflections
    z₀ = -rand(N_locations) .+ reflections

    return x₀, y₀, z₀
end

N = 16
ν = 1
closure = ScalarDiffusivity(ν=ν)

grid = RectilinearGrid(topology=(Periodic, Periodic, Periodic), size=(N,N,N), extent=(1,1,1))
model = NonhydrostaticModel(grid=grid, advection=WENO(order=9), closure=closure,
                           auxiliary_fields=(; ∫∫εdVdt=0.0, ∫∫ε2dVdt=0.0, ∫εdV_prev=0.0))

# A kind of convoluted way to create x-periodic, resolved initial noise
σx = 4grid.Δxᶜᵃᵃ # x length scale of the noise
σy = 4grid.Δyᵃᶜᵃ # x length scale of the noise
σz = 4grid.Δzᵃᵃᶜ # z length scale of the noise

N_gaussians = 16 # How many Gaussians do we want sprinkled throughout the domain?
Random.seed!(772)
xₚ, yₚ, zₚ = periodic_locations(N_gaussians)

resolved_noise(x, y, z) = sum(@. exp(-(x-xₚ)^2/σx^2 -(y-yₚ)^2/σy^2 -(z-zₚ)^2/σz^2))
set!(model, u=resolved_noise)
u, v, w = model.velocities

using Statistics
u.data.parent .-= mean(u)
v.data.parent .-= mean(v)
w.data.parent .-= mean(w)

Δt = 0.01grid.Δxᶜᵃᵃ/maximum(u)
simulation = Simulation(model; Δt=Δt, stop_time=0.1)

wizard = TimeStepWizard(cfl=0.05, diffusive_cfl=0.05)
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
    @printf("Time: %s, Δt: %s,  e: %f, max(speed): %f, ∫∫εdVdt: %f \n", prettytime(sim), prettytime(sim.Δt), interior(∫edV)[1,1,1], maximum(speed), model.auxiliary_fields.∫∫εdVdt)
end
simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))


∫edV_t⁰ = parent(∫edV)[1,1,1]
function accumulate_ε(sim)
    compute!(∫εdV)
    increment = sim.Δt * model.auxiliary_fields.∫εdV_prev
    model.auxiliary_fields = (; model.auxiliary_fields..., ∫∫εdVdt = model.auxiliary_fields.∫∫εdVdt + increment)

    compute!(∫ε2dV)
    increment = sim.Δt * parent(∫ε2dV)[1,1,1]
    model.auxiliary_fields = (; model.auxiliary_fields..., ∫∫ε2dVdt = model.auxiliary_fields.∫∫ε2dVdt + increment)
    
    model.auxiliary_fields = (; model.auxiliary_fields..., ∫εdV_prev = parent(∫εdV)[1,1,1])
    return nothing
end
simulation.callbacks[:integrate_ε] = Callback(accumulate_ε)

get_∫∫εdVdt(model) = model.auxiliary_fields.∫∫εdVdt
get_∫∫ε2dVdt(model) = model.auxiliary_fields.∫∫ε2dVdt

outputs = (; u, v, w, 
           ε, ∫εdV,
           ε2, ∫ε2dV,
           e, ∫edV,
           ∫∫εdVdt = get_∫∫εdVdt,
           ∫∫ε2dVdt = get_∫∫ε2dVdt,
           )

dt = simulation.Δt
filename = "ke_dissip"
simulation.output_writers[:tracer] = NetCDFOutputWriter(model, outputs;
                                                        filename = filename,
                                                        schedule = TimeInterval(dt),
                                                        dimensions = (; ∫∫εdVdt = (), ∫∫ε2dVdt = ()),
                                                        overwrite_existing = true)
run!(simulation)


compute!(∫edV)
∫edV_tᶠ = parent(∫edV)[1,1,1]
∫∫εdVdt_tᶠ = ∫edV_t⁰- model.auxiliary_fields.∫∫εdVdt
@show abs_error = (abs(∫∫εdVdt_tᶠ - ∫edV_tᶠ)/∫edV_t⁰)




using NCDatasets, GLMakie

ds = NCDataset(simulation.output_writers[:tracer].filepath, "r")

∂ₜ∫edV = -diff(ds["∫edV"]) / dt

∫edV = ds["∫edV"]

∫∫εdVdt = ds["∫∫εdVdt"]
∫∫εdVdt = ∫edV[1] .- ∫∫εdVdt

∫∫ε2dVdt = ds["∫∫ε2dVdt"]
∫∫ε2dVdt = ∫edV[1] .- ∫∫ε2dVdt

fig = Figure(resolution = (800, 800))

ax1 = Axis(fig[1, 1]; title = "KE")
ax2 = Axis(fig[2, 1]; title = "KE dissip rate")

lines!(ax1, ds["time"], ∫edV, label="∫edV")
lines!(ax1, ds["time"], ∫∫εdVdt, label="∫∫εdVdt (conservative form)", linestyle=:dashdot)
lines!(ax1, ds["time"], ∫∫ε2dVdt, label="∫∫ε2dVdt (non-conservative form)", linestyle=:dot)
axislegend(ax1)

lines!(ax2, ds["time"][2:end], ∂ₜ∫edV, label="∂(∫edV)/∂ₜ")
lines!(ax2, ds["time"], ds["∫εdV"], label="∫εdV (conservative form)", linestyle=:dashdot)
lines!(ax2, ds["time"], ds["∫ε2dV"], label="∫ε2dV (non-conservative form)", linestyle=:dot)
axislegend(ax2)

close(ds)
