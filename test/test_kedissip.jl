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

grid = RectilinearGrid(topology=(Periodic, Periodic, Bounded), size=(N,N,N), extent=(1,1,1))
model = NonhydrostaticModel(grid=grid, advection=WENO(order=5), closure=closure)

# A kind of convoluted way to create periodic, resolved initial noise
σx = 2grid.Δxᶜᵃᵃ # x length scale of the noise
σy = 2grid.Δyᵃᶜᵃ # x length scale of the noise
σz = 2grid.Δzᵃᵃᶜ # z length scale of the noise

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
simulation = Simulation(model; Δt=Δt, stop_time=0.02)

wizard = TimeStepWizard(cfl=0.05, diffusive_cfl=0.05)
simulation.callbacks[:wizard] = Callback(wizard)

include("test_kernels.jl")

@compute e = Field(Oceanostics.TKEBudgetTerms.KineticEnergy(model))

@compute ε = Field(Oceanostics.TKEBudgetTerms.IsotropicPseudoViscousDissipationRate(model))
@compute ε2 = Field(Oceanostics.TKEBudgetTerms.IsotropicViscousDissipationRate(model))
@compute εt2 = Field(KernelFunctionOperation{Center, Center, Center}(uᵢ∂ⱼ_τᵢⱼᶜᶜᶜ, model.grid; computed_dependencies=dependencies2))

@compute ∫εdV  = Field(Integral(ε))
@compute ∫ε2dV = Field(Field(Integral(ε2)))
@compute ∫εt2dV = Field(Integral(εt2))
@compute ∫edV  = Field(Integral(e))

using Printf
function progress(sim)
    compute!(∫edV)
    @printf("Time: %s, Δt: %s,  e: %f, ∫∫εdVdt: %f \n", prettytime(sim), prettytime(sim.Δt), interior(∫edV)[1,1,1], ∫∫εdVdt[])
end
simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))


∫∫εdVdt = Ref(0.0)
∫∫ε2dVdt = Ref(0.0)
∫∫εt2dVdt = Ref(0.0)

∫εdV_prev = Ref(0.0)
∫ε2dV_prev = Ref(0.0)
∫εt2dV_prev = Ref(0.0)

∫edV_t⁰ = parent(∫edV)[1,1,1]
function accumulate_ε(sim)
    increment = sim.Δt * ∫εdV_prev[]
    ∫∫εdVdt[] += increment

    increment = sim.Δt * ∫ε2dV_prev[]
    ∫∫ε2dVdt[] += increment

    increment = sim.Δt * ∫εt2dV_prev[]
    ∫∫εt2dVdt[] += increment
    
    compute!(∫εdV)
    ∫εdV_prev[] = parent(∫εdV)[1,1,1]
    compute!(∫ε2dV)
    ∫ε2dV_prev[] = parent(∫ε2dV)[1,1,1]
    compute!(∫εt2dV)
    ∫εt2dV_prev[] = parent(∫εt2dV)[1,1,1]
    return nothing
end
simulation.callbacks[:integrate_ε] = Callback(accumulate_ε)

get_∫∫εdVdt(model) = ∫∫εdVdt[]
get_∫∫ε2dVdt(model) = ∫∫ε2dVdt[]
get_∫∫εt2dVdt(model) = ∫∫εt2dVdt[]

outputs = (; u, v, w, 
           e, ∫edV,
           ε, ∫εdV,
           ∫∫εdVdt = get_∫∫εdVdt,
           ε2, ∫ε2dV,
           ∫∫ε2dVdt = get_∫∫ε2dVdt,
           εt2, ∫εt2dV,
           ∫∫εt2dVdt = get_∫∫εt2dVdt,
           )

dt = simulation.Δt
filename = "ke_dissip"
simulation.output_writers[:tracer] = NetCDFOutputWriter(model, outputs;
                                                        filename = filename,
                                                        schedule = TimeInterval(dt),
                                                        dimensions = (; ∫∫εdVdt = (), 
                                                                      ∫∫ε2dVdt = (), 
                                                                      ∫∫εt2dVdt = (),
                                                                      ),
                                                        overwrite_existing = true)
run!(simulation)




compute!(∫edV)
∫edV_tᶠ = parent(∫edV)[1,1,1]
∫∫εdVdt_tᶠ = ∫edV_t⁰- ∫∫εdVdt[]
@show abs_error = (abs(∫∫εdVdt_tᶠ - ∫edV_tᶠ)/∫edV_t⁰)




using NCDatasets, GLMakie

ds = NCDataset(simulation.output_writers[:tracer].filepath, "r")

∂ₜ∫edV = -diff(ds["∫edV"]) / dt

∫edV = ds["∫edV"]

∫∫εdVdt = ds["∫∫εdVdt"]
∫∫εdVdt = ∫edV[1] .- ∫∫εdVdt

∫∫ε2dVdt = ds["∫∫ε2dVdt"]
∫∫ε2dVdt = ∫edV[1] .- ∫∫ε2dVdt

∫∫εt2dVdt = ds["∫∫εt2dVdt"]
∫∫εt2dVdt = ∫edV[1] .- ∫∫εt2dVdt

fig = Figure(resolution = (800, 800))

ax1 = Axis(fig[1, 1]; title = "KE")
ax2 = Axis(fig[2, 1]; title = "KE dissip rate")

lines!(ax1, ds["time"], ∫edV, label="∫edV")
lines!(ax1, ds["time"], ∫∫εdVdt, label="∫∫εdVdt (conservative form)", linestyle=:dashdot)
lines!(ax1, ds["time"], ∫∫ε2dVdt, label="∫∫ε2dVdt (non-conservative form)", linestyle=:dot)
lines!(ax1, ds["time"], ∫∫εt2dVdt, label="∫∫εt2dVdt (uᵢ∂ⱼ_τᵢⱼ)", linestyle=:dash)
axislegend(ax1)

lines!(ax2, ds["time"][2:end], ∂ₜ∫edV, label="∂ₜ∫edV")
lines!(ax2, ds["time"], ds["∫εdV"], label="∫εdV (conservative form)", linestyle=:dashdot)
lines!(ax2, ds["time"], ds["∫ε2dV"], label="∫ε2dV (non-conservative form)", linestyle=:dot)
lines!(ax2, ds["time"], ds["∫εt2dV"], label="∫εt2dV (uᵢ∂ⱼ_τᵢⱼ)", linestyle=:dash)
axislegend(ax2)

close(ds)
