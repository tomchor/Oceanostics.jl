using Oceananigans
using Oceananigans.Fields: @compute

grid = RectilinearGrid(size=(64, 64), x=(-5, 5), z=(-5, 5), topology=(Periodic, Flat, Bounded))
model = NonhydrostaticModel(; grid, timestepper = :RungeKutta3,
                            advection = UpwindBiasedFifthOrder(),
                            closure = ScalarDiffusivity(ν=2e-5, κ=2e-5),
                            buoyancy = BuoyancyTracer(), tracers = :b)

# Initial conditions
Ri= 0.1
h = 1/4
shear_flow(x, y, z, t) = tanh(z)
stratification(x, y, z, t, p) = p.h * p.Ri * tanh(z / p.h)

noise(x, y, z) = 1e-2*randn()
shear_flow(x, y, z) = tanh(z) + noise(x, y, z)
stratification(x, y, z) = h * Ri * tanh(z / h)
set!(model, u=shear_flow, b=stratification)

# Adaptive-time-step simulation
simulation = Simulation(model, Δt=0.1, stop_time=100)

wizard = TimeStepWizard(cfl=0.05, max_Δt=1)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(2))

# KernelFunctionOperation for KE evolution
using Oceananigans.Grids: Center
using Oceananigans.Operators
using Oceananigans.Models.NonhydrostaticModels: u_velocity_tendency, v_velocity_tendency, w_velocity_tendency

@inline ψf(i, j, k, grid, ψ, f, args...) = ψ[i, j, k] * f(i, j, k, grid, args...)
@inline function uᵢ∂ₜuᵢᶜᶜᶜ(i, j, k, grid, advection, coriolis, stokes_drift, closure, immersed_bc, buoyancy, background_fields,
                                          velocities, tracers, auxiliary_fields, diffusivity_fields, forcing,
                                          pHY′, clock)
        u∂ₜu = ℑxᶜᵃᵃ(i, j, k, grid, ψf, velocities.u, u_velocity_tendency, advection, coriolis, stokes_drift, closure, immersed_bc, buoyancy, background_fields,
                                                                           velocities, tracers, auxiliary_fields, diffusivity_fields, forcing,
                                                                           pHY′, clock)

        v∂ₜv = ℑyᵃᶜᵃ(i, j, k, grid, ψf, velocities.v, v_velocity_tendency, advection, coriolis, stokes_drift, closure, immersed_bc, buoyancy, background_fields,
                                                                           velocities, tracers, auxiliary_fields, diffusivity_fields, forcing,
                                                                           pHY′, clock)

        w∂ₜw = ℑzᵃᵃᶜ(i, j, k, grid, ψf, velocities.w, w_velocity_tendency, advection, coriolis, stokes_drift, closure, immersed_bc, buoyancy, background_fields,
                                                                           velocities, tracers, auxiliary_fields, diffusivity_fields, forcing,
                                                                           clock)

    return u∂ₜu + v∂ₜv + w∂ₜw
end

function KineticEnergyTendency(model)
    dependencies = (model.advection, model.coriolis, model.stokes_drift, model.closure,
                    model.velocities.u.boundary_conditions.immersed, model.buoyancy, model.background_fields,
                    model.velocities, model.tracers, model.auxiliary_fields, model.diffusivity_fields, model.forcing,
                    model.pressures.pHY′, model.clock)
    return KernelFunctionOperation{Center, Center, Center}(uᵢ∂ₜuᵢᶜᶜᶜ, model.grid, dependencies...)
end

ε = Field(Average(KineticEnergyTendency(model)))
ε̄ = Field(Average(ε))

# Diagnostics from state variables
u, v, w = model.velocities
b = model.tracers.b
@compute K̄ = Field(Average(Field(@at (Center, Center, Center) (u^2 + v^2)/2)))

# Time integration of evolution
∫ε̄dt = Ref(parent(K̄)[1,1,1])
function accumulate_integrals(sim)
    compute!(ε̄)
    increment = sim.Δt * parent(ε̄)[1,1,1]
    ∫ε̄dt[] += increment
    return nothing
end
simulation.callbacks[:time_integration] = Callback(accumulate_integrals)

# Outputs and NetCDFWriter
get_∫ε̄dt(model) = ∫ε̄dt[]
outputs = (; u, w, b, K̄, ∫ε̄dt = get_∫ε̄dt)
fname = "kh"
simulation.output_writers[:nc] = NetCDFOutputWriter(model, outputs; filename = "$fname.nc",
                                                    schedule = TimeInterval(1),
                                                    dimensions = Dict("∫ε̄dt" => ()),
                                                    overwrite_existing = true)


# Nice progress message and run simulation
using Printf
progress(sim) = @printf("Iteration: %d, time: %.2f, Δt: %.4f\n", iteration(sim), sim.model.clock.time, sim.Δt)
simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))
run!(simulation)


# Now we plot stuff
using Rasters
ds = RasterStack(simulation.output_writers[:nc].filepath)

using GLMakie
n = Observable(1)

uₙ = @lift permutedims(ds[:u][Ti=$n, yC=Near(0)], (:xF, :zC))
wₙ = @lift permutedims(ds[:w][Ti=$n, yC=Near(0)], (:xC, :zF))
bₙ = @lift permutedims(ds[:b][Ti=$n, yC=Near(0)], (:xC, :zC))


times = dims(ds, :Ti)
frames = 1:length(times)
title = @lift @sprintf "Time = %.2f" times[$n]

fig = Figure(resolution=(800, 600))

fig[1, 1:5] = Label(fig, title, fontsize=24, tellwidth=false)

kwargs = (xlabel="x", ylabel="z",)
ax1 = Axis(fig[2, 1]; title = "u", kwargs...)
ax2 = Axis(fig[2, 2]; title = "w", kwargs...)
ax3 = Axis(fig[2, 4]; title = "b", kwargs...)
axb = Axis(fig[3, 1:5]; title = "KE", xlabel="Time")

ulim = (-.75, +.75)

hm1 = heatmap!(ax1, uₙ; colormap = :balance, colorrange = ulim)
hm2 = heatmap!(ax2, wₙ; colormap = :balance, colorrange = ulim)
Colorbar(fig[2, 3], hm2)

hm3 = heatmap!(ax3, bₙ; colormap = :balance)
Colorbar(fig[2, 5], hm3)

lines!(axb, Array(times), Array(ds[:K̄]), label="avg KE")
lines!(axb, Array(times), Array(ds[:∫ε̄dt]), label="∫ᵗ(uᵢ∂uᵢ/∂t)dt")
vlines!(axb, @lift times[$n])
axislegend(position=:lb)

record(fig, "$fname.mp4", frames, framerate=10) do i
       @info "Plotting frame $i of $(frames[end])..."
       n[] = i
end
