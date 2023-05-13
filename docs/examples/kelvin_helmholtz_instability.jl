# # Two dimensional turbulence example
#
# This example simulates a simple 2D Kelvin-Helmholtz instability
#
# Before starting, make sure you have the required packages installed for this example, which can be
# done with
#
# ```julia
# using Pkg
# pkg"add Oceananigans, Oceanostics, CairoMakie, Rasters"
# ```

# ## Model and simulation setup

# We begin by creating a model with an isotropic diffusivity and fifth-order advection on a `xz`
# 128² grid using a buoyancy `b` as the active scalar. We'll work here with nondimensional
# quantities.

using Oceananigans

N = 128
L = 10
grid = RectilinearGrid(size=(N, N), x=(-L/2, +L/2), z=(-L/2, +L/2), topology=(Periodic, Flat, Bounded))

model = NonhydrostaticModel(; grid, timestepper = :RungeKutta3,
                            advection = UpwindBiasedFifthOrder(),
                            closure = ScalarDiffusivity(ν=2e-5, κ=2e-5),
                            buoyancy = BuoyancyTracer(), tracers = :b)

# We use hyperbolic tangent functions for the initial conditions and set the maximum Richardson
# number below the threshold of 1/4. We also add some grid-scale small-amplitude noise to `u` to
# kick the instability off:

Ri₀ = 0.1; h = 1/4
shear_flow(x, y, z, t) = tanh(z)
stratification(x, y, z, t, p) = p.h * p.Ri₀ * tanh(z / p.h)

noise(x, y, z) = 1e-2*randn()
shear_flow(x, y, z) = tanh(z) + noise(x, y, z)
stratification(x, y, z) = h * Ri₀ * tanh(z / h)
set!(model, u=shear_flow, b=stratification)

# With the model above we create an adaptive-time-step simulation:

simulation = Simulation(model, Δt=0.1, stop_time=100)

wizard = TimeStepWizard(cfl=0.8, max_Δt=1)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(2))


# ## Model diagnostics
#
# We set-up a progress messenger using the `SimpleProgressMessenger`, which, as the name suggests,
# displays simple information about the simulation

using Oceanostics

progress = SimpleProgressMessenger()
simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))


# We can also define some useful diagnostics for of the flow, starting with the `RichardsonNumber`

Ri = RichardsonNumber(model)
Q = QVelocityGradientTensorInvariant(model)

∫χᴰ = Integral(TracerVarianceDissipationRate(model, :b))
∫χ = Integral(TracerVarianceDiffusiveTerm(model, :b))


# Outputs and NetCDFWriter
output_fields = (; Ri, Q, model.tracers.b, ∫χ, ∫χᴰ)
filename = "kelvin_helmholtz"
simulation.output_writers[:nc] = NetCDFOutputWriter(model, output_fields,
                                                    filename = joinpath(@__DIR__, filename),
                                                    schedule = TimeInterval(0.6),
                                                    overwrite_existing = true)



# ## Run the simulation and process results
#
# To run the simulation:

run!(simulation)

# Now we'll read the results using Rasters.jl, which works somewhat similarly to Python's Xarray and
# can speed-up the work the workflow

using Rasters

ds = RasterStack(simulation.output_writers[:nc].filepath)

# In order to plot results, we use Makie.jl, for which Rasters.jl already has some recipes

using GLMakie
n = Observable(1)

uₙ = @lift ds.Ri[Ti=$n, yC=Near(0)]
wₙ = @lift ds.Q[Ti=$n, yC=Near(0)]
bₙ = @lift ds.b[Ti=$n, yC=Near(0)]


times = dims(ds, :Ti)
frames = 1:length(times)
title = @lift "Time = " * string(round(times[$n], digits=2))

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

lines!(axb, Array(times), Array(ds.∫χ), label="avg KE")
lines!(axb, Array(times), Array(ds.∫χᴰ), label="∫ᵗ(uᵢ∂uᵢ/∂t)dt")
vlines!(axb, @lift times[$n])
axislegend(position=:lb)

@info "Animating..."
record(fig, filename * ".mp4", frames, framerate=10) do i
       n[] = i
end
