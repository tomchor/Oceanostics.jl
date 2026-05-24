# # [Two-dimensional turbulence example](@id two_d_turbulence_example)
#
# In this example we simulate a 2D flow initialized with random-noise velocities and a passive tracer ``c`` with
# a smooth sine/cosine initial condition. We then use Oceanostics to close the volume-integrated
# kinetic-energy and tracer-variance (``c^2``) budgets.
#
# Before starting, make sure you have the required packages installed for this example, which can
# be done with
#
# ```julia
# using Pkg
# pkg"add Oceananigans, Oceanostics, CairoMakie"
# ```

# ## Model and simulation setup

# We begin by creating a model with an isotropic diffusivity and a fourth-order centered
# advection scheme on a 256² grid, with one passive tracer `c`. Using a centered scheme
# avoids numerical dissipation, so the volume-integrated KE and ``c^2`` budgets reduce to purely
# dissipative balances and we can close them against ``\varepsilon`` and ``\chi`` alone.

using Oceananigans

grid = RectilinearGrid(size=(256, 256), extent=(2π, 2π), topology=(Periodic, Periodic, Flat))

model = NonhydrostaticModel(grid; timestepper = :RungeKutta3,
                            advection = Centered(order=4),
                            tracers = :c,
                            closure = ScalarDiffusivity(ν=1e-5, κ=1e-3))

# Grid-scale white noise is not really *resolved* by the grid, so instead we build a randomized
# but well-resolved velocity initial condition as a sum of `N_blobs` Gaussian bumps with random
# centers and random amplitudes. Each bump is ``\sigma_b \approx 4\Delta x`` wide and the
# periodic copies of each center are summed in so the resulting field is smooth across the
# periodic boundary. The tracer keeps a smooth sine/cosine pattern.

using Random, Statistics

u, v, w = model.velocities
c = model.tracers.c

Random.seed!(772)
N_blobs = 32
σ_blob  = 10 * minimum_xspacing(grid)
xc      = grid.Lx * rand(N_blobs)
yc      = grid.Ly * rand(N_blobs)
amp_u   = randn(N_blobs) # random Gaussian amplitudes for u
amp_v   = randn(N_blobs) # ... and for v

# Sum of blobs and their periodic images at (dx, dy) ∈ {-2π, 0, 2π}²
blob_sum(x, y, amp) = sum(amp[k] * exp(-((x - xc[k] - dx)^2 + (y - yc[k] - dy)^2) / σ_blob^2)
                          for k in 1:N_blobs, dx in (-2π, 0, 2π), dy in (-2π, 0, 2π))

uᵢ(x, y) = blob_sum(x, y, amp_u)
vᵢ(x, y) = blob_sum(x, y, amp_v)
cᵢ(x, y) = sin(2x) * cos(3y) + cos(x) * sin(2y)

set!(model, u=uᵢ, v=vᵢ, c=cᵢ)

u .-= mean(u)
v .-= mean(v)

# We use this model to create a simulation with a `TimeStepWizard` to maximize the Δt

Δt = 0.2 * minimum_xspacing(grid) / maximum(u) # Start with a conservative Δt
simulation = Simulation(model; Δt, stop_time=80)

wizard = TimeStepWizard(cfl=0.8, diffusive_cfl=0.8)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(5))


# ## Model diagnostics
#
# Up until now we have only used Oceananigans, but we can make use of Oceanostics for the first
# diagnostic we'll set-up: a progress messenger. Here we use a `BasicMessenger`, which,
# as the name suggests, displays basic information about the simulation

using Oceanostics

progress = ProgressMessengers.BasicMessenger()
simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

# We define the visualization fields — speed, vorticity, kinetic energy `KE` — and the
# dissipation rates `ε` and `χ`, which we will use to close the budgets.

using Oceananigans.AbstractOperations: @at

speed     = @at (Center, Center, Center) sqrt(u^2 + v^2)
vorticity = ∂x(v) - ∂y(u)
KE        = KineticEnergyEquation.KineticEnergy(model)
ε         = KineticEnergyEquation.DissipationRate(model)
χ         = TracerVarianceEquation.TracerVarianceDissipationRate(model, :c)

# To close the budgets we also define the relevant volume integrals as scalar outputs. For a 2D
# periodic domain with no forcing or buoyancy, advection and pressure-redistribution terms
# volume-integrate to (essentially) zero by incompressibility, so the volume-integrated KE and
# ``c^2`` evolution equations reduce to
#
# ```math
# \frac{d}{dt} \int \mathrm{KE}\, dV = -\int \varepsilon\, dV,\qquad
# \frac{d}{dt} \int c^2\, dV = -\int \chi\, dV.
# ```
#
# A caveat: a discretized version of the continuum KE equation (such as the one above) is not guaranteed to exactly conserve energy at
# the *discrete* level. To get strict discrete conservation of energy one would have to derive a discrete
# KE equation directly from the discrete momentum equations — using both the current and
# previous time-step velocities. We are not doing that here: we compute ``\varepsilon``
# from the current model state and finite-difference snapshots of ``\int \mathrm{KE}\, dV``
# independently. The two relations are consistent in the continuum limit but only approximately
# at the discrete level, so we expect the KE budget to close only approximately.

∫KE = Integral(KE)
∫c² = Integral(c^2)
∫ε  = Integral(ε)
∫χ  = Integral(χ)

# We output snapshots in consecutive-iteration pairs every 0.6 time units. The
# `ConsecutiveIterations` schedule writes a second snapshot one model step after each scheduled
# output time, which lets us finite-difference the integrated quantities across that single
# step to estimate ``d/dt`` accurately — no time-integration accumulator is needed.

output_fields = (; speed, vorticity, KE, c, ∫KE, ∫c², ∫ε, ∫χ)

using NCDatasets
filename = "two_dimensional_turbulence"
simulation.output_writers[:nc] = NetCDFWriter(model, output_fields,
                                              filename = joinpath(@__DIR__, filename),
                                              schedule = ConsecutiveIterations(TimeInterval(0.6)),
                                              overwrite_existing = true)


# ## Run the simulation and process results
#
# To run the simulation:

run!(simulation)

# Read snapshot fields and the integrated scalars.

filepath    = simulation.output_writers[:nc].filepath
speed_t     = FieldTimeSeries(filepath, "speed")
vorticity_t = FieldTimeSeries(filepath, "vorticity")
KE_t        = FieldTimeSeries(filepath, "KE")
c_t         = FieldTimeSeries(filepath, "c")

ds = NCDataset(filepath)
times = ds["time"][:]
∫KE_t = ds["∫KE"][:]
∫c²_t = ds["∫c²"][:]
∫ε_t  = ds["∫ε"][:]
∫χ_t  = ds["∫χ"][:]
close(ds)

# `ConsecutiveIterations` arranges the snapshot times in pairs:
# `(t₀, t₀+Δt_model, t₀+0.6, t₀+0.6+Δt_model, …)`. Pair `k` has indices `(2k-1, 2k)`; we obtain
# ``d/dt`` from a one-step finite difference inside each pair, evaluated at the pair midpoint.

idx1     = 1:2:length(times) - 1   # primary snapshots
idx2     = 2:2:length(times)       # consecutive-iteration snapshots
Δt_pair  = times[idx2] .- times[idx1]
t_pair   = @. 0.5 * (times[idx1] + times[idx2])

dKEdt    = (∫KE_t[idx2] .- ∫KE_t[idx1]) ./ Δt_pair
dc²dt    = (∫c²_t[idx2] .- ∫c²_t[idx1]) ./ Δt_pair

# Source terms at the pair midpoint
ε_pair   = @. 0.5 * (∫ε_t[idx1] + ∫ε_t[idx2])
χ_pair   = @. 0.5 * (∫χ_t[idx1] + ∫χ_t[idx2])

KE_resid = @. dKEdt - (-ε_pair)
c²_resid = @. dc²dt - (-χ_pair)


# ## Plotting
#
# We use Makie.jl, which has recipes for Oceananigans `Field`s.

using CairoMakie

set_theme!(Theme(fontsize = 20))
fig = Figure()

axis_kwargs = (aspect = DataAspect(),
               height = 250, width = 250,
               xticksvisible = false, yticksvisible = false,
               xticklabelsvisible = false, yticklabelsvisible = false)

ax_speed = Axis(fig[2, 1]; title = "Speed",          axis_kwargs...)
ax_ω     = Axis(fig[2, 2]; title = "Vorticity",      axis_kwargs...)
ax_KE    = Axis(fig[2, 3]; title = "Kinetic energy", axis_kwargs...)
ax_c     = Axis(fig[2, 4]; title = "Tracer c",       axis_kwargs...)

# Animate only the primary snapshots (one frame per pair) — the consecutive iterations are
# essentially identical to their pair-mates by eye.

n = Observable(1)

speedₙ = @lift speed_t[idx1[$n]]
ωₙ     = @lift vorticity_t[idx1[$n]]
KEₙ    = @lift KE_t[idx1[$n]]
cₙ     = @lift c_t[idx1[$n]]

hm_speed = heatmap!(ax_speed, speedₙ, colormap = :magma, colorrange=(0, 1.5))
Colorbar(fig[3, 1], hm_speed; vertical=false, height=8, ticklabelsize=12)

hm_ω = heatmap!(ax_ω, ωₙ, colormap = :balance, colorrange=(-10, 10))
Colorbar(fig[3, 2], hm_ω; vertical=false, height=8, ticklabelsize=12)

hm_KE = heatmap!(ax_KE, KEₙ, colormap = :plasma, colorrange=(0, 0.5))
Colorbar(fig[3, 3], hm_KE; vertical=false, height=8, ticklabelsize=12)

hm_c = heatmap!(ax_c, cₙ, colormap = :balance, colorrange=(-1.5, 1.5))
Colorbar(fig[3, 4], hm_c; vertical=false, height=8, ticklabelsize=12)

# Volume-integrated KE budget — `d(∫KE)/dt` against `-∫ε dV`, with the residual.

budget_kwargs = (height = 180, width = 1080)

ax_KEbud = Axis(fig[4, 1:4]; title = "Volume-integrated KE budget", budget_kwargs...)
lines!(ax_KEbud, t_pair, dKEdt,   label = "d(∫KE)/dt")
lines!(ax_KEbud, t_pair, -ε_pair, label = "-∫ε dV")
lines!(ax_KEbud, t_pair, KE_resid, label = "residual", color = :black, linestyle = :dash)
axislegend(ax_KEbud; labelsize = 10, position = :rb)

# Volume-integrated c² budget — `d(∫c²)/dt` against `-∫χ dV`, with the residual.

ax_c²bud = Axis(fig[5, 1:4]; title = "Volume-integrated ∫c² budget", xlabel = "Time", budget_kwargs...)
lines!(ax_c²bud, t_pair, dc²dt,   label = "d(∫c²)/dt")
lines!(ax_c²bud, t_pair, -χ_pair, label = "-∫χ dV")
lines!(ax_c²bud, t_pair, c²_resid, label = "residual", color = :black, linestyle = :dash)
axislegend(ax_c²bud; labelsize = 10, position = :rb)

# Time marker on both budget panels

tₙ = @lift t_pair[$n]
vlines!(ax_KEbud, tₙ, color = :black, linestyle = :dot)
vlines!(ax_c²bud, tₙ, color = :black, linestyle = :dot)

title = @lift "Time = " * string(round(t_pair[$n], digits=2))
Label(fig[1, 1:4], title, fontsize=24, tellwidth=false);

# Adjust the total figure size based on our panels and record a movie.

resize_to_layout!(fig)
@info "Animating..."
record(fig, filename * ".mp4", 1:length(t_pair), framerate=24) do i
    n[] = i
end
nothing #hide

# ![](two_dimensional_turbulence.mp4)
#
# The two bottom panels show the volume-integrated KE and ``c^2`` budgets: ``d/dt`` of the
# integrated quantity is compared against ``-\int \varepsilon\, dV`` (respectively
# ``-\int \chi\, dV``), the only term that survives volume-integration for a periodic
# incompressible flow with a centered advection scheme. The residual shows the gap between them.
#
# Outputting with `ConsecutiveIterations(TimeInterval(...))` is the trick that makes this
# diagnosis possible: the writer emits two snapshots one model step apart at every output time,
# so the finite-difference time derivative is a single-step approximation rather than a coarse
# difference between widely spaced outputs. We can therefore read both ``\int \mathrm{KE}\, dV``
# and the source terms straight off disk and close the budget without any in-simulation
# time-integration callback.
#
# Because we use a centered advection scheme, the volume-integrated advection contributions
# vanish exactly and the residual collapses to the timestepping discretization error — a clean
# closure of the budget against the explicit dissipation alone.
#
# Note that `KineticEnergyDissipationRate` (`ε`) and `TracerVarianceDissipationRate` (`χ`) are
# implemented in an energy/variance-conserving form (i.e., they use the exact same
# discretizations and interpolations as Oceananigans), so the explicit-dissipation pieces of the
# budgets are exact and any residual we see is dominated by the one-step finite-difference
# truncation in ``d/dt``.
