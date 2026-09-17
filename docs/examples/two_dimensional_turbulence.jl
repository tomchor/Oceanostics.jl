# # [Two-dimensional turbulence example](@id two_d_turbulence_example)
#
# In this example we simulate a 2D flow initialized with random-noise velocities and a passive tracer ``c`` with
# a smooth sine/cosine initial condition. We then use Oceanostics to close the volume-integrated
# kinetic energy and tracer variance (``c^2``) budgets.
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
# dissipative balances and we can close them against ``\varepsilon_k`` and ``\chi`` alone.

using Oceananigans

grid = RectilinearGrid(size=(256, 256), extent=(2π, 2π), topology=(Periodic, Periodic, Flat))

model = NonhydrostaticModel(grid; timestepper = :RungeKutta3,
                            advection = Centered(order=4),
                            tracers = :c,
                            closure = ScalarDiffusivity(ν=1e-4, κ=1e-3))

# Grid-scale white noise is not really *resolved* by the grid, so instead we build a randomized
# but well-resolved velocity initial condition as a sum of `N_blobs` Gaussian bumps with random
# centers and random amplitudes. Each bump is ``\sigma_b \approx 10\Delta x`` wide and the
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

# Sum of blobs and their periodic images at (dx, dy) ∈ {-Lx, 0, Lx} × {-Ly, 0, Ly}
blob_sum(x, y, amp) = sum(amp[k] * exp(-((x - xc[k] - dx)^2 + (y - yc[k] - dy)^2) / σ_blob^2)
                          for k  in 1:N_blobs,
                              dx in (-grid.Lx, 0, grid.Lx),
                              dy in (-grid.Ly, 0, grid.Ly))

uᵢ(x, y) = blob_sum(x, y, amp_u)
vᵢ(x, y) = blob_sum(x, y, amp_v)
cᵢ(x, y) = sin(2x) * cos(3y) + cos(x) * sin(2y)

set!(model, u=uᵢ, v=vᵢ, c=cᵢ)

u .-= mean(u)
v .-= mean(v)

# We use this model to create a simulation with a `TimeStepWizard` to maximize the Δt

u_max = max(maximum(abs, u), maximum(abs, v)) # peak speed magnitude (not signed max)
Δt = 0.2 * minimum_xspacing(grid) / u_max      # Start with a conservative Δt
simulation = Simulation(model; Δt, stop_time=80)

wizard = TimeStepWizard(cfl=0.8, diffusive_cfl=0.8)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(5))


# ## Model diagnostics
#
# Up until now we have only used Oceananigans, but we can make use of Oceanostics for the first
# diagnostic we'll set-up: a progress messenger. Here we use a `BasicMessenger`, which,
# as the name suggests, displays only basic information about the simulation

using Oceanostics

progress = ProgressMessengers.BasicMessenger()
simulation.callbacks[:progress] = Callback(progress, IterationInterval(200))

# We define the visualization fields — speed, vorticity, kinetic energy `eₖ` — and the
# dissipation rates `εₖ` and `χ`, which we will use to close KE and tracer variance budgets.

using Oceananigans.AbstractOperations: @at

speed     = @at (Center, Center, Center) √(u^2 + v^2)
vorticity = ∂x(v) - ∂y(u)
eₖ        = KineticEnergyEquation.KineticEnergy(model)
εₖ        = KineticEnergyEquation.DissipationRate(model)
χ         = TracerVarianceEquation.TracerVarianceDissipationRate(model, :c)

# Note that `KineticEnergyEquation.DissipationRate` (`εₖ`) and `TracerVarianceEquation.TracerVarianceDissipationRate`
# (`χ`) --- which can also be called as `KineticEnergyDissipationRate` and `TracerVarianceDissipationRate` --- are
# implemented using the same kernels as Oceananigans (and therefore use the same interpolations and
# discretizations).

# To close the budgets we also define the relevant volume integrals as scalar outputs. For a 2D
# periodic domain with no forcing or buoyancy, advection and pressure-redistribution terms
# volume-integrate to zero due to incompressibility, so the volume-integrated KE and
# ``c^2`` evolution equations reduce to
#
# ```math
# \frac{d}{dt} \int e_k\, \mathrm{d}V = -\int \varepsilon_k\, \mathrm{d}V,\qquad
# \frac{d}{dt} \int c^2\, \mathrm{d}V = -\int \chi\, \mathrm{d}V.
# ```
#
# A caveat: a discretized version of the continuum KE equation (such as the one above) is not guaranteed to exactly conserve energy at
# the *discrete* level. To get strict discrete conservation of energy one would have to derive a discrete
# KE equation directly from the discrete momentum equations — using both the current and
# previous time-step velocities. We are not doing that here: we compute ``\varepsilon_k``
# from the current model state and difference ``\int e_k\, \mathrm{d}V`` across a time step
# independently. The two relations are consistent in the continuum limit but only approximately
# at the discrete level for a well-resolved flow, so we expect the KE budget to close only approximately.

∫eₖ = Integral(eₖ)
∫c² = Integral(c^2)
∫εₖ = Integral(εₖ)
∫χ  = Integral(χ)

# The two tendencies come from `TimeDerivative`, which differences its operand across one model step
# while the simulation runs. The writer registers a callback that updates it on the iteration before
# each output as well as at the output itself, so each record carries ``d/dt`` at its own time, already
# divided by the elapsed time and already alongside the source term it has to balance.

∂ₜ∫eₖ = TimeDerivative(∫eₖ)
∂ₜ∫c² = TimeDerivative(∫c²)

# We use two NetCDF writers. A *visualization* writer outputs the 2D snapshot fields and a *budget*
# writer only the (cheap) integrated scalars, both on `TimeInterval(0.6)`. Separating the two avoids
# writing the heavy 2D fields twice per output time.

using NCDatasets
filename = "two_dimensional_turbulence"

simulation.output_writers[:nc] = NetCDFWriter(model, (; speed, vorticity, eₖ, c),
                                              filename = joinpath(@__DIR__, filename),
                                              schedule = TimeInterval(0.6),
                                              overwrite_files = true)

simulation.output_writers[:budget] = NetCDFWriter(model, (; ∂ₜ∫eₖ, ∂ₜ∫c², ∫εₖ, ∫χ),
                                                  filename = joinpath(@__DIR__, filename * "_budget"),
                                                  schedule = TimeInterval(0.6),
                                                  overwrite_files = true)


# ## Run the simulation and process results
#
# To run the simulation:

run!(simulation)

# Read visualization snapshots from the `:nc` writer.

snap_filepath = simulation.output_writers[:nc].filepath
speed_t       = FieldTimeSeries(snap_filepath, "speed")
vorticity_t   = FieldTimeSeries(snap_filepath, "vorticity")
eₖ_t          = FieldTimeSeries(snap_filepath, "eₖ")
c_t           = FieldTimeSeries(snap_filepath, "c")

ds = NCDataset(snap_filepath)
times = ds["time"][:]
close(ds)

# Read the budget scalars from the `:budget` writer. Every record carries both tendencies and both
# source terms at the same time, so there is nothing left to pair up. The one exception is the first
# record, at the start of the run, where a `TimeDerivative` has no earlier state to difference against
# and is written as zero; the budget starts from the second.

bud_filepath = simulation.output_writers[:budget].filepath
ds_bud = NCDataset(bud_filepath)
nb     = 2:length(ds_bud["time"])

t_bud    = ds_bud["time"][nb]
deₖdt    = ds_bud["∂ₜ∫eₖ"][nb]
dc²dt    = ds_bud["∂ₜ∫c²"][nb]
εₖ_bud   = ds_bud["∫εₖ"][nb]
χ_bud    = ds_bud["∫χ"][nb]
close(ds_bud)

# Budget residuals in sum-to-zero form: the negative tendency plus the source term. Plotting every
# curve with these signs makes them add up to the residual, which stays near zero.
eₖ_resid = @. -deₖdt - εₖ_bud
c²_resid = @. -dc²dt - χ_bud

using Test                                #hide
rms(x) = √(sum(abs2, x) / length(x))      #hide
@test rms(eₖ_resid) < 0.02 * rms(deₖdt);  #hide
@test rms(eₖ_resid) < 0.02 * rms(εₖ_bud);  #hide
@test rms(c²_resid) < 0.01 * rms(dc²dt);  #hide
@test rms(c²_resid) < 0.01 * rms(χ_bud);   #hide


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
ax_eₖ    = Axis(fig[2, 3]; title = "Kinetic energy", axis_kwargs...)
ax_c     = Axis(fig[2, 4]; title = "Tracer c",       axis_kwargs...)

# Each frame is one visualization snapshot.

n = Observable(1)

speedₙ = @lift speed_t[$n]
ωₙ     = @lift vorticity_t[$n]
eₖₙ    = @lift eₖ_t[$n]
cₙ     = @lift c_t[$n]

hm_speed = heatmap!(ax_speed, speedₙ, colormap = :magma, colorrange=(0, 1.5))
Colorbar(fig[3, 1], hm_speed; vertical=false, height=8, ticklabelsize=12)

hm_ω = heatmap!(ax_ω, ωₙ, colormap = :balance, colorrange=(-10, 10))
Colorbar(fig[3, 2], hm_ω; vertical=false, height=8, ticklabelsize=12)

hm_eₖ = heatmap!(ax_eₖ, eₖₙ, colormap = :plasma, colorrange=(0, 0.5))
Colorbar(fig[3, 3], hm_eₖ; vertical=false, height=8, ticklabelsize=12)

hm_c = heatmap!(ax_c, cₙ, colormap = :balance, colorrange=(-1.5, 1.5))
Colorbar(fig[3, 4], hm_c; vertical=false, height=8, ticklabelsize=12)

# Volume-integrated KE budget: the negative tendency `-d(∫eₖ)/dt` and `-∫εₖ dV`, which sum to the residual.

budget_kwargs = (height = 180, width = 1080)

ax_eₖbud = Axis(fig[4, 1:4]; title = "Volume-integrated kinetic energy budget", budget_kwargs...)
lines!(ax_eₖbud, t_bud, -deₖdt,   label = "-d(∫eₖ)/dt")
lines!(ax_eₖbud, t_bud, -εₖ_bud,  label = "-∫εₖ dV")
lines!(ax_eₖbud, t_bud, eₖ_resid, label = "residual", color = :black, linestyle = :dash)
axislegend(ax_eₖbud; labelsize = 10, position = :rb)

# Volume-integrated c² budget: the negative tendency `-d(∫c²)/dt` and `-∫χ dV`, which sum to the residual.

ax_c²bud = Axis(fig[5, 1:4]; title = "Volume-integrated tracer variance budget", xlabel = "Time", budget_kwargs...)
lines!(ax_c²bud, t_bud, -dc²dt,  label = "-d(∫c²)/dt")
lines!(ax_c²bud, t_bud, -χ_bud,   label = "-∫χ dV")
lines!(ax_c²bud, t_bud, c²_resid, label = "residual", color = :black, linestyle = :dash)
axislegend(ax_c²bud; labelsize = 10, position = :rb)

# Time marker on both budget panels (using the snapshot time shown in the heatmaps)

tₙ = @lift times[$n]
vlines!(ax_eₖbud, tₙ, color = :black, linestyle = :dot)
vlines!(ax_c²bud, tₙ, color = :black, linestyle = :dot)

title = @lift "Time = " * string(round(times[$n], digits=2))
Label(fig[1, 1:4], title, fontsize=24, tellwidth=false);

# Adjust the total figure size based on our panels and record a movie.

resize_to_layout!(fig)
@info "Animating..."
record(fig, filename * ".mp4", 1:length(times), framerate=24) do i
    n[] = i
end
nothing #hide

# ![](two_dimensional_turbulence.mp4)
#
# The two bottom panels show the volume-integrated KE and ``c^2`` budgets. Each plots the negative
# tendency ``-d/dt`` of the integrated quantity alongside ``-\int \varepsilon_k\, \mathrm{d}V`` (respectively
# ``-\int \chi\, \mathrm{d}V``), the only source term that survives volume-integration for a periodic
# incompressible flow with a centered advection scheme. With the tendency negated, the two curves add
# up to the residual, which stays near zero.
