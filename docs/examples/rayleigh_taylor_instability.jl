# # [Rayleigh-Taylor instability](@id rayleigh_taylor_example)
#
# This example simulates a three-dimensional [Rayleigh-Taylor
# instability](https://en.wikipedia.org/wiki/Rayleigh%E2%80%93Taylor_instability): a heavy fluid
# initially resting above a light one, which is unstable under gravity. All of the kinetic energy is
# released from the potential energy stored in the unstable stratification, with no fluxes through the
# boundaries. We run it as a large-eddy simulation (LES) and use Oceanostics to close the volume-integrated
# coarse-grained (filtered-flow) kinetic-energy budget, in which the sub-filter scales enter through a
# cross-scale flux and the modeled dissipation.
#
# Before starting, make sure you have the required packages installed for this example, which can be
# done with
#
# ```julia
# using Pkg
# pkg"add Oceananigans, Oceanostics, CairoMakie, NCDatasets"
# ```

# ## Model and simulation setup

using Oceananigans

# We work with nondimensional quantities. We take the domain height `H` as the length scale and the
# buoyancy jump `Δb` across the initial interface as the buoyancy scale, so the free-fall velocity
# `U = √(Δb H)` and the free-fall time `τ = √(H / Δb)` follow. Setting `H = Δb = 1` makes `U = τ = 1`,
# and time is measured in free-fall units:

H  = 1     # domain height (length scale)
Δb = 1     # buoyancy jump across the interface (buoyancy scale)
U  = sqrt(Δb * H)   # free-fall velocity scale
τ  = sqrt(H / Δb)   # free-fall time scale

# We use a cuboid that is periodic in the two horizontal directions and bounded in the
# vertical, with the unstable interface at mid-depth `z = 0`:

N = 48
grid = RectilinearGrid(size=(N, N÷2, N), x=(-H/2, H/2), y=(-H/4, H/4), z=(-H/2, H/2),
                       topology=(Periodic, Periodic, Bounded))

# ## Closure and model
#
# Rayleigh-Taylor mixing is fully three-dimensional and turbulent, so we model the unresolved motions
# with a Large Eddy Simulation (LES) closure:

closure = SmagorinskyLilly(C=0.3) # A large C adds eddy viscosity, keeping this very coarse example well behaved

# We build a `NonhydrostaticModel` with a fourth-order centered advection scheme, a third-order
# Runge-Kutta timestepper, and a buoyancy `b` as the active tracer. A centered scheme is non-dissipative
# and, unlike an upwind scheme, adds no numerical dissipation of its own, so essentially all of the
# modeled dissipation comes from the LES closure.

model = NonhydrostaticModel(grid; timestepper = :RungeKutta3,
                            advection = Centered(order=4),
                            closure,
                            buoyancy = BuoyancyTracer(), tracers = :b)

# ## Initial condition
#
# The initial buoyancy is a hyperbolic-tangent profile that is *heavy on top*: it decreases from
# `+Δb/2` at the bottom to `−Δb/2` at the top, so `∂b/∂z < 0` and the stratification is unstable. The
# interface is thin (its half-thickness `δ` is about one grid spacing) and we perturb it with
# small-amplitude random noise localized to the interface, which seeds a broad band of horizontal
# wavelengths and produces a multi-mode, turbulent instability rather than a single growing bubble. We
# seed the random number generator so the run is reproducible:

δ = 0.02 * H                    # interface half-thickness
b₀(z) = -(Δb / 2) * tanh(z / δ) # +Δb/2 at the bottom, −Δb/2 at the top

using Random
Random.seed!(43)
noise_amplitude = 1e-2
bᵢ(x, y, z) = b₀(z) + noise_amplitude * Δb * randn() * exp(-(z / δ)^2)

set!(model, b=bᵢ)

# ## Simulation
#
# We create an adaptive-time-step simulation. The initial step is set conservatively from the
# horizontal grid spacing and the free-fall velocity, and a `TimeStepWizard` adapts it as the mixing
# layer accelerates:

Δx = minimum_xspacing(grid)
simulation = Simulation(model, Δt = 0.1 * Δx / U, stop_time = 10τ)

conjure_time_step_wizard!(simulation, IterationInterval(5), cfl=0.7, max_change=1.1)

# ## Model diagnostics
#
# We report progress with the `TimedMessenger`, which prints, among other things, the wall-clock
# duration of each time step:

using Oceanostics

progress = ProgressMessengers.TimedMessenger()
simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

# ### Coarse-grained kinetic-energy budget
#
# Rayleigh-Taylor turbulence converts potential energy into kinetic energy (KE) across a wide range of
# scales, so we follow it with a coarse-graining (filtered-flow) analysis in the spirit of [Aluie et
# al. (2018)](https://doi.org/10.1175/JPO-D-17-0100.1), closing the volume-integrated budget of the
# filtered KE ``\overline{K} = \tfrac{1}{2}\,\overline{u}_i\overline{u}_i``. We apply a
# Gaussian filter of width `ℓ` (a few grid cells) in the two horizontal directions, which are
# statistically homogeneous; the vertical, singled out by gravity and bounded by the walls, is left
# unfiltered.
#
# There is no mean flow and no boundary forcing, so the only source is buoyancy production. Volume
# integrated (advection and pressure work integrate to zero, because the flow is doubly periodic and
# `w = 0` with free slip at the top and bottom), the budget reads
#
# ```math
# \frac{d}{dt} \int \overline{K}\, dV
#   = \int \overline{w}\,\overline{b}\, dV
#   - \int \Pi_K\, dV
#   - \int \overline{\varepsilon}\, dV ,
# ```
#
# with a buoyancy production ``\overline{w}\,\overline{b}`` (the conversion of potential into filtered
# kinetic energy), the cross-scale kinetic-energy flux ``\Pi_K = -\tau^{ij}\overline{S}^{ij}`` to
# sub-filter scales ([`KineticEnergyCrossScaleFlux`](@ref)), and the coarse-grained viscous dissipation
# ``\overline{\varepsilon}`` of the filtered flow ([`CoarseGrainedKineticEnergyDissipationRate`](@ref)).
# With an LES closure, ``\overline{\varepsilon}`` is the dissipation carried out by the modeled
# (Smagorinsky-Lilly) stress acting on the filtered flow. Note that ``\Pi_K`` and
# ``\overline{\varepsilon}`` are two separate sinks of ``\overline{K}``: the first moves energy across
# the filter scale, the second removes it through the modeled stress. In the code below,
# ``\overline{K}``, ``\Pi_K`` and ``\overline{\varepsilon}`` are written `Kˡ`, `Πₖ` and `εˡ`.

using Oceananigans.AbstractOperations: @at

ℓ  = 4 * Δx                          # filter scale (full width at half maximum)
σℓ = ℓ / (2 * sqrt(2 * log(2)))      # corresponding Gaussian standard deviation
filt = GaussianFilter(; dims=(1, 2), σ=σℓ, boundary=:shrink)

u, v, w = model.velocities
b = model.tracers.b
ū, v̄, w̄, b̄ = filt(u), filt(v), filt(w), filt(b)

Kˡ = @at (Center, Center, Center) (ū^2 + v̄^2 + w̄^2) / 2   # filtered (coarse-grained) kinetic energy
wb = @at (Center, Center, Center) (w̄ * b̄)                 # buoyancy production
Πₖ = KineticEnergyCrossScaleFlux(model, filt)             # cross-scale flux to sub-filter scales
εˡ = CoarseGrainedKineticEnergyDissipationRate(model, filt)  # coarse-grained dissipation

# The budget needs only the (cheap) volume integrals of these terms:

∫Kˡ = Integral(Kˡ)
∫wb = Integral(wb)
∫Πₖ = Integral(Πₖ)
∫εˡ = Integral(εˡ)

# ## Output
#
# We use two NetCDF writers. A *snapshot* writer stores a vertical (`x`–`z`) slice of the buoyancy `b`
# and the cross-scale flux `Πₖ` at a fixed `y` index (the flow is periodic and statistically homogeneous
# in `y`, so the particular plane makes no difference), while a *budget* writer stores only the
# integrated scalars on `ConsecutiveIterations(TimeInterval(τ/5))`, which takes a second sample one model
# step after each output time. That lets us finite-difference `∫Kˡ` across that single step to estimate
# `d/dt`, exactly as in the [Kelvin-Helmholtz example](@ref kelvin_helmholtz_example).

using NCDatasets
filename = joinpath(@__DIR__, "rayleigh_taylor_instability")

j_mid = N ÷ 2
simulation.output_writers[:fields] = NetCDFWriter(model, (; b, Πₖ),
                                                  filename = filename,
                                                  schedule = TimeInterval(τ / 5),
                                                  indices = (:, j_mid, :),
                                                  overwrite_existing = true)   
   
simulation.output_writers[:budget] = NetCDFWriter(model, (; ∫Kˡ, ∫wb, ∫Πₖ, ∫εˡ),
                                                  filename = filename * "_budget",
                                                  schedule = ConsecutiveIterations(TimeInterval(τ / 5)),
                                                  overwrite_existing = true)

# ## Run the simulation and process results
#
# To run the simulation:

run!(simulation)

# We read the buoyancy and cross-scale-flux slices, and their `x`–`z` coordinates, back with
# `NCDataset` (both fields live at cell centers, so their coordinates are `x_caa` and `z_aac`); the
# singleton `y` dimension of the slice is dropped:

using CairoMakie

ds = NCDataset(simulation.output_writers[:fields].filepath)
times = ds["time"][:]
x_caa = ds["x_caa"][:]
z_aac = ds["z_aac"][:]
b_arr = ds["b"][:, 1, :, :]
Π_arr = ds["Πₖ"][:, 1, :, :]
close(ds)

# The integrated budget scalars come in consecutive-iteration pairs `(2k-1, 2k)`; a one-step finite
# difference inside each pair gives `d(∫Kˡ)/dt`, and each source term is evaluated at the pair
# midpoint. The residual measures how well the coarse-grained budget closes.

bud_filepath = simulation.output_writers[:budget].filepath
ds_bud = NCDataset(bud_filepath)
times_bud = ds_bud["time"][:]
∫Kˡ_t = ds_bud["∫Kˡ"][:]
∫wb_t = ds_bud["∫wb"][:]
∫Πₖ_t = ds_bud["∫Πₖ"][:]
∫εˡ_t = ds_bud["∫εˡ"][:]
close(ds_bud)

i1 = 1:2:length(times_bud)-1   # primary snapshots
i2 = 2:2:length(times_bud)     # consecutive-iteration snapshots
Δt_pair = times_bud[i2] .- times_bud[i1]
t_pair = @. 0.5 * (times_bud[i1] + times_bud[i2])

dKˡdt   = (∫Kˡ_t[i2] .- ∫Kˡ_t[i1]) ./ Δt_pair
wb_pair = @. 0.5 * (∫wb_t[i1] + ∫wb_t[i2])
Πₖ_pair = @. 0.5 * (∫Πₖ_t[i1] + ∫Πₖ_t[i2])
εˡ_pair = @. 0.5 * (∫εˡ_t[i1] + ∫εˡ_t[i2])

resid = @. dKˡdt - (wb_pair - Πₖ_pair - εˡ_pair)

using Test                              #hide
rms(x) = √(sum(abs2, x) / length(x))    #hide
@test rms(resid) < 0.06 * rms(dKˡdt);   #hide

# ## Plotting
#
# We build the figure: on top, the vertical slices of buoyancy `b` (the spikes and bubbles) and of the
# cross-scale flux `Πₖ`; below, the volume-integrated coarse-grained kinetic-energy budget.

set_theme!(Theme(fontsize=18))
fig = Figure(size=(1000, 850))

n = Observable(1)

axb = Axis(fig[2, 1]; title="buoyancy, b", xlabel="x", ylabel="z", aspect=1)
axΠ = Axis(fig[2, 3]; title="cross-scale KE flux, Πₖ", xlabel="x", ylabel="z", aspect=1)

blim = 0.8 * maximum(abs, b_arr[:, :, 1])
Πlim = 0.8 * maximum(abs, Π_arr[:, :, end])

bₙ = @lift b_arr[:, :, $n]
Πₙ = @lift Π_arr[:, :, $n]

hmb = heatmap!(axb, x_caa, z_aac, bₙ; colormap=:balance, colorrange=(-blim, blim))
Colorbar(fig[2, 2], hmb)

hmΠ = heatmap!(axΠ, x_caa, z_aac, Πₙ; colormap=:balance, colorrange=(-Πlim, Πlim))
Colorbar(fig[2, 4], hmΠ)

# The bottom panel shows the volume-integrated coarse-grained kinetic-energy budget: `d(∫Kˡ)/dt`
# against the single source that feeds it, the buoyancy production `∫w̄b̄ dV`, and the two sinks that
# drain it, the cross-scale flux `−∫Πₖ dV` and the coarse-grained dissipation `−∫εˡ dV`, together with
# the residual.

ax_bud = Axis(fig[3, 1:4]; xlabel="time [free-fall units]", title="Coarse-grained KE budget")
lines!(ax_bud, t_pair ./ τ, dKˡdt,   label="d(∫Kˡ)/dt")
lines!(ax_bud, t_pair ./ τ, wb_pair,  label="∫w̄b̄ dV  (buoyancy production)")
lines!(ax_bud, t_pair ./ τ, -Πₖ_pair, label="−∫Πₖ dV  (flux to sub-filter scales)")
lines!(ax_bud, t_pair ./ τ, -εˡ_pair, label="−∫εˡ dV  (dissipation)")
lines!(ax_bud, t_pair ./ τ, resid,   label="residual", color=:black, linestyle=:dash)
axislegend(ax_bud; position=:rt, labelsize=10)

vlines!(ax_bud, @lift(times[$n] / τ), color=:black, linestyle=:dash)

title = @lift "Rayleigh-Taylor instability, t = " * string(round(times[$n] / τ, digits=2)) * " τ"
fig[1, 1:4] = Label(fig, title, fontsize=22, tellwidth=false)

@info "Animating..."
record(fig, "rayleigh_taylor_instability.mp4", 1:length(times), framerate=12) do i
    n[] = i
end

# ![](rayleigh_taylor_instability.mp4)
#
# As the heavy fluid falls in spikes and the light fluid rises in bubbles (left), the flow rolls up and
# breaks into a turbulent mixing layer. The cross-scale flux `Πₖ` (right) marks where kinetic energy
# crosses the filter scale, mostly forward (downscale, `Πₖ > 0`) along the sharpening edges of the
# spikes and bubbles, with patches of backscatter (`Πₖ < 0`). The bottom panel shows the
# volume-integrated coarse-grained kinetic-energy budget: buoyancy production feeds the filtered kinetic
# energy, which is drained by two separate sinks, the cross-scale flux `∫Πₖ dV` that hands energy to the
# sub-filter scales and the modeled dissipation `∫εˡ dV`. The small residual shows how well the budget
# closes.
