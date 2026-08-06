# # [Rayleigh-Taylor instability](@id rayleigh_taylor_example)
#
# This example simulates a three-dimensional [Rayleigh-Taylor
# instability](https://en.wikipedia.org/wiki/Rayleigh%E2%80%93Taylor_instability): a heavy fluid
# initially resting above a light one, which is unstable under gravity. All of the kinetic energy is
# released from the potential energy stored in the unstable stratification, with no fluxes through the
# boundaries. We run it as a large-eddy simulation (LES) and use Oceanostics to close the volume-integrated
# sub-filter-scale kinetic-energy budget, that is, the budget of the kinetic energy carried by the scales
# that a low-pass filter removes from the flow.
#
# Before starting, make sure you have the required packages installed for this example, which can be
# done with
#
# ```julia
# using Pkg
# pkg"add Oceananigans, Oceanostics, CairoMakie, NCDatasets"
# ```

# ## Parameters and grid

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
# We model the stresses with a Large Eddy Simulation (LES) closure:

closure = SmagorinskyLilly(C=0.3) # A large C increases eddy viscosity, keeping this very coarse example well behaved

# We build a `NonhydrostaticModel` with a fourth-order centered advection scheme, a third-order
# Runge-Kutta timestepper, and a buoyancy `b` as the active tracer. A centered scheme is non-dissipative
# and adds no numerical dissipation of its own, so essentially all of the dissipation comes from the LES closure.

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
simulation = Simulation(model, Δt = 0.1 * Δx / U, stop_time = 7τ)
conjure_time_step_wizard!(simulation, IterationInterval(2), cfl=0.8, max_change=1.1)

# ## Model diagnostics
#
# We report progress with the `TimedMessenger`

using Oceanostics
progress = ProgressMessengers.TimedMessenger()
add_callback!(simulation, progress, IterationInterval(100))

# ### Sub-filter-scale kinetic-energy budget
#
# Rayleigh-Taylor turbulence converts potential energy into kinetic energy (KE) across a wide range of
# scales, so we follow it with a filtering analysis in the spirit of [Aluie et
# al. (2018)](https://doi.org/10.1175/JPO-D-17-0100.1). A low-pass filter (overbar) splits each field
# into a filtered part and a sub-filter remainder. Here we budget the kinetic energy carried by the
# scales the filter removes,
#
# ```math
# K^s = \tfrac{1}{2}\,\tau^{ii} , \qquad
# \tau^{ij} = \overline{u^i u^j} - \overline{u}^i\,\overline{u}^j ,
# ```
#
# where ``\tau^{ij}`` is the sub-filter stress ([`subfilter_stress_tensor`](@ref)), so ``K^s`` itself is
# computed by [`SubFilterKineticEnergy`](@ref). We apply a Gaussian
# filter of width `ℓ` in the two horizontal directions, which are statistically
# homogeneous; the vertical direction is left unfiltered.
#
# Volume integrated (the transport terms integrate to zero, because the flow is doubly periodic and
# `w = 0` with free slip at the top and bottom), the budget reads
#
# ```math
# \frac{d}{dt} \int K^s\, dV
#   = \int \Pi_K\, dV
#   + \int \tau(w, b)\, dV
#   - \int \varepsilon_K^s\, dV ,
# ```
#
# with two sources and one sink:
#
# - ``\Pi_K = -\tau^{ij}\overline{S}^{ij}`` is the cross-scale kinetic-energy flux
#   ([`KineticEnergyCrossScaleFlux`](@ref)), the rate at which the filtered scales hand kinetic energy
#   down to the sub-filter scales.
# - ``\tau(w, b) = \overline{wb} - \overline{w}\,\overline{b}`` is the sub-filter buoyancy flux (a
#   `subfilter_covariance`), which converts sub-filter potential energy into sub-filter kinetic energy.
# - ``\varepsilon_K^s = \overline{\varepsilon} - \varepsilon^{\ell}`` is the sub-filter dissipation
#   ([`SubFilterKineticEnergyDissipationRate`](@ref)): the filtered total dissipation ``\varepsilon``
#   ([`KineticEnergyDissipationRate`](@ref Oceanostics.KineticEnergyEquation.DissipationRate)) minus the
#   dissipation ``\varepsilon^{\ell}`` of the filtered flow
#   ([`FilteredKineticEnergyDissipationRate`](@ref)). For a constant viscosity it reduces to
#   ``2\nu[\overline{S^{ij}S^{ij}} - \overline{S}^{ij}\overline{S}^{ij}] \ge 0``, a strictly positive
#   sink; with an LES closure it is the dissipation that the modeled stress carries out on the sub-filter scales.
#
# In the code below, ``K^s``, ``\Pi_K``, ``\tau(w,b)`` and ``\varepsilon_K^s`` are written `Kˢ`, `Πₖ`,
# `wbˢ` and `εˢ`.

using Oceananigans.AbstractOperations: @at

ℓ  = 8 * Δx                          # filter scale (full width at half maximum of the Gaussian kernel)
σℓ = ℓ / (2 * sqrt(2 * log(2)))      # corresponding Gaussian standard deviation
gfilter = GaussianFilter(dims=(1, 2), σ=σℓ)

u, v, w = model.velocities
b = model.tracers.b

Kˢ  = SubFilterKineticEnergy(model, gfilter)     # sub-filter kinetic energy ½τⁱⁱ
Πₖ  = KineticEnergyCrossScaleFlux(model, gfilter)  # cross-scale flux from filtered scales
wbˢ = subfilter_covariance(w, b, gfilter)          # sub-filter buoyancy flux τ(w, b)
εˢ  = SubFilterKineticEnergyDissipationRate(model, gfilter)  # sub-filter dissipation ε̄ − εˡ

# The budget needs only the (cheap) volume integrals of these terms:

∫Kˢ  = ∫dV(Kˢ)
∫Πₖ  = ∫dV(Πₖ)
∫wbˢ = ∫dV(wbˢ)
∫εˢ  = ∫dV(εˢ)

# For the movie we also keep the filtered kinetic energy
# ``\overline{K} = \tfrac{1}{2}\,\overline{u}_i\overline{u}_i`` ([`FilteredKineticEnergy`](@ref)), the
# filtered counterpart of ``K^s``. Together the two show how the filter splits the flow's kinetic energy
# between the scales it keeps and the scales it removes:

## `FilteredKineticEnergy` materializes the filtered velocities internally, so the multi-direction filter
## runs on its fast staged path (see the filter performance notes and `check_filter_staging`).
Kˡ = FilteredKineticEnergy(model, gfilter)  # kinetic energy of the filtered flow

# ## Output
#
# We use two NetCDF writers. A snapshot writer stores vertical (`x`–`z`) slices of the buoyancy `b`,
# the cross-scale flux `Πₖ` and the two kinetic energies `Kˡ` and `Kˢ`, at a fixed `y` index (the flow is
# periodic and statistically homogeneous in `y`, so the particular plane makes no difference), while a
# budget writer stores only the integrated scalars on `ConsecutiveIterations(TimeInterval(τ/5))`, which
# takes a second sample one model step after each output time. That lets us finite-difference `∫Kˢ` across
# that single step to estimate `d/dt`, exactly as in the
# [Kelvin-Helmholtz example](@ref kelvin_helmholtz_example).

using NCDatasets
filename = joinpath(@__DIR__, "rayleigh_taylor_instability")

simulation.output_writers[:fields] = NetCDFWriter(model, (; b, Πₖ, Kˡ, Kˢ),
                                                  filename = filename,
                                                  schedule = TimeInterval(τ / 5),
                                                  indices = (:, 1, :),
                                                  overwrite_existing = true)

simulation.output_writers[:budget] = NetCDFWriter(model, (; ∫Kˢ, ∫Πₖ, ∫wbˢ, ∫εˢ),
                                                  filename = filename * "_budget",
                                                  schedule = ConsecutiveIterations(TimeInterval(τ / 5)),
                                                  overwrite_existing = true)

# ## Run the simulation and process results
#
# To run the simulation:

run!(simulation)

# We read the four field slices, and their `x`–`z` coordinates, back with `NCDataset` (all of them live
# at cell centers, so their coordinates are `x_caa` and `z_aac`); the singleton `y` dimension of the
# slice is dropped:

using CairoMakie

ds = NCDataset(simulation.output_writers[:fields].filepath)
times = ds["time"][:]
x_caa = ds["x_caa"][:]
z_aac = ds["z_aac"][:]
b_arr = ds["b"][:, 1, :, :]
Π_arr = ds["Πₖ"][:, 1, :, :]
Kˡ_arr = ds["Kˡ"][:, 1, :, :]
Kˢ_arr = ds["Kˢ"][:, 1, :, :]
close(ds)

# The integrated budget scalars come in consecutive-iteration pairs `(2k-1, 2k)`; a one-step finite
# difference inside each pair gives `d(∫Kˢ)/dt`, and each budget term is evaluated at the pair
# midpoint. The residual measures how well the sub-filter-scale budget closes.

bud_filepath = simulation.output_writers[:budget].filepath
ds_bud = NCDataset(bud_filepath)
times_bud = ds_bud["time"][:]
∫Kˢ_t = ds_bud["∫Kˢ"][:]
∫Πₖ_t = ds_bud["∫Πₖ"][:]
∫wbˢ_t = ds_bud["∫wbˢ"][:]
∫εˢ_t = ds_bud["∫εˢ"][:]
close(ds_bud)

i1 = 1:2:length(times_bud)-1   # primary snapshots
i2 = 2:2:length(times_bud)     # consecutive-iteration snapshots
Δt_pair = times_bud[i2] .- times_bud[i1]
t_pair = @. 0.5 * (times_bud[i1] + times_bud[i2])

dKˢdt    = (∫Kˢ_t[i2] .- ∫Kˢ_t[i1]) ./ Δt_pair
Πₖ_pair  = @. 0.5 * (∫Πₖ_t[i1] + ∫Πₖ_t[i2]);
wbˢ_pair = @. 0.5 * (∫wbˢ_t[i1] + ∫wbˢ_t[i2]);
εˢ_pair  = @. 0.5 * (∫εˢ_t[i1] + ∫εˢ_t[i2]);

# Residual in sum-to-zero form: the negative tendency plus the sources, so the plotted curves add to it
resid = @. -dKˢdt + Πₖ_pair + wbˢ_pair - εˢ_pair

using Test                                                          #hide
rms(x) = √(sum(abs2, x) / length(x))                                #hide
budget_scale = rms(@. abs(Πₖ_pair) + abs(wbˢ_pair) + abs(εˢ_pair))  #hide
@test rms(resid) < 0.02 * budget_scale;                             #hide

# ## Plotting
#
# We build the figure in three rows: the vertical slices of the buoyancy `b` (the spikes and bubbles) and
# of the cross-scale flux `Πₖ` on top; the two kinetic energies that the filter separates, `Kˡ` and `Kˢ`,
# in the middle; and the volume-integrated sub-filter-scale kinetic-energy budget at the bottom.

set_theme!(Theme(fontsize=18))
fig = Figure(size=(1000, 1200))

n = Observable(1)

axb = Axis(fig[2, 1]; title="buoyancy, b", xlabel="x", ylabel="z", aspect=1)
axΠ = Axis(fig[2, 3]; title="cross-scale KE flux, Πₖ", xlabel="x", ylabel="z", aspect=1)

blim = 0.8 * maximum(abs, b_arr[:, :, 1])
Πlim = 0.5 * maximum(abs, Π_arr)

bₙ = @lift b_arr[:, :, $n]
Πₙ = @lift Π_arr[:, :, $n]

hmb = heatmap!(axb, x_caa, z_aac, bₙ; colormap=:balance, colorrange=(-blim, blim))
Colorbar(fig[2, 2], hmb)

hmΠ = heatmap!(axΠ, x_caa, z_aac, Πₙ; colormap=:balance, colorrange=(-Πlim, Πlim))
Colorbar(fig[2, 4], hmΠ)

# The middle row splits the kinetic energy across the filter scale: on the left the filtered
# energy `Kˡ`, on the right the sub-filter energy `Kˢ` whose budget the bottom panel closes.
# Both are non-negative, so they get a sequential colormap, and each gets its own colour scale because
# the two differ by orders of magnitude.

axKˡ = Axis(fig[3, 1]; title="filtered KE, Kˡ", xlabel="x", ylabel="z", aspect=1)
axKˢ = Axis(fig[3, 3]; title="sub-filter-scale KE, Kˢ", xlabel="x", ylabel="z", aspect=1)

Kˡlim = maximum(Kˡ_arr)
Kˢlim = maximum(Kˢ_arr)

Kˡₙ = @lift Kˡ_arr[:, :, $n]
Kˢₙ = @lift Kˢ_arr[:, :, $n]

hmKˡ = heatmap!(axKˡ, x_caa, z_aac, Kˡₙ; colormap=:magma, colorrange=(0, Kˡlim))
Colorbar(fig[3, 2], hmKˡ)

hmKˢ = heatmap!(axKˢ, x_caa, z_aac, Kˢₙ; colormap=:magma, colorrange=(0, Kˢlim))
Colorbar(fig[3, 4], hmKˢ)

# The bottom panel shows the volume-integrated sub-filter-scale kinetic-energy budget. We plot the
# negative tendency `−d(∫Kˢ)/dt` together with the two sources that feed it, the cross-scale flux
# `∫Πₖ dV` handed down from the filtered scales and the sub-filter buoyancy flux `∫τ(w,b) dV`, and the
# single sink that drains it, the sub-filter dissipation `−∫εˢ dV`. With the tendency negated, the four
# curves sum to the residual.

ax_bud = Axis(fig[4, 1:4]; xlabel="time [free-fall units]", title="Sub-filter-scale KE budget")
lines!(ax_bud, t_pair ./ τ, -dKˢdt,   label="−d(∫Kˢ)/dt")
lines!(ax_bud, t_pair ./ τ, Πₖ_pair,  label="∫Πₖ dV  (flux from filtered scales)")
lines!(ax_bud, t_pair ./ τ, wbˢ_pair, label="∫τ(w,b) dV  (sub-filter buoyancy flux)")
lines!(ax_bud, t_pair ./ τ, -εˢ_pair, label="−∫εˢ dV  (sub-filter dissipation)")
lines!(ax_bud, t_pair ./ τ, resid,    label="residual", color=:black, linestyle=:dash)
axislegend(ax_bud; position=:lt, labelsize=10)

vlines!(ax_bud, @lift(times[$n] / τ), color=:black, linestyle=:dash)

title = @lift "Rayleigh-Taylor instability, t = " * string(round(times[$n] / τ, digits=2)) * " τ"
fig[1, 1:4] = Label(fig, title, fontsize=22, tellwidth=false)

@info "Animating..."
record(fig, "rayleigh_taylor_instability.mp4", 1:length(times), framerate=12) do i
    n[] = i
end

# ![](rayleigh_taylor_instability.mp4)
#
# As the heavy fluid falls in spikes and the light fluid rises in bubbles, the flow rolls up and
# breaks into a turbulent mixing layer. The bottom panel shows the volume-integrated SFS KE budget.
# The sub-filter buoyancy flux `∫τ(w,b) dV` and dissipation `∫εˢ dV` are the dominant terms, with the
# tendency `−d(∫Kˢ)/dt` in third place. The residual is small, which shows that the budget closes well
# and that the filtering analysis is consistent with the simulation.
