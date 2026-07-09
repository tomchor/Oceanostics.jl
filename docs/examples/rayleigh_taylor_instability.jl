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
simulation = Simulation(model, Δt = 0.1 * Δx / U, stop_time = 7τ)

conjure_time_step_wizard!(simulation, IterationInterval(5), cfl=0.7, max_change=1.1)

# ## Model diagnostics
#
# We report progress with the `TimedMessenger`, which prints, among other things, the wall-clock
# duration of each time step:

using Oceanostics

progress = ProgressMessengers.TimedMessenger()
simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

# ### Sub-filter-scale kinetic-energy budget
#
# Rayleigh-Taylor turbulence converts potential energy into kinetic energy (KE) across a wide range of
# scales, so we follow it with a coarse-graining analysis in the spirit of [Aluie et
# al. (2018)](https://doi.org/10.1175/JPO-D-17-0100.1). A low-pass filter (overbar) splits each field
# into a filtered part and a sub-filter remainder. Here we budget the kinetic energy carried by the
# scales the filter removes,
#
# ```math
# K^s = \tfrac{1}{2}\,\tau^{ii} , \qquad
# \tau^{ij} = \overline{u^i u^j} - \overline{u}^i\,\overline{u}^j ,
# ```
#
# where ``\tau^{ij}`` is the sub-filter stress ([`subfilter_stress_tensor`](@ref)). We apply a Gaussian
# filter of width `ℓ` in the two horizontal directions, which are statistically
# homogeneous; the vertical, singled out by gravity and bounded by the walls, is left unfiltered.
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
#   ([`KineticEnergyCrossScaleFlux`](@ref)), the rate at which the resolved scales hand kinetic energy
#   down to the sub-filter scales. Mind the sign: the same term drains the *filtered* kinetic energy,
#   so what is a sink there is a source here.
# - ``\tau(w, b) = \overline{wb} - \overline{w}\,\overline{b}`` is the sub-filter buoyancy flux (a
#   `subfilter_covariance`), which converts sub-filter potential energy into sub-filter kinetic energy.
# - ``\varepsilon_K^s = \overline{\varepsilon} - \varepsilon^{\ell}`` is the sub-filter dissipation: the
#   filtered total dissipation ``\varepsilon``
#   ([`KineticEnergyDissipationRate`](@ref Oceanostics.KineticEnergyEquation.DissipationRate)) minus the
#   dissipation ``\varepsilon^{\ell}`` of the filtered flow
#   ([`CoarseGrainedKineticEnergyDissipationRate`](@ref)). For a constant viscosity it reduces to
#   ``2\nu[\overline{S^{ij}S^{ij}} - \overline{S}^{ij}\overline{S}^{ij}] \ge 0``, a strictly positive
#   sink; with an LES closure it is the dissipation that the modeled (Smagorinsky-Lilly) stress carries
#   out on the sub-filter scales.
#
# In the code below, ``K^s``, ``\Pi_K``, ``\tau(w,b)`` and ``\varepsilon_K^s`` are written `Kˢ`, `Πₖ`,
# `wbˢ` and `εˢ`.

using Oceananigans.AbstractOperations: @at

ℓ  = 8 * Δx                          # filter scale (full width at half maximum of the Gaussian kernel)
σℓ = ℓ / (2 * sqrt(2 * log(2)))      # corresponding Gaussian standard deviation
gfilter = GaussianFilter(; dims=(1, 2), σ=σℓ)

u, v, w = model.velocities
b = model.tracers.b

## `collocate_diagonals` puts τ₁₁, τ₂₂ and τ₃₃ at cell centers so we can trace them into Kˢ = ½τⁱⁱ
τᵢⱼ = subfilter_stress_tensor(model, gfilter; collocate_diagonals=true)

Kˢ  = @at (Center, Center, Center) (τᵢⱼ.τ₁₁ + τᵢⱼ.τ₂₂ + τᵢⱼ.τ₃₃) / 2  # sub-filter kinetic energy ½τⁱⁱ
Πₖ  = KineticEnergyCrossScaleFlux(model, gfilter)                     # cross-scale flux from resolved scales
wbˢ = subfilter_covariance(w, b, gfilter)                             # sub-filter buoyancy flux τ(w, b)

ε   = KineticEnergyDissipationRate(model)                        # dissipation of the full flow
εˡ  = CoarseGrainedKineticEnergyDissipationRate(model, gfilter)  # dissipation of the filtered flow
εˢ  = gfilter(ε) - εˡ                                            # sub-filter dissipation

# The budget needs only the (cheap) volume integrals of these terms:

∫Kˢ  = Integral(Kˢ)
∫Πₖ  = Integral(Πₖ)
∫wbˢ = Integral(wbˢ)
∫εˢ  = Integral(εˢ)

# For the movie we also keep the coarse-grained kinetic energy
# ``\overline{K} = \tfrac{1}{2}\,\overline{u}_i\overline{u}_i``, the resolved counterpart of ``K^s``.
# Together the two show how the filter splits the flow's kinetic energy between the scales it keeps and
# the scales it removes:

ū, v̄, w̄ = gfilter(u), gfilter(v), gfilter(w)
Kˡ = @at (Center, Center, Center) (ū^2 + v̄^2 + w̄^2) / 2   # coarse-grained kinetic energy ½ūᵢūᵢ

# ## Output
#
# We use two NetCDF writers. A *snapshot* writer stores vertical (`x`–`z`) slices of the buoyancy `b`,
# the cross-scale flux `Πₖ` and the two kinetic energies `Kˡ` and `Kˢ`, at a fixed `y` index (the flow is
# periodic and statistically homogeneous in `y`, so the particular plane makes no difference), while a
# *budget* writer stores only the integrated scalars on `ConsecutiveIterations(TimeInterval(τ/5))`, which
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
Πₖ_pair  = @. 0.5 * (∫Πₖ_t[i1] + ∫Πₖ_t[i2])
wbˢ_pair = @. 0.5 * (∫wbˢ_t[i1] + ∫wbˢ_t[i2])
εˢ_pair  = @. 0.5 * (∫εˢ_t[i1] + ∫εˢ_t[i2])

resid = @. dKˢdt - (Πₖ_pair + wbˢ_pair - εˢ_pair)

using Test                                              #hide
rms(x) = √(sum(abs2, x) / length(x))                    #hide
budget_terms = (dKˢdt, Πₖ_pair, wbˢ_pair, εˢ_pair)      #hide
@test rms(resid) < 0.1 * minimum(rms, budget_terms);    #hide

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

# The middle row splits the kinetic energy across the filter scale: on the left the coarse-grained
# (resolved) energy `Kˡ`, on the right the sub-filter energy `Kˢ` whose budget the bottom panel closes.
# Both are non-negative, so they get a sequential colormap, and each gets its own colour scale because
# the two differ by orders of magnitude.

axKˡ = Axis(fig[3, 1]; title="coarse-grained KE, Kˡ", xlabel="x", ylabel="z", aspect=1)
axKˢ = Axis(fig[3, 3]; title="sub-filter-scale KE, Kˢ", xlabel="x", ylabel="z", aspect=1)

Kˡlim = maximum(Kˡ_arr)
Kˢlim = maximum(Kˢ_arr)

Kˡₙ = @lift Kˡ_arr[:, :, $n]
Kˢₙ = @lift Kˢ_arr[:, :, $n]

hmKˡ = heatmap!(axKˡ, x_caa, z_aac, Kˡₙ; colormap=:magma, colorrange=(0, Kˡlim))
Colorbar(fig[3, 2], hmKˡ)

hmKˢ = heatmap!(axKˢ, x_caa, z_aac, Kˢₙ; colormap=:magma, colorrange=(0, Kˢlim))
Colorbar(fig[3, 4], hmKˢ)

# The bottom panel shows the volume-integrated sub-filter-scale kinetic-energy budget: `d(∫Kˢ)/dt`
# against the two sources that feed it, the cross-scale flux `∫Πₖ dV` handed down from the resolved
# scales and the sub-filter buoyancy flux `∫τ(w,b) dV`, and the single sink that drains it, the
# sub-filter dissipation `−∫εˢ dV`, together with the residual.

ax_bud = Axis(fig[4, 1:4]; xlabel="time [free-fall units]", title="Sub-filter-scale KE budget")
lines!(ax_bud, t_pair ./ τ, dKˢdt,    label="d(∫Kˢ)/dt")
lines!(ax_bud, t_pair ./ τ, Πₖ_pair,  label="∫Πₖ dV  (flux from resolved scales)")
lines!(ax_bud, t_pair ./ τ, wbˢ_pair, label="∫τ(w,b) dV  (sub-filter buoyancy flux)")
lines!(ax_bud, t_pair ./ τ, -εˢ_pair, label="−∫εˢ dV  (sub-filter dissipation)")
lines!(ax_bud, t_pair ./ τ, resid,    label="residual", color=:black, linestyle=:dash)
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
# volume-integrated sub-filter-scale kinetic-energy budget. Of the two sources, the larger one here is
# the sub-filter buoyancy flux `∫τ(w,b) dV`: buoyancy injects kinetic energy directly at small scales,
# rather than only at large ones. That is what we should expect, since the initial interface is about one
# grid spacing thick, so most of the buoyancy variance already sits below the filter scale. The
# cross-scale flux `∫Πₖ dV` is weak and even slightly negative (net backscatter) while the interface is
# still smooth, and only becomes a genuine downscale source once the mixing layer turns turbulent. The
# modeled dissipation `∫εˢ dV` grows to match the two sources, and once it overtakes them the sub-filter
# energy decays (`d(∫Kˢ)/dt < 0`). The small residual shows how well the budget closes.
