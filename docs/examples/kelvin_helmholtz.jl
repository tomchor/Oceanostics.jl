# # [Kelvin-Helmholtz instability](@id kelvin_helmholtz_example)
#
# In this example we simulate a simple 2D Kelvin-Helmholtz instability and then use
# Oceanostics to close the volume-integrated kinetic-energy budget of the filtered flow.
#
# Before starting, make sure you have the required packages installed for this example, which can be
# done with
#
# ```julia
# using Pkg
# pkg"add Oceananigans, Oceanostics, CairoMakie"
# ```

# ## Model and simulation setup

using Oceananigans

# We work with nondimensional quantities, following the standard nondimensionalization of the
# stratified shear layer ([Kaminski and Smyth, 2019](https://doi.org/10.1017/jfm.2018.973)). We
# nondimensionalize the Boussinesq equations using the shear-layer half-width `h` as the length
# scale and the velocity scale `U` (half the velocity difference across the layer), so that time is
# measured in units of `h / U`. The flow is then governed by three nondimensional numbers — the
# Richardson number `Ri₀`, the Reynolds number `Re = U h / ν`, and the Prandtl number `Pr = ν / κ` —
# from which the viscosity `ν` and the buoyancy diffusivity `κ` follow:

U   = 1     # velocity scale (half the velocity difference across the shear layer)
h   = 1     # length scale (shear-layer half-width)
Ri₀ = 0.1   # Richardson number
Re  = 4e3   # Reynolds number (bounded by the grid; see the resolution note below)
Pr  = 1     # Prandtl number

ν = U * h / Re   # viscosity
κ = ν / Pr       # buoyancy diffusivity

# We begin by creating a model with this isotropic diffusivity and centered advection on a `xz`
# grid, using a buoyancy `b` as the active scalar. We make the box one wavelength of the most
# unstable Kelvin-Helmholtz mode wide (`k_max = 0.4446 / h`; [Michalke,
# 1964](https://doi.org/10.1017/S0022112064000908)), so that the perturbation we seed below fits
# periodically:

N = 256
k_max = 0.4446 / h   # most unstable KH wavenumber (Michalke, 1964)
Lx = 2π / k_max      # one most-unstable wavelength
Lz = 10
grid = RectilinearGrid(size=(N, N), x=(-Lx/2, +Lx/2), z=(-Lz/2, +Lz/2), topology=(Periodic, Flat, Bounded))

model = NonhydrostaticModel(grid; timestepper = :RungeKutta3,
                            advection = Centered(order=4), # A centered scheme is used here to minimize numerical dissipation
                            closure = ScalarDiffusivity(; ν, κ),
                            buoyancy = BuoyancyTracer(), tracers = :b)

# We use hyperbolic tangent profiles with the same length scale `h` for both the shear flow and
# the stratification. The buoyancy jump `B₀ = U² Ri₀ / h` is chosen so that the gradient Richardson
# number `N² / (∂u/∂z)²` reaches its minimum value `Ri₀ = 0.1` — below the classical stability
# threshold of 1/4 — at the center of the shear layer (`z = 0`), where the flow is most unstable. To
# kick off the instability we perturb the vertical velocity `w` with the most unstable mode
# `sin(k_max x)`, localized to the shear layer by a Gaussian envelope `exp(-z²)` and given a random
# amplitude. We seed the random number generator so the perturbation — and hence the movie — is
# reproducible:

B₀ = U^2 * Ri₀ / h
perturbation_amplitude = 5e-2

shear_flow(x, z) = U * tanh(z / h)
stratification(x, z) = B₀ * tanh(z / h)
perturbation(x, z) = perturbation_amplitude * abs(randn()) * exp(-z^2) * sin(x * k_max - π)

using Random
Random.seed!(43)
set!(model, u=shear_flow, b=stratification, w=perturbation)

#
# Next create an adaptive-time-step simulation using the model above. The initial time step is set
# conservatively from the horizontal grid spacing and velocity scale; the `TimeStepWizard` below
# adapts it as the flow evolves:

Δx = minimum_xspacing(grid)
simulation = Simulation(model, Δt = 0.2 * Δx / U, stop_time=120)
conjure_time_step_wizard!(simulation, IterationInterval(2), cfl=0.8, max_Δt=1)

# ## Model diagnostics
#
# We set-up a progress messenger using the `TimedMessenger`, which displays, among other
# information, the time step duration

using Oceanostics

progress = ProgressMessengers.TimedMessenger()
simulation.callbacks[:progress] = Callback(progress, IterationInterval(200))


# We can also define some useful diagnostics of the flow, starting with the `RichardsonNumber`

Ri = RichardsonNumber(model)

# We also set-up the `QVelocityGradientTensorInvariant`, which is usually used for visualizing
# vortices in the flow:

Q = QVelocityGradientTensorInvariant(model)

# Q is one of the velocity gradient tensor invariants and it measures the amount of vorticity versus
# the strain in the flow and, when it's positive, indicates a vortex. This method of vortex
# visualization is called the [Q-criterion](https://tinyurl.com/mwv6fskc).

# ### Coarse-grained kinetic energy budget
#
# Kelvin-Helmholtz billows draw kinetic energy from the mean shear and pass it down to ever-smaller
# scales, so this is a natural flow in which to look at a *coarse-grained* (filtered) kinetic-energy
# budget in the spirit of [Aluie et al. (2018)](https://doi.org/10.1175/JPO-D-17-0100.1). We define a
# box filter whose width is comparable to the shear-layer half-width `h` and use it to build every
# term in the budget of the filtered kinetic energy ``\overline{K} = \tfrac{1}{2}\overline{u}_i\overline{u}_i``.
# Volume-integrated — advection and pressure work integrate to zero, since the flow is periodic in `x`
# and `w = 0` with free slip at the `z` walls — that budget reads
#
# ```math
# \frac{d}{dt} \int \overline{K}\, dV
#   = \int \overline{w}\,\overline{b}\, dV
#   - \int \Pi_K\, dV
#   - \int \overline{\varepsilon}\, dV ,
# ```
#
# with a buoyancy production ``\overline{w}\,\overline{b}`` (the conversion between filtered kinetic and
# potential energy), the cross-scale kinetic-energy flux ``\Pi_K`` to subfilter scales
# ([`KineticEnergyCrossScaleFlux`](@ref)), and viscous dissipation due to the coarse-grained
# flow ``\overline{\varepsilon}`` ([`FilteredKineticEnergyDissipationRate`](@ref)).

using Oceananigans.AbstractOperations: @at

# A box filter is specified by its stencil size `N` in grid points rather than by a physical width, so
# we pick the odd `N` whose stencil spans roughly the shear-layer half-width `h`. The grid is slightly
# anisotropic here (`Δx ≈ 0.11 h`, `Δz ≈ 0.078 h`), so no single `N` matches `h` exactly in both
# directions; `N = 11` brackets it, spanning `11Δx ≈ 1.2 h` in `x` and `11Δz ≈ 0.86 h` in `z`.

bfilter = BoxFilter(; dims=(1, 3), N=11, boundary=:shrink)  # stencil ≈ h wide, the shear-layer half-width

u, w = model.velocities.u, model.velocities.w
b = model.tracers.b
## Materialize each filtered field so the multi-direction filter takes its fast staged (separable)
## path; composing the raw `bfilter(u)` into `ū^2 + w̄^2` below would instead run it fused (see the
## filter performance notes and `check_filter_staging`).
ū, w̄, b̄ = Field(bfilter(u)), Field(bfilter(w)), Field(bfilter(b))

Kˡ = @at (Center, Center, Center) (ū^2 + w̄^2) / 2   # filtered kinetic energy ½ūᵢūᵢ
w̄b̄ = @at (Center, Center, Center) (w̄ * b̄)           # buoyancy production of the filtered flow
Πₖ = KineticEnergyCrossScaleFlux(model, bfilter; dims=(1, 3))
εˡ = FilteredKineticEnergyDissipationRate(model, bfilter)

# The budget only needs the (cheap) volume integrals of these terms:

∫Kˡ = Integral(Kˡ)
∫w̄b̄ = Integral(w̄b̄)
∫Πₖ = Integral(Πₖ)
∫εˡ = Integral(εˡ)

# ### Available and background potential energy
#
# Billows do more than cascade kinetic energy. They overturn the stratification, carrying dense fluid
# over light, and diffusion across the folded interface then mixes the two irreversibly. To separate
# that irreversible mixing from the reversible stirring that precedes it, we split the potential energy
# following [Winters et al. (1995)](https://doi.org/10.1017/S002211209500125X). Sorting the buoyancy
# field adiabatically into the state of minimum potential energy assigns every parcel a reference height
# ``z^\star``, and the potential energy splits into
#
# ```math
# \underbrace{-bz}_{E_p} = \underbrace{-b z^\star}_{E_b} + \underbrace{-b(z - z^\star)}_{E_a} ,
# ```
#
# a background part ``E_b`` locked into the sorted state, which only irreversible mixing can change, and
# an available part ``E_a`` that the flow can still trade back and forth with kinetic energy. Both are
# built from the same reference height, so we compute ``z^\star`` once with
# [`reference_height`](@ref) and hand it to both diagnostics:

z✶ = reference_height(model)

∫K   = Integral(KineticEnergyEquation.KineticEnergy(model))
∫Eₚ  = Integral(PotentialEnergy(model))
∫E_b = Integral(BackgroundPotentialEnergy(model, z✶))
∫E_a = Integral(AvailablePotentialEnergy(model, z✶))

# Sorting couples every cell in the domain to every other one, so `z✶` cannot be a pointwise kernel like
# the other diagnostics: it is re-sorted from scratch each time these integrals are written out. Here
# that is a sort of `N²` values per output, negligible next to a time step.

# We use two NetCDF writers. A *snapshot* writer stores the 2D fields on a plain `TimeInterval(1)`,
# while a *budget* writer stores only the integrated scalars on `ConsecutiveIterations(TimeInterval(1))`
# — a second sample one model step after each output time — which lets us finite-difference `∫Kˡ` across
# that single step to estimate `d/dt`, exactly as in the
# [Two-dimensional turbulence example](@ref two_d_turbulence_example).

using NCDatasets
filename = "kelvin_helmholtz"

simulation.output_writers[:nc] = NetCDFWriter(model, (; Ri, Q, b, w̄b̄, Πₖ, εˡ),
                                              filename=joinpath(@__DIR__, filename),
                                              schedule=TimeInterval(1),
                                              overwrite_existing=true)

simulation.output_writers[:budget] = NetCDFWriter(model, (; ∫Kˡ, ∫w̄b̄, ∫Πₖ, ∫εˡ, ∫K, ∫Eₚ, ∫E_b, ∫E_a),
                                                  filename=joinpath(@__DIR__, filename * "_budget"),
                                                  schedule=ConsecutiveIterations(TimeInterval(1)),
                                                  overwrite_existing=true)


# ## Run the simulation and process results
#
# To run the simulation:

run!(simulation)

# Now we'll read the snapshot fields using `FieldTimeSeries`

filepath = simulation.output_writers[:nc].filepath
Ri_t = FieldTimeSeries(filepath, "Ri")
Q_t  = FieldTimeSeries(filepath, "Q")
b_t  = FieldTimeSeries(filepath, "b")
w̄b̄_t = FieldTimeSeries(filepath, "w̄b̄")
Πₖ_t = FieldTimeSeries(filepath, "Πₖ")
εˡ_t = FieldTimeSeries(filepath, "εˡ")

ds = NCDataset(filepath)
times = ds["time"][:]
close(ds)

# The integrated budget scalars come in consecutive-iteration pairs `(2k-1, 2k)`; a one-step finite
# difference inside each pair gives `d(∫Kˡ)/dt`, and each source term is evaluated at the pair midpoint.

bud_filepath = simulation.output_writers[:budget].filepath
ds_bud = NCDataset(bud_filepath)
times_bud = ds_bud["time"][:]
∫Kˡ_t     = ds_bud["∫Kˡ"][:]
∫w̄b̄_t     = ds_bud["∫w̄b̄"][:]
∫Πₖ_t     = ds_bud["∫Πₖ"][:]
∫εˡ_t     = ds_bud["∫εˡ"][:]
∫K_t      = ds_bud["∫K"][:]
∫Eₚ_t     = ds_bud["∫Eₚ"][:]
∫E_b_t    = ds_bud["∫E_b"][:]
∫E_a_t    = ds_bud["∫E_a"][:]
close(ds_bud)

i1 = 1:2:length(times_bud)-1   # primary snapshots
i2 = 2:2:length(times_bud)       # consecutive-iteration snapshots
Δt_pair = times_bud[i2] .- times_bud[i1]
t_pair = @. 0.5 * (times_bud[i1] + times_bud[i2])

dKˡdt   = (∫Kˡ_t[i2] .- ∫Kˡ_t[i1]) ./ Δt_pair
w̄b̄_pair = @. 0.5 * (∫w̄b̄_t[i1] + ∫w̄b̄_t[i2])
Πₖ_pair = @. 0.5 * (∫Πₖ_t[i1] + ∫Πₖ_t[i2])
εˡ_pair = @. 0.5 * (∫εˡ_t[i1] + ∫εˡ_t[i2])

# Residual in sum-to-zero form: the negative tendency plus the three sources, so the plotted curves add to it
resid = @. -dKˡdt + w̄b̄_pair - Πₖ_pair - εˡ_pair

using Test                              #hide
rms(x) = √(sum(abs2, x) / length(x))    #hide
@test rms(resid) < 0.06 * rms(dKˡdt);   #hide

# For the potential energy we plot the three reservoirs against their initial values, since what the
# instability does to them is small next to the energies themselves. The available potential energy
# needs no offset: the initial stratification is monotonic in `z` and uniform in `x`, so it is already
# its own sorted state and starts with none.

ΔK   = ∫K_t[i1]   .- ∫K_t[1]
ΔE_b = ∫E_b_t[i1] .- ∫E_b_t[1]
E_a  = ∫E_a_t[i1]

## The split is exact cell by cell, so the only error in the integral is the Float32 output    #hide
@test maximum(abs, ∫Eₚ_t .- ∫E_b_t .- ∫E_a_t) < 1e-6 * maximum(abs, ∫Eₚ_t)                     #hide
@test abs(E_a[1]) < 1e-10 * maximum(abs, ∫Eₚ_t)   # a sorted initial state holds no APE        #hide
@test minimum(E_a) > -1e-10 * maximum(abs, ∫Eₚ_t) # the integrated APE is non-negative         #hide
@test 1 < argmax(E_a) < length(E_a)               # APE peaks partway through: stir, then mix  #hide
@test ΔE_b[end] > 0.3 * maximum(E_a)              # much of that APE ends up locked into E_b   #hide
## E_b only ever rises, as the continuous equations demand: the scheme is resolved enough here   #hide
## that its buoyancy overshoots never read to the sort as spurious unmixing                      #hide
@test minimum(diff(ΔE_b)) > -1e-6 * maximum(abs, ∫Eₚ_t);                                        #hide


# ## Plotting
#
# We now use Makie to create the figure and its axes

using CairoMakie

set_theme!(Theme(fontsize=24))
fig = Figure()

kwargs = (xlabel="x", ylabel="z", height=150, width=250)
ax1 = Axis(fig[2, 1]; title="Ri", kwargs...)
ax2 = Axis(fig[2, 2]; title="Q", kwargs...)
ax3 = Axis(fig[2, 3]; title="b", kwargs...);

# Next we use `Observable`s to lift the values and plot heatmaps and their colorbars

n = Observable(1)

Riₙ = @lift Ri_t[$n]
hm1 = heatmap!(ax1, Riₙ; colormap=:bwr, colorrange=(-1, +1))
Colorbar(fig[3, 1], hm1, vertical=false, height=8)

Qₙ  = @lift Q_t[$n]
hm2 = heatmap!(ax2, Qₙ; colormap=:inferno, colorrange=(0, 0.2))
Colorbar(fig[3, 2], hm2, vertical=false, height=8)

bₙ = @lift b_t[$n]
hm3 = heatmap!(ax3, bₙ; colormap=:balance, colorrange=(-B₀, +B₀))
Colorbar(fig[3, 3], hm3, vertical=false, height=8);

# The second row shows the (local) budget terms as 2D fields: the buoyancy production `w̄b̄`, the
# cross-scale kinetic-energy flux `Πₖ`, and the coarse-grained dissipation `εˡ`. Each gets a symmetric
# (or, for the sign-definite `εˡ`, one-sided) color range set from its own peak magnitude over the run.

maxabs(fts) = maximum(maximum(abs, interior(fts[k])) for k in 1:length(times))
wb_lim = maxabs(w̄b̄_t)
Π_lim  = maxabs(Πₖ_t)
ε_lim  = maxabs(εˡ_t)

ax4 = Axis(fig[4, 1]; title="w̄b̄", kwargs...)
ax5 = Axis(fig[4, 2]; title="Πₖ", kwargs...)
ax6 = Axis(fig[4, 3]; title="εˡ", kwargs...)

w̄b̄ₙ = @lift w̄b̄_t[$n]
hm4 = heatmap!(ax4, w̄b̄ₙ; colormap=:balance, colorrange=(-wb_lim, wb_lim))
Colorbar(fig[5, 1], hm4, vertical=false, height=8)

Πₖₙ = @lift Πₖ_t[$n]
hm5 = heatmap!(ax5, Πₖₙ; colormap=:balance, colorrange=(-Π_lim, Π_lim))
Colorbar(fig[5, 2], hm5, vertical=false, height=8)

εˡₙ = @lift εˡ_t[$n]
hm6 = heatmap!(ax6, εˡₙ; colormap=:magma, colorrange=(0, ε_lim))
Colorbar(fig[5, 3], hm6, vertical=false, height=8);

# The bottom panel shows the volume-integrated coarse-grained kinetic-energy budget. We plot the negative
# tendency `−d(∫Kˡ)/dt` together with its three sources: buoyancy production `∫w̄b̄ dV`, the cross-scale
# flux `−∫Πₖ dV`, and the coarse-grained dissipation `−∫εˡ dV`. With the tendency negated, the four curves
# sum to the residual.

ax_bud = Axis(fig[6, 1:3]; xlabel="Time", title="Coarse-grained KE budget", height=140)
lines!(ax_bud, t_pair, -dKˡdt, label="−d(∫Kˡ)/dt")
lines!(ax_bud, t_pair, w̄b̄_pair, label="∫w̄b̄ dV")
lines!(ax_bud, t_pair, -Πₖ_pair, label="−∫Πₖ dV")
lines!(ax_bud, t_pair, -εˡ_pair, label="−∫εˡ dV")
lines!(ax_bud, t_pair, resid, label="residual", color=:black, linestyle=:dash)
axislegend(ax_bud; position=:lb, labelsize=10)

# The last panel tracks the three energy reservoirs through the same event.

ax_pe = Axis(fig[7, 1:3]; xlabel="Time", title="Energy reservoirs (change from t = 0)", height=140)
lines!(ax_pe, times_bud[i1], ΔK, label="Δ∫K dV")
lines!(ax_pe, times_bud[i1], E_a, label="∫Eₐ dV")
lines!(ax_pe, times_bud[i1], ΔE_b, label="Δ∫E_b dV")
axislegend(ax_pe; position=:lt, labelsize=10)

# Now we mark the time by placing a vertical line in the bottom panels and adding a helpful title

tₙ = @lift times[$n]
vlines!(ax_bud, tₙ, color=:black, linestyle=:dash)
vlines!(ax_pe, tₙ, color=:black, linestyle=:dash)

title = @lift "Time = " * string(round(times[$n], digits=2))
fig[1, 1:3] = Label(fig, title, fontsize=24, tellwidth=false);

# Finally, we adjust the figure dimensions to fit all the panels and record a movie

resize_to_layout!(fig)

@info "Animating..."
record(fig, filename * ".mp4", 1:length(times), framerate=10) do i
    n[] = i
end

# ![](kelvin_helmholtz.mp4)
#
# The sixth panel shows the volume-integrated coarse-grained kinetic-energy budget. As the billows
# grow and overturn, the filtered flow mostly loses kinetic energy to potential energy (`∫w̄b̄ dV < 0`) and
# feeds the subfilter scales through the cross-scale flux (`−∫Πₖ dV`), while the coarse-grained viscous
# dissipation `∫εˡ dV` stays comparatively small at this Reynolds number. The residual (dashed), the
# sum of the negative tendency `−d(∫Kˡ)/dt` and the three source terms, stays small. As in the
# [Two-dimensional turbulence example](@ref two_d_turbulence_example), the centered scheme contributes no
# numerical dissipation of its own, so the budget closes against the explicit `∫εˡ dV` alone with a negligible residual.
#
# The last panel follows the same event through the three energy reservoirs, and shows where that
# potential energy goes. Kinetic energy drains away and available potential energy takes its place
# almost one for one while the billow rolls up, which is the reversible half of the exchange: the flow
# is lifting dense fluid and could in principle get all of it back. `∫Eₐ dV` peaks near `t ≈ 51`; past
# that the overturns break down, it decays, and the background potential energy climbs. That climb is
# what the split is for. It is the irreversible part, the record of buoyancy that diffusion has
# actually mixed across density surfaces, and by the end of the run it has taken up a little over half
# of the peak available potential energy. The renewed rise in `∫Eₐ dV` after `t ≈ 90` comes with kinetic
# energy still draining, so the shear is converting kinetic energy into available potential energy
# again rather than recovering any of what was mixed.
#
# Note that `∫E_b dV` never decreases. That is what the continuous equations demand, since only
# irreversible mixing can move the background state, but it is not automatic on a grid. A centered
# scheme carries no numerical dissipation and is correspondingly free to overshoot, and buoyancy
# extrema beyond the initial range `±B₀` read to the sort as fluid denser and lighter than anything the
# flow started with, which pushes `E_b` down. Here the overshoot stays mild, peaking around `1.5 B₀` at
# `t ≈ 69` while the billow breaks down, and never reverses the rise. That monotonicity is a useful
# check on resolution: the background potential energy is the standard measure of spurious diapycnal
# transport precisely because it is sensitive to this, and it registers such an artifact when the
# kinetic-energy budget above is entirely blind to it.
