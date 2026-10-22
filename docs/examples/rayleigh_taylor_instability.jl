# # [Rayleigh-Taylor instability](@id rayleigh_taylor_example)
#
# This example simulates a three-dimensional [Rayleigh-Taylor
# instability](https://en.wikipedia.org/wiki/Rayleigh%E2%80%93Taylor_instability): a heavy fluid
# initially resting above a light one, which is unstable under gravity. All of the kinetic energy is
# released from the potential energy stored in the unstable stratification, with no fluxes through the
# boundaries. We run it as a large-eddy simulation (LES) and use Oceanostics to close the volume-integrated
# subfilter-scale kinetic-energy budget, that is, the budget of the kinetic energy carried by the scales
# that a low-pass filter removes from the flow, and the budget of the available potential energy carried
# by the same scales.
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

closure = SmagorinskyLilly(C=0.3) # A large C increases eddy viscosity, keeping this very coarse example well-resolved

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

# ### Subfilter-scale kinetic-energy budget
#
# Rayleigh-Taylor turbulence converts potential energy into kinetic energy (KE) across a wide range of
# scales, so we follow it with a filtering analysis in the spirit of [Aluie et
# al. (2018)](https://doi.org/10.1175/JPO-D-17-0100.1). A low-pass filter (overbar) splits each field
# into a filtered part and a subfilter remainder. Here we budget the kinetic energy carried by the
# scales the filter removes,
#
# ```math
# e_k^s = \tfrac{1}{2}\,\tau(u_i, u_i) , \qquad
# \tau(u_i, u_j) = \overline{u_i u_j} - \overline{u}_i\,\overline{u}_j ,
# ```
#
# where ``\tau(u_i, u_j)`` is the subfilter stress ([`subfilter_stress_tensor`](@ref)), so ``e_k^s`` itself
# is computed by [`SubFilterKineticEnergy`](@ref). We apply a Gaussian
# filter of width `ℓ` in the two horizontal directions, which are statistically
# homogeneous; the vertical direction is left unfiltered.
#
# Volume integrated (the transport terms integrate to zero, because the flow is doubly periodic and
# `w = 0` with free slip at the top and bottom), the budget reads
#
# ```math
# \frac{d}{dt} \int e_k^s\, \mathrm{d}V
#   = \int \Pi_k\, \mathrm{d}V
#   + \int \tau(w, b)\, \mathrm{d}V
#   - \int \varepsilon_k^s\, \mathrm{d}V ,
# ```
#
# with two sources and one sink:
#
# - ``\Pi_k = -\tau(u_i, u_j)\overline{S}_{ij}`` is the cross-scale kinetic-energy flux
#   ([`KineticEnergyCrossScaleFlux`](@ref)), the rate at which the filtered scales hand kinetic energy
#   down to the subfilter scales.
# - ``\tau(w, b) = \overline{wb} - \overline{w}\,\overline{b}`` is the subfilter buoyancy flux (a
#   `subfilter_covariance`), which converts subfilter potential energy into subfilter kinetic energy.
#   The filter here is purely horizontal, so this equals the ``\tau^l(w, b_r)`` of the
#   [subfilter kinetic energy budget](@ref "Subfilter kinetic energy equation"): the reference
#   profile ``b^\star(z)`` is constant along the filtered directions and drops out of the covariance.
# - ``\varepsilon_k^s = \overline{\varepsilon_k} - \varepsilon_k^l`` is the subfilter dissipation
#   ([`SubFilterKineticEnergyDissipationRate`](@ref)): the filtered total dissipation ``\varepsilon_k``
#   ([`KineticEnergyDissipationRate`](@ref Oceanostics.KineticEnergyEquation.DissipationRate)) minus the
#   dissipation ``\varepsilon_k^l`` of the filtered flow
#   ([`FilteredKineticEnergyDissipationRate`](@ref)). For a constant viscosity it reduces to
#   ``2\nu[\overline{S_{ij}S_{ij}} - \overline{S}_{ij}\overline{S}_{ij}] \ge 0``, a strictly positive
#   sink; with an LES closure it is the dissipation that the modeled stress carries out on the subfilter scales.
#
# In the code below, ``e_k^s``, ``\Pi_k``, ``\tau(w,b)`` and ``\varepsilon_k^s`` are written `eₖˢ`,
# `Πₖ`, `wbˢ` and `εₖˢ`.

using Oceananigans.AbstractOperations: @at

ℓ  = 8 * Δx                          # filter scale (full width at half maximum of the Gaussian kernel)
σℓ = ℓ / (2 * sqrt(2 * log(2)))      # corresponding Gaussian standard deviation
gfilter = GaussianFilter(dims=(1, 2), σ=σℓ)

u, v, w = model.velocities
b = model.tracers.b

eₖˢ = SubFilterKineticEnergy(model, gfilter)       # subfilter kinetic energy ½τᵢᵢ
Πₖ  = KineticEnergyCrossScaleFlux(model, gfilter)  # cross-scale flux from filtered scales
wbˢ = subfilter_covariance(w, b, gfilter)          # subfilter buoyancy flux τ(w, b)
εₖˢ = SubFilterKineticEnergyDissipationRate(model, gfilter)  # subfilter dissipation ε̄ₖ − εₖˡ

# The budget needs only the (cheap) volume integrals of these terms:

∫eₖˢ = Integral(eₖˢ)
∫Πₖ  = Integral(Πₖ)
∫wbˢ = Integral(wbˢ)
∫εₖˢ = Integral(εₖˢ)

∂ₜ∫eₖˢ = TimeDerivative(∫eₖˢ)

# ### Subfilter available potential energy budget
#
# The same filter splits the local available potential energy (APE) ``e_a`` of the
# [Available potential energy equation](@ref), measured against the sorted reference profile
# ``b^\star(z, t)`` of the full buoyancy, into the APE of the filtered buoyancy, ``e_a^l``, and the
# subfilter remainder ``e_a^s = \overline{e_a} - e_a^l`` ([`SubFilterAvailablePotentialEnergy`](@ref)),
# which a horizontal filter keeps non-negative. Volume integrated, its transport terms vanish as well,
# since the top and bottom are also insulating, and the budget reads
#
# ```math
# \frac{d}{dt} \int e_a^s\, \mathrm{d}V
#   = - \int \tau(w, b_r)\, \mathrm{d}V
#   + \int \Pi_a\, \mathrm{d}V
#   - \int \varepsilon_a^s\, \mathrm{d}V
#   + \int R^s\, \mathrm{d}V ,
# ```
#
# where
#
# - ``\tau(w, b_r) = \overline{w b_r} - \overline{w}\, \overline{b_r}`` converts subfilter APE into subfilter
#   kinetic energy ([`SubFilterAvailablePotentialToKineticEnergyConversion`](@ref)); it is the
#   ``\tau(w, b)`` that feeds ``e_k^s`` above.
# - ``\Pi_a`` is the cross-scale APE flux ([`AvailablePotentialEnergyCrossScaleFlux`](@ref)), the rate at
#   which the filtered scales hand APE down to the subfilter scales.
# - ``\varepsilon_a^s = \overline{\varepsilon_a} - \varepsilon_a^l`` is the subfilter APE dissipation
#   ([`SubFilterAvailablePotentialEnergyDissipationRate`](@ref)), the APE that the closure's buoyancy
#   diffusion removes from the subfilter scales.
# - ``R^s = \overline{R} - R^l`` is the subfilter reference tendency
#   ([`SubFilterAvailablePotentialEnergyReferenceTendency`](@ref)). Mixing reshapes the sorted profile,
#   which changes the APE of every parcel at a rate ``R``. That rate integrates to zero over the domain
#   while its filtered counterpart ``R^l`` does not, so ``R^s`` moves APE between the filtered and the
#   subfilter scales.
#
# In the code below these are written `eₐˢ`, `wbᵣˢ`, `Πₐ`, `εₐˢ` and `Rˢ`. All of them are measured
# against one sorted column of the buoyancy, which `ProfileLookup` shares between them, and ``R^s``
# integrates the time derivative of its profile. Reading a `TimeDerivative` does not update it, so a
# callback does, on the iteration before each output and at the output, the same step over which the
# budget writer differences ``\int e_a^s\, \mathrm{d}V``:

z✶_column = reference_height(model, method=VerticalSort())
lookup    = ProfileLookup(z✶_column)

∂ₜb✶ = TimeDerivative(reference_buoyancy(z✶_column))
simulation.callbacks[:∂ₜb✶] = Callback(∂ₜb✶, PrecedingIterations(TimeInterval(τ / 5)))

eₐˢ  = SubFilterAvailablePotentialEnergy(model, gfilter; method=lookup)
wbᵣˢ = SubFilterAvailablePotentialToKineticEnergyConversion(model, gfilter; method=lookup)
Πₐ   = AvailablePotentialEnergyCrossScaleFlux(model, gfilter; method=lookup)
εₐˢ  = SubFilterAvailablePotentialEnergyDissipationRate(model, gfilter; method=lookup)
Rˢ   = SubFilterAvailablePotentialEnergyReferenceTendency(model, gfilter, ∂ₜb✶; method=lookup)

∫wbᵣˢ = Integral(wbᵣˢ)
∫Πₐ   = Integral(Πₐ)
∫εₐˢ  = Integral(εₐˢ)
∫Rˢ   = Integral(Rˢ)

∂ₜ∫eₐˢ = TimeDerivative(Integral(eₐˢ))

# ## Output
#
# We use two NetCDF writers. A snapshot writer stores vertical (`x`–`z`) slices of the buoyancy `b`,
# the cross-scale flux `Πₖ` and the two subfilter energies `eₐˢ` and `eₖˢ`, at a fixed `y` index (the
# flow is periodic and statistically homogeneous in `y`, so the particular plane makes no difference),
# while a budget writer stores only the integrated scalars. Both run on `TimeInterval(τ/5)`.

using NCDatasets
filename = joinpath(@__DIR__, "rayleigh_taylor_instability")

simulation.output_writers[:fields] = NetCDFWriter(model, (; b, Πₖ, eₐˢ, eₖˢ),
                                                  filename = filename,
                                                  schedule = TimeInterval(τ / 5),
                                                  indices = (:, 1, :),
                                                  overwrite_files = true)

simulation.output_writers[:budget] = NetCDFWriter(model, (; ∂ₜ∫eₖˢ, ∫Πₖ, ∫wbˢ, ∫εₖˢ, ∂ₜ∫eₐˢ, ∫wbᵣˢ, ∫Πₐ, ∫εₐˢ, ∫Rˢ),
                                                  filename = filename * "_budget",
                                                  schedule = TimeInterval(τ / 5),
                                                  overwrite_files = true)

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
eₐˢ_arr = ds["eₐˢ"][:, 1, :, :]
eₖˢ_arr = ds["eₖˢ"][:, 1, :, :]
close(ds)

# Every budget record carries both tendencies and their budget terms at the same time. The first
# record has no earlier state to difference against and is written as zero, so the budgets start from
# the second. The residuals measure how well the two subfilter budgets close.

bud_filepath = simulation.output_writers[:budget].filepath
ds_bud = NCDataset(bud_filepath)
nb     = 2:length(ds_bud["time"])

t_bud    = ds_bud["time"][nb]
deₖˢdt   = ds_bud["∂ₜ∫eₖˢ"][nb]
Πₖ_bud   = ds_bud["∫Πₖ"][nb]
wbˢ_bud  = ds_bud["∫wbˢ"][nb]
εₖˢ_bud  = ds_bud["∫εₖˢ"][nb]
deₐˢdt   = ds_bud["∂ₜ∫eₐˢ"][nb]
wbᵣˢ_bud = ds_bud["∫wbᵣˢ"][nb]
Πₐ_bud   = ds_bud["∫Πₐ"][nb]
εₐˢ_bud  = ds_bud["∫εₐˢ"][nb]
Rˢ_bud   = ds_bud["∫Rˢ"][nb]
close(ds_bud)

# Residuals in sum-to-zero form: the negative tendency plus the right-hand side, so the plotted curves
# add to it

eₖˢ_resid = @. -deₖˢdt + Πₖ_bud + wbˢ_bud - εₖˢ_bud
eₐˢ_resid = @. -deₐˢdt - wbᵣˢ_bud + Πₐ_bud - εₐˢ_bud + Rˢ_bud

using Test                                                                    #hide
rms(x) = √(sum(abs2, x) / length(x))                                          #hide
budget_scale = rms(@. abs(Πₖ_bud) + abs(wbˢ_bud) + abs(εₖˢ_bud))              #hide
@test rms(eₖˢ_resid) < 0.02 * budget_scale                                    #hide
ape_scale = rms(@. abs(wbᵣˢ_bud) + abs(Πₐ_bud) + abs(εₐˢ_bud) + abs(Rˢ_bud)) #hide
@test rms(eₐˢ_resid) < 0.01 * ape_scale                                       #hide
## ∫Rˢ dV is many times the residual, so the budget does not close without it #hide
@test rms(eₐˢ_resid) < 0.2 * rms(Rˢ_bud);                                     #hide

# ## Plotting
#
# We build the figure in four rows: the vertical slices of the buoyancy `b` (the spikes and bubbles) and
# of the cross-scale flux `Πₖ` on top; the two subfilter energies `eₐˢ` and `eₖˢ` in the middle; and the
# two volume-integrated subfilter budgets at the bottom.

set_theme!(Theme(fontsize=18))
fig = Figure(size=(1000, 1500))

n = Observable(1)

axb = Axis(fig[2, 1]; title="buoyancy, b", xlabel="x", ylabel="z", aspect=1)
axΠ = Axis(fig[2, 3]; title="cross-scale kinetic energy flux, Πₖ", xlabel="x", ylabel="z", aspect=1)

blim = 0.8 * maximum(abs, b_arr[:, :, 1])
Πlim = 0.5 * maximum(abs, Π_arr)

bₙ = @lift b_arr[:, :, $n]
Πₙ = @lift Π_arr[:, :, $n]

hmb = heatmap!(axb, x_caa, z_aac, bₙ; colormap=:balance, colorrange=(-blim, blim))
Colorbar(fig[2, 2], hmb)

hmΠ = heatmap!(axΠ, x_caa, z_aac, Πₙ; colormap=:balance, colorrange=(-Πlim, Πlim))
Colorbar(fig[2, 4], hmΠ)

# The middle row shows the two subfilter energies whose budgets the bottom panels close: the available
# potential energy `eₐˢ` on the left and the kinetic energy `eₖˢ` on the right. Both are non-negative,
# so they get a sequential colormap, each with its own colour scale.

axeₐˢ = Axis(fig[3, 1]; title="subfilter available potential energy, eₐˢ", xlabel="x", ylabel="z", aspect=1)
axeₖˢ = Axis(fig[3, 3]; title="subfilter kinetic energy, eₖˢ", xlabel="x", ylabel="z", aspect=1)

eₐˢlim = maximum(eₐˢ_arr)
eₖˢlim = maximum(eₖˢ_arr)

eₐˢₙ = @lift eₐˢ_arr[:, :, $n]
eₖˢₙ = @lift eₖˢ_arr[:, :, $n]

hmeₐˢ = heatmap!(axeₐˢ, x_caa, z_aac, eₐˢₙ; colormap=:magma, colorrange=(0, eₐˢlim))
Colorbar(fig[3, 2], hmeₐˢ)

hmeₖˢ = heatmap!(axeₖˢ, x_caa, z_aac, eₖˢₙ; colormap=:magma, colorrange=(0, eₖˢlim))
Colorbar(fig[3, 4], hmeₖˢ)

# The fourth row shows the volume-integrated subfilter-scale kinetic-energy budget. We plot the
# negative tendency `−d(∫eₖˢ)/dt` together with the two sources that feed it, the cross-scale flux
# `∫Πₖ dV` handed down from the filtered scales and the subfilter buoyancy flux `∫τ(w,b) dV`, and the
# single sink that drains it, the subfilter dissipation `−∫εₖˢ dV`. With the tendency negated, the four
# curves sum to the residual.

ax_bud = Axis(fig[4, 1:4]; xlabel="time [free-fall units]", title="Subfilter kinetic energy budget")
lines!(ax_bud, t_bud ./ τ, -deₖˢdt,   label="−d(∫eₖˢ)/dt")
lines!(ax_bud, t_bud ./ τ, Πₖ_bud,    label="∫Πₖ dV  (flux from filtered scales)")
lines!(ax_bud, t_bud ./ τ, wbˢ_bud,   label="∫τ(w,b) dV  (subfilter buoyancy flux)")
lines!(ax_bud, t_bud ./ τ, -εₖˢ_bud,  label="−∫εₖˢ dV  (subfilter dissipation)")
lines!(ax_bud, t_bud ./ τ, eₖˢ_resid, label="residual", color=:black, linestyle=:dash)
axislegend(ax_bud; position=:lt, labelsize=10)

# The bottom row shows the subfilter APE budget in the same form: the negative tendency
# `−d(∫eₐˢ)/dt`, the conversion `−∫τˡ(w,bᵣ) dV` into subfilter kinetic energy, the cross-scale flux
# `∫Πₐ dV`, the subfilter dissipation `−∫εₐˢ dV` and the reference tendency `∫Rˢ dV`.

ax_ape = Axis(fig[5, 1:4]; xlabel="time [free-fall units]", title="Subfilter available potential energy budget")
lines!(ax_ape, t_bud ./ τ, -deₐˢdt,   label="−d(∫eₐˢ)/dt")
lines!(ax_ape, t_bud ./ τ, -wbᵣˢ_bud, label="−∫τˡ(w,bᵣ) dV  (conversion to subfilter KE)")
lines!(ax_ape, t_bud ./ τ, Πₐ_bud,    label="∫Πₐ dV  (flux from filtered scales)")
lines!(ax_ape, t_bud ./ τ, -εₐˢ_bud,  label="−∫εₐˢ dV  (subfilter dissipation)")
lines!(ax_ape, t_bud ./ τ, Rˢ_bud,    label="∫Rˢ dV  (reference tendency)")
lines!(ax_ape, t_bud ./ τ, eₐˢ_resid, label="residual", color=:black, linestyle=:dash)
axislegend(ax_ape; position=:lt, labelsize=10)

for ax in (ax_bud, ax_ape)
    vlines!(ax, @lift(times[$n] / τ), color=:black, linestyle=:dash)
end

title = @lift "Rayleigh-Taylor instability, t = " * string(round(times[$n] / τ, digits=2)) * " τ"
fig[1, 1:4] = Label(fig, title, fontsize=22, tellwidth=false)

@info "Animating..."
record(fig, "rayleigh_taylor_instability.mp4", 1:length(times), framerate=12) do i
    n[] = i
end

# ![](rayleigh_taylor_instability.mp4)
#
# As the heavy fluid falls in spikes and the light fluid rises in bubbles, the flow rolls up and
# breaks into a turbulent mixing layer. In the subfilter KE budget, the subfilter buoyancy flux
# `∫τ(w,b) dV` and dissipation `∫εₖˢ dV` are the dominant terms, with the tendency `−d(∫eₖˢ)/dt` in
# third place. In the subfilter APE budget, the cross-scale flux `∫Πₐ dV` is the main source. About
# half of what it supplies is converted into subfilter kinetic energy, through the mirror image of the
# `∫τ(w,b) dV` curve above, and most of the rest is dissipated. The reference tendency `∫Rˢ dV` is the
# smallest of the four terms on the right-hand side, but still many times the residual. Both residuals
# are small, which shows that the budgets close well and that the filtering analysis is consistent with
# the simulation.
