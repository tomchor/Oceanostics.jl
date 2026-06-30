# # [Spatial filtering and subfilter fluxes](@id spatial_filtering_example)
#
# In this example we use Oceanostics' [`GaussianFilter`](@ref) to coarse-grain a two-dimensional
# turbulent flow. Spatial filtering splits a field into a smooth, large-scale (filtered) part and a
# small-scale (subfilter) fluctuation,
#
# ```math
# \psi = \bar{\psi} + \psi', \qquad \psi' \equiv \psi - \bar{\psi},
# ```
#
# where the overbar denotes a Gaussian-weighted local average.
# Beyond visualization, filtering lets us build *subfilter* (sub-filter-scale) diagnostics such as the
# subfilter tracer flux ``\tau_i = \overline{u_i c} - \bar{u}_i \bar{c}``, which represents the
# transport carried by scales smaller than the filter width. We close the example with the *cross-scale
# kinetic energy flux*, which measures the energy exchanged between the filtered and subfilter scales.
#
# Before starting, make sure you have the required packages installed for this example, which can
# be done with
#
# ```julia
# using Pkg
# pkg"add Oceananigans, Oceanostics, CairoMakie"
# ```

# ## Model and simulation setup
#
# To get a turbulent field to filter, we follow the
# [Two-dimensional turbulence example](@ref two_d_turbulence_example): we run the same
# doubly-periodic 2D turbulence simulation, initialized with a smooth, well-resolved sum of Gaussian
# velocity blobs and a sine/cosine tracer that the flow stirs into thin filaments. See that example
# for a detailed walk-through of the setup; here we only summarize it. (The
# grid here is uniform, but [`GaussianFilter`](@ref) also supports stretched directions, choosing a
# fast precomputed-weight path on uniform directions and a node-distance path on variably spaced
# ones.)

using Oceananigans

grid = RectilinearGrid(size=(256, 256), extent=(2π, 2π), topology=(Periodic, Periodic, Flat))

model = NonhydrostaticModel(grid; timestepper = :RungeKutta3,
                            advection = Centered(order=4),
                            tracers = :c,
                            closure = ScalarDiffusivity(ν=1e-4, κ=1e-3))

# Initial conditions:

using Random, Statistics

u, v, w = model.velocities
c = model.tracers.c

Random.seed!(772)
N_blobs = 24
σ_blob  = 8 * minimum_xspacing(grid)
xc      = grid.Lx * rand(N_blobs)
yc      = grid.Ly * rand(N_blobs)
amp_u   = randn(N_blobs)
amp_v   = randn(N_blobs)

## Sum of blobs and their periodic images so the field is smooth across the periodic boundary
blob_sum(x, y, amp) = sum(amp[k] * exp(-((x - xc[k] - dx)^2 + (y - yc[k] - dy)^2) / σ_blob^2)
                          for k  in 1:N_blobs,
                              dx in (-grid.Lx, 0, grid.Lx),
                              dy in (-grid.Ly, 0, grid.Ly))

uᵢ(x, y) = blob_sum(x, y, amp_u)
vᵢ(x, y) = blob_sum(x, y, amp_v)
cᵢ(x, y) = sin(3x) * cos(2y)

set!(model, u=uᵢ, v=vᵢ, c=cᵢ)
u .-= mean(u)
v .-= mean(v)

# We run a short simulation with an adaptive time step to let the flow develop a range of scales:

u_max = max(maximum(abs, u), maximum(abs, v))
simulation = Simulation(model; Δt = 0.2 * minimum_xspacing(grid) / u_max, stop_time = 30)

wizard = TimeStepWizard(cfl=0.8, diffusive_cfl=0.8)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(5))

run!(simulation)

# ## Scale separation
#
# We now filter the snapshot at the end of the run. We pick a Gaussian filter with a kernel standard deviation
# (serving as the filter width) of ``\sigma = 8\Delta`` (a few grid cells wide):

using Oceanostics

Δ = minimum_xspacing(grid)
σ = 8Δ

filter = GaussianFilter(; dims=(1, 2), σ=σ)

ω  = ∂x(v) - ∂y(u) # vorticity
ω̄  = filter(ω)     # filtered (large-scale) vorticity
ω′ = Field(ω - ω̄)  # subfilter fluctuation

# We plot the vorticity, filtered vorticity, and subfilter fluctuations side by side:

using CairoMakie
set_theme!(Theme(fontsize = 18))

axis_kwargs = (aspect = DataAspect(),
               height = 200, width = 200,
               xticksvisible = false, yticksvisible = false,
               xticklabelsvisible = false, yticklabelsvisible = false)

fig_ω = Figure()
ax_ω  = Axis(fig_ω[1, 1]; title = "Vorticity ω", axis_kwargs...)
ax_ω̄  = Axis(fig_ω[1, 2]; title = "Filtered ω̄",  axis_kwargs...)
ax_ω′ = Axis(fig_ω[1, 3]; title = "ω′ = ω − ω̄",  axis_kwargs...)

ω_range = (-10, 10)
heatmap!(ax_ω,  ω;  colormap = :balance, colorrange = ω_range)
heatmap!(ax_ω̄,  ω̄;  colormap = :balance, colorrange = ω_range)
hm_ω = heatmap!(ax_ω′, ω′; colormap = :balance, colorrange = ω_range)
Colorbar(fig_ω[1, 4], hm_ω)

resize_to_layout!(fig_ω)
fig_ω

# The Gaussian filter splits the vorticity into a smooth filtered field ``\bar{\omega}`` and the
# fine-scale residual ``\omega'`` it removes.

# ## Filter-width sweep
#
# The amount of structure that survives depends on ``\sigma``: the wider the kernel, the more scales
# are averaged away. We filter the vorticity at three increasing widths — ``2\Delta``, ``4\Delta``
# and ``8\Delta`` — and plot each result as it is computed.

σ_sweep = (4Δ, 8Δ, 16Δ)
ω̄_sweep = [GaussianFilter(ω; dims=(1, 2), σ=s) for s in σ_sweep]   # one filter per width, applied to ω

fig_sweep = Figure()
for (i, s) in enumerate(σ_sweep)
    ax = Axis(fig_sweep[1, i]; title = "ω̄, σ = $(round(Int, s / Δ))Δ", axis_kwargs...)
    hm = heatmap!(ax, ω̄_sweep[i]; colormap = :balance, colorrange = ω_range)
    i == length(σ_sweep) && Colorbar(fig_sweep[1, i + 1], hm)
end

resize_to_layout!(fig_sweep)
fig_sweep

# Wider kernels (``\sigma = 2\Delta \to 8\Delta``) progressively erase smaller scales.

# ## Subfilter tracer flux
#
# Filtering also lets us quantify transport by subfilter scales. The subfilter tracer flux is
# ``\tau_i = \overline{u_i c} - \bar{u}_i \bar{c}``: the difference between the filtered advective
# flux and the flux carried by the filtered fields. We reuse the same `filter` object on each piece —
# the two velocities, the tracer, and the two advective products ``u_i c`` — before combining: one
# filter object, applied to five different fields.

using Oceananigans.AbstractOperations: @at

ū  = filter(u)
v̄  = filter(v)
c̄  = filter(c)

τx = filter(u * c) - ū * c̄   # overline(u c) - ū c̄
τy = filter(v * c) - v̄ * c̄
τ  = √(τx^2 + τy^2)          # subfilter flux magnitude

# Finally we plot the tracer, its filtered version, and the magnitude of the subfilter flux:

fig_τ = Figure()
ax_c  = Axis(fig_τ[1, 1]; title = "Tracer c",      axis_kwargs...)
ax_c̄  = Axis(fig_τ[1, 2]; title = "Filtered c̄",    axis_kwargs...)
ax_τ  = Axis(fig_τ[1, 3]; title = "Subfilter |τ|", axis_kwargs...)

heatmap!(ax_c, c;  colormap = :balance, colorrange = (-1, 1))
heatmap!(ax_c̄, c̄;  colormap = :balance, colorrange = (-1, 1))
hm_τ = heatmap!(ax_τ, τ; colormap = :magma)
Colorbar(fig_τ[1, 4], hm_τ)

resize_to_layout!(fig_τ)
fig_τ

# The magnitude of the subfilter flux ``\tau`` is largest precisely along the thin tracer filaments —
# the small-scale structure that the filtered fields ``\bar{u}_i\,\bar{c}`` cannot represent on their
# own.

# ## Cross-scale kinetic energy flux
#
# We can also calculate the kinetic energy flux across the filter scale:
#
# ```math
# \Pi_K = -\tau_{ij}\,\bar{S}_{ij}, \qquad
# \tau_{ij} = \overline{u_i u_j} - \bar{u}_i \bar{u}_j, \qquad
# \bar{S}_{ij} = \tfrac{1}{2}\left(\partial_j \bar{u}_i + \partial_i \bar{u}_j\right),
# ```
#
# where ``\tau_{ij}`` is the subfilter (momentum) stress tensor — the tensor analogue of the subfilter
# tracer flux above — and ``\bar{S}_{ij}`` is the strain rate of the filtered velocity. ``\Pi_K > 0``
# is a forward (downscale) transfer from filtered to subfilter scales, while ``\Pi_K < 0`` is
# backscatter from small to large scales, which is prominent in two-dimensional turbulence and its
# inverse energy cascade (see [Aluie et al. (2018)](https://doi.org/10.1175/JPO-D-17-0100.1)).
#
# Rather than assembling ``\tau_{ij}`` and ``\bar{S}_{ij}`` by hand as we did for the tracer flux,
# Oceanostics packages a [`KineticEnergyCrossScaleFlux`](@ref) which accepts the same `filter` we built
# above and returns ``\Pi_K``:

Πₖ = KineticEnergyCrossScaleFlux(model, filter; dims=(1, 2))

# We show it next to the filtered kinetic energy ``\tfrac{1}{2}(\bar{u}^2 + \bar{v}^2)``, reusing the
# filtered velocities ``\bar{u}``, ``\bar{v}`` from the previous section:

K̄ = (ū^2 + v̄^2) / 2

fig_Π = Figure()
ax_K = Axis(fig_Π[1, 1]; title = "Filtered KE ½(ū² + v̄²)", axis_kwargs...)
ax_Π = Axis(fig_Π[1, 3]; title = "Cross-scale flux Πₖ",    axis_kwargs...)

hm_K = heatmap!(ax_K, K̄; colormap = :magma)
Colorbar(fig_Π[1, 2], hm_K)

Πlim = 0.95 * maximum(abs, interior(Πₖ))
hm_Π = heatmap!(ax_Π, Πₖ; colormap = :balance, colorrange = (-Πlim, Πlim))
Colorbar(fig_Π[1, 4], hm_Π)

resize_to_layout!(fig_Π)
fig_Π

# ## Coarse-grained kinetic energy dissipation
#
# The cross-scale flux ``\Pi_K`` shuffles kinetic energy *between* the filtered and subfilter scales, but
# it is not a true sink. Viscosity acting on the *filtered* flow is, and Oceanostics exposes that
# resolved-scale sink as [`CoarseGrainedKineticEnergyDissipationRate`](@ref):
#
# ```math
# \overline{\varepsilon} = \frac{\partial \overline{u}_i}{\partial x_j}\,\overline{F}_{ij},
# ```
#
# the [`KineticEnergyDissipationRate`](@ref Oceanostics.KineticEnergyEquation.DissipationRate) ``\partial_j u_i\,F_{ij}`` evaluated on the filtered
# velocities (``\nu`` from the model's closure). It reuses the same `filter`, lives at cell centers, and
# for this constant-viscosity run equals ``2\nu\,\overline{S}_{ij}\overline{S}_{ij}``:

ε̄ = CoarseGrainedKineticEnergyDissipationRate(model, filter)

fig_ε = Figure()
ax_ε = Axis(fig_ε[1, 1]; title = "Coarse-grained KE dissipation ε̄", axis_kwargs...)
hm_ε = heatmap!(ax_ε, ε̄; colormap = :magma)
Colorbar(fig_ε[1, 2], hm_ε)

resize_to_layout!(fig_ε)
fig_ε

# Red (``\Pi_K > 0``) marks forward transfer to subfilter scales and blue (``\Pi_K < 0``) marks
# backscatter to larger scales. Both concentrate in the strained regions between vortices, where the
# filtered strain ``\bar{S}_{ij}`` — and hence the scale-to-scale energy exchange — is largest.
