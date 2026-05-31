# # [Spatial filtering and subfilter fluxes](@id spatial_filtering_example)
#
# In this example we use Oceanostics' [`GaussianFilter`](@ref) to coarse-grain a two-dimensional
# turbulent flow. Spatial filtering splits a field into a smooth, large-scale (resolved) part and a
# small-scale (subfilter) fluctuation,
#
# ```math
# \psi = \bar{\psi} + \psi', \qquad \psi' \equiv \psi - \bar{\psi},
# ```
#
# where the overbar denotes a Gaussian-weighted local average with standard deviation ``\sigma``.
# Beyond visualization, filtering lets us build *subfilter* (subgrid-scale) diagnostics such as the
# subfilter tracer flux ``\tau_i = \overline{u_i c} - \bar{u}_i \bar{c}``, which represents the
# transport carried by scales smaller than ``\sigma``.
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
# We generate a turbulent field with a quick doubly-periodic 2D turbulence simulation — the same
# setup as the [Two-dimensional turbulence example](@ref two_d_turbulence_example), but on a
# smaller grid and run for a shorter time, since here we only need a developed multi-scale field to
# filter. A `GaussianFilter` requires **uniform spacing** along each filtered direction, which a
# doubly-periodic `RectilinearGrid` satisfies in both ``x`` and ``y``.

using Oceananigans

grid = RectilinearGrid(size=(128, 128), extent=(2π, 2π), topology=(Periodic, Periodic, Flat))

model = NonhydrostaticModel(grid; timestepper = :RungeKutta3,
                            advection = Centered(order=4),
                            tracers = :c,
                            closure = ScalarDiffusivity(ν=1e-4, κ=1e-3))

# We initialize the velocities as a smooth, well-resolved sum of Gaussian blobs (so the flow is
# resolved rather than grid-scale noise) and the tracer with a smooth sine/cosine pattern that
# turbulence will stir into thin filaments.

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
# We now filter the snapshot at the end of the run. We pick a filter width ``\sigma`` of a few grid
# cells and split the vorticity into resolved and subfilter parts. Vorticity lives at
# ``(f, f, c)``, so we interpolate it to cell centers before filtering (and for plotting).

using Oceananigans.AbstractOperations: @at
using Oceanostics

Δ   = minimum_xspacing(grid)
σ_f = 6Δ

ω  = Field(@at (Center, Center, Center) (∂x(v) - ∂y(u)))   # vorticity at (Center, Center, Center)
ω̄  = Field(GaussianFilter(ω; dims=(1, 2), σ=σ_f))          # resolved (large-scale) vorticity
ω′ = Field(ω - ω̄)                                           # subfilter fluctuation

# A normalized Gaussian filter removes small-scale variance while (on a periodic domain) preserving
# the field mean, so the filtered field is necessarily smoother than the original:

using Test                                              #hide
@test var(interior(ω̄)) < var(interior(ω))              #hide

# ## Filter-width sweep
#
# The amount of structure that survives depends on ``\sigma``: the wider the kernel, the more
# scales are averaged away. We filter the vorticity at three increasing widths to make this
# explicit.

σ_sweep = (2Δ, 4Δ, 8Δ)
ω̄_sweep = [Field(GaussianFilter(ω; dims=(1, 2), σ=s)) for s in σ_sweep]

@test var(interior(ω̄_sweep[1])) > var(interior(ω̄_sweep[2])) > var(interior(ω̄_sweep[3]))   #hide

# ## Subfilter tracer flux
#
# Filtering also lets us quantify transport by unresolved scales. The subfilter tracer flux is
# ``\tau_i = \overline{u_i c} - \bar{u}_i \bar{c}``: the difference between the filtered advective
# flux and the flux carried by the filtered fields. We interpolate the velocities to centers, build
# the products as `Field`s, filter each piece, and combine.

uᶜ = Field(@at (Center, Center, Center) u)
vᶜ = Field(@at (Center, Center, Center) v)

ū  = Field(GaussianFilter(uᶜ; dims=(1, 2), σ=σ_f))
v̄  = Field(GaussianFilter(vᶜ; dims=(1, 2), σ=σ_f))
c̄  = Field(GaussianFilter(c;  dims=(1, 2), σ=σ_f))

ūc̄ = Field(GaussianFilter(Field(uᶜ * c); dims=(1, 2), σ=σ_f))   # = overline(u c)
v̄c̄ = Field(GaussianFilter(Field(vᶜ * c); dims=(1, 2), σ=σ_f))   # = overline(v c)

τx = Field(ūc̄ - ū * c̄)
τy = Field(v̄c̄ - v̄ * c̄)
τ  = Field(sqrt(τx^2 + τy^2))   # subfilter flux magnitude

# ## Plotting
#
# We assemble a static figure. The top row shows the scale separation of vorticity, the middle row
# the filter-width sweep, and the bottom row the tracer together with its filtered version and the
# subfilter flux magnitude.

using CairoMakie

set_theme!(Theme(fontsize = 18))
fig = Figure()

axis_kwargs = (aspect = DataAspect(),
               height = 200, width = 200,
               xticksvisible = false, yticksvisible = false,
               xticklabelsvisible = false, yticklabelsvisible = false)

## Row 1: scale separation of vorticity
ax_ω  = Axis(fig[1, 1]; title = "Vorticity ω",        axis_kwargs...)
ax_ω̄  = Axis(fig[1, 2]; title = "Filtered ω̄",          axis_kwargs...)
ax_ω′ = Axis(fig[1, 3]; title = "Residual ω′ = ω − ω̄", axis_kwargs...)

ω_range = (-10, 10)
heatmap!(ax_ω,  ω;  colormap = :balance, colorrange = ω_range)
heatmap!(ax_ω̄,  ω̄;  colormap = :balance, colorrange = ω_range)
hm_ω = heatmap!(ax_ω′, ω′; colormap = :balance, colorrange = ω_range)
Colorbar(fig[1, 4], hm_ω)

## Row 2: filter-width sweep of the filtered vorticity
for (i, s) in enumerate(σ_sweep)
    ax = Axis(fig[2, i]; title = "ω̄, σ = $(round(s / Δ, digits=1))Δ", axis_kwargs...)
    heatmap!(ax, ω̄_sweep[i]; colormap = :balance, colorrange = ω_range)
end

## Row 3: tracer, filtered tracer, and subfilter flux magnitude
ax_c  = Axis(fig[3, 1]; title = "Tracer c",      axis_kwargs...)
ax_c̄  = Axis(fig[3, 2]; title = "Filtered c̄",     axis_kwargs...)
ax_τ  = Axis(fig[3, 3]; title = "Subfilter |τ|", axis_kwargs...)

heatmap!(ax_c, c;  colormap = :balance, colorrange = (-1, 1))
heatmap!(ax_c̄, c̄;  colormap = :balance, colorrange = (-1, 1))
hm_τ = heatmap!(ax_τ, τ; colormap = :magma)
Colorbar(fig[3, 4], hm_τ)

resize_to_layout!(fig)
fig

# The top row shows how the Gaussian filter splits the vorticity into a smooth resolved field
# ``\bar{\omega}`` and the fine-scale residual ``\omega'`` it removes. The middle row shows that
# wider kernels (`σ = 2Δ → 8Δ`) progressively erase smaller scales. The bottom row shows the
# tracer ``c``, its filtered version ``\bar{c}``, and the magnitude of the subfilter flux
# ``\tau``, which is largest precisely along the thin tracer filaments — the small-scale structure
# that the filtered fields ``\bar{u}_i\,\bar{c}`` cannot represent on their own.
