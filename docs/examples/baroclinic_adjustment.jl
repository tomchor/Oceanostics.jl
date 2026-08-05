# # [Baroclinic adjustment and the potential energy budget](@id baroclinic_adjustment_example)
#
# In this example we spin up a pair of submesoscale fronts in a doubly-periodic channel, let them go
# baroclinically unstable, and close the volume-integrated potential energy budget against the
# kinetic energy one.
#
# Before starting, make sure you have the required packages installed for this example, which can be
# done with
#
# ```julia
# using Pkg
# pkg"add Oceananigans, Oceanostics, CairoMakie, NCDatasets"
# ```

using Oceananigans
using Oceananigans.Units

# ## Parameters
#
# A 1 km by 2 km channel, 70 m deep, with a 30 m mixed layer over a stratified interior. The two
# fronts are 250 m wide and carry a cross-front buoyancy gradient `M²`:

Lx = 1kilometers
Ly = 2kilometers
H_target = 70.0      # [m] requested depth; the stretched grid below lands a little deeper
h  = 30.0            # [m] mixed-layer depth

f       = 1e-4       # [s⁻¹] Coriolis frequency
M²      = 3e-8       # [s⁻²] cross-front buoyancy gradient inside each front
N²_ml   = 9e-8       # [s⁻²] mixed-layer stratification (z > -h)
N²_int  = 1.8e-6     # [s⁻²] interior stratification (z < -h)
w_front = 250.0      # [m]   width of each front

# ## Grid
#
# Doubly periodic in the horizontal and bounded in the vertical. Periodicity in `y` is what the double
# front buys us, and it is what makes the transport terms of both budgets integrate to zero. The
# vertical coordinate is surface-intensified: a constant 2 m spacing over the top 32 m resolves the
# mixed layer the instability lives in, and stretches below it.
#
# The cells come out roughly 16 m by 16 m by 2 to 8 m, within an order of magnitude of isotropic. That
# matters for the closure below: `Smagorinsky` builds its filter width from the cell volume, so it is
# only meaningful on a grid that is not wildly stretched in one direction.

z = ReferenceToStretchedDiscretization(extent = H_target,
                                       bias = :right, bias_edge = 0,   # fine spacing at the surface
                                       constant_spacing = 2,
                                       constant_spacing_extent = 32,
                                       stretching = PowerLawStretching(1.08))

grid = RectilinearGrid(size = (64, 128, length(z)),
                       x = (0, Lx), y = (-Ly/2, Ly/2), z = z,
                       topology = (Periodic, Periodic, Bounded))

H = grid.Lz   # [m] the depth the grid actually has

@info "Grid: $(size(grid)), depth $(round(H, digits=1)) m, Δx = $(round(minimum_xspacing(grid), digits=1)) m, Δz from $(round(minimum_zspacing(grid), digits=2)) m to $(round(maximum(zspacings(grid, Center())), digits=2)) m"

# ## Derived quantities
#
# The mixed layer sets the scale of the instability: its deformation radius `Ld = N h / f`, the
# fastest-growing wavelength `λ ≈ 3.9 Ld`, and the growth rate `σ ≈ 0.31 M²/N`. The balanced Richardson
# number `Ri = N²/α²` is 1 in the mixed layer, which is what lets the resolved strain reach the
# stratification, and so what lets the closure below do anything at all.

α     = M² / f              # [s⁻¹]   thermal-wind shear inside a front
N_ml  = √N²_ml              # [s⁻¹]
N_int = √N²_int             # [s⁻¹]
Ld    = N_ml * h / f        # [m]     mixed-layer deformation radius
λ     = 3.9 * Ld            # [m]     fastest-growing wavelength
Ri_ml = N²_ml / α^2         # []      balanced Richardson number in the mixed layer
Δb    = M² * w_front        # [m s⁻²] buoyancy jump across each front
Ū     = α * H               # [m s⁻¹] thermal-wind velocity scale
σ     = 0.31 * M² / N_ml    # [s⁻¹]   growth rate of the mixed-layer mode

@info "Ld = $(round(Ld, digits=1)) m, λ = $(round(λ, digits=1)) m in a $(round(Int, Ly)) m channel, Ri = $(round(Ri_ml, digits=2))"
@info "Δb = $(round(Δb, sigdigits=3)) m s⁻², Ū = $(round(100Ū, digits=2)) cm/s, growth time 1/σ = $(prettytime(1/σ))"

# ## Closure
#
# We use a `Smagorinsky` closure with a constant coefficient to model turbulence stresses. We use
# a high coefficient to make sure the instability is well-resolved in this very coarse example:

using Oceananigans.TurbulenceClosures.Smagorinskys: Smagorinsky
closure = Smagorinsky(coefficient=0.3)

# ## Model
#
# The advection scheme is `Centered`, which adds no dissipation of its own, so every sink in the
# budgets below is one we write down and compute. We set up an `f`-plane Coriolis:

model = NonhydrostaticModel(grid;
                            coriolis = FPlane(; f),
                            buoyancy = BuoyancyTracer(),
                            tracers = :b,
                            timestepper = :RungeKutta3,
                            advection = Centered(order=4),
                            closure = closure)

# ## Initial condition
#
# The buoyancy is the two-layer stratification plus two ramps of width `w_front`, one rising at
# `y = -Ly/4` and one falling at `y = +Ly/4`. Their difference returns to zero at the edges of the
# domain, so `b` matches across the periodic boundary:

ramp(y, w) = min(max(0, y/w + 1/2), 1)

y₁ = -Ly/4
y₂ = +Ly/4

double_ramp(y, w) = ramp(y - y₁, w) - ramp(y - y₂, w)

## the stratification integrated from the surface down, continuous across the mixed-layer base
b_strat(z) = ifelse(z ≥ -h, N²_ml * z, -N²_ml * h + N²_int * (z + h))

# The velocity starts in thermal-wind balance with those fronts, referenced to the bottom, and a little
# noise on the buoyancy seeds the instability:

using Random
Random.seed!(8675309)

ϵb = 1e-2 * Δb    # noise amplitude, a percent of the front's buoyancy jump

bᵢ(x, y, z) = b_strat(z) + Δb * double_ramp(y, w_front) + ϵb * randn()

ramp_prime(y, w) = (-w/2 < y < w/2) ? 1/w : zero(y)
double_ramp_prime(y, w) = ramp_prime(y - y₁, w) - ramp_prime(y - y₂, w)

uᵢ(x, y, z) = -(Δb * double_ramp_prime(y, w_front) / f) * (z + H)

set!(model, u=uᵢ, b=bᵢ)
nothing #hide

# ## Simulation

max_Δt = min(minimum_xspacing(grid) / Ū, 1 / N_int)   # Smagorinsky sets ν from the flow, so no fixed diffusive bound

simulation = Simulation(model, Δt = max_Δt, stop_time = 8days)
conjure_time_step_wizard!(simulation, IterationInterval(5), cfl = 0.7, max_Δt = max_Δt)

using Oceanostics
add_callback!(simulation, ProgressMessengers.TimedMessenger(), IterationInterval(100))

# ## The potential energy budget
#
# Given `eₚ = -bz`, we multiply the budget equation for `b` by `-z`. Pulling `z` inside each derivative splits terms into a transport and a
# conversion, and over this domain every transport integrates away, leaving just the two conversions:
#
# ```math
# \frac{d}{dt}\int e_p\, dV = -\int wb\, dV + \int \Phi\, dV ,
# ```
#
# with ``wb``
# ([`PotentialToKineticEnergyConversion`](@ref Oceanostics.KineticEnergyEquation.PotentialEnergyConversion))
# the exchange with the kinetic energy and ``\Phi = \kappa\,\partial b/\partial z``
# ([`PotentialEnergyDiffusiveVerticalBuoyancyFlux`](@ref Oceanostics.PotentialEnergyEquation.DiffusiveVerticalBuoyancyFlux))
# the work diffusion does against gravity.

eₚ = PotentialEnergy(model)
wb = PotentialToKineticEnergyConversion(model)
Φ  = PotentialEnergyDiffusiveVerticalBuoyancyFlux(model)

# ## The kinetic energy budget
#
# The other side of the exchange, written with `eₖ = ½uᵢuᵢ` to match `eₚ`. `wb` is the same term in
# both and carries opposite signs, so it cancels from their sum:
#
# ```math
# \frac{d}{dt}\int e_k\, dV = \int wb\, dV - \int \varepsilon_k\, dV .
# ```

eₖ = KineticEnergy(model)
εₖ = KineticEnergyDissipationRate(model)

∫eₚ = ∫dV(eₚ)
∫Φ  = ∫dV(Φ)
∫eₖ = ∫dV(eₖ)
∫wb = ∫dV(wb)
∫εₖ = ∫dV(εₖ)

# For the movie we keep the surface vorticity, buoyancy, and the closure's own eddy viscosity, which
# shows where `Smagorinsky` is actually acting:

u, v, w = model.velocities
b = model.tracers.b
ζ = ∂x(v) - ∂y(u)
νₑ = viscosity(model)

# ## Output
#
# A *snapshot* writer for the surface maps and a *budget* writer for the volume integrals, the latter
# on `ConsecutiveIterations`, which takes a second sample one model step after each output time so we
# can finite-difference `d/dt` across that step.

using NCDatasets
filename = joinpath(@__DIR__, "baroclinic_adjustment")

simulation.output_writers[:fields] =
    NetCDFWriter(model, (; ζ, b, νₑ),
                 filename = filename,
                 schedule = TimeInterval(3hours),
                 indices = (:, :, grid.Nz),
                 overwrite_existing = true)

simulation.output_writers[:budget] =
    NetCDFWriter(model, (; ∫eₚ, ∫Φ, ∫eₖ, ∫wb, ∫εₖ),
                 filename = filename * "_budget",
                 schedule = ConsecutiveIterations(TimeInterval(3hours)),
                 overwrite_existing = true)

# ## Run the simulation

run!(simulation)

# ## Process the results

using CairoMakie

ds = NCDataset(simulation.output_writers[:fields].filepath)
times = ds["time"][:]
x_faa = ds["x_faa"][:]; y_afa = ds["y_afa"][:]
x_caa = ds["x_caa"][:]; y_aca = ds["y_aca"][:]
ζ_arr = ds["ζ"][:, :, 1, :]
b_arr = ds["b"][:, :, 1, :]
ν_arr = ds["νₑ"][:, :, 1, :]
close(ds)

ζlim = maximum(abs, ζ_arr)
νlim = maximum(ν_arr)

# The budget scalars come in consecutive-iteration pairs `(2k-1, 2k)`; a one-step finite difference
# inside each pair gives the tendencies, and every source term is averaged over the same pair.

ds_b = NCDataset(simulation.output_writers[:budget].filepath)
t_bud    = ds_b["time"][:]
eₚ_bud = ds_b["∫eₚ"][:]
Φ_bud  = ds_b["∫Φ"][:]
eₖ_bud = ds_b["∫eₖ"][:]
wb_bud = ds_b["∫wb"][:]
εₖ_bud = ds_b["∫εₖ"][:]
close(ds_b)

idx1 = 1:2:length(t_bud) - 1
idx2 = 2:2:length(t_bud)

Δt_pair = t_bud[idx2] .- t_bud[idx1]
t_pair  = @. 0.5 * (t_bud[idx1] + t_bud[idx2])

deₚdt = (eₚ_bud[idx2] .- eₚ_bud[idx1]) ./ Δt_pair
deₖdt = (eₖ_bud[idx2] .- eₖ_bud[idx1]) ./ Δt_pair

pair_mean(x) = @. 0.5 * (x[idx1] + x[idx2])

Φ_pair  = pair_mean(Φ_bud)
wb_pair = pair_mean(wb_bud)
εₖ_pair = pair_mean(εₖ_bud);

# Both budgets in sum-to-zero form: every curve is plotted with the sign it carries here, so each panel
# below adds up to its residual.

eₚ_resid = @. -deₚdt - wb_pair + Φ_pair
eₖ_resid = @. -deₖdt + wb_pair - εₖ_pair

using Test                                                                            #hide
using Statistics: mean                                                                #hide
rms(x) = √(sum(abs2, x) / length(x))                                                  #hide
eₚ_terms = (deₚdt, wb_pair, Φ_pair)                                                   #hide
eₖ_terms = (deₖdt, wb_pair, εₖ_pair)                                                  #hide
## Each budget closes to a small fraction of its own largest term.                                   #hide
@test rms(eₚ_resid) < 0.03 * maximum(rms, eₚ_terms)                                   #hide
@test rms(eₖ_resid) < 0.05 * maximum(rms, eₖ_terms)                                   #hide
## `wb` is the same term in both, so it cancels from their sum                        #hide
@test rms(eₚ_resid .+ eₖ_resid) < 0.05 * rms(wb_pair)                                 #hide
## The adjustment does what it should: the fronts slump and the potential energy they release turns  #hide
## into kinetic energy, so the conversion is positive on average.                                    #hide
@test mean(wb_pair) > 0                                                               #hide
## It leads the kinetic energy budget. It does not lead the potential energy one — with this closure #hide
## the diffusive term is an order of magnitude larger — but the budget resolves it far above its own #hide
## residual, which is the claim that matters.                                                        #hide
@test rms(wb_pair) > 0.5 * rms(deₖdt)                                                 #hide
@test rms(wb_pair) > 10 * rms(eₚ_resid)                                               #hide
@test rms(Φ_pair)  > rms(wb_pair)                                                     #hide
## The closure is active everywhere, which is what a constant-coefficient Smagorinsky does and what  #hide
## the Lilly-corrected one would not at this resolution.                                             #hide
@test minimum(εₖ_pair) > 0                                                            #hide
@test all(εₖ_pair .≥ 0);                                                              #hide

# ## Plotting

set_theme!(Theme(fontsize = 18))
fig = Figure(size = (1500, 950))

n = Observable(1)

panel_kwargs = (xlabel = "x [m]", ylabel = "y [m]", aspect = DataAspect(), height = 240)
axζ = Axis(fig[2, 1]; title = "vertical vorticity, ζ",  panel_kwargs...)
axb = Axis(fig[2, 3]; title = "surface buoyancy, b",    panel_kwargs...)
axν = Axis(fig[2, 5]; title = "eddy viscosity, νₑ",     panel_kwargs...)

ζₙ = @lift ζ_arr[:, :, $n]
bₙ = @lift b_arr[:, :, $n]
νₙ = @lift ν_arr[:, :, $n]

hmζ = heatmap!(axζ, x_faa, y_afa, ζₙ; colormap = :balance, colorrange = (-ζlim, ζlim))
Colorbar(fig[2, 2], hmζ)

hmb = heatmap!(axb, x_caa, y_aca, bₙ; colormap = :thermal)
Colorbar(fig[2, 4], hmb)

hmν = heatmap!(axν, x_caa, y_aca, νₙ; colormap = :tempo, colorrange = (0, νlim))
Colorbar(fig[2, 6], hmν)

budget_kwargs = (xlabel = "time [days]", ylabel = "[m⁵ s⁻³]")

ax_p = Axis(fig[3, 1:6]; title = "Volume-integrated potential energy budget", budget_kwargs...)
lines!(ax_p, t_pair ./ day, -deₚdt,    label = "-d(∫eₚ)/dt")
lines!(ax_p, t_pair ./ day, -wb_pair,  label = "-∫wb dV")
lines!(ax_p, t_pair ./ day,  Φ_pair,   label = "∫Φ dV")
lines!(ax_p, t_pair ./ day,  eₚ_resid, label = "residual", color = :black, linestyle = :dash)
axislegend(ax_p; position = :rt, labelsize = 10, nbanks = 2)

ax_k = Axis(fig[4, 1:6]; title = "Volume-integrated kinetic energy budget", budget_kwargs...)
lines!(ax_k, t_pair ./ day, -deₖdt,    label = "-d(∫eₖ)/dt")
lines!(ax_k, t_pair ./ day,  wb_pair,  label = "∫wb dV")
lines!(ax_k, t_pair ./ day, -εₖ_pair,  label = "-∫εₖ dV")
lines!(ax_k, t_pair ./ day,  eₖ_resid, label = "residual", color = :black, linestyle = :dash)
axislegend(ax_k; position = :rt, labelsize = 10, nbanks = 2)

vlines!(ax_p, @lift(times[$n] / day), color = :black, linestyle = :dot)
vlines!(ax_k, @lift(times[$n] / day), color = :black, linestyle = :dot)

title = @lift "Baroclinic adjustment, t = " * prettytime(times[$n])
fig[1, 1:6] = Label(fig, title, fontsize = 22, tellwidth = false)

@info "Animating..."
record(fig, "baroclinic_adjustment.mp4", 1:length(times), framerate = 8) do i
    n[] = i
end
set_theme!() #hide
nothing #hide

# ![](baroclinic_adjustment.mp4)
