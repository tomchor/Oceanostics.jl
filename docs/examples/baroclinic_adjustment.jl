# # [Baroclinic adjustment and the potential energy budget](@id baroclinic_adjustment_example)
#
# In this example we spin up a pair of mesoscale fronts in a doubly-periodic channel, let them go
# baroclinically unstable, and close the volume-integrated *potential* energy budget against the
# kinetic energy one. The setup follows the double-front baroclinic adjustment of
# [Wenegrat's CoarseGrainedBCI](https://github.com/wenegrat/CoarseGrainedBCI), itself a periodic
# variant of Oceananigans' own baroclinic adjustment example.
#
# Two fronts rather than one is what makes the potential energy budget well posed. A single front with
# a uniform background gradient, as in [the Eady example](@ref eady_example), has a buoyancy field that
# grows without bound in the cross-front direction, so `∫eₚ dV` over the domain is infinite and only
# the perturbation part of it is finite. Here the buoyancy rises across one front and falls back across
# the other, so `b` is genuinely periodic, the whole field is prognostic, and `∫eₚ dV = -∫bz dV` is a
# finite number the budget can track.
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
# A 1000 km square channel, 1 km deep, on a β-plane at 45°S. The fronts are 100 km wide and carry a
# cross-front buoyancy gradient `M²` on top of a uniform stratification `N²`:

Lx = Ly = 1000kilometers
Lz = 1kilometers

N²    = 1e-5      # [s⁻²] background stratification
M²    = 1e-7      # [s⁻²] cross-front buoyancy gradient
Δy    = 100kilometers   # width of each front
latitude = -45

Δb = Δy * M²                        # buoyancy jump across each front
f₀ = 2 * 7.292115e-5 * sind(latitude)   # [s⁻¹] Coriolis parameter at that latitude
U  = M² * Lz / abs(f₀)              # [m s⁻¹] thermal-wind velocity scale

# ## Grid
#
# Doubly periodic in the horizontal and bounded in the vertical. Periodicity in `y` is what the double
# front buys us, and it is what makes the transport terms of both budgets integrate to zero:

grid = RectilinearGrid(size = (64, 64, 16),
                       x = (0, Lx), y = (-Ly/2, Ly/2), z = (-Lz, 0),
                       topology = (Periodic, Periodic, Bounded))

# ## Closure
#
# The dissipation comes from an anisotropic Laplacian whose viscosity is set per direction from a
# grid-Péclet criterion, `ν = (U/Pe)·Δ`, following the reference setup. Tying `ν` to the grid spacing
# this way makes it shrink as the grid is refined, and it comes out much smaller in the vertical than
# the horizontal because the cells are 15 km wide and 60 m tall. The two targets are set separately
# since there is no reason one tuned value should transfer between directions:

Δx, Δy_grid, Δz = Lx / size(grid, 1), Ly / size(grid, 2), Lz / size(grid, 3)

Pe_h, Pe_v = 100, 50      # target cell Péclet numbers, UΔ/ν

νh = (U / Pe_h) * √(Δx * Δy_grid)   # [m² s⁻¹]
νv = (U / Pe_v) * Δz                # [m² s⁻¹]

@info "Thermal-wind scale U = $(round(U, digits=3)) m/s, νh = $(round(νh, digits=1)) m² s⁻¹, νv = $(round(νv, digits=3)) m² s⁻¹"

closure = (HorizontalScalarDiffusivity(ν=νh, κ=νh),
           VerticalScalarDiffusivity(ν=νv, κ=νv))

# ## Model
#
# The advection scheme is `Centered`, which adds no dissipation of its own, so every sink in the
# budgets below is one we write down and compute:

model = NonhydrostaticModel(grid;
                            coriolis = BetaPlane(; latitude),
                            buoyancy = BuoyancyTracer(),
                            tracers = :b,
                            timestepper = :RungeKutta3,
                            advection = Centered(order=4),
                            closure = closure)

# ## Initial condition
#
# The buoyancy is a uniform stratification plus two ramps of width `Δy`, one rising at `y = -Ly/4` and
# one falling at `y = +Ly/4`. Their difference returns to zero at the edges of the domain, so `b`
# matches across the periodic boundary:

ramp(y, Δy) = min(max(0, y/Δy + 1/2), 1)

y₁ = -Ly/4
y₂ = +Ly/4

double_ramp(y, Δy) = ramp(y - y₁, Δy) - ramp(y - y₂, Δy)

# The velocity starts in thermal-wind balance with those fronts, referenced to mid-depth, and a little
# noise on the buoyancy seeds the instability:

using Random
Random.seed!(8675309)

ϵb = 1e-2 * Δb    # noise amplitude, a percent of the front's buoyancy jump

bᵢ(x, y, z) = N² * z + Δb * double_ramp(y, Δy) + ϵb * randn()

ramp_prime(y, Δy) = (-Δy/2 < y < Δy/2) ? 1/Δy : zero(y)
double_ramp_prime(y, Δy) = ramp_prime(y - y₁, Δy) - ramp_prime(y - y₂, Δy)

β = model.coriolis.β
f_cor(y) = f₀ + β * y

z_ref = -Lz/2
uᵢ(x, y, z) = -(Δb * double_ramp_prime(y, Δy) / f_cor(y)) * (z - z_ref)

set!(model, u=uᵢ, b=bᵢ)
nothing #hide

# ## Simulation

simulation = Simulation(model, Δt = 1minute, stop_time = 20days)
conjure_time_step_wizard!(simulation, IterationInterval(5), cfl = 0.2, diffusive_cfl = 0.2, max_Δt = 20minutes)

using Oceanostics
add_callback!(simulation, ProgressMessengers.TimedMessenger(), IterationInterval(100))

# ## The potential energy budget
#
# `eₚ = -bz` obeys `-z` times the equation the model steps for `b`, so its terms are that equation's
# terms weighted by `-z`. Over this domain the transports integrate away and the budget reduces to the
# two conversion terms:
#
# ```math
# \frac{d}{dt}\int e_p\, dV = \int \mathrm{ADV}\, dV + \int \mathrm{DIFF}\, dV
#                           = -\int wb\, dV + \int \Phi\, dV ,
# ```
#
# where ``\mathrm{ADV} = z\,\partial_j(u_jb)`` ([`PotentialEnergyAdvection`](@ref)) and
# ``\mathrm{DIFF} = z\,\partial_jq_j`` ([`PotentialEnergyDiffusion`](@ref)) are the terms as the model
# actually computes them, while ``wb``
# ([`PotentialToKineticEnergyConversion`](@ref Oceanostics.KineticEnergyEquation.PotentialEnergyConversion))
# and ``\Phi = \kappa\,\partial b/\partial z``
# ([`PotentialEnergyDiffusiveBuoyancyFlux`](@ref Oceanostics.PotentialEnergyEquation.DiffusiveBuoyancyFlux))
# are what they collapse to once `z` is pulled inside the derivative and the transports drop out. We
# output both forms so the identity can be checked rather than assumed.
#
# [`PotentialEnergyTendency`](@ref) is the whole right-hand side in one term, taken off Oceananigans'
# own buoyancy tendency. Writing it alongside the finite-differenced `d(∫eₚ)/dt` is the sharpest check
# available: the two are computed by completely different routes.
#
# There is no `BackgroundField` here, so
# [`PotentialEnergyBackgroundAdvection`](@ref) is identically zero and does not appear.

eₚ   = PotentialEnergy(model)
TEND = PotentialEnergyTendency(model)
ADV  = PotentialEnergyAdvection(model)
DIFF = PotentialEnergyDiffusion(model)
Φ    = PotentialEnergyDiffusiveBuoyancyFlux(model)

# ## The kinetic energy budget
#
# The other side of the exchange. `wb` carries opposite signs in the two budgets, so it cancels from
# their sum:
#
# ```math
# \frac{d}{dt}\int K\, dV = \int wb\, dV - \int \varepsilon\, dV .
# ```

K  = KineticEnergy(model)
wb = PotentialToKineticEnergyConversion(model)
ε  = KineticEnergyDissipationRate(model)

∫eₚ   = Integral(eₚ)
∫TEND = Integral(TEND)
∫ADV  = Integral(ADV)
∫DIFF = Integral(DIFF)
∫Φ    = Integral(Φ)
∫K    = Integral(K)
∫wb   = Integral(wb)
∫ε    = Integral(ε)

# For the movie we keep the surface vorticity and buoyancy:

u, v, w = model.velocities
b = model.tracers.b
ζ = ∂x(v) - ∂y(u)

# ## Output
#
# A *snapshot* writer for the surface maps and a *budget* writer for the volume integrals, the latter
# on `ConsecutiveIterations`, which takes a second sample one model step after each output time so we
# can finite-difference `d/dt` across that step.

using NCDatasets
filename = joinpath(@__DIR__, "baroclinic_adjustment")

simulation.output_writers[:fields] =
    NetCDFWriter(model, (; ζ, b),
                 filename = filename,
                 schedule = TimeInterval(12hours),
                 indices = (:, :, grid.Nz),
                 overwrite_existing = true)

simulation.output_writers[:budget] =
    NetCDFWriter(model, (; ∫eₚ, ∫TEND, ∫ADV, ∫DIFF, ∫Φ, ∫K, ∫wb, ∫ε),
                 filename = filename * "_budget",
                 schedule = ConsecutiveIterations(TimeInterval(12hours)),
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
close(ds)

ζlim = maximum(abs, ζ_arr)
blim = maximum(abs, b_arr .- sum(b_arr) / length(b_arr))

# The budget scalars come in consecutive-iteration pairs `(2k-1, 2k)`; a one-step finite difference
# inside each pair gives the tendencies, and every source term is averaged over the same pair.

ds_b = NCDataset(simulation.output_writers[:budget].filepath)
t_bud    = ds_b["time"][:]
eₚ_bud   = ds_b["∫eₚ"][:]
TEND_bud = ds_b["∫TEND"][:]
ADV_bud  = ds_b["∫ADV"][:]
DIFF_bud = ds_b["∫DIFF"][:]
Φ_bud    = ds_b["∫Φ"][:]
K_bud    = ds_b["∫K"][:]
wb_bud   = ds_b["∫wb"][:]
ε_bud    = ds_b["∫ε"][:]
close(ds_b)

idx1 = 1:2:length(t_bud) - 1
idx2 = 2:2:length(t_bud)

Δt_pair = t_bud[idx2] .- t_bud[idx1]
t_pair  = @. 0.5 * (t_bud[idx1] + t_bud[idx2])

deₚdt = (eₚ_bud[idx2] .- eₚ_bud[idx1]) ./ Δt_pair
dKdt  = (K_bud[idx2]  .- K_bud[idx1])  ./ Δt_pair

pair_mean(x) = @. 0.5 * (x[idx1] + x[idx2])

TEND_pair = pair_mean(TEND_bud)
ADV_pair  = pair_mean(ADV_bud)
DIFF_pair = pair_mean(DIFF_bud)
Φ_pair    = pair_mean(Φ_bud)
wb_pair   = pair_mean(wb_bud)
ε_pair    = pair_mean(ε_bud)

# Both budgets in sum-to-zero form, plus the two collapses that let `∫wb dV` and `∫Φ dV` stand in for
# the advective and diffusive terms:

eₚ_resid = @. -deₚdt + ADV_pair + DIFF_pair
K_resid  = @. -dKdt  + wb_pair  - ε_pair

adv_collapse  = @. ADV_pair  + wb_pair    # should vanish: ∫z∂ⱼ(uⱼb) dV = -∫wb dV
diff_collapse = @. DIFF_pair - Φ_pair     # should vanish: ∫z∂ⱼqⱼ dV    =  ∫Φ dV
tend_check    = @. TEND_pair - deₚdt      # the model's own tendency against the finite difference

using Test                                                                            #hide
using Statistics: mean                                                                #hide
rms(x) = √(sum(abs2, x) / length(x))                                                  #hide
eₚ_terms = (deₚdt, ADV_pair, DIFF_pair)                                               #hide
K_terms  = (dKdt, wb_pair, ε_pair)                                                    #hide
## Each budget closes to a small fraction of its own largest term. The potential energy one closes   #hide
## to parts in a hundred thousand: its terms come off Oceananigans' own buoyancy tendency, so the    #hide
## only discrepancy is the finite-differenced `d/dt`.                                                #hide
@test rms(eₚ_resid) < 1e-3 * maximum(rms, eₚ_terms)                                   #hide
@test rms(K_resid)  < 0.05 * maximum(rms, K_terms)                                    #hide
## `wb` is the same term in both, so it cancels from their sum                        #hide
@test rms(eₚ_resid .+ K_resid) < 0.05 * rms(wb_pair)                                  #hide
## `PotentialEnergyTendency` comes off the model's own buoyancy tendency rather than off a finite    #hide
## difference of `∫eₚ`, so agreeing with that difference is an end-to-end check of the whole set.    #hide
@test rms(tend_check) < 1e-3 * rms(deₚdt)                                             #hide
## The two continuum collapses. `∫DIFF = ∫Φ` telescopes and holds to roundoff; `∫ADV = -∫wb` is      #hide
## second order in the grid spacing and holds to a fraction of a percent.                            #hide
@test rms(diff_collapse) < 1e-8 * rms(Φ_pair)                                         #hide
@test rms(adv_collapse)  < 0.02 * rms(wb_pair)                                        #hide
## The adjustment does what it should: the fronts slump and release potential energy into kinetic    #hide
## energy. `∫ADV dV` is the conversion seen from the potential energy side, so it is negative.       #hide
@test mean(wb_pair)  > 0                                                              #hide
@test mean(ADV_pair) < 0                                                              #hide
@test K_bud[idx1][end] > 10 * K_bud[idx1][1]                                          #hide
@test all(ε_pair .≥ 0);                                                               #hide

# ## Plotting

set_theme!(Theme(fontsize = 18))
fig = Figure(size = (1100, 1000))

n = Observable(1)

axζ = Axis(fig[2, 1]; title = "vertical vorticity, ζ", xlabel = "x [km]", ylabel = "y [km]", aspect = 1)
axb = Axis(fig[2, 3]; title = "surface buoyancy, b", xlabel = "x [km]", ylabel = "y [km]", aspect = 1)

ζₙ = @lift ζ_arr[:, :, $n]
bₙ = @lift b_arr[:, :, $n]

hmζ = heatmap!(axζ, x_faa ./ 1e3, y_afa ./ 1e3, ζₙ; colormap = :balance, colorrange = (-ζlim, ζlim))
Colorbar(fig[2, 2], hmζ)

hmb = heatmap!(axb, x_caa ./ 1e3, y_aca ./ 1e3, bₙ; colormap = :thermal)
Colorbar(fig[2, 4], hmb)

budget_kwargs = (xlabel = "time [days]", ylabel = "[m⁵ s⁻³]")

ax_p = Axis(fig[3, 1:4]; title = "Volume-integrated potential energy budget", budget_kwargs...)
lines!(ax_p, t_pair ./ day, -deₚdt,     label = "-d(∫eₚ)/dt")
lines!(ax_p, t_pair ./ day,  ADV_pair,  label = "∫ADV dV  (= -∫wb dV)")
lines!(ax_p, t_pair ./ day,  DIFF_pair, label = "∫DIFF dV  (= ∫Φ dV)")
lines!(ax_p, t_pair ./ day,  eₚ_resid,  label = "residual", color = :black, linestyle = :dash)
axislegend(ax_p; position = :rt, labelsize = 10, nbanks = 2)

ax_K = Axis(fig[4, 1:4]; title = "Volume-integrated kinetic energy budget", budget_kwargs...)
lines!(ax_K, t_pair ./ day, -dKdt,    label = "-d(∫K)/dt")
lines!(ax_K, t_pair ./ day,  wb_pair, label = "∫wb dV  (buoyancy conversion)")
lines!(ax_K, t_pair ./ day, -ε_pair,  label = "-∫ε dV  (dissipation)")
lines!(ax_K, t_pair ./ day,  K_resid, label = "residual", color = :black, linestyle = :dash)
axislegend(ax_K; position = :rt, labelsize = 10, nbanks = 2)

vlines!(ax_p, @lift(times[$n] / day), color = :black, linestyle = :dot)
vlines!(ax_K, @lift(times[$n] / day), color = :black, linestyle = :dot)

title = @lift "Baroclinic adjustment, t = " * prettytime(times[$n])
fig[1, 1:4] = Label(fig, title, fontsize = 22, tellwidth = false)

@info "Animating..."
record(fig, "baroclinic_adjustment.mp4", 1:length(times), framerate = 8) do i
    n[] = i
end
set_theme!() #hide
nothing #hide

# ![](baroclinic_adjustment.mp4)
#
# Both fronts go unstable and roll up into mesoscale eddies, and the two budget panels show the energy
# moving between the reservoirs. `∫wb dV` is the through line: the eddies slump the fronts and what
# the potential energy loses to that conversion appears as `∫K dV`. Since it enters the two budgets
# with opposite signs it cancels from their sum, which is the sense in which this is one exchange
# rather than two independent balances.
#
# `∫eₚ dV` still rises over the run, which is not a contradiction. The horizontal and vertical
# diffusivities act on the background stratification everywhere, all the time, and diffusion working
# against gravity raises potential energy. That steady source, `∫DIFF dV = ∫Φ dV`, is larger than the
# eddies' release, so the net drift is upward with the conversion riding on top of it. Splitting the
# two apart is exactly what the budget is for.
#
# The potential energy panel is the one this example exists for. `∫ADV dV` and `∫DIFF dV` are plotted
# as the module computes them, `-z` times the advective and diffusive terms of the buoyancy equation,
# and they land on `-∫wb dV` and `∫Φ dV`. The diffusive pair agree to roundoff, since that collapse
# telescopes on the discrete grid. The advective pair agree to about a tenth of a percent, since that
# one differs by a transport term which integrates to zero only in the continuum. Keeping the `-z ×`
# form is what makes the terms sum to `PotentialEnergyTendency` exactly instead, cell by cell, and the
# tendency check above confirms that end to end: the model's own buoyancy tendency, weighted by `-z`
# and integrated, matches a finite difference of `∫eₚ dV` to parts in a hundred thousand.
#
# The kinetic energy residual is larger, around a percent of its largest term, which is the usual
# level for these examples: the discrete `K` equation is not derived from the discrete momentum
# equation the model steps, so the two sides agree only to truncation error. The same caveat applies
# to [the two-dimensional turbulence example](@ref two_d_turbulence_example). The potential energy
# budget escapes it because its terms are the model's own tendency taken apart rather than rederived.
