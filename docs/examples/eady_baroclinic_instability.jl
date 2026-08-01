# # [Eady baroclinic instability](@id eady_example)
#
# This example simulates the [Eady problem](https://en.wikipedia.org/wiki/Eady_model): the
# baroclinic instability of a uniformly stratified, uniformly sheared flow held in thermal-wind
# balance on an ``f``-plane. The setup is a port of the classic
# [Eady turbulence example](https://numericalearth.github.io/OceananigansMuseum/v0.74.1/generated/eady_turbulence/)
# to up-to-date Oceananigans syntax, with the bottom drag removed. We then use Oceanostics to close the
# volume-integrated kinetic and potential energy budgets of the growing eddies.
#
# The mean flow enters as a pair of `BackgroundField`s, so the model steps *perturbations* about it.
# That is what makes this example worth doing: each background field puts a term of its own into the
# budget of the perturbation energy it acts on. The buoyancy one is small next to the exchange between
# the two budgets and still twenty times the residual, so a budget that leaves it out does not close.
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
# We collect all the primitive parameters of the problem up front, before constructing anything.

## Domain size
Lx = Ly = 1e6   # [m]   horizontal extent (1000 km)
H  = 4e3        # [m]   depth

## Physical parameters
f = 1e-4        # [s⁻¹] Coriolis frequency
α = 5 * f       # [s⁻¹] geostrophic shear ∂U/∂z
N = 1e-3        # [s⁻¹] buoyancy frequency

# ## Derived dynamical quantities
#
# From those parameters we can form the quantities that characterize the flow: the cross-front and
# vertical buoyancy gradients, the deformation radius `Ld = N H / f`, the wavelength of the
# fastest-growing Eady mode `λ ≈ 3.9 Ld`, the balanced Richardson number `Ri = N²/α²`, the
# thermal-wind velocity scale `Ū = α H`, and the maximum Eady growth rate `σ ≈ 0.31 M²/N`.

M² = α * f          # [s⁻²]   cross-front buoyancy gradient |∂B/∂y|
N² = N^2            # [s⁻²]   vertical buoyancy gradient
Ld = N * H / f      # [m]     first baroclinic deformation radius
λ  = 3.9 * Ld       # [m]     wavelength of the fastest-growing Eady mode
Ri = N² / α^2       # []      balanced Richardson number
Ū  = α * H          # [m s⁻¹] peak background (thermal-wind) velocity
σ  = 0.31 * M² / N  # [s⁻¹]   maximum Eady growth rate

@info "Deformation radius Ld = $(round(Ld/1e3, digits=1)) km, fastest Eady wavelength λ = $(round(λ/1e3, digits=1)) km"
@info "Balanced Richardson number Ri = $(round(Ri, digits=2)), thermal-wind velocity Ū = $(round(Ū, digits=3)) m/s, Eady growth time 1/σ = $(prettytime(1/σ))"

# ## Grid
#
# We use a mesoscale-resolving grid, periodic in the horizontal and bounded in the vertical:

grid = RectilinearGrid(size = (48, 48, 16), extent = (Lx, Ly, H))

# ## Coriolis and background state
#
# The flow is set up on an ``f``-plane. The background velocity increases linearly with height, and
# the background buoyancy combines the geostrophic (cross-front) component with a stable
# stratification. They are in thermal-wind balance, ``f\,\partial_z U = -\partial_y B``:

coriolis = FPlane(f = f)

U(x, y, z, t, p) = + p.α * (z + p.H)
B(x, y, z, t, p) = - p.α * p.f * y + p.N^2 * z

background_parameters = (; α, f, N, H)
U_field = BackgroundField(U, parameters=background_parameters)
B_field = BackgroundField(B, parameters=background_parameters)

# ## Turbulence closures
#
# We dissipate variance with a Laplacian vertical diffusivity and a biharmonic horizontal
# diffusivity, applied simultaneously as a tuple of two closures:

Δx = minimum_xspacing(grid)
κ₂z = 2e-2                # [m² s⁻¹] Laplacian vertical viscosity and diffusivity
κ₄h = 1e-1 / day * Δx^4   # [m⁴ s⁻¹] biharmonic horizontal viscosity and diffusivity

vertical_diffusivity   = VerticalScalarDiffusivity(ν=κ₂z, κ=κ₂z)
biharmonic_diffusivity = HorizontalScalarBiharmonicDiffusivity(ν=κ₄h, κ=κ₄h)

# ## Model
#
# We build a `NonhydrostaticModel` with a third-order Runge-Kutta timestepper, the buoyancy `b` as the
# active tracer, and the background fields and closures defined above. There is no bottom drag, so the
# vertical boundaries are free-slip.
#
# The advection scheme is `Centered`, not `WENO`: an upwind scheme dissipates energy implicitly, and
# that dissipation appears in neither ``\varepsilon`` nor any other term we compute, so it would show up
# as a residual. With a centered scheme every sink in the two budgets is one we write down, and the
# biharmonic closure handles the grid scale.

model = NonhydrostaticModel(grid;
                            advection = Centered(order=4),
                            timestepper = :RungeKutta3,
                            coriolis = coriolis,
                            tracers = :b,
                            buoyancy = BuoyancyTracer(),
                            background_fields = (b=B_field, u=U_field),
                            closure = (vertical_diffusivity, biharmonic_diffusivity))

# ## Initial condition
#
# We seed the instability with small-amplitude random noise, damped toward the top and bottom
# boundaries so it projects onto interior modes, and then remove any net horizontal-mean velocity the
# noise introduces. The random seed is fixed so the run is reproducible:

using Random
Random.seed!(772)

Ξ(z) = randn() * z / H * (z / H + 1) # noise that vanishes at z = 0 and z = -H

Ũ = 1e-1 * α * H   # velocity-noise amplitude
B̃ = 1e-2 * α * f   # buoyancy-noise amplitude

uᵢ(x, y, z) = Ũ * Ξ(z)
vᵢ(x, y, z) = Ũ * Ξ(z)
bᵢ(x, y, z) = B̃ * Ξ(z)

set!(model, u=uᵢ, v=vᵢ, b=bᵢ)

using Statistics: mean
parent(model.velocities.u) .-= mean(interior(model.velocities.u))
parent(model.velocities.v) .-= mean(interior(model.velocities.v))
nothing #hide

# ## Simulation
#
# The initial time step is set from the most restrictive of the advective and diffusive limits, and a
# `TimeStepWizard` adapts it as the eddies spin up:

max_Δt = min(Δx / Ū, Δx^4 / κ₄h, Δx^2 / κ₂z, 1 / N)

simulation = Simulation(model, Δt = max_Δt, stop_time = 20days)

wizard = TimeStepWizard(cfl=0.7, max_change=1.1, max_Δt=max_Δt)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(5))

# We report progress with a simple messenger:

using Oceanostics.ProgressMessengers: TimedMessenger
add_callback!(simulation, TimedMessenger(), IterationInterval(100))

# ## Energy budgets
#
# The model's `u` and `b` are perturbations about `U` and `B`, so `K = ½uᵢuᵢ` and `eₚ = -bz` are the
# *perturbation* energies, and each background field puts a term of its own into the budget of the
# energy it acts on. Volume integrated over the doubly-periodic, free-slip domain, advection, the
# pressure work and the two transport terms all drop out, and what is left is
#
# ```math
# \frac{d}{dt}\int K\, dV = \int \mathrm{SP}\, dV + \int wb\, dV - \int \varepsilon\, dV , \qquad
# \frac{d}{dt}\int e_p\, dV = \int \mathrm{ADV}_B\, dV - \int wb\, dV + \int \Phi\, dV .
# ```
#
# The kinetic energy picks up the shear production ``\mathrm{SP} = -u_i u_j \partial_j U_i``, which
# here is ``-\alpha\, u w``, and is drained by the dissipation ``\varepsilon``
# ([`KineticEnergyDissipationRate`](@ref Oceanostics.KineticEnergyEquation.DissipationRate)). The potential
# energy picks up ``\mathrm{ADV}_B = z\,\partial_j(u_j B)``, the advection of the background buoyancy by
# the perturbation flow ([`PotentialEnergyBackgroundAdvection`](@ref)), and exchanges with the
# stratification through the diffusive buoyancy flux ``\Phi = \kappa\,\partial b/\partial z``
# ([`PotentialEnergyDiffusiveBuoyancyFlux`](@ref Oceanostics.PotentialEnergyEquation.DiffusiveBuoyancyFlux)). The buoyancy conversion ``wb``
# ([`PotentialToKineticEnergyConversion`](@ref Oceanostics.KineticEnergyEquation.PotentialEnergyConversion)) is the one term the two budgets share, with opposite
# signs, so it cancels from their sum.
#
# ``\mathrm{SP}`` is
# [`TurbulentKineticEnergyShearProductionRate`](@ref) handed the background velocity as the mean flow:
# the model's velocities are already the perturbations, so we pass them straight through rather than
# letting the constructor subtract a mean.

using Oceanostics
using Oceananigans.Fields: ZeroField

u, v, w = model.velocities
U_background = model.background_fields.velocities.u

K  = KineticEnergy(model)
SP = TurbulentKineticEnergyShearProductionRate(u, v, w, U_background, ZeroField(), ZeroField())
wb = PotentialToKineticEnergyConversion(model)
ε  = KineticEnergyDissipationRate(model)

eₚ    = PotentialEnergy(model)
ADV_B = PotentialEnergyBackgroundAdvection(model)
Φ     = PotentialEnergyDiffusiveBuoyancyFlux(model)

∫K     = Integral(K)
∫SP    = Integral(SP)
∫wb    = Integral(wb)
∫ε     = Integral(ε)
∫eₚ    = Integral(eₚ)
∫ADV_B = Integral(ADV_B)
∫Φ     = Integral(Φ)

# For the movie we also keep the surface vertical vorticity `ζ` and the surface buoyancy perturbation:

ζ = ∂x(v) - ∂y(u)
b = model.tracers.b

# ## Output
#
# We use two NetCDF writers: a *snapshot* writer for the surface fields, and a *budget* writer for the
# volume integrals on `ConsecutiveIterations(TimeInterval(8hours))`, which takes a second sample one
# model step after each output time. That lets us finite-difference the integrated energies across the
# step to estimate `d/dt`, as in the [two-dimensional turbulence example](@ref two_d_turbulence_example).

using NCDatasets
filename = joinpath(@__DIR__, "eady_baroclinic_instability")

simulation.output_writers[:fields] =
    NetCDFWriter(model, (; ζ, b),
                 filename = filename,
                 schedule = TimeInterval(8hours),
                 indices = (:, :, grid.Nz),
                 overwrite_existing = true)

simulation.output_writers[:budget] =
    NetCDFWriter(model, (; ∫K, ∫SP, ∫wb, ∫ε, ∫eₚ, ∫ADV_B, ∫Φ),
                 filename = filename * "_budget",
                 schedule = ConsecutiveIterations(TimeInterval(8hours)),
                 overwrite_existing = true)

# ## Run the simulation and process results
#
# To run the simulation:

run!(simulation)

# We read the surface snapshots and the integrated budget scalars back with NCDatasets:

using CairoMakie

ds = NCDataset(simulation.output_writers[:fields].filepath)
times = ds["time"][:]
x_faa = ds["x_faa"][:]; y_afa = ds["y_afa"][:]   # ζ at (Face, Face)
x_caa = ds["x_caa"][:]; y_aca = ds["y_aca"][:]   # b at (Center, Center)
ζ_arr = ds["ζ"][:, :, 1, :]
b_arr = ds["b"][:, :, 1, :]
close(ds)

ζlim = maximum(abs, ζ_arr)
blim = maximum(abs, b_arr)

# The integrated budget scalars come in consecutive-iteration pairs `(2k-1, 2k)`; a one-step finite
# difference inside each pair gives the tendencies, and each source term is evaluated at the pair
# midpoint.

ds_b = NCDataset(simulation.output_writers[:budget].filepath)
t_bud  = ds_b["time"][:]
K_bud  = ds_b["∫K"][:]
SP_bud = ds_b["∫SP"][:]
wb_bud = ds_b["∫wb"][:]
ε_bud  = ds_b["∫ε"][:]
eₚ_bud = ds_b["∫eₚ"][:]
ADV_bud = ds_b["∫ADV_B"][:]
Φ_bud   = ds_b["∫Φ"][:]
close(ds_b)

idx1 = 1:2:length(t_bud) - 1   # primary snapshots
idx2 = 2:2:length(t_bud)       # consecutive-iteration snapshots

Δt_pair = t_bud[idx2] .- t_bud[idx1]
t_pair  = @. 0.5 * (t_bud[idx1] + t_bud[idx2])

dKdt  = (K_bud[idx2]  .- K_bud[idx1])  ./ Δt_pair
deₚdt = (eₚ_bud[idx2] .- eₚ_bud[idx1]) ./ Δt_pair

pair_mean(x) = @. 0.5 * (x[idx1] + x[idx2])

SP_pair  = pair_mean(SP_bud)
wb_pair  = pair_mean(wb_bud)
ε_pair   = pair_mean(ε_bud)
ADV_pair = pair_mean(ADV_bud)
Φ_pair   = pair_mean(Φ_bud)

# Both budgets are written in sum-to-zero form: every curve is plotted with the sign it carries here, so
# the panels below add up to the residual.

K_resid  = @. -dKdt  + SP_pair + wb_pair - ε_pair
eₚ_resid = @. -deₚdt + ADV_pair - wb_pair + Φ_pair

using Test                                                                         #hide
rms(x) = √(sum(abs2, x) / length(x))                                               #hide
## both budgets close to a small fraction of their own tendency ...                #hide
@test rms(K_resid)  < 0.05 * rms(dKdt)                                             #hide
@test rms(eₚ_resid) < 0.01 * rms(deₚdt)                                            #hide
## ... and of `wb`, the largest term in either of them                             #hide
@test rms(K_resid)  < 0.04 * rms(wb_pair)                                          #hide
@test rms(eₚ_resid) < 0.01 * rms(wb_pair)                                          #hide
## `wb` is the same term in both, so it cancels from the sum of the two budgets    #hide
@test rms(K_resid .+ eₚ_resid) < 0.05 * rms(wb_pair)                               #hide
## The background buoyancy term is the point of this example: small next to `wb`, and still twenty  #hide
## times the residual of its own budget, so dropping it leaves a visible imbalance rather than      #hide
## something lost in the truncation error.                                                          #hide
@test rms(@. eₚ_resid + ADV_pair) > 5 * rms(eₚ_resid)                              #hide
## The background *shear* term is a different story at `Ri = 25`: the eddies live off the potential #hide
## energy in the tilted isopycnals, not off the mean shear, so `SP` is a percent of `wb` and below  #hide
## the residual. It is in the budget for completeness, not because it carries the flow.             #hide
@test rms(SP_pair) < 0.05 * rms(wb_pair);                                          #hide

# The instability also does what it should: the eddies grow by orders of magnitude, drawing on the
# potential energy the tilted isopycnals hold.

K_int  = K_bud[idx1]
eₚ_int = eₚ_bud[idx1]

@test K_int[end] > 100 * K_int[1]              # the perturbations grow, and by a lot     #hide
@test mean(wb_pair) > 0                        # fed by the conversion eₚ → K             #hide
@test eₚ_int[end] < 0                          # which drains eₚ well below where it started  #hide
## and the potential energy released exceeds the kinetic energy gained, the rest having gone to    #hide
## dissipation                                                                                     #hide
@test -(eₚ_int[end] - eₚ_int[1]) > K_int[end] - K_int[1]                           #hide
@test all(ε_pair .≥ 0);                        # viscosity only ever removes energy       #hide

# ## Plotting
#
# Every term grows by four orders of magnitude as the instability develops, so each budget is plotted
# normalized by the instantaneous `∫K dV`. That turns the curves into rates per day and keeps the early,
# small-amplitude stage readable next to the saturated one; the residual then reads directly as a
# relative error.

set_theme!(Theme(fontsize = 18))
fig = Figure(size = (1100, 1000))

n = Observable(1)

axζ = Axis(fig[2, 1]; title = "vertical vorticity, ζ", xlabel="x [km]", ylabel="y [km]", aspect=1)
axb = Axis(fig[2, 3]; title = "buoyancy perturbation, b", xlabel="x [km]", ylabel="y [km]", aspect=1)

ζₙ = @lift ζ_arr[:, :, $n]
bₙ = @lift b_arr[:, :, $n]

hmζ = heatmap!(axζ, x_faa ./ 1e3, y_afa ./ 1e3, ζₙ; colormap = :balance, colorrange = (-ζlim, ζlim))
Colorbar(fig[2, 2], hmζ)

hmb = heatmap!(axb, x_caa ./ 1e3, y_aca ./ 1e3, bₙ; colormap = :balance, colorrange = (-blim, blim))
Colorbar(fig[2, 4], hmb)

rate = day ./ pair_mean(K_bud)   # normalization: turns every term into a rate per day
budget_kwargs = (xlabel = "time [days]", ylabel = "[day⁻¹]")

ax_K = Axis(fig[3, 1:4]; title = "Volume-integrated kinetic energy budget, normalized by ∫K dV", budget_kwargs...)
lines!(ax_K, t_pair ./ day, -dKdt    .* rate, label = "-d(∫K)/dt")
lines!(ax_K, t_pair ./ day,  SP_pair .* rate, label = "∫SP dV  (background shear production)")
lines!(ax_K, t_pair ./ day,  wb_pair .* rate, label = "∫wb dV  (buoyancy conversion)")
lines!(ax_K, t_pair ./ day, -ε_pair  .* rate, label = "-∫ε dV  (dissipation)")
lines!(ax_K, t_pair ./ day,  K_resid .* rate, label = "residual", color = :black, linestyle = :dash)
axislegend(ax_K; position = :rt, labelsize = 10, nbanks = 2)

ax_p = Axis(fig[4, 1:4]; title = "Volume-integrated potential energy budget, normalized by ∫K dV", budget_kwargs...)
lines!(ax_p, t_pair ./ day, -deₚdt    .* rate, label = "-d(∫eₚ)/dt")
lines!(ax_p, t_pair ./ day,  ADV_pair .* rate, label = "∫ADV_B dV  (background buoyancy advection)")
lines!(ax_p, t_pair ./ day, -wb_pair  .* rate, label = "-∫wb dV  (buoyancy conversion)")
lines!(ax_p, t_pair ./ day,  Φ_pair   .* rate, label = "∫Φ dV  (diffusive buoyancy flux)")
lines!(ax_p, t_pair ./ day,  eₚ_resid .* rate, label = "residual", color = :black, linestyle = :dash)
axislegend(ax_p; position = :rt, labelsize = 10, nbanks = 2)

vlines!(ax_K, @lift(times[$n] / day), color = :black, linestyle = :dot)
vlines!(ax_p, @lift(times[$n] / day), color = :black, linestyle = :dot)

title = @lift "Eady turbulence, t = " * prettytime(times[$n])
fig[1, 1:4] = Label(fig, title, fontsize = 22, tellwidth = false)

@info "Animating..."
record(fig, "eady_baroclinic_instability.mp4", 1:length(times), framerate = 12) do i
    n[] = i
end
set_theme!() #hide
nothing #hide

# ![](eady_baroclinic_instability.mp4)
#
# The front becomes baroclinically unstable and rolls up into mesoscale eddies, and the two budget
# panels show where their energy comes from. `∫wb dV` runs through both with opposite signs and is the
# largest term in either: the eddies flatten the tilted isopycnals, `∫eₚ dV` falls, and the released
# potential energy shows up as `∫K dV`, with dissipation taking back a sizeable fraction on the way.
# That is the textbook picture of baroclinic instability, and at `Ri = 25` it is essentially the whole
# story: the direct shear production `∫SP dV` is about a percent of the conversion and takes both
# signs, so the eddies live off the potential energy in the tilted isopycnals rather than off the mean
# shear. It is small enough here to sit below the residual, and is kept in the budget for completeness.
#
# The background buoyancy term `∫ADV_B dV` is comparably small, since it is the depth-weighted flow
# across the background gradient rather than a production of anything positive definite. Unlike `∫SP dV`
# it is well clear of the noise: some twenty times the residual of the `eₚ` budget, so leaving it out,
# as one would if the background field were forgotten, turns a budget that closes to seven parts in ten
# thousand into one off by nearly a fifth. `∫Φ dV` is smaller still, which is what a mesoscale run at
# this resolution should look like.
#
# Both residuals stay near zero, and cannot be exactly zero: the discrete `K` and `eₚ` equations are not
# derived from the discrete momentum and buoyancy equations the model steps, so the two sides agree only
# to the truncation error of a well-resolved flow. The same caveat applies to
# [the two-dimensional turbulence example](@ref two_d_turbulence_example).
