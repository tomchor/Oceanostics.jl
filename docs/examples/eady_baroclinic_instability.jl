# # Eady baroclinic instability
#
# This example simulates the [Eady problem](https://en.wikipedia.org/wiki/Eady_model): the
# baroclinic instability of a uniformly stratified, uniformly sheared flow held in thermal-wind
# balance on an ``f``-plane. The setup is a port of the classic
# [Eady turbulence example](https://numericalearth.github.io/OceananigansMuseum/v0.74.1/generated/eady_turbulence/)
# to up-to-date Oceananigans syntax, with the bottom drag removed. We then use Oceanostics to close a
# *sub-filter-scale* (coarse-grained) kinetic-energy budget of the developing mesoscale eddies.
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
α = 10 * f      # [s⁻¹] geostrophic shear ∂U/∂z
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
κ₂z = 1e-2                # [m² s⁻¹] Laplacian vertical viscosity and diffusivity
κ₄h = 1e-1 / day * Δx^4   # [m⁴ s⁻¹] biharmonic horizontal viscosity and diffusivity

vertical_diffusivity   = VerticalScalarDiffusivity(ν=κ₂z, κ=κ₂z)
biharmonic_diffusivity = HorizontalScalarBiharmonicDiffusivity(ν=κ₄h, κ=κ₄h)

# ## Model
#
# We build a `NonhydrostaticModel` with fifth-order `WENO` advection, a third-order Runge-Kutta
# timestepper, the buoyancy `b` as the active tracer, and the background fields and closures defined
# above. Following the request, there is no bottom drag, so the vertical boundaries are free-slip.

model = NonhydrostaticModel(grid;
                            advection = WENO(order=5),
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
# noise introduces:

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

# ## Simulation
#
# The initial time step is set from the most restrictive of the advective and diffusive limits, and a
# `TimeStepWizard` adapts it as the eddies spin up:

max_Δt = min(Δx / Ū, Δx^4 / κ₄h, Δx^2 / κ₂z, 1 / N)

simulation = Simulation(model, Δt = max_Δt, stop_time = 16days)

wizard = TimeStepWizard(cfl=0.85, max_change=1.1, max_Δt=max_Δt)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

# We report progress with a simple messenger:

using Printf

start_time = time_ns()
progress(sim) = @printf("i: % 6d, sim time: % 10s, wall time: % 10s, Δt: % 10s, CFL: %.2e\n",
                        sim.model.clock.iteration, prettytime(sim.model.clock.time),
                        prettytime(1e-9 * (time_ns() - start_time)), prettytime(sim.Δt),
                        AdvectiveCFL(sim.Δt)(sim.model))

simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

# ## Sub-filter-scale kinetic-energy budget
#
# As the front rolls up, kinetic energy is passed from the large baroclinic eddies down to smaller
# scales. We quantify that transfer with a *coarse-graining* (filtered-flow) analysis in the spirit of
# [Aluie et al. (2018)](https://doi.org/10.1175/JPO-D-17-0100.1). A low-pass Gaussian filter of width
# `ℓ` splits the flow into a resolved (`> ℓ`) and a sub-filter (`< ℓ`) part, and we build the budget of
# the *sub-filter* kinetic energy ``K^s = \tfrac{1}{2}\,\tau_{ii}``, where
# ``\tau_{ij} = \overline{u_i u_j} - \overline{u}_i\overline{u}_j`` is the sub-filter-scale stress. We
# filter horizontally (`dims = (1, 2)`), since the flow is statistically homogeneous in the horizontal
# and the deformation-scale eddies are the resolved structures.
#
# Volume-integrated, the sub-filter kinetic energy is fed by the cross-scale flux ``\Pi_K`` from the
# resolved scales and drained by sub-filter-scale dissipation ``\varepsilon^s``,
#
# ```math
# \frac{d}{dt} \int K^s\, dV \;\approx\; \int \Pi_K\, dV \;-\; \int \varepsilon^s\, dV ,
# ```
#
# with the cross-scale kinetic-energy flux ``\Pi_K = -\tau^{ij}\overline{S}^{ij}``
# ([`KineticEnergyCrossScaleFlux`](@ref)). The mean geostrophic shear injects energy at large scales
# only, so it does not appear directly in the sub-filter budget. We obtain ``\varepsilon^s`` as the
# budget residual, since the total-viscous-dissipation diagnostic requires a three-dimensional-formulation
# closure and this run uses a vertical Laplacian plus a horizontal biharmonic closure.

using Oceanostics
using Oceananigans.AbstractOperations: @at

ℓ  = 3 * Δx                          # filter scale (full width at half maximum)
σℓ = ℓ / (2 * sqrt(2 * log(2)))      # corresponding Gaussian standard deviation
filt = GaussianFilter(; dims=(1, 2), σ=σℓ, boundary=:shrink)

u, v, w = model.velocities
ū, v̄, w̄ = filt(u), filt(v), filt(w)

Kˡ = @at (Center, Center, Center) (ū^2 + v̄^2 + w̄^2) / 2       # resolved (filtered) kinetic energy
τ  = subfilter_stress_tensor(model, filt; collocate_diagonals=true)
Kˢ = @at (Center, Center, Center) (τ.τ₁₁ + τ.τ₂₂ + τ.τ₃₃) / 2  # sub-filter-scale kinetic energy ½τᵢᵢ
Πₖ = KineticEnergyCrossScaleFlux(model, filt)                  # resolved → sub-filter KE flux
εˡ = CoarseGrainedKineticEnergyDissipationRate(model, filt)    # dissipation of the filtered flow

∫Kˡ = Integral(Kˡ)
∫Kˢ = Integral(Kˢ)
∫Πₖ = Integral(Πₖ)
∫εˡ = Integral(εˡ)

# For the movie we also keep the surface vertical vorticity `ζ` and the cross-scale flux `Πₖ`:

ζ = ∂x(v) - ∂y(u)

# ## Output
#
# We use two NetCDF writers: a *snapshot* writer for the surface fields, and a *budget* writer for the
# volume integrals on `ConsecutiveIterations(TimeInterval(4hours))` — a second sample one model step
# after each output time — which lets us finite-difference `∫Kˢ` across that step to estimate `d/dt`, as
# in the [Kelvin-Helmholtz example](@ref).

using NCDatasets
filename = joinpath(@__DIR__, "eady_baroclinic_instability")

simulation.output_writers[:fields] =
    NetCDFWriter(model, (; ζ, Πₖ),
                 filename = filename,
                 schedule = TimeInterval(4hours),
                 indices = (:, :, grid.Nz),
                 overwrite_existing = true)

simulation.output_writers[:budget] =
    NetCDFWriter(model, (; ∫Kˡ, ∫Kˢ, ∫Πₖ, ∫εˡ),
                 filename = filename * "_budget",
                 schedule = ConsecutiveIterations(TimeInterval(4hours)),
                 overwrite_existing = true)

# ## Run the simulation and process results
#
# To run the simulation:

run!(simulation)

# We read the surface snapshots and the integrated budget scalars back with NCDatasets:

using CairoMakie

ds = NCDataset(simulation.output_writers[:fields].filepath)
times = ds["time"][:]
x_faa = ds["x_faa"][:]; y_afa = ds["y_afa"][:]   # ζ  at (Face, Face)
x_caa = ds["x_caa"][:]; y_aca = ds["y_aca"][:]   # Πₖ at (Center, Center)
ζ_arr = ds["ζ"][:, :, 1, :]
Π_arr = ds["Πₖ"][:, :, 1, :]
close(ds)

ζlim = maximum(abs, ζ_arr)
Πlim = maximum(abs, Π_arr)

# The integrated budget scalars come in consecutive-iteration pairs `(2k-1, 2k)`; a one-step finite
# difference inside each pair gives `d(∫Kˢ)/dt`, and the cross-scale flux is evaluated at the pair
# midpoint. The inferred sub-filter dissipation is the residual `∫Πₖ - d(∫Kˢ)/dt`.

ds_b = NCDataset(simulation.output_writers[:budget].filepath)
tb    = ds_b["time"][:]
∫Kˢ_t = ds_b["∫Kˢ"][:]
∫Πₖ_t = ds_b["∫Πₖ"][:]
close(ds_b)

i1 = 1:2:length(tb)-1
i2 = 2:2:length(tb)
Δtp = tb[i2] .- tb[i1]
tp  = @. 0.5 * (tb[i1] + tb[i2])

dKˢdt = (∫Kˢ_t[i2] .- ∫Kˢ_t[i1]) ./ Δtp
Πₖ_p  = @. 0.5 * (∫Πₖ_t[i1] + ∫Πₖ_t[i2])
ε_sfs = Πₖ_p .- dKˢdt                     # inferred sub-filter-scale dissipation

# We now build the figure: surface maps of `ζ` and `Πₖ` on top, and the sub-filter-scale kinetic-energy
# budget below.

set_theme!(Theme(fontsize = 18))
fig = Figure(size = (1100, 850))

n = Observable(1)

axζ = Axis(fig[2, 1]; title = "vertical vorticity, ζ", xlabel="x [km]", ylabel="y [km]", aspect=1)
axΠ = Axis(fig[2, 3]; title = "cross-scale KE flux, Πₖ", xlabel="x [km]", ylabel="y [km]", aspect=1)

ζₙ = @lift ζ_arr[:, :, $n]
Πₙ = @lift Π_arr[:, :, $n]

hmζ = heatmap!(axζ, x_faa ./ 1e3, y_afa ./ 1e3, ζₙ; colormap = :balance, colorrange = (-ζlim, ζlim))
Colorbar(fig[2, 2], hmζ)

hmΠ = heatmap!(axΠ, x_caa ./ 1e3, y_aca ./ 1e3, Πₙ; colormap = :balance, colorrange = (-Πlim, Πlim))
Colorbar(fig[2, 4], hmΠ)

ax_bud = Axis(fig[3, 1:4]; xlabel="time [days]", ylabel="[m⁵ s⁻³]",
              title="Sub-filter-scale KE budget (ℓ = $(round(Int, ℓ/1e3)) km)")
lines!(ax_bud, tp ./ day, dKˢdt, label="d(∫Kˢ)/dt")
lines!(ax_bud, tp ./ day, Πₖ_p,  label="∫Πₖ dV  (resolved → sub-filter flux)")
lines!(ax_bud, tp ./ day, ε_sfs, label="∫Πₖ − d(∫Kˢ)/dt  (inferred sub-filter dissipation)",
       color=:black, linestyle=:dash)
axislegend(ax_bud; position=:lt, labelsize=11)

vlines!(ax_bud, @lift(times[$n] / day), color=:black, linestyle=:dash)

title = @lift "Eady turbulence, t = " * prettytime(times[$n])
fig[1, 1:4] = Label(fig, title, fontsize=22, tellwidth=false)

@info "Animating..."
record(fig, "eady_baroclinic_instability.mp4", 1:length(times), framerate=12) do i
    n[] = i
end

# ![](eady_baroclinic_instability.mp4)
#
# As the front becomes baroclinically unstable it rolls up into mesoscale eddies (left), and the
# cross-scale flux `Πₖ` (right) marks where kinetic energy is passed across the filter scale: mostly
# forward (downscale, `Πₖ > 0`) along the sharpening eddy edges, with patches of backscatter
# (`Πₖ < 0`) elsewhere. The bottom panel shows the volume-integrated sub-filter-scale budget: the
# cross-scale flux `∫Πₖ dV` builds up as the eddies grow, feeding the sub-filter kinetic energy
# `d(∫Kˢ)/dt`, with the residual giving the sub-filter-scale dissipation that removes it.
