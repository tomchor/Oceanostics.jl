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

simulation = Simulation(model, Δt = max_Δt, stop_time = 10days)

wizard = TimeStepWizard(cfl=0.85, max_change=1.1, max_Δt=max_Δt)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(5))

# We report progress with a simple messenger:

using Oceanostics.ProgressMessengers: TimedMessenger
add_callback!(simulation, TimedMessenger(), IterationInterval(20))

# ## Coarse-grained kinetic-energy budget
#
# As the front rolls up, kinetic energy moves between scales. We follow it with a *coarse-graining*
# (filtered-flow) analysis in the spirit of [Aluie et al. (2018)](https://doi.org/10.1175/JPO-D-17-0100.1),
# closing the volume-integrated budget of the filtered kinetic energy
# ``\overline{K} = \tfrac{1}{2}\,\overline{u}_i\overline{u}_i``. A low-pass Gaussian filter of width `ℓ`
# defines the resolved (`> ℓ`) scales; we apply it horizontally (`dims = (1, 2)`), since the flow is
# statistically homogeneous in the horizontal.
#
# Because the mean geostrophic flow `U(z) = α(z + H)` is imposed as a `BackgroundField`, it is not part
# of the resolved velocity, so its shear `∂U/∂z = α` enters the budget as an explicit production term.
# Volume-integrated (advection and pressure work integrate to zero on the doubly-periodic, free-slip
# domain), the budget reads
#
# ```math
# \frac{d}{dt} \int \overline{K}\, dV
#   = -α \int \overline{u}\,\overline{w}\, dV
#   + \int \overline{w}\,\overline{b}\, dV
#   - \int \Pi_K\, dV
#   - \int \overline{\varepsilon}\, dV ,
# ```
#
# with a production ``-α\,\overline{u}\,\overline{w}`` by the background shear, a buoyancy production
# ``\overline{w}\,\overline{b}`` (conversion between filtered kinetic and potential energy), the
# cross-scale kinetic-energy flux ``\Pi_K = -\tau^{ij}\overline{S}^{ij}`` to sub-filter scales
# ([`KineticEnergyCrossScaleFlux`](@ref)), and the coarse-grained dissipation ``\overline{\varepsilon}``
# ([`CoarseGrainedKineticEnergyDissipationRate`](@ref)).

using Oceanostics
using Oceananigans.AbstractOperations: @at

ℓ  = 3 * Δx                          # filter scale (full width at half maximum)
σℓ = ℓ / (2 * sqrt(2 * log(2)))      # corresponding Gaussian standard deviation
filt = GaussianFilter(; dims=(1, 2), σ=σℓ, boundary=:shrink)

u, v, w = model.velocities
b = model.tracers.b
ū, v̄, w̄, b̄ = filt(u), filt(v), filt(w), filt(b)

Kˡ = @at (Center, Center, Center) (ū^2 + v̄^2 + w̄^2) / 2   # filtered (coarse-grained) kinetic energy
uw = @at (Center, Center, Center) (ū * w̄)                 # resolved flux of along-front momentum, ū w̄
Pu = -α * uw                                              # production by the background shear α = ∂U/∂z
wb = @at (Center, Center, Center) (w̄ * b̄)                 # buoyancy production
Πₖ = KineticEnergyCrossScaleFlux(model, filt)             # cross-scale flux to sub-filter scales
εˡ = CoarseGrainedKineticEnergyDissipationRate(model, filt)  # coarse-grained dissipation

∫Kˡ = Integral(Kˡ)
∫Pu = Integral(Pu)
∫wb = Integral(wb)
∫Πₖ = Integral(Πₖ)
∫εˡ = Integral(εˡ)

# For the movie we also keep the surface vertical vorticity `ζ` and the cross-scale flux `Πₖ`:

ζ = ∂x(v) - ∂y(u)

# ## Output
#
# We use two NetCDF writers: a *snapshot* writer for the surface fields, and a *budget* writer for the
# volume integrals on `ConsecutiveIterations(TimeInterval(4hours))`, which takes a second sample one
# model step after each output time. That lets us finite-difference `∫K̄` across the step to estimate
# `d/dt`, as in the [Kelvin-Helmholtz example](@ref kelvin_helmholtz_example).

using NCDatasets
filename = joinpath(@__DIR__, "eady_baroclinic_instability")

simulation.output_writers[:fields] =
    NetCDFWriter(model, (; ζ, Πₖ),
                 filename = filename,
                 schedule = TimeInterval(4hours),
                 indices = (:, :, grid.Nz),
                 overwrite_existing = true)

simulation.output_writers[:budget] =
    NetCDFWriter(model, (; ∫Kˡ, ∫Pu, ∫wb, ∫Πₖ, ∫εˡ),
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
# difference inside each pair gives `d(∫K̄)/dt`, and each source term is evaluated at the pair midpoint.
# The residual measures how well the coarse-grained budget closes.

ds_b = NCDataset(simulation.output_writers[:budget].filepath)
tb    = ds_b["time"][:]
∫Kˡ_t = ds_b["∫Kˡ"][:]
∫Pu_t = ds_b["∫Pu"][:]
∫wb_t = ds_b["∫wb"][:]
∫Πₖ_t = ds_b["∫Πₖ"][:]
∫εˡ_t = ds_b["∫εˡ"][:]
close(ds_b)

i1 = 1:2:length(tb)-1
i2 = 2:2:length(tb)
Δtp = tb[i2] .- tb[i1]
tp  = @. 0.5 * (tb[i1] + tb[i2])

dKdt = (∫Kˡ_t[i2] .- ∫Kˡ_t[i1]) ./ Δtp
Pu_p = @. 0.5 * (∫Pu_t[i1] + ∫Pu_t[i2])
wb_p = @. 0.5 * (∫wb_t[i1] + ∫wb_t[i2])
Πₖ_p = @. 0.5 * (∫Πₖ_t[i1] + ∫Πₖ_t[i2])
εˡ_p = @. 0.5 * (∫εˡ_t[i1] + ∫εˡ_t[i2])
resid = dKdt .- (Pu_p .+ wb_p .- Πₖ_p .- εˡ_p)

# We now build the figure: surface maps of `ζ` and `Πₖ` on top, and the coarse-grained kinetic-energy
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
              title="Coarse-grained KE budget (ℓ = $(round(Int, ℓ/1e3)) km)")
lines!(ax_bud, tp ./ day, dKdt,   label="d(∫K̄)/dt")
lines!(ax_bud, tp ./ day, Pu_p,   label="−α∫ū w̄ dV  (background shear production)")
lines!(ax_bud, tp ./ day, wb_p,   label="∫w̄ b̄ dV  (buoyancy production)")
lines!(ax_bud, tp ./ day, -Πₖ_p,  label="−∫Πₖ dV  (flux to sub-filter scales)")
lines!(ax_bud, tp ./ day, -εˡ_p,  label="−∫ε̄ dV  (dissipation)")
lines!(ax_bud, tp ./ day, resid,  label="residual", color=:black, linestyle=:dash)
axislegend(ax_bud; position=:lt, labelsize=10)

vlines!(ax_bud, @lift(times[$n] / day), color=:black, linestyle=:dash)

title = @lift "Eady turbulence, t = " * prettytime(times[$n])
fig[1, 1:4] = Label(fig, title, fontsize=22, tellwidth=false)

@info "Animating..."
record(fig, "eady_baroclinic_instability.mp4", 1:length(times), framerate=12) do i
    n[] = i
end

# ![](eady_baroclinic_instability.mp4)
#
# As the front becomes baroclinically unstable it rolls up into mesoscale eddies (left), while the
# cross-scale flux `Πₖ` (right) marks where kinetic energy crosses the filter scale: mostly forward
# (downscale, `Πₖ > 0`) along the sharpening eddy edges, with patches of backscatter (`Πₖ < 0`). The
# bottom panel shows the volume-integrated coarse-grained kinetic-energy budget: the background-shear
# and buoyancy productions feed the filtered kinetic energy, part of it passes to the sub-filter scales
# through `∫Πₖ dV` and is dissipated, and the small residual shows how well the budget closes.
