# # [Geostrophic adjustment of a free surface](@id geostrophic_adjustment_example)
#
# In this example a rotating ocean at rest adjusts to a mound in its free surface. Gravity spreads the
# mound, the rotation keeps it from flattening completely, and the ocean settles into a geostrophic eddy
# while inertia-gravity waves carry the rest of the released energy out through open boundaries. All of
# the energy starts in the free surface, so this is the simplest flow whose energetics the free surface
# runs, and its final state is known exactly.
#
# The linear dynamics conserve the potential vorticity of each water column, ``\zeta - f\eta/H``, with
# ``\zeta`` the vertical vorticity (Gill, 1982, chapter 7). The ocean starts at rest and ends
# geostrophic, with ``\zeta = (g/f) \nabla^2 \eta``, so the initial surface ``\eta_i`` sets the final
# one ``\eta_f`` through
#
# ```math
# \eta_f - R^2 \nabla^2 \eta_f = \eta_i ,
# ```
#
# where ``R = \sqrt{gH}/f`` is the deformation radius. We choose the final state, a Gaussian eddy of
# radius ``L``, and derive the initial surface from it:
#
# ```math
# \eta_f = A \, e^{-r^2/2L^2} , \qquad
# \eta_i = A \, e^{-r^2/2L^2} \left(1 + 2\beta - \beta \frac{r^2}{L^2}\right) , \qquad
# \beta = \frac{R^2}{L^2} .
# ```
#
# The eddy turns with the geostrophic velocity
# ``(u, v) = (g/f) \, (-\partial_y \eta_f, \partial_x \eta_f)``, clockwise for ``A > 0``. Its surface
# holds the potential energy ``\tfrac{1}{2} g A^2 \pi L^2``, against
# ``\tfrac{1}{2} g A^2 \pi L^2 (1 + 2\beta + 2\beta^2)`` at the start, so the surface releases
#
# ```math
# \Delta E_s = \int \tfrac{1}{2} g \left(\eta_i^2 - \eta_f^2\right) \mathrm{d}A
#            = g A^2 \pi L^2 \beta (1 + \beta) ,
# ```
#
# and the eddy keeps
#
# ```math
# E_k = \int \tfrac{1}{2} H \left(u^2 + v^2\right) \mathrm{d}A = \tfrac{1}{2} g A^2 \pi L^2 \beta ,
# ```
#
# a fraction ``1/[2(1 + \beta)]`` of it. Here the eddy is as wide as the deformation radius, so
# ``\beta = 1``: the eddy keeps a quarter of the released energy and the waves carry the other three
# quarters away.
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
# A homogeneous ocean 100 m deep on an ``f``-plane, with a single level. The domain is a square ten
# deformation radii wide, centred on the eddy, and its cells are a twelfth of a deformation radius:

H = 100   # [m]   depth
f = 1e-4  # [s⁻¹] Coriolis frequency
A = 0.1   # [m]   height of the final eddy
g = Oceananigans.defaults.gravitational_acceleration

c = √(g * H)   # [m s⁻¹] speed of the surface gravity waves
R = c / f      # [m]     deformation radius
L = R          # [m]     radius of the final eddy
β = R^2 / L^2

grid = RectilinearGrid(size = (120, 120, 1), x = (-5R, 5R), y = (-5R, 5R), z = (-H, 0),
                       topology = (Bounded, Bounded, Bounded))

@info "R = $(round(R/1e3, digits=1)) km, c = $(round(c, digits=1)) m/s, Δx = $(round(grid.Lx/grid.Nx/1e3, digits=1)) km"

# ## Open boundaries
#
# The waves leave through all four sides. A split-explicit free surface steps the depth-integrated
# transport ``(U, V)`` separately from the velocities, and its open boundaries take the condition of
# Flather (1976), `GravityWaveRadiationBoundaryCondition`: on each side the transport follows the
# surface as it does in an outgoing gravity wave, ``U = \sqrt{gH} \, \eta`` on the east side, and no
# wave comes in. The free surface gets the matching radiation condition automatically. The velocities
# ``u`` and ``v`` take `ObliqueRadiation`, the radiation condition of Raymond and Kuo (1984), which acts
# on the part of the flow that varies with depth. This flow has none, so here the barotropic condition
# does all of the work.

using Oceananigans.BoundaryConditions: ObliqueRadiation

U_open = GravityWaveRadiationBoundaryCondition((0, 0))
u_open = NormalFlowBoundaryCondition(0; scheme = ObliqueRadiation())

U_bcs = FieldBoundaryConditions(grid, (Face(), Center(), nothing), west = U_open, east = U_open)
V_bcs = FieldBoundaryConditions(grid, (Center(), Face(), nothing), south = U_open, north = U_open)
u_bcs = FieldBoundaryConditions(west = u_open, east = u_open)
v_bcs = FieldBoundaryConditions(south = u_open, north = u_open)
nothing #hide

# ## Model
#
# These are the linear shallow-water equations: a `HydrostaticFreeSurfaceModel` with no buoyancy, no
# advection of momentum and no closure.

model = HydrostaticFreeSurfaceModel(grid;
                                    coriolis = FPlane(; f),
                                    free_surface = SplitExplicitFreeSurface(grid; substeps = 10),
                                    momentum_advection = nothing,
                                    buoyancy = nothing,
                                    tracers = (),
                                    closure = nothing,
                                    boundary_conditions = (u = u_bcs, v = v_bcs, U = U_bcs, V = V_bcs),
                                    timestepper = :SplitRungeKutta3)

ηᵢ(x, y) = A * exp(-(x^2 + y^2) / 2L^2) * (1 + 2β - β * (x^2 + y^2) / L^2)
set!(model, η = (x, y, z) -> ηᵢ(x, y))

# ## Simulation
#
# The fastest waves reach the boundaries in 14 hours, and four days hold five and a half inertial
# periods. The free surface steps the gravity waves in substeps, so the time step is set by the budgets
# below: their time derivatives are backward differences over one step, which follow the exchange of
# energy, a matter of hours, only if the step is short. We take one minute:

simulation = Simulation(model, Δt = 1minute, stop_time = 4days)

using Oceanostics
add_callback!(simulation, ProgressMessengers.TimedMessenger(), IterationInterval(1000))

# ## The energy budgets
#
# The ocean starts at rest, so all of its energy is the free surface's, `eₛ = ½gη²` per unit area
# ([`FreeSurfacePotentialEnergy`](@ref Oceanostics.PotentialEnergyEquation.FreeSurfacePotentialEnergy)).
# The kinetic energy `eₖ = ½(u² + v²)` changes only through the work of the pressure, which here is the
# surface pressure ``g\eta`` alone, since there is no buoyancy. The Coriolis force does no work, so
#
# ```math
# \frac{d}{dt}\int e_k \, \mathrm{d}V = -\int \texttt{PR} \, \mathrm{d}V , \qquad
# \frac{d}{dt}\int e_s \, \mathrm{d}A = \int \texttt{PR} \, \mathrm{d}V
#     - \oint g H \eta \, u_n \, \mathrm{d}l ,
# ```
#
# with ``\texttt{PR} = u_j \partial_j (g\eta)``
# ([`KineticEnergyPressureRedistribution`](@ref Oceanostics.KineticEnergyEquation.KineticEnergyPressureRedistribution))
# and ``u_n`` the outward velocity on the boundary. Integrated over the domain, the work done on the
# water, ``-\int \texttt{PR} \, \mathrm{d}V``, is the energy the surface gives up,
# ``-\int g\eta \, \partial_t\eta \, \mathrm{d}A``, less the work of the surface pressure on the water
# that leaves through the boundary. That boundary term is the energy the waves carry out, and it needs
# only the flow on the boundary, so we compute it from the output.

eₖ = KineticEnergy(model)
eₛ = FreeSurfacePotentialEnergy(model)
PR = KineticEnergyPressureRedistribution(model)

∫eₖ = ∫dV(eₖ)
∫eₛ = Integral(eₛ, dims=(1, 2))
∫PR = ∫dV(PR)

∂ₜ∫eₖ = TimeDerivative(∫eₖ)
∂ₜ∫eₛ = TimeDerivative(∫eₛ)

# For the movie, the comparison with the exact solution and the boundary term we also keep ``u``, ``v``
# and ``\eta``, averaged over their single vertical level so that none of them carries a vertical
# dimension:

u, v, w = model.velocities
uₛ = Average(u, dims=3)
vₛ = Average(v, dims=3)
ηₛ = Average(model.free_surface.displacement, dims=3)

# ## Output
#
# One writer for the fields and the integrals, every half hour:

using NCDatasets
filename = joinpath(@__DIR__, "geostrophic_adjustment")

simulation.output_writers[:output] =
    NetCDFWriter(model, (; u = uₛ, v = vₛ, η = ηₛ, ∫eₖ, ∫eₛ, ∂ₜ∫eₖ, ∂ₜ∫eₛ, ∫PR),
                 filename = filename,
                 schedule = TimeInterval(30minutes),
                 overwrite_files = true)

# ## Run the simulation

run!(simulation)

# ## Process the results

ds = NCDataset(simulation.output_writers[:output].filepath)
t = ds["time"][:]
x = ds["x_caa"][:]
y = ds["y_aca"][:]
u_arr = ds["u"][:, :, :]
v_arr = ds["v"][:, :, :]
η_arr = ds["η"][:, :, :]
nothing #hide

# The first record of a time derivative has no earlier state to difference against and is written as
# zero, so the budgets start from the second.

nb = 2:length(t)
deₖdt  = ds["∂ₜ∫eₖ"][nb]
deₛdt  = ds["∂ₜ∫eₛ"][nb]
PR_bud = ds["∫PR"][nb]
Eₖ_tot = ds["∫eₖ"][:]
Eₛ_tot = ds["∫eₛ"][:]
close(ds)

# The work of the surface pressure on the water leaving through the four sides, with ``\eta`` from the
# cells along each side and the velocity on the side itself:

Δ = x[2] - x[1]

function boundary_work(η, u, v)
    west  = -sum(η[1, :]   .* u[1, :])
    east  =  sum(η[end, :] .* u[end, :])
    south = -sum(η[:, 1]   .* v[:, 1])
    north =  sum(η[:, end] .* v[:, end])
    return g * H * (west + east + south + north) * Δ
end

F = [boundary_work(η_arr[:, :, n], u_arr[:, :, n], v_arr[:, :, n]) for n in eachindex(t)]
radiated = cumsum([0; (F[2:end] .+ F[1:end-1]) ./ 2 .* diff(t)])
nothing #hide

# The exact final state, and the velocities moved onto the cell centres where ``\eta`` lives:

η_final(x, y) = A * exp(-(x^2 + y^2) / 2L^2)
u_final(x, y) =  g / f * y / L^2 * η_final(x, y)
v_final(x, y) = -g / f * x / L^2 * η_final(x, y)

u_ccc = (u_arr[1:end-1, :, :] .+ u_arr[2:end, :, :]) ./ 2
v_ccc = (v_arr[:, 1:end-1, :] .+ v_arr[:, 2:end, :]) ./ 2
nothing #hide

# The eddy carries inertial oscillations that the waves leave behind, so the comparison uses the flow
# averaged over the last two inertial periods:

using Statistics: mean
T_inertial = 2π / f
window = t .≥ t[end] - 2T_inertial
time_mean(a) = dropdims(mean(a[:, :, window], dims=3), dims=3)

η̄ = time_mean(η_arr)
ū = time_mean(u_ccc)
v̄ = time_mean(v_ccc)

ηf = η_final.(x, y')
uf = u_final.(x, y')
vf = v_final.(x, y')

η_error = maximum(abs, η̄ .- ηf) / A
velocity_error = maximum(hypot.(ū .- uf, v̄ .- vf)) / maximum(hypot.(uf, vf))

ΔEₛ_exact = g * A^2 * π * L^2 * β * (1 + β)
Eₖ_exact  = g * A^2 * π * L^2 * β / 2

ΔEₛ = Eₛ_tot[1] - mean(Eₛ_tot[window])
Eₖ  = mean(Eₖ_tot[window])

@info "The surface released $(round(ΔEₛ / ΔEₛ_exact, digits=3)) of gA²πL²β(1+β)"
@info "The eddy kept $(round(Eₖ / ΔEₛ, digits=3)) of it, against 1/4"
@info "The waves carried $(round(radiated[end] / ΔEₛ, digits=3)) of it out of the domain, against 3/4"
@info "Largest departure from the exact surface: $(round(100η_error, digits=2))% of A"
@info "Largest departure from the exact velocity: $(round(100velocity_error, digits=2))% of the eddy's fastest speed"

# Both budgets in sum-to-zero form, and the energy in the domain, which drops by what the waves carry
# out through the boundaries and by what the time stepping dissipates:

eₖ_resid = @. deₖdt + PR_bud
eₛ_resid = @. deₛdt - PR_bud + F[nb]
E_tot = Eₖ_tot .+ Eₛ_tot
dissipated = E_tot[1] - E_tot[end] - radiated[end]

@info "The time stepping dissipated $(round(100dissipated / ΔEₛ_exact, digits=2))% of the released energy"

# The energy over time, as fractions of the exact release. Kinetic energy swings with the inertial
# oscillations, so we also show its average over the preceding inertial period, and the same for the
# energy the surface has released:

N = round(Int, T_inertial / (t[2] - t[1]))   # records per inertial period
period_mean(a) = [mean(a[n-N+1:n]) for n in N:length(a)]
t_mean = t[N:end]

released = (Eₛ_tot[1] .- Eₛ_tot) ./ ΔEₛ_exact
kinetic  = Eₖ_tot ./ ΔEₛ_exact
nothing #hide

using Test                                                                                     #hide
rms(x) = √(sum(abs2, x) / length(x))                                                           #hide
## The settled flow matches the exact solution, field by field and in its energy               #hide
@test η_error < 0.01                                                                           #hide
@test velocity_error < 0.02                                                                    #hide
@test abs(ΔEₛ / ΔEₛ_exact - 1) < 0.02                                                          #hide
@test abs(Eₖ / Eₖ_exact - 1) < 0.03                                                            #hide
@test abs(radiated[end] / (ΔEₛ_exact - Eₖ_exact) - 1) < 0.03                                   #hide
## Both budgets close to about a percent, and the time stepping dissipates little energy       #hide
@test rms(eₖ_resid) < 0.03 * rms(PR_bud)                                                       #hide
@test rms(eₛ_resid) < 0.03 * rms(PR_bud)                                                       #hide
@test abs(dissipated) < 0.02 * ΔEₛ_exact;                                                      #hide

# ## Plotting

using CairoMakie

set_theme!(Theme(fontsize = 18))
fig = Figure(size = (1000, 1550))

n = Observable(1)

xR = x ./ R
yR = y ./ R
j₀ = searchsortedfirst(y, 0)   # the row of cells just north of the eddy's centre

ηₙ  = @lift η_arr[:, :, $n]
η′ₙ = @lift η_arr[:, :, $n] .- ηf
η₀ₙ = @lift η_arr[:, j₀, $n]
v₀ₙ = @lift v_ccc[:, j₀, $n]

maps = fig[2, 1] = GridLayout()
map_kwargs = (xlabel = "x / R", ylabel = "y / R", aspect = DataAspect())

ax_η = Axis(maps[1, 1]; title = "free surface", map_kwargs...)
hm_η = heatmap!(ax_η, xR, yR, ηₙ, colormap = :balance, colorrange = (-1.5A, 1.5A))
Colorbar(maps[1, 2], hm_η, label = "η [m]")

ax_w = Axis(maps[1, 3]; title = "departure from the final state", map_kwargs...)
hm_w = heatmap!(ax_w, xR, yR, η′ₙ, colormap = :balance, colorrange = (-0.2A, 0.2A))
Colorbar(maps[1, 4], hm_w, label = rich("η - η", subscript("f"), " [m]"))

sections = fig[3, 1] = GridLayout()
section_kwargs = (xlabel = "x / R", limits = ((-5, 5), nothing))

axη = Axis(sections[1, 1]; title = "free surface through the centre", ylabel = "η [m]", section_kwargs...)
lines!(axη, xR, η₀ₙ, label = "model")
lines!(axη, xR, ηf[:, j₀], color = :black, linestyle = :dash, label = "exact final state")
axislegend(axη; position = :rt, labelsize = 14)

axv = Axis(sections[1, 2]; title = "v through the centre", ylabel = "[m s⁻¹]", section_kwargs...)
lines!(axv, xR, v₀ₙ, label = "model")
lines!(axv, xR, vf[:, j₀], color = :black, linestyle = :dash, label = "exact final state")
axislegend(axv; position = :rt, labelsize = 14)

ax_e = Axis(fig[4, 1]; title = "Energy in the domain and through its boundaries",
            xlabel = "time [days]", ylabel = "fraction of gA²πL²β(1+β)")
colors = Makie.wong_colors()
lines!(ax_e, t ./ day, released, color = (colors[1], 0.4), linewidth = 1)
lines!(ax_e, t ./ day, kinetic,  color = (colors[2], 0.4), linewidth = 1)
lines!(ax_e, t_mean ./ day, period_mean(released), color = colors[1], linewidth = 3,
       label = "released by the surface")
lines!(ax_e, t_mean ./ day, period_mean(kinetic), color = colors[2], linewidth = 3,
       label = "kinetic")
lines!(ax_e, t ./ day, radiated ./ ΔEₛ_exact, color = colors[3], linewidth = 3,
       label = "carried out by the waves")
hlines!(ax_e, [1, 1/4, 3/4], color = :black, linestyle = :dash)
axislegend(ax_e; position = :rc, labelsize = 16)

ax_b = Axis(fig[5, 1]; title = "Domain-integrated energy budgets", xlabel = "time [days]", ylabel = "[m⁴ s⁻³]")
lines!(ax_b, t[nb] ./ day, deₖdt,    label = "d(∫eₖ)/dt")
lines!(ax_b, t[nb] ./ day, -PR_bud,  label = "-∫PR dV", linestyle = :dash)
lines!(ax_b, t[nb] ./ day, F[nb],    label = "∮gHηuₙ dl")
lines!(ax_b, t[nb] ./ day, eₖ_resid, label = "kinetic residual", color = :black)
lines!(ax_b, t[nb] ./ day, eₛ_resid, label = "surface residual", color = :gray)
axislegend(ax_b; position = :rt, labelsize = 16, nbanks = 2)

for ax in (ax_e, ax_b)
    vlines!(ax, @lift(t[$n] / day), color = :black, linestyle = :dot)
end

title = @lift "Geostrophic adjustment, t = " * prettytime(t[$n])
fig[1, 1] = Label(fig, title, fontsize = 22, tellwidth = false)

rowsize!(fig.layout, 1, Fixed(50))
resize_to_layout!(fig)

@info "Animating..."
record(fig, "geostrophic_adjustment.mp4", 1:length(t), framerate = 24) do i
    n[] = i
end
set_theme!() #hide
nothing #hide

# ![](geostrophic_adjustment.mp4)
