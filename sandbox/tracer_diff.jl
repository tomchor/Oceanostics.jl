using Oceananigans
using Oceanostics
using Statistics

N = 8
κ = 1

grid = RectilinearGrid(topology=(Periodic, Periodic, Periodic),
                       size=(N,N,N), extent=(1,1,1))
model = NonhydrostaticModel(grid=grid, tracers=:c,
                            closure=ScalarDiffusivity(κ=κ))


# A kind of convoluted way to create x-periodic, resolved initial noise
σx = 2grid.Δxᶜᵃᵃ # x length scale of the noise
σy = 2grid.Δyᵃᶜᵃ # x length scale of the noise
σz = 2grid.Δzᵃᵃᶜ # z length scale of the noise

N = 2^4 # How many Gaussians do we want sprinkled throughout the domain?
x₀ = grid.Lx * rand(N); y₀ = grid.Ly * rand(N); z₀ = -grid.Lz * rand(N) # Locations of the Gaussians

xₚ = x₀ .+ (grid.Lx .* [-2;;-1;;0;;1;;2]) # Make that noise periodic by "infinite" horizontal reflection
yₚ = y₀ .+ (grid.Ly .* [-2;;-1;;0;;1;;2]) # Make that noise periodic by "infinite" horizontal reflection
zₚ = z₀ .+ (grid.Lz .* [-2;;-1;;0;;1;;2]) # Make that noise periodic by "infinite" horizontal reflection

resolved_noise(x, y, z) = sum(@. exp(-(x-xₚ)^2/σx^2 -(y-yₚ)^2/σy^2 -(z-zₚ)^2/σz^2))

set!(model, c=resolved_noise)
c = model.tracers.c

simulation = Simulation(model; Δt=grid.Δxᶜᵃᵃ^2/κ/100, stop_time=0.1)
simulation.callbacks[:progress] = Callback(TimedProgressMessenger(), IterationInterval(10))
wizard = TimeStepWizard(max_change=1.02, diffusive_cfl=0.1)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(4))

χ = Oceanostics.FlowDiagnostics.IsotropicTracerVarianceDissipationRate(model, :c)
ε_q = @at (Center, Center, Center) κ * (∂x(∂x(c^2)) + ∂y(∂y(c^2)) + ∂z(∂z(c^2)))
c2 = c^2

χ_int = Integral(χ, dims=(1,2,3)) # Can't integrate in all directions due to #2857
ε_q_int = Integral(ε_q, dims=(1,2,3)) # Can't integrate in all directions due to #2857
c2_int = Integral(c2, dims=(1,2,3))

outputs = (; χ, χ_int,
           ε_q, ε_q_int,
           c2, c2_int,
           )
simulation.output_writers[:tracer] = NetCDFOutputWriter(model, outputs;
                                                        filename = "tracer_diff.nc",
                                                        schedule = TimeInterval(1e-5),
                                                        overwrite_existing = true)

using Oceanostics: TimedProgressMessenger, SingleLineProgressMessenger

run!(simulation)



@info "Starting to plot video..."
using NCDatasets, CairoMakie
 
xu, yu, zu = nodes(model.velocities.u)
xw, yw, zw = nodes(model.velocities.w)
xB, yB, zB = nodes(B)
 
u_lims = w_lims = (-2.5e-4, +2.5e-4)
 
ds = NCDataset(simulation.output_writers[:fields].filepath, "r")
 
fig = Figure(resolution = (800, 800))
 
axis_kwargs = (xlabel = "Across-slope distance (x)",
               ylabel = "Slope-normal\ndistance (z)",
               limits = ((0, grid.Lx), (0, grid.Lz)),
               )
 
ax_u = Axis(fig[2, 1]; title = "u", axis_kwargs...)
ax_w = Axis(fig[3, 1]; title = "w", axis_kwargs...)
ax_B = Axis(fig[4, 1]; title = "B", axis_kwargs...)
 
n = Observable(1)
 
u = @lift ds["u"][:, 1, :, $n]
hm_u = heatmap!(ax_u, xu, zu, u, colorrange = u_lims, colormap = :balance)
Colorbar(fig[2, 2], hm_u; label = "m s⁻¹")
 
w = @lift ds["w"][:, 1, :, $n]
hm_w = heatmap!(ax_w, xw, zw, w, colorrange = w_lims, colormap = :balance)
Colorbar(fig[3, 2], hm_w; label = "m s⁻¹")
 
B = @lift ds["B"][:, 1, :, $n]
hm_B = contourf!(ax_B, xB, zB, B, colormap = :viridis, levels = 30)
Colorbar(fig[3, 2], hm_w; label = "m s⁻¹")
 
times = collect(ds["time"])
title = @lift "t = " * string(prettytime(times[$n]))
fig[1, :] = Label(fig, title, fontsize=20, tellwidth=false)
 
frames = 1:length(times)
record(fig, "tbbl.mp4", frames, framerate=12) do i
    msg = string("Plotting frame ", i, " of ", frames[end])
    if i%5 == 0 print(msg * " \r") end
    n[] = i
end
close(ds)

