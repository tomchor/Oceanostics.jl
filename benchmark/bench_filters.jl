using Oceananigans
using Oceananigans: location, fill_halo_regions!
using Oceananigans.Fields: Field
using CUDA
using Printf
using Statistics: median

using Oceanostics: BoxFilter, GaussianFilter

const ARCH = CUDA.functional() ? GPU() : CPU()

function make_grid(N)
    return RectilinearGrid(ARCH,
                           size = (N, N, N),
                           x = (0, 1), y = (0, 1), z = (0, 1),
                           halo = (8, 8, 8),
                           topology = (Periodic, Periodic, Periodic))
end

function make_field(grid)
    c = CenterField(grid)
    set!(c, (x, y, z) -> sin(2π*x) * cos(2π*y) + 0.1 * z)
    fill_halo_regions!(c)
    return c
end

# Sync inside @elapsed so the recorded time covers launch + execution.
# Report the minimum across `samples` because we are after how fast the GPU
# can do this when nothing else is competing — outliers come from the GC,
# driver, or other concurrent work, none of which we want in the comparison.
function time_compute(target; samples=50)
    compute!(target); CUDA.synchronize()
    compute!(target); CUDA.synchronize()
    times = Float64[]
    for _ in 1:samples
        CUDA.synchronize()
        t = @elapsed begin
            compute!(target)
            CUDA.synchronize()
        end
        push!(times, t)
    end
    return minimum(times), median(times)
end

# A Field whose operand is the raw filter exercises our staged `compute!`
# override (for ≥2D filters). Wrapping the filter inside `1.0 * filter`
# changes the operand to a UnaryOperation so the override doesn't match,
# and the original fused single-kernel evaluation runs instead.
build_staged(filter_kfo) = Field(filter_kfo)
build_fused(filter_kfo)  = Field(1.0 * filter_kfo)

function bench_case(filter_name, make_filter, N, dims, width; samples=50)
    grid = make_grid(N)
    c = make_field(grid)
    filt = make_filter(c, dims, width)
    Ncells = N^3
    N_sten = 2*width + 1

    staged = build_staged(filt)
    fused  = build_fused(filt)

    t_st, _ = time_compute(staged; samples=samples)
    t_fu, _ = time_compute(fused;  samples=samples)

    speedup = t_fu / t_st
    @printf("  %-14s  grid=%-4s  dims=%-9s  N_sten=%-3d   fused=%9.3f ms (%6.2f Gpts/s)   staged=%9.3f ms (%6.2f Gpts/s)   speedup=%5.2fx\n",
            filter_name, string(N)*"³", string(dims), N_sten,
            t_fu*1e3, Ncells/t_fu/1e9, t_st*1e3, Ncells/t_st/1e9, speedup)
    return (; filter_name, N, dims, N_sten, t_fused=t_fu, t_staged=t_st, speedup)
end

# Each filter exposes a `width`-shaped sweep:
# - BoxFilter takes the total stencil width directly via N=2*width+1.
# - GaussianFilter takes σ in physical units; we set σ so that σ_in_cells
#   ≈ width/2, which gives `infer_width` ≈ width and N_sten = 2*width+1.
make_box(c, dims, width)      = BoxFilter(c; dims=dims, N=2*width+1)
make_gauss(c, dims, width) = let grid = c.grid, N = size(grid, 1), σ = (width/2)/N
    GaussianFilter(c; dims=dims, σ=σ, N=2*width+1)
end

println("\n=== Filter benchmark ===  arch=$ARCH")
println("    'fused'  = single-kernel Nᵈ evaluation (forced via `1.0 * filter`)")
println("    'staged' = compute! override that runs d separable 1D passes")
println("    1D filters: both paths use the same kernel; the staged column reflects the @unroll fix.")
println()

println("[Single-direction]")
for (name, mk) in (("BoxFilter", make_box), ("GaussianFilter", make_gauss))
    for N in (128, 256), width in (2, 4, 8, 16)
        bench_case(name, mk, N, (1,), width)
    end
    println()
end

println("[Two-direction (x,y)]")
for (name, mk) in (("BoxFilter", make_box), ("GaussianFilter", make_gauss))
    for N in (128, 256), width in (2, 4, 8)
        bench_case(name, mk, N, (1, 2), width)
    end
    println()
end

println("[Three-direction (x,y,z)]")
for (name, mk) in (("BoxFilter", make_box), ("GaussianFilter", make_gauss))
    for N in (64, 128, 256), width in (2, 4, 8)
        bench_case(name, mk, N, (1, 2, 3), width)
    end
    println()
end
