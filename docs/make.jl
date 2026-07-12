pushfirst!(LOAD_PATH, joinpath(@__DIR__, "..")) # add Oceanostics environment
using Pkg; Pkg.instantiate()

using Documenter
using Literate

using Oceananigans
using Oceanostics

#+++ Examples
EXAMPLES_DIR = joinpath(@__DIR__, "examples")
OUTPUT_DIR   = joinpath(@__DIR__, "src/generated")

examples = ["Two-dimensional turbulence"   => "two_dimensional_turbulence",
            "Kelvin-Helmholtz instability" => "kelvin_helmholtz",
            "Rayleigh-Taylor instability"  => "rayleigh_taylor_instability",
            "Tilted bottom boundary layer" => "tilted_bottom_boundary_layer",
            "Spatial filtering"            => "spatial_filtering",
            ]

example_pages = [ k => "generated/$v.md" for (k, v) in examples ]

"""
    generate_example(slug)

Run `examples/\$slug.jl` through Literate with `execute = true`. This runs the example
here (simulation, figures, movie, and the hidden budget-closure `@test`s) and bakes the
outputs into the generated Markdown, so Documenter includes the page without re-executing
it. Because execution no longer happens inside `makedocs`, the examples can be built one
per CI job in parallel instead of serially. `#hide` lines are still executed but omitted
from the rendered code, exactly as under the previous Documenter-executed `@example` path.
"""
generate_example(slug) =
    Literate.markdown(joinpath(EXAMPLES_DIR, slug * ".jl"), OUTPUT_DIR;
                      execute = true, flavor = Literate.DocumenterFlavor())

# Single-example mode: when OCEANOSTICS_DOCS_EXAMPLE names one example, build only that
# page and stop. This is what each parallel CI job runs; the resulting page and media are
# uploaded as an artifact and later collected by the assemble/`makedocs` job.
single_example = get(ENV, "OCEANOSTICS_DOCS_EXAMPLE", "")
if single_example != ""
    @info "Building single example: $single_example"
    generate_example(single_example)
    exit(0)
end

# Assemble mode: generate any example page that isn't already present, then build the site.
# A plain `julia docs/make.jl` (nothing pre-generated) builds every example serially, so
# local builds keep working; in CI the matrix jobs pre-generate the pages and drop them in
# `OUTPUT_DIR`, so these are skipped.
for (_, slug) in examples
    if isfile(joinpath(OUTPUT_DIR, slug * ".md"))
        @info "Reusing pre-built example page: $slug"
    else
        @info "Building example: $slug"
        generate_example(slug)
    end
end
#---


#+++ Organize pages and HTML format
pages = ["Home" => "index.md",
         "Budget equations" => ["Tracer equation"                        => "tracer_equation.md",
                                "Momentum equation"                      => "momentum_equation.md",
                                "Kinetic energy equation"                => "kinetic_energy_equation.md",
                                "Coarse-grained kinetic energy equation" => "coarse_grained_kinetic_energy_equation.md",
                                "Turbulent kinetic energy equation"      => "turbulent_kinetic_energy_equation.md",
                                "Tracer variance equation"               => "tracer_variance_equation.md",
                                "Potential energy equation"              => "potential_energy_equation.md",
                                ],
         "Flow diagnostics" => "flow_diagnostics.md",
         "Progress messengers" => "progress_messengers.md",
         "Spatial filters" => "filters.md",
         "Examples" => example_pages,
         "Function library" => "library.md",
        ]

CI = get(ENV, "CI", nothing) == "true"

format = Documenter.HTML(collapselevel = 1,
                         prettyurls = CI, # Makes links work when building locally
                         mathengine = MathJax3(),
                         warn_outdated = true,
                         )
#---

#+++ Make the docs
makedocs(sitename = "Oceanostics.jl",
         authors = "Tomas Chor and contributors",
         pages = pages,
         modules = [Oceanostics],
         doctest = true,
         clean = true,
         format = format,
         checkdocs = :none,
         doctestfilters = [r"with \d+ methods?"], # method count drifts with Oceananigans versions
         )
#---

#+++ Cleanup any output files, e.g., .jld2 or .nc, created by docs. Otherwise they are pushed up in the docs branch in the repo
"""
    recursive_find(directory, pattern)

Return list of filepaths within `directory` that contains the `pattern::Regex`.
"""
recursive_find(directory, pattern) =
    mapreduce(vcat, walkdir(directory)) do (root, dirs, files)
        joinpath.(root, filter(contains(pattern), files))
    end

files = []
for pattern in [r"\.jld2", r"\.nc"]
    global files = vcat(files, recursive_find(@__DIR__, pattern))
end

for file in files
    rm(file)
end
#---

#+++ Deploy thedocs
if CI
    deploydocs(repo = "github.com/tomchor/Oceanostics.jl.git",
               versions = ["stable" => "v^", "v#.#.#", "dev" => "dev"],
               devbranch = "main",
               forcepush = true,
               push_preview = false,
               branch_previews = "doc-previews",
               )
end
#---
