pushfirst!(LOAD_PATH, joinpath(@__DIR__, "..")) # add Oceanostics environment
using Pkg; Pkg.instantiate()

using Base64

using Documenter
using DocumenterCodeBlocks: CodeBlocks
using Literate

using Oceananigans
using Oceanostics

#+++ Examples
EXAMPLES_DIR = joinpath(@__DIR__, "examples")
OUTPUT_DIR   = joinpath(@__DIR__, "src/generated")

examples = [
    "Two-dimensional turbulence"   => "two_dimensional_turbulence", # KE and tracer variance budgets (~8 min)
    "Baroclinic adjustment"        => "baroclinic_adjustment", # PE and KE budgets, double front (~20 min)
    "Spatial filtering"            => "spatial_filtering", # Spatial filtering examples (~5 min)
    "Kelvin-Helmholtz instability" => "kelvin_helmholtz", # Filtered KE budget (~8 min)
    "Rayleigh-Taylor instability"  => "rayleigh_taylor_instability", # SFS budget (~6 min)
    "Lock release"                 => "lock_release", # APE and reference profiles calculation (~ 5 min)
    ]

example_pages = [ k => "generated/$v.md" for (k, v) in examples ]

"""
    externalize_images(name)

Return a Literate `postprocess` function that moves inlined figures out of the page. For
Makie figures, Literate's `DocumenterFlavor` embeds the `text/html` representation, which
is a base64 PNG inside an `<img>` tag, directly into the Markdown via `@raw html`. A page
with several figures then blows past Documenter's `size_threshold`. This rewrites each such
block into a standalone `\$(name)-figN.png` file (next to the page, so it travels with the
artifact) referenced with `![](...)`, matching the lean pages Documenter produced when it
executed the `@example` blocks itself.
"""
function externalize_images(name)
    return function (content::AbstractString)
        counter = Ref(0)
        pattern = r"```@raw html\s*\n<img[^>]*src=\"data:image/png;base64,\s*([A-Za-z0-9+/=]+)\"[^>]*>\s*\n```"
        return replace(content, pattern => function (block)
            b64 = match(pattern, block).captures[1]
            counter[] += 1
            file = "$(name)-fig$(counter[]).png"
            write(joinpath(OUTPUT_DIR, file), base64decode(b64))
            return "![]($(file))"
        end)
    end
end

"""
    colorize_ansi_output(content)

Retag Literate's executed-output blocks so Documenter renders their ANSI escape sequences as
colors. `Literate.markdown(..., execute = true)` writes each block's captured stdout/stderr into
an info-less ```` ```` ```` fence, and Documenter's HTML writer only runs a code block through
its ANSI-to-HTML printer when the block is tagged `documenter-ansi`; otherwise the escapes are
emitted verbatim and the `ProgressMessengers` output reads as literal `^[[93m` (issue #284).
Under the previous Documenter-executed `@example` path this came for free, because Documenter
colorized the output it had captured itself.

Prose fences carry three backticks and Literate's own code chunks carry a `julia` info string,
so an info-less fence of four or more backticks is unambiguously executed output.
"""
function colorize_ansi_output(content::AbstractString)
    lines = String.(split(content, '\n'))
    fence = 0 # backtick count that opened the block we are inside; 0 when outside a block
    for (i, line) in enumerate(lines)
        m = match(r"^(`{3,})(.*)$", line)
        m === nothing && continue
        ticks, info = length(m[1]), strip(m[2])
        if fence == 0
            fence = ticks
            isempty(info) && ticks >= 4 && (lines[i] = m[1] * "documenter-ansi")
        elseif ticks >= fence && isempty(info)
            fence = 0
        end
    end
    return join(lines, '\n')
end

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
                      execute = true, flavor = Literate.DocumenterFlavor(),
                      postprocess = colorize_ansi_output ∘ externalize_images(slug))

# Single-example mode: when OCEANOSTICS_DOCS_EXAMPLE names one example, build only that
# page and stop. This is what each parallel CI job runs; the resulting page and media are
# uploaded as an artifact and later collected by the assemble/`makedocs` job.
single_example = get(ENV, "OCEANOSTICS_DOCS_EXAMPLE", "")
if single_example != ""
    single_example in last.(examples) ||
        error("OCEANOSTICS_DOCS_EXAMPLE = \"$single_example\" is not a known example; " *
              "expected one of $(last.(examples))")
    @info "Building single example: $single_example"
    generate_example(single_example)
    exit(0)
end

# Reuse-vs-regenerate policy for the example pages:
#  - CI assemble job (OCEANOSTICS_DOCS_ASSEMBLE=true): the pages were built by the parallel
#    example jobs and their media downloaded into `OUTPUT_DIR`. Reuse them exactly as-is and
#    never run a simulation here (that would be slow and at -O0). Error loudly if one is
#    missing, since that means a broken or expired artifact, not "build it now". This keeps
#    the CI path from depending on artifact-vs-checkout mtime ordering.
#  - Local build (no flag): reuse a page only if it is newer than its `.jl` source, else
#    regenerate it. Delete `OUTPUT_DIR` (or edit the example) to force a rebuild.
reuse_pages = get(ENV, "OCEANOSTICS_DOCS_ASSEMBLE", "") == "true"
for (_, slug) in examples
    page   = joinpath(OUTPUT_DIR, slug * ".md")
    source = joinpath(EXAMPLES_DIR, slug * ".jl")
    if reuse_pages
        isfile(page) || error("Assemble mode: pre-built page $(page) is missing; " *
                              "expected it from the example jobs' artifacts.")
        @info "Reusing pre-built example page: $slug"
    elseif isfile(page) && mtime(page) >= mtime(source)
        @info "Reusing up-to-date example page: $slug"
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
                                "Filtered kinetic energy equation"       => "filtered_kinetic_energy_equation.md",
                                "Sub-filter kinetic energy equation"     => "subfilter_kinetic_energy_equation.md",
                                "Turbulent kinetic energy equation"      => "turbulent_kinetic_energy_equation.md",
                                "Tracer variance equation"               => "tracer_variance_equation.md",
                                "Potential energy equation"              => "potential_energy_equation.md",
                                "Background potential energy equation"   => "background_potential_energy_equation.md",
                                "Available potential energy equation"    => "available_potential_energy_equation.md",
                                ],
         "Flow diagnostics" => "flow_diagnostics.md",
         "Progress messengers" => "progress_messengers.md",
         "Spatial filters" => "filters.md",
         "Examples" => example_pages,
         "Validation" => ["Sorted reference state" => "validation/reference_state.md"],
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
# `CodeBlocks()` tokenizes Julia code blocks at build time with JuliaSyntax instead of leaving
# them to highlight.js in the browser. That buys a real parse (so function calls, types and
# macros get their own colors), line numbers with linkable gutters, and hover popups linking
# identifiers to their docstrings. It only touches `julia`/`julia-repl`/`jldoctest` blocks, so
# the executed-output blocks retagged by `colorize_ansi_output` above are left alone.
makedocs(sitename = "Oceanostics.jl",
         authors = "Tomas Chor and contributors",
         pages = pages,
         modules = [Oceanostics],
         doctest = true,
         clean = true,
         format = format,
         plugins = [CodeBlocks()],
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
