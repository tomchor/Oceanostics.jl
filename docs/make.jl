using Documenter
using Literate
using Glob

using Oceananigans
using Oceanostics

#+++ Run examples
EXAMPLES_DIR = joinpath(@__DIR__, "examples")
OUTPUT_DIR   = joinpath(@__DIR__, "src/generated")

examples = ["Two-dimensional turbulence"   => "two_dimensional_turbulence",
            "Kelvin-Helmholtz instability" => "kelvin_helmholtz",
            "Tilted bottom boundary layer" => "tilted_bottom_boundary_layer",
            ]

example_codes = [ v * ".jl" for (k, v) in examples ]
example_pages = [ k => "generated/$v.md" for (k, v) in examples ]

for example in example_codes
    example_filepath = joinpath(EXAMPLES_DIR, example)
    Literate.markdown(example_filepath, OUTPUT_DIR; flavor = Literate.DocumenterFlavor())
end
#---


#+++ Organize pages and HTML format
pages = ["Home" => "index.md",
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
         strict = :doctest,
         clean = true,
         forcepush = true,
         format = format,
         checkdocs = :exports
         )
#---

#+++ Cleanup any output files, e.g., .jld2 or .nc, created by docs. Otherwise they are pushed up in the docs branch in the repo
for file in vcat(glob("docs/*.jld2"), glob("docs/*.nc"), glob("docs/generated/*.nc"), glob("docs/generated/*.nc"))
    rm(file)
end
#---

#+++ Deploy thedocs
if CI
    deploydocs(repo = "github.com/tomchor/Oceanostics.jl.git",
               push_preview = true,
               )
end
#---
