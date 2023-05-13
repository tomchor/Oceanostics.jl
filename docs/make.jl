pushfirst!(LOAD_PATH, joinpath(@__DIR__, "..")) # add Oceanostics environment to docs
using Pkg
CI = get(ENV, "CI", nothing) == "true" || get(ENV, "GITHUB_TOKEN", nothing) !== nothing
CI && Pkg.instantiate()

using Documenter
using Literate

using Oceananigans
using Oceanostics

#+++ Run examples
EXAMPLES_DIR = joinpath(@__DIR__, "examples")
OUTPUT_DIR   = joinpath(@__DIR__, "src/generated")

examples = ["Two-dimensional turbulence"   => "two_dimensional_turbulence",
            "Kelvin-Helmholtz instability" => "kelvin_helmholtz",
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


format = Documenter.HTML(collapselevel = 1,
                         prettyurls = CI, # Makes links work when building locally
                         mathengine = MathJax3(),
                         warn_outdated = true,
                         )
#---

#+++ Make and deploy docs
makedocs(sitename = "Oceanostics.jl",
         authors = "Tomas Chor and contributors",
         pages = pages,
         modules = [Oceanostics],
         doctest = true,
         strict = :doctest,
         clean = true,
         format = format,
         checkdocs = :exports
         )

if CI
    deploydocs(repo = "github.com/tomchor/Oceanostics.jl.git",
               push_preview = true,
               )
end
#---
