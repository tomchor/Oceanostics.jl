pushfirst!(LOAD_PATH, joinpath(@__DIR__, "..")) # add Oceanostics to environment
using Documenter, Example

using Oceananigans
using Oceanostics


pages = ["Home" => "index.md",]

makedocs(sitename = "Oceanostics.jl",
         authors = "Tomas Chor and contributors",
         pages = pages,
         modules = [Oceanostics],
         )
