# Build one page of the docs, over and over, from a REPL:
#
#     julia --project=docs
#     julia> include("docs/build_single_page.jl")   # re-include after every edit to the page
#
# `PAGE` picks which page, and `DOCTEST` runs the `jldoctest` blocks written on it (off by default,
# since the prose and the math are what usually need iterating on; the doctests inside docstrings
# belong to the module and only run in the full build):
#
#     julia> PAGE = "filtered_available_potential_energy_equation.md"; include("docs/build_single_page.jl")
#     julia> DOCTEST = true; include("docs/build_single_page.jl")
#
# Both stay set for the rest of the session. The full docs are still built with `docs/make.jl`; this
# is a fast loop for one page, not a substitute for it.

#+++ Environment (only the first `include` of a session pays for this)
using Pkg
if !@isdefined(SINGLE_PAGE_ENVIRONMENT_READY)
    Base.active_project() == joinpath(@__DIR__, "Project.toml") || Pkg.activate(@__DIR__)

    # `Oceanostics` comes from the parent environment, as in `make.jl`.
    parent_environment = normpath(joinpath(@__DIR__, ".."))
    parent_environment in LOAD_PATH || pushfirst!(LOAD_PATH, parent_environment)

    try
        Pkg.instantiate()
    catch # a manifest older than `Project.toml` can be missing a dependency it has since gained
        Pkg.resolve()
        Pkg.instantiate()
    end

    SINGLE_PAGE_ENVIRONMENT_READY = true
end

using Documenter
using DocumenterCodeBlocks: CodeBlocks
using Oceananigans
using Oceanostics
#---

#+++ Stage a source directory holding only this page
# `makedocs` expands *every* `.md` it finds under `source`, so aiming it at `docs/src` would drag in
# the example pages and every other module page on each rebuild. Giving the one page a source
# directory of its own keeps a rebuild to a few seconds. The copy is refreshed on every include, so
# it always renders whatever is currently on disk.
page = @isdefined(PAGE) ? PAGE : "available_potential_energy_equation.md"
page_path = joinpath(@__DIR__, "src", page)
isfile(page_path) || error("$(page_path) does not exist")

staging = joinpath(@__DIR__, "build_single_page") # `docs/build_*` is gitignored
source  = joinpath(staging, "src")
build   = joinpath(staging, "html")

# Staged as `index.md` so it *is* the landing page: Documenter warns about a site without one, and
# a browser (or `LiveServer.serve`) then lands on the page with no path to type.
rm(source, recursive=true, force=true)
mkpath(source)
cp(page_path, joinpath(source, "index.md"))

# The page's own first heading is all the navigation menu needs.
heading = match(r"^#[ \t]+(.+)$"m, read(page_path, String))
title = heading === nothing ? page : String(strip(heading.captures[1]))
#---

#+++ Build
first_build = !isdir(build)

makedocs(sitename = "Oceanostics.jl",
         root = @__DIR__,
         source = source,
         build = build,
         pages = [title => "index.md"],
         doctest = @isdefined(DOCTEST) ? DOCTEST : false,
         doctestfilters = [r"with \d+ methods?"], # as in `make.jl`: method counts drift with Oceananigans
         clean = true,
         format = Documenter.HTML(collapselevel = 1, prettyurls = false, mathengine = MathJax3()),
         plugins = [CodeBlocks()],
         checkdocs = :none,
         # Links into the pages this build leaves out cannot resolve, so cross-references warn
         # instead of failing. Everything else (a malformed `@docs` block, a failing doctest) still
         # errors, which is the point of building at all.
         warnonly = [:cross_references],
         )

html = joinpath(build, "index.html")
@info "Built $(page)" html

# Open a tab on the first build; after that the tab only needs a refresh. `using LiveServer;
# serve(dir=build)` in a second REPL does the refreshing for you (`build` stays defined out here).
if first_build
    try
        run(pipeline(`xdg-open $(html)`, stdout=devnull, stderr=devnull), wait=false)
    catch
        @info "Could not open a browser; point one at the path above"
    end
end
#---
