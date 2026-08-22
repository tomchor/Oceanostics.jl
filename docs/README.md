# Documentation

All files for the docs are here. Docs are built using
[Documenter.jl](https://documenter.juliadocs.org/stable/) and a quick intro to how to build them can
be found [here](https://documenter.juliadocs.org/stable/man/guide/).

As per `make.jl`, docs are deployed to the `gh-pages` of this repo.

To build the docs locally, from the main directory of a local clone of the repository run


```
julia --color=yes --project -e 'using Pkg; Pkg.instantiate()'; julia --color=yes --project=docs/ -e 'using Pkg; Pkg.instantiate()'; JULIA_DEBUG=Documenter julia --color=yes --project=docs docs/make.jl
```

If the docs are built successfully you can view them by opening `docs/build/index.html` from
your favorite browser.

## Building a single page

A full build takes minutes, most of it spent running the examples, which is a poor loop when what you
are editing is the prose of one page. `docs/build_single_page.jl` renders one page on its own:

```
julia --project=docs
julia> include("docs/build_single_page.jl")
```

The first include loads Documenter and Oceanostics and opens the rendered page
(`docs/build_single_page/html/index.html`) in a browser. Every include after that rebuilds it in about
a second, so the loop is edit, re-include, refresh the tab. `PAGE` selects a different page and
`DOCTEST = true` runs that page's `jldoctest` blocks; the header of the script has the details. Links
into the pages a single-page build leaves out cannot resolve, so they are reported as warnings and the
build carries on.
