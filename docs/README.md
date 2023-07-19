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
