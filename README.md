# Oceanostics

[![Build Status](https://img.shields.io/badge/documentation-in%20development-orange)](https://tomchor.github.io/Oceanostics.jl/dev/)
[![Stable documentation](https://img.shields.io/badge/documentation-stable%20release-blue?style=flat-square)](https://tomchor.github.io/Oceanostics.jl/stable/)


Oceanostics is a Julia package created to facilitate obtaining diagnostic quantities in
[Oceananigans](https://github.com/CliMA/Oceananigans.jl) simulations. It was created solely as a
companion package to [Oceananigans](https://github.com/CliMA/Oceananigans.jl), so we refer users
first to the [Oceananigans documentation](https://clima.github.io/OceananigansDocumentation/stable/)
before getting started with Oceanostics.


## Installation

To install the latest stable version from Julia:
```julia
julia> ]
(@v1.8) pkg> add Oceanostics
```

If you want the latest developments (which may or may not be unstable) you can add the latest
version from Github in the `main` branch:

```julia
julia> ]
(@v1.8) pkg> add Oceanostics#main
```

