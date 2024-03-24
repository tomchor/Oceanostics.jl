module PotentialEnergyEquationTerms

using DocStringExtensions

export PotentialEnergy

using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans: Models.seawater_density
using Oceananigans: Models.model_geopotential_height
using Oceananigans.Grids: Center, Face
using Oceanostics: validate_location, validate_buoyancy


"""
    $(SIGNATURES)

Return a `KernelFunctionOperation` to compute the `PotentialEnergy` per unit volume,
```math
Eₚ = \\frac{gρz}{ρ₀}
```
at each grid `location` in `model`.

**NOTE:** A `BoussinesqEquationOfState` must be used in the `model` to calculate
`seawater_density`. See the [relevant documentation](https://clima.github.io/OceananigansDocumentation/dev/model_setup/buoyancy_and_equation_of_state/#Idealized-nonlinear-equations-of-state)
for how to set `SeawaterBuoyancy` using a `BoussinesqEquationOfState`.

Example
=======

The default behaviour of `PotentialEnergy` uses the *in-situ density* in the calculation:
```jldoctest
julia> using Oceananigans, SeawaterPolynomials.TEOS10

julia> using Oceanostics.PotentialEnergyEquationTerms: PotentialEnergy

julia> grid = RectilinearGrid(size=100, z=(-1000, 0), topology=(Flat, Flat, Bounded))
1×1×100 RectilinearGrid{Float64, Flat, Flat, Bounded} on CPU with 0×0×3 halo
├── Flat x
├── Flat y
└── Bounded  z ∈ [-1000.0, 0.0]   regularly spaced with Δz=10.0

julia> tracers = (:T, :S)
(:T, :S)

julia> eos = TEOS10EquationOfState()
BoussinesqEquationOfState{Float64}:
    ├── seawater_polynomial: TEOS10SeawaterPolynomial{Float64}
    └── reference_density: 1020.0

julia> buoyancy = SeawaterBuoyancy(equation_of_state=eos)
SeawaterBuoyancy{Float64}:
├── gravitational_acceleration: 9.80665
└── equation_of_state: BoussinesqEquationOfState{Float64}

julia> model = NonhydrostaticModel(; grid, buoyancy, tracers)
NonhydrostaticModel{CPU, RectilinearGrid}(time = 0 seconds, iteration = 0)
├── grid: 1×1×100 RectilinearGrid{Float64, Flat, Flat, Bounded} on CPU with 0×0×3 halo
├── timestepper: QuasiAdamsBashforth2TimeStepper
├── tracers: (T, S)
├── closure: Nothing
├── buoyancy: SeawaterBuoyancy with g=9.80665 and BoussinesqEquationOfState{Float64} with ĝ = NegativeZDirection()
└── coriolis: Nothing

julia> PotentialEnergy(model)
KernelFunctionOperation at (Center, Center, Center)
├── grid: 1×1×100 RectilinearGrid{Float64, Flat, Flat, Bounded} on CPU with 0×0×3 halo
├── kernel_function: gρz_ccc (generic function with 1 method)
└── arguments: ("KernelFunctionOperation at (Center, Center, Center)", "KernelFunctionOperation at (Center, Center, Center)", "(g=9.80665,)")
```

To use a reference density set a constant value for the keyword argument `geopotential_height`
and pass this the function. For example,
```jldoctest
julia> using Oceananigans, SeawaterPolynomials.TEOS10;

julia> using Oceanostics.PotentialEnergyEquationTerms: PotentialEnergy;

julia> grid = RectilinearGrid(size=100, z=(-1000, 0), topology=(Flat, Flat, Bounded));

julia> tracers = (:T, :S);

julia> eos = TEOS10EquationOfState();

julia> buoyancy = SeawaterBuoyancy(equation_of_state=eos);

julia> model = NonhydrostaticModel(; grid, buoyancy, tracers);

julia> geopotential_height = 0; # density variable will be σ₀

julia> PotentialEnergy(model; geopotential_height)
KernelFunctionOperation at (Center, Center, Center)
├── grid: 1×1×100 RectilinearGrid{Float64, Flat, Flat, Bounded} on CPU with 0×0×3 halo
├── kernel_function: gρz_ccc (generic function with 1 method)
└── arguments: ("KernelFunctionOperation at (Center, Center, Center)", "KernelFunctionOperation at (Center, Center, Center)", "(g=9.80665,)")
```
"""
@inline function PotentialEnergy(model; geopotential_height = model_geopotential_height(model), location = (Center, Center, Center))

    validate_location(location, "PotentialEnergy")
    validate_buoyancy(model.buoyancy)

    grid = model.grid
    ρ = seawater_density(model; geopotential_height)
    Z = model_geopotential_height(model)
    parameters = (g = model.buoyancy.model.gravitational_acceleration,
                  ρ₀ = model.buoyancy.model.equation_of_state.reference_density)

    return KernelFunctionOperation{Center, Center, Center}(gρz_ccc, grid, ρ, Z, parameters)
end

@inline gρz_ccc(i, j, k, grid, ρ, Z, p) = (p.g / p.ρ₀) * ρ[i, j, k] * Z[i, j, k]

end # module
