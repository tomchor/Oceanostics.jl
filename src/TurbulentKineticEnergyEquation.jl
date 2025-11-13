module TurbulentKineticEnergyEquation
using DocStringExtensions

export TurbulentKineticEnergy
export IsotropicDissipationRate, TurbulentKineticEnergyIsotropicDissipationRate
export XShearProductionRate, TurbulentKineticEnergyXShearProductionRate,
       YShearProductionRate, TurbulentKineticEnergyYShearProductionRate,
       ZShearProductionRate, TurbulentKineticEnergyZShearProductionRate,
       ShearProductionRate, TurbulentKineticEnergyShearProductionRate

using Oceananigans.Operators
using Oceananigans.AbstractOperations
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Grids: Center
using Oceananigans.Fields: ZeroField

using Oceanostics: validate_location, CustomKFO
using Oceanostics.KineticEnergyEquation: KineticEnergyIsotropicDissipationRate

# Some useful operators
@inline ψ²(i, j, k, grid, ψ) = @inbounds ψ[i, j, k]^2

@inline ψ′²(i, j, k, grid, ψ, ψ̄) = @inbounds (ψ[i, j, k] - ψ̄[i, j, k])^2
@inline ψ′²(i, j, k, grid, ψ, ψ̄::Number) = @inbounds (ψ[i, j, k] - ψ̄)^2

#+++ TurbulentKineticEnergy
@inline turbulent_kinetic_energy_ccc(i, j, k, grid, u, v, w, U, V, W) = (ℑxᶜᵃᵃ(i, j, k, grid, ψ′², u, U) +
                                                                         ℑyᵃᶜᵃ(i, j, k, grid, ψ′², v, V) +
                                                                         ℑzᵃᵃᶜ(i, j, k, grid, ψ′², w, W)) / 2

const TurbulentKineticEnergy = CustomKFO{<:typeof(turbulent_kinetic_energy_ccc)}

"""
    $(SIGNATURES)

Calculate the turbulent kinetic energy of `model`.

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(; grid);

julia> TKE = TurbulentKineticEnergyEquation.TurbulentKineticEnergy(model)
KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: turbulent_kinetic_energy_ccc (generic function with 1 method)
└── arguments: ("Field", "Field", "Field", "Oceananigans.Fields.ZeroField", "Oceananigans.Fields.ZeroField", "Oceananigans.Fields.ZeroField")
```
"""
function TurbulentKineticEnergy(model, u, v, w; U=ZeroField(), V=ZeroField(), W=ZeroField(), location = (Center, Center, Center))
    validate_location(location, "TurbulentKineticEnergy")
    return KernelFunctionOperation{Center, Center, Center}(turbulent_kinetic_energy_ccc, model.grid,
                                                           u, v, w, U, V, W)
end

TurbulentKineticEnergy(model; kwargs...) = TurbulentKineticEnergy(model, model.velocities...; kwargs...)
#---

#+++ TurbulentKineticEnergyIsotropicDissipationRate
"""
    $(SIGNATURES)

Calculate the Turbulent Kinetic Energy Isotropic Dissipation Rate, defined as

    ε = 2 ν S'ᵢⱼS'ᵢⱼ,

where S'ᵢⱼ is the strain rate tensor, for a fluid with an isotropic turbulence closure (i.e., a
turbulence closure where ν (eddy or not) is the same for all directions.

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(; grid, closure=ScalarDiffusivity(ν=1e-4));

julia> TurbulentKineticEnergyEquation.IsotropicDissipationRate(model)
KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: isotropic_viscous_dissipation_rate_ccc (generic function with 1 method)
└── arguments: ("Field", "Field", "Field", "NamedTuple")
```
"""
@inline TurbulentKineticEnergyIsotropicDissipationRate(u, v, w, args...; U=ZeroField(), V=ZeroField(), W=ZeroField(), location = (Center, Center, Center)) =
    KineticEnergyIsotropicDissipationRate((u - U), (v - V), (w - W), args...; location)

@inline function TurbulentKineticEnergyIsotropicDissipationRate(model; U=ZeroField(), V=ZeroField(), W=ZeroField(), kwargs...)
    u, v, w = model.velocities
    return TurbulentKineticEnergyIsotropicDissipationRate((u - U), (v - V), (w - W), model.closure, model.closure_fields, model.clock; kwargs...)
end

const IsotropicDissipationRate = TurbulentKineticEnergyIsotropicDissipationRate
#---

#+++ TurbulentKineticEnergyXShearProductionRate
@inline function shear_production_rate_x_ccc(i, j, k, grid, u′, v′, w′, U, V, W)
    u′_int = ℑxᶜᵃᵃ(i, j, k, grid, u′) # F, C, C  → C, C, C

    ∂xU = ∂xᶜᶜᶜ(i, j, k, grid, U) # F, C, C  → C, C, C
    u′u′ = ℑxᶜᵃᵃ(i, j, k, grid, ψ², u′)
    u′u′∂xU = u′u′ * ∂xU

    ∂xV = ℑxyᶜᶜᵃ(i, j, k, grid, ∂xᶠᶠᶜ, V) # C, F, C  → F, F, C  → C, C, C
    v′u′ = ℑyᵃᶜᵃ(i, j, k, grid, v′) * u′_int
    v′u′∂xV = v′u′ * ∂xV

    ∂xW = ℑxzᶜᵃᶜ(i, j, k, grid, ∂xᶠᶜᶠ, W) # C, C, F  → F, C, F  → C, C, C
    w′u′ = ℑzᵃᵃᶜ(i, j, k, grid, w′) * u′_int
    w′u′∂xW = w′u′ * ∂xW

    return -(u′u′∂xU + v′u′∂xV + w′u′∂xW)
end

const TurbulentKineticEnergyXShearProductionRate = CustomKFO{<:typeof(shear_production_rate_x_ccc)}
const XShearProductionRate = TurbulentKineticEnergyXShearProductionRate

"""
    $(SIGNATURES)

Calculate the shear production rate in the `model`'s `x` direction:

    XSHEAR = uᵢ′u′∂x(Uᵢ)

where `uᵢ′` is the velocity perturbation in the `i` direction, `u′` is the velocity perturbation in the `x` direction,
`Uᵢ` is the background velocity in the `i` direction, and `∂x` is the horizontal derivative.

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(; grid);

julia> XSHEAR = TurbulentKineticEnergyEquation.XShearProductionRate(model)
KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: shear_production_rate_x_ccc (generic function with 1 method)
└── arguments: ("Field", "Field", "Field", "Oceananigans.Fields.ZeroField", "Oceananigans.Fields.ZeroField", "Oceananigans.Fields.ZeroField")
```
"""
function TurbulentKineticEnergyXShearProductionRate(u′, v′, w′, U, V, W;
                                                    grid = u′.grid,
                                                    location = (Center, Center, Center))
    validate_location(location, "TurbulentKineticEnergyXShearProductionRate")
    return KernelFunctionOperation{Center, Center, Center}(shear_production_rate_x_ccc, grid,
                                                           u′, v′, w′, U, V, W)
end


function TurbulentKineticEnergyXShearProductionRate(model; U=ZeroField(), V=ZeroField(), W=ZeroField(), kwargs...)
    u, v, w = model.velocities
    return TurbulentKineticEnergyXShearProductionRate(u-U, v-V, w-W, U, V, W; kwargs...)
end
#---

#+++ TurbulentKineticEnergyYShearProductionRate
@inline function shear_production_rate_y_ccc(i, j, k, grid, u′, v′, w′, U, V, W)
    v′_int = ℑyᵃᶜᵃ(i, j, k, grid, v′) # C, F, C  → C, C, C

    ∂yU = ℑxyᶜᶜᵃ(i, j, k, grid, ∂yᶠᶠᶜ, U) # F, C, C  → F, F, C  → C, C, C
    u′v′ = ℑxᶜᵃᵃ(i, j, k, grid, u′) * v′_int
    u′v′∂yU = u′v′ * ∂yU

    ∂yV = ∂yᶜᶜᶜ(i, j, k, grid, V) # C, F, C  → C, C C
    v′v′ = ℑyᵃᶜᵃ(i, j, k, grid, ψ², v′) # C, F, C  → C, C, C
    v′v′∂yV = v′v′ * ∂yV

    ∂yW = ℑyzᵃᶜᶜ(i, j, k, grid, ∂yᶜᶠᶠ, W) # C, C, F  → C, F, F  → C, C, C
    w′v′ = ℑzᵃᵃᶜ(i, j, k, grid, w′) * v′_int
    w′v′∂yW = w′v′ * ∂yW

    return -(u′v′∂yU + v′v′∂yV + w′v′∂yW)
end

const TurbulentKineticEnergyYShearProductionRate = CustomKFO{<:typeof(shear_production_rate_y_ccc)}
const YShearProductionRate = TurbulentKineticEnergyYShearProductionRate

"""
    $(SIGNATURES)

Calculate the shear production rate in the `model`'s `y` direction:

    YSHEAR = uᵢ′v′∂y(Uᵢ)

where `uᵢ′` is the velocity perturbation in the `i` direction, `v′` is the velocity perturbation in the `y` direction,
`Uᵢ` is the background velocity in the `i` direction, and `∂y` is the vertical derivative.

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(; grid);

julia> YSHEAR = TurbulentKineticEnergyEquation.YShearProductionRate(model)
KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: shear_production_rate_y_ccc (generic function with 1 method)
└── arguments: ("Field", "Field", "Field", "Oceananigans.Fields.ZeroField", "Oceananigans.Fields.ZeroField", "Oceananigans.Fields.ZeroField")
```
"""
function TurbulentKineticEnergyYShearProductionRate(u′, v′, w′, U, V, W;
                                                    grid = u′.grid,
                                                    location = (Center, Center, Center))
    validate_location(location, "TurbulentKineticEnergyYShearProductionRate")
    return KernelFunctionOperation{Center, Center, Center}(shear_production_rate_y_ccc, grid,
                                                           u′, v′, w′, U, V, W)
end

function TurbulentKineticEnergyYShearProductionRate(model; U=ZeroField(), V=ZeroField(), W=ZeroField(), kwargs...)
    u, v, w = model.velocities
    return TurbulentKineticEnergyYShearProductionRate(u-U, v-V, w-W, U, V, W; kwargs...)
end
#---

#+++ TurbulentKineticEnergyZShearProductionRate
@inline function shear_production_rate_z_ccc(i, j, k, grid, u′, v′, w′, U, V, W)
    w′_int = ℑzᵃᵃᶜ(i, j, k, grid, w′) # C, C, F  → C, C, C

    ∂zU = ℑxzᶜᵃᶜ(i, j, k, grid, ∂zᶠᶜᶠ, U) # F, C, C  → F, C, F  → C, C, C
    u′w′ = ℑxᶜᵃᵃ(i, j, k, grid, u′) * w′_int
    u′w′∂zU = u′w′ * ∂zU

    ∂zV = ℑyzᵃᶜᶜ(i, j, k, grid, ∂zᶜᶠᶠ, V) # C, F, C  → C, F, F  → C, C, C
    v′w′ = ℑyᵃᶜᵃ(i, j, k, grid, v′) * w′_int
    v′w′∂zV = v′w′ * ∂zV

    ∂zW = ∂zᶜᶜᶜ(i, j, k, grid, W) # C, C, F  → C, C, C
    w′w′ = ℑzᵃᵃᶜ(i, j, k, grid, ψ², w′) # C, C, F  → C, C, C
    w′w′∂zW = w′w′ * ∂zW

    return - (u′w′∂zU + v′w′∂zV + w′w′∂zW)
end

const TurbulentKineticEnergyZShearProductionRate = CustomKFO{<:typeof(shear_production_rate_z_ccc)}
const ZShearProductionRate = TurbulentKineticEnergyZShearProductionRate

"""
    $(SIGNATURES)

Calculate the shear production rate in the `model`'s `z` direction:

    ZSHEAR = uᵢ′w′∂z(Uᵢ)

where `uᵢ′` is the velocity perturbation in the `i` direction, `w′` is the vertical velocity perturbation,
`Uᵢ` is the background velocity in the `i` direction, and `∂z` is the vertical derivative.

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(; grid);

julia> ZSHEAR = TurbulentKineticEnergyEquation.ZShearProductionRate(model)
KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: shear_production_rate_z_ccc (generic function with 1 method)
└── arguments: ("Field", "Field", "Field", "Oceananigans.Fields.ZeroField", "Oceananigans.Fields.ZeroField", "Oceananigans.Fields.ZeroField")
```
"""
function TurbulentKineticEnergyZShearProductionRate(u′, v′, w′, U, V, W;
                                                    grid = u′.grid,
                                                    location = (Center, Center, Center))
    validate_location(location, "TurbulentKineticEnergyZShearProductionRate")
    return KernelFunctionOperation{Center, Center, Center}(shear_production_rate_z_ccc, grid,
                                                           u′, v′, w′, U, V, W)
end

function TurbulentKineticEnergyZShearProductionRate(model; U=ZeroField(), V=ZeroField(), W=ZeroField(), kwargs...)
    u, v, w = model.velocities
    return TurbulentKineticEnergyZShearProductionRate(u-U, v-V, w-W, U, V, W; kwargs...)
end
#---

#+++ TurbulentKineticEnergyShearProductionRate
@inline function shear_production_rate_ccc(args...)
    return shear_production_rate_x_ccc(args...) +
           shear_production_rate_y_ccc(args...) +
           shear_production_rate_z_ccc(args...)
end

const TurbulentKineticEnergyShearProductionRate = CustomKFO{<:typeof(shear_production_rate_ccc)}
const ShearProductionRate = TurbulentKineticEnergyShearProductionRate

"""
    $(SIGNATURES)

Calculate the total shear production rate (sum of the shear production rates in the `model`'s `x`, `y` and `z` directions):

    SHEAR = XSHEAR + YSHEAR + ZSHEAR = uᵢ′uⱼ′∂ⱼ(Uᵢ)

where `XSHEAR`, `YSHEAR` and `ZSHEAR` are the shear production rates in the `x`, `y` and `z` directions, respectively,
`uᵢ′` and `uⱼ′` are the velocity perturbations in the `i` and `j` directions, respectively, and `∂ⱼ` is the derivative in the `j` direction.

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(; grid);

julia> SHEAR = TurbulentKineticEnergyEquation.ShearProductionRate(model)
KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: shear_production_rate_ccc (generic function with 1 method)
└── arguments: ("Field", "Field", "Field", "Oceananigans.Fields.ZeroField", "Oceananigans.Fields.ZeroField", "Oceananigans.Fields.ZeroField")
```
"""
function TurbulentKineticEnergyShearProductionRate(u′, v′, w′, U, V, W;
                                                   grid = u′.grid,
                                                   location = (Center, Center, Center))
    validate_location(location, "TurbulentKineticEnergyShearProductionRate")
    return KernelFunctionOperation{Center, Center, Center}(shear_production_rate_ccc, grid,
                                                           u′, v′, w′, U, V, W)
end

@inline TurbulentKineticEnergyShearProductionRate(model; U=ZeroField(), V=ZeroField(), W=ZeroField(), kwargs...) =
    TurbulentKineticEnergyShearProductionRate(model.velocities..., U, V, W; kwargs...)
#---

end