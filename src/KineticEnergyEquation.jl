module KineticEnergyEquation
using DocStringExtensions

export KineticEnergy
export KineticEnergyTendency
export Advection, KineticEnergyAdvection
export Stress, KineticEnergyStress
export Forcing, KineticEnergyForcing
export PressureRedistribution, KineticEnergyPressureRedistribution
export BuoyancyProduction, KineticEnergyBuoyancyProduction
export DissipationRate, KineticEnergyDissipationRate
export KineticEnergyIsotropicDissipationRate

using Oceananigans: NonhydrostaticModel, HydrostaticFreeSurfaceModel, fields
using Oceananigans.Operators
using Oceananigans.AbstractOperations
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Grids: Center, Face
using Oceananigans.Fields: ZeroField
using Oceananigans.Models.NonhydrostaticModels: u_velocity_tendency, v_velocity_tendency, w_velocity_tendency
using Oceananigans.Advection: div_𝐯u, div_𝐯v, div_𝐯w
using Oceananigans.TurbulenceClosures: viscous_flux_ux, viscous_flux_uy, viscous_flux_uz,
                                       viscous_flux_vx, viscous_flux_vy, viscous_flux_vz,
                                       viscous_flux_wx, viscous_flux_wy, viscous_flux_wz,
                                       ∂ⱼ_τ₁ⱼ, ∂ⱼ_τ₂ⱼ, ∂ⱼ_τ₃ⱼ
using Oceananigans.TurbulenceClosures: immersed_∂ⱼ_τ₁ⱼ, immersed_∂ⱼ_τ₂ⱼ, immersed_∂ⱼ_τ₃ⱼ
using Oceananigans.BuoyancyFormulations: x_dot_g_bᶠᶜᶜ, y_dot_g_bᶜᶠᶜ, z_dot_g_bᶜᶜᶠ

using Oceanostics: _νᶜᶜᶜ
using Oceanostics: validate_location, validate_dissipative_closure, perturbation_fields, CustomKFO

# Some useful operators
@inline ψ²(i, j, k, grid, ψ) = @inbounds ψ[i, j, k]^2
@inline fψ_plus_gφ²(i, j, k, grid, f, ψ, g, φ) = (f(i, j, k, grid, ψ) + g(i, j, k, grid, φ))^2

#++++ KineticEnergy
@inline kinetic_energy_ccc(i, j, k, grid, u, v, w) = (ℑxᶜᵃᵃ(i, j, k, grid, ψ², u) +
                                                      ℑyᵃᶜᵃ(i, j, k, grid, ψ², v) +
                                                      ℑzᵃᵃᶜ(i, j, k, grid, ψ², w)) / 2

const KineticEnergy = CustomKFO{<:typeof(kinetic_energy_ccc)}

"""
    $(SIGNATURES)

Calculate the kinetic energy of `model` manually specifying `u`, `v` and `w`.

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid);

julia> KE = KineticEnergyEquation.KineticEnergy(model, model.velocities...)
KineticEnergy KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: kinetic_energy_ccc (generic function with 1 method)
└── arguments: ("Field", "Field", "Field")
└── computes: kinetic energy  ½uᵢuᵢ
```
"""
function KineticEnergy(model, u, v, w; location = (Center, Center, Center))
    validate_location(location, "KineticEnergy")
    return KernelFunctionOperation{Center, Center, Center}(kinetic_energy_ccc, model.grid, u, v, w)
end

"""
    $(SIGNATURES)

Calculate the kinetic energy of `model`.

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid);

julia> KE = KineticEnergyEquation.KineticEnergy(model)
KineticEnergy KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: kinetic_energy_ccc (generic function with 1 method)
└── arguments: ("Field", "Field", "Field")
└── computes: kinetic energy  ½uᵢuᵢ
```
"""
KineticEnergy(model; kwargs...) = KineticEnergy(model, model.velocities...; kwargs...)
#------

#+++ KineticEnergyTendency
@inline ψf(i, j, k, grid, ψ, f, args...) = @inbounds ψ[i, j, k] * f(i, j, k, grid, args...)

@inline function uᵢGᵢᶜᶜᶜ(i, j, k, grid, advection,
                                        coriolis,
                                        stokes_drift,
                                        closure,
                                        u_immersed_bc,
                                        v_immersed_bc,
                                        w_immersed_bc,
                                        buoyancy,
                                        background_fields,
                                        velocities,
                                        tracers,
                                        auxiliary_fields,
                                        closure_fields,
                                        pHY′,
                                        clock,
                                        forcings)
    common_args = (buoyancy, background_fields, velocities, tracers, auxiliary_fields, closure_fields, pHY′, clock)
    u∂ₜu = ℑxᶜᵃᵃ(i, j, k, grid, ψf, velocities.u, u_velocity_tendency, advection, coriolis, stokes_drift, closure, u_immersed_bc, common_args..., forcings.u)
    v∂ₜv = ℑyᵃᶜᵃ(i, j, k, grid, ψf, velocities.v, v_velocity_tendency, advection, coriolis, stokes_drift, closure, v_immersed_bc, common_args..., forcings.v)
    w∂ₜw = ℑzᵃᵃᶜ(i, j, k, grid, ψf, velocities.w, w_velocity_tendency, advection, coriolis, stokes_drift, closure, w_immersed_bc, common_args..., forcings.w)
    return u∂ₜu + v∂ₜv + w∂ₜw
end

const KineticEnergyTendency = CustomKFO{<:typeof(uᵢGᵢᶜᶜᶜ)}

"""
    $(SIGNATURES)

Return a `KernelFunctionOperation` that computes the tendency uᵢGᵢ of the KE, excluding the nonhydrostatic
pressure contribution:

    KET = ½∂ₜuᵢ² = uᵢGᵢ - uᵢ∂ᵢpₙₕₛ

```jldoctest
julia> using Oceananigans

julia> grid = RectilinearGrid(size = (1, 1, 4), extent = (1, 1, 1));

julia> model = NonhydrostaticModel(grid);

julia> using Oceanostics.KineticEnergyEquation: KineticEnergyTendency

julia> ke_tendency = KineticEnergyTendency(model)
KineticEnergyTendency KernelFunctionOperation at (Center, Center, Center)
├── grid: 1×1×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×1×3 halo
├── kernel_function: uᵢGᵢᶜᶜᶜ (generic function with 1 method)
└── arguments: ("Centered", "Nothing", "Nothing", "Nothing", "Nothing", "Nothing", "Nothing", "Nothing", "Oceananigans.Models.NonhydrostaticModels.BackgroundFields", "NamedTuple", "NamedTuple", "NamedTuple", "Nothing", "Nothing", "Clock", "NamedTuple")
└── computes: kinetic energy tendency  uᵢGᵢ (excl. nonhydrostatic pressure)
```
"""
function KineticEnergyTendency(model::NonhydrostaticModel; location = (Center, Center, Center))
    validate_location(location, "KineticEnergyTendency")
    dependencies = (model.advection,
                    model.coriolis,
                    model.stokes_drift,
                    model.closure,
                    u_immersed_bc = model.velocities.u.boundary_conditions.immersed,
                    v_immersed_bc = model.velocities.v.boundary_conditions.immersed,
                    w_immersed_bc = model.velocities.w.boundary_conditions.immersed,
                    model.buoyancy,
                    model.background_fields,
                    model.velocities,
                    model.tracers,
                    model.auxiliary_fields,
                    model.closure_fields,
                    model.pressures.pHY′,
                    model.clock,
                    model.forcing,)
    return KernelFunctionOperation{Center, Center, Center}(uᵢGᵢᶜᶜᶜ, model.grid, dependencies...)
end
#---

#+++ KineticEnergyAdvection
@inline function uᵢ∂ⱼuⱼuᵢᶜᶜᶜ(i, j, k, grid, velocities, advection)
    u∂ⱼuⱼu = ℑxᶜᵃᵃ(i, j, k, grid, ψf, velocities.u, div_𝐯u, advection, velocities, velocities.u)
    v∂ⱼuⱼv = ℑyᵃᶜᵃ(i, j, k, grid, ψf, velocities.v, div_𝐯v, advection, velocities, velocities.v)
    w∂ⱼuⱼw = ℑzᵃᵃᶜ(i, j, k, grid, ψf, velocities.w, div_𝐯w, advection, velocities, velocities.w)
    return u∂ⱼuⱼu + v∂ⱼuⱼv + w∂ⱼuⱼw
end

const KineticEnergyAdvection = CustomKFO{<:typeof(uᵢ∂ⱼuⱼuᵢᶜᶜᶜ)}
const Advection = KineticEnergyAdvection

"""
    $(SIGNATURES)

Return a `KernelFunctionOperation` that computes the advection term, defined as

    ADV = uᵢ∂ⱼ(uᵢuⱼ)

By default, the buoyancy production will be calculated using the resolved `velocities` and
users cab use the keyword `velocities` to modify that behavior:

```jldoctest
julia> using Oceananigans

julia> grid = RectilinearGrid(size = (1, 1, 4), extent = (1,1,1));

julia> model = NonhydrostaticModel(grid);

julia> using Oceanostics.KineticEnergyEquation: KineticEnergyAdvection

julia> ADV = KineticEnergyAdvection(model)
KineticEnergyAdvection KernelFunctionOperation at (Center, Center, Center)
├── grid: 1×1×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×1×3 halo
├── kernel_function: uᵢ∂ⱼuⱼuᵢᶜᶜᶜ (generic function with 1 method)
└── arguments: ("NamedTuple", "Centered")
└── computes: kinetic energy advection  uᵢ∂ⱼ(uᵢuⱼ)
```
"""
function KineticEnergyAdvection(model::NonhydrostaticModel; velocities = model.velocities, location = (Center, Center, Center))
    validate_location(location, "KineticEnergyAdvection")
    return KernelFunctionOperation{Center, Center, Center}(uᵢ∂ⱼuⱼuᵢᶜᶜᶜ, model.grid, velocities, model.advection)
end
#---

#+++ KineticEnergyStress
@inline function uᵢ∂ⱼ_τᵢⱼᶜᶜᶜ(i, j, k, grid, closure,
                                            closure_fields,
                                            clock,
                                            model_fields,
                                            buoyancy)

    u∂ⱼ_τ₁ⱼ = ℑxᶜᵃᵃ(i, j, k, grid, ψf, model_fields.u, ∂ⱼ_τ₁ⱼ, closure, closure_fields, clock, model_fields, buoyancy)
    v∂ⱼ_τ₂ⱼ = ℑyᵃᶜᵃ(i, j, k, grid, ψf, model_fields.v, ∂ⱼ_τ₂ⱼ, closure, closure_fields, clock, model_fields, buoyancy)
    w∂ⱼ_τ₃ⱼ = ℑzᵃᵃᶜ(i, j, k, grid, ψf, model_fields.w, ∂ⱼ_τ₃ⱼ, closure, closure_fields, clock, model_fields, buoyancy)

    return u∂ⱼ_τ₁ⱼ+ v∂ⱼ_τ₂ⱼ + w∂ⱼ_τ₃ⱼ
end

const KineticEnergyStress = CustomKFO{<:typeof(uᵢ∂ⱼ_τᵢⱼᶜᶜᶜ)}
const Stress = KineticEnergyStress

"""
    $(SIGNATURES)

Return a `KernelFunctionOperation` that computes the diffusive term of the KE prognostic equation:

```
    DIFF = uᵢ∂ⱼτᵢⱼ
```

where `uᵢ` are the velocity components and `τᵢⱼ` is the diffusive flux of `i` momentum in the
`j`-th direction.

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid; closure=ScalarDiffusivity(ν=1e-4));

julia> DIFF = KineticEnergyEquation.KineticEnergyStress(model)
KineticEnergyStress KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: uᵢ∂ⱼ_τᵢⱼᶜᶜᶜ (generic function with 1 method)
└── arguments: ("ScalarDiffusivity", "Nothing", "Clock", "NamedTuple", "Nothing")
└── computes: kinetic energy stress/diffusion  uᵢ∂ⱼτᵢⱼ
```
"""
function KineticEnergyStress(model; location = (Center, Center, Center))
    validate_location(location, "KineticEnergyStress")
    model_fields = fields(model)

    if model isa HydrostaticFreeSurfaceModel
        model_fields = (; model_fields..., w=ZeroField())
    end
    dependencies = (model.closure,
                    model.closure_fields,
                    model.clock,
                    fields(model),
                    model.buoyancy)
    return KernelFunctionOperation{Center, Center, Center}(uᵢ∂ⱼ_τᵢⱼᶜᶜᶜ, model.grid, dependencies...)
end
#---

#+++ KineticEnergyForcing
@inline function uᵢFᵤᵢᶜᶜᶜ(i, j, k, grid, forcings,
                                         clock,
                                         model_fields)

    uFᵘ = ℑxᶜᵃᵃ(i, j, k, grid, ψf, model_fields.u, forcings.u, clock, model_fields)
    vFᵛ = ℑyᵃᶜᵃ(i, j, k, grid, ψf, model_fields.v, forcings.v, clock, model_fields)
    wFʷ = ℑzᵃᵃᶜ(i, j, k, grid, ψf, model_fields.w, forcings.w, clock, model_fields)

    return uFᵘ+ vFᵛ + wFʷ
end

const KineticEnergyForcing = CustomKFO{<:typeof(uᵢFᵤᵢᶜᶜᶜ)}
const Forcing = KineticEnergyForcing

"""
    $(SIGNATURES)

Return a `KernelFunctionOperation` that computes the forcing term of the KE prognostic equation:

```
    FORC = uᵢFᵤᵢ
```

where `uᵢ` are the velocity components and `Fᵤᵢ` is the forcing term(s) in the `uᵢ`
prognostic equation (i.e. the forcing for `uᵢ`).

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid);

julia> FORC = KineticEnergyEquation.KineticEnergyForcing(model)
KineticEnergyForcing KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: uᵢFᵤᵢᶜᶜᶜ (generic function with 1 method)
└── arguments: ("NamedTuple", "Clock", "NamedTuple")
└── computes: kinetic energy forcing  uᵢFᵤᵢ
```
"""
function KineticEnergyForcing(model::NonhydrostaticModel; location = (Center, Center, Center))
    validate_location(location, "KineticEnergyForcing")
    model_fields = fields(model)

    dependencies = (model.forcing,
                    model.clock,
                    fields(model))
    return KernelFunctionOperation{Center, Center, Center}(uᵢFᵤᵢᶜᶜᶜ, model.grid, dependencies...)
end
#---

#+++ KineticEnergyPressureRedistribution
@inline function uᵢ∂ᵢpᶜᶜᶜ(i, j, k, grid, velocities, pressure)
    u∂x_p = ℑxᶜᵃᵃ(i, j, k, grid, ψf, velocities.u, ∂xᶠᶜᶜ, pressure)
    v∂y_p = ℑyᵃᶜᵃ(i, j, k, grid, ψf, velocities.v, ∂yᶜᶠᶜ, pressure)
    w∂z_p = ℑzᵃᵃᶜ(i, j, k, grid, ψf, velocities.w, ∂zᶜᶜᶠ, pressure)
    return u∂x_p + v∂y_p + w∂z_p
end

const KineticEnergyPressureRedistribution = CustomKFO{<:typeof(uᵢ∂ᵢpᶜᶜᶜ)}
const PressureRedistribution = KineticEnergyPressureRedistribution

"""
    $(SIGNATURES)

Return a `KernelFunctionOperation` that computes the pressure redistribution term:

    PR = uᵢ∂ᵢp

where `p` is the pressure. By default `p` is taken to be the total pressure (nonhydrostatic + hydrostatic):

```jldoctest ∇u⃗p_example
julia> using Oceananigans

julia> grid = RectilinearGrid(size = (1, 1, 4), extent = (1,1,1));

julia> model = NonhydrostaticModel(grid);

julia> using Oceanostics.KineticEnergyEquation: KineticEnergyPressureRedistribution

julia> ∇u⃗p = KineticEnergyPressureRedistribution(model)
KineticEnergyPressureRedistribution KernelFunctionOperation at (Center, Center, Center)
├── grid: 1×1×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×1×3 halo
├── kernel_function: uᵢ∂ᵢpᶜᶜᶜ (generic function with 1 method)
└── arguments: ("NamedTuple", "Field")
└── computes: kinetic energy pressure redistribution  uᵢ∂ᵢp
```

We can also pass `velocities` and `pressure` keywords to perform more specific calculations. The
example below illustrates calculation of the nonhydrostatic contribution to the pressure
redistrubution term:

```jldoctest ∇u⃗p_example
julia> ∇u⃗pNHS = KineticEnergyPressureRedistribution(model, pressure=model.pressures.pNHS)
KineticEnergyPressureRedistribution KernelFunctionOperation at (Center, Center, Center)
├── grid: 1×1×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×1×3 halo
├── kernel_function: uᵢ∂ᵢpᶜᶜᶜ (generic function with 1 method)
└── arguments: ("NamedTuple", "Field")
└── computes: kinetic energy pressure redistribution  uᵢ∂ᵢp
```
"""
function KineticEnergyPressureRedistribution(model::NonhydrostaticModel; velocities = model.velocities,
                                    pressure = model.pressures.pHY′ == nothing ? model.pressures.pNHS : sum(model.pressures),
                                    location = (Center, Center, Center))
    validate_location(location, "KineticEnergyPressureRedistribution")
    return KernelFunctionOperation{Center, Center, Center}(uᵢ∂ᵢpᶜᶜᶜ, model.grid, velocities, pressure)
end
#---

#+++ KineticEnergyBuoyancyProduction
@inline function uᵢbᵢᶜᶜᶜ(i, j, k, grid, velocities, buoyancy_model, tracers)
    ubˣ = ℑxᶜᵃᵃ(i, j, k, grid, ψf, velocities.u, x_dot_g_bᶠᶜᶜ, buoyancy_model, tracers)
    vbʸ = ℑyᵃᶜᵃ(i, j, k, grid, ψf, velocities.v, y_dot_g_bᶜᶠᶜ, buoyancy_model, tracers)
    wbᶻ = ℑzᵃᵃᶜ(i, j, k, grid, ψf, velocities.w, z_dot_g_bᶜᶜᶠ, buoyancy_model, tracers)
    return ubˣ + vbʸ + wbᶻ
end

const KineticEnergyBuoyancyProduction = CustomKFO{<:typeof(uᵢbᵢᶜᶜᶜ)}
const BuoyancyProduction = KineticEnergyBuoyancyProduction

"""
    $(SIGNATURES)

Return a `KernelFunctionOperation` that computes the buoyancy production term, defined as

    BP = uᵢbᵢ

where bᵢ is the component of the buoyancy acceleration in the `i`-th direction (which is zero for x
and y, except when `gravity_unit_vector` isn't aligned with the grid's z-direction) and all three
components of `i=1,2,3` are added up.

By default, the buoyancy production will be calculated using the resolved `velocities` and
`tracers`:

```jldoctest wb_example
julia> using Oceananigans

julia> grid = RectilinearGrid(size = (1, 1, 4), extent = (1,1,1));

julia> model = NonhydrostaticModel(grid; buoyancy=BuoyancyTracer(), tracers=:b);

julia> using Oceanostics.KineticEnergyEquation: BuoyancyProduction

julia> wb = BuoyancyProduction(model)
KineticEnergyBuoyancyProduction KernelFunctionOperation at (Center, Center, Center)
├── grid: 1×1×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×1×3 halo
├── kernel_function: uᵢbᵢᶜᶜᶜ (generic function with 1 method)
└── arguments: ("NamedTuple", "BuoyancyForce", "NamedTuple")
└── computes: kinetic energy buoyancy production  uᵢbᵢ
```

If we want to calculate only the _turbulent_ buoyancy production rate, we can do so by passing
turbulent perturbations to the `velocities` and/or `tracers` options):

```jldoctest wb_example
julia> w′ = Field(model.velocities.w - Field(Average(model.velocities.w)));

julia> b′ = Field(model.tracers.b - Field(Average(model.tracers.b)));

julia> w′b′ = BuoyancyProduction(model, velocities=(u=model.velocities.u, v=model.velocities.v, w=w′), tracers=(b=b′,))
KineticEnergyBuoyancyProduction KernelFunctionOperation at (Center, Center, Center)
├── grid: 1×1×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×1×3 halo
├── kernel_function: uᵢbᵢᶜᶜᶜ (generic function with 1 method)
└── arguments: ("NamedTuple", "BuoyancyForce", "NamedTuple")
└── computes: kinetic energy buoyancy production  uᵢbᵢ
```
"""
function BuoyancyProduction(model::NonhydrostaticModel; velocities = model.velocities, tracers = model.tracers, location = (Center, Center, Center))
    validate_location(location, "BuoyancyProduction")
    return KernelFunctionOperation{Center, Center, Center}(uᵢbᵢᶜᶜᶜ, model.grid, velocities, model.buoyancy, tracers)
end
#---

#+++ KineticEnergyDissipationRate
# ∂ⱼu₁ ⋅ τ₁ⱼ
Axᶜᶜᶜ_δuᶜᶜᶜ_τ₁₁ᶜᶜᶜ(i, j, k, grid, closure, K_fields, clo, fields, b) = -Axᶜᶜᶜ(i, j, k, grid) * δxᶜᵃᵃ(i, j, k, grid, fields.u) * viscous_flux_ux(i, j, k, grid, closure, K_fields, clo, fields, b)
Ayᶠᶠᶜ_δuᶠᶠᶜ_τ₁₂ᶠᶠᶜ(i, j, k, grid, closure, K_fields, clo, fields, b) = -Ayᶠᶠᶜ(i, j, k, grid) * δyᵃᶠᵃ(i, j, k, grid, fields.u) * viscous_flux_uy(i, j, k, grid, closure, K_fields, clo, fields, b)
Azᶠᶜᶠ_δuᶠᶜᶠ_τ₁₃ᶠᶜᶠ(i, j, k, grid, closure, K_fields, clo, fields, b) = -Azᶠᶜᶠ(i, j, k, grid) * δzᵃᵃᶠ(i, j, k, grid, fields.u) * viscous_flux_uz(i, j, k, grid, closure, K_fields, clo, fields, b)

# ∂ⱼu₂ ⋅ τ₂ⱼ
Axᶠᶠᶜ_δvᶠᶠᶜ_τ₂₁ᶠᶠᶜ(i, j, k, grid, closure, K_fields, clo, fields, b) = -Axᶠᶠᶜ(i, j, k, grid) * δxᶠᵃᵃ(i, j, k, grid, fields.v) * viscous_flux_vx(i, j, k, grid, closure, K_fields, clo, fields, b)
Ayᶜᶜᶜ_δvᶜᶜᶜ_τ₂₂ᶜᶜᶜ(i, j, k, grid, closure, K_fields, clo, fields, b) = -Ayᶜᶜᶜ(i, j, k, grid) * δyᵃᶜᵃ(i, j, k, grid, fields.v) * viscous_flux_vy(i, j, k, grid, closure, K_fields, clo, fields, b)
Azᶜᶠᶠ_δvᶜᶠᶠ_τ₂₃ᶜᶠᶠ(i, j, k, grid, closure, K_fields, clo, fields, b) = -Azᶜᶠᶠ(i, j, k, grid) * δzᵃᵃᶠ(i, j, k, grid, fields.v) * viscous_flux_vz(i, j, k, grid, closure, K_fields, clo, fields, b)

# ∂ⱼu₃ ⋅ τ₃ⱼ
Axᶠᶜᶠ_δwᶠᶜᶠ_τ₃₁ᶠᶜᶠ(i, j, k, grid, closure, K_fields, clo, fields, b) = -Axᶠᶜᶠ(i, j, k, grid) * δxᶠᵃᵃ(i, j, k, grid, fields.w) * viscous_flux_wx(i, j, k, grid, closure, K_fields, clo, fields, b)
Ayᶜᶠᶠ_δwᶜᶠᶠ_τ₃₂ᶜᶠᶠ(i, j, k, grid, closure, K_fields, clo, fields, b) = -Ayᶜᶠᶠ(i, j, k, grid) * δyᵃᶠᵃ(i, j, k, grid, fields.w) * viscous_flux_wy(i, j, k, grid, closure, K_fields, clo, fields, b)
Azᶜᶜᶜ_δwᶜᶜᶜ_τ₃₃ᶜᶜᶜ(i, j, k, grid, closure, K_fields, clo, fields, b) = -Azᶜᶜᶜ(i, j, k, grid) * δzᵃᵃᶜ(i, j, k, grid, fields.w) * viscous_flux_wz(i, j, k, grid, closure, K_fields, clo, fields, b)

@inline viscous_dissipation_rate_ccc(i, j, k, grid, closure_fields, fields, p) =
    (Axᶜᶜᶜ_δuᶜᶜᶜ_τ₁₁ᶜᶜᶜ(i, j, k, grid,         p.closure, closure_fields, p.clock, fields, p.buoyancy) + # C, C, C
     ℑxyᶜᶜᵃ(i, j, k, grid, Ayᶠᶠᶜ_δuᶠᶠᶜ_τ₁₂ᶠᶠᶜ, p.closure, closure_fields, p.clock, fields, p.buoyancy) + # F, F, C  → C, C, C
     ℑxzᶜᵃᶜ(i, j, k, grid, Azᶠᶜᶠ_δuᶠᶜᶠ_τ₁₃ᶠᶜᶠ, p.closure, closure_fields, p.clock, fields, p.buoyancy) + # F, C, F  → C, C, C

     ℑxyᶜᶜᵃ(i, j, k, grid, Axᶠᶠᶜ_δvᶠᶠᶜ_τ₂₁ᶠᶠᶜ, p.closure, closure_fields, p.clock, fields, p.buoyancy) + # F, F, C  → C, C, C
     Ayᶜᶜᶜ_δvᶜᶜᶜ_τ₂₂ᶜᶜᶜ(i, j, k, grid,         p.closure, closure_fields, p.clock, fields, p.buoyancy) + # C, C, C
     ℑyzᵃᶜᶜ(i, j, k, grid, Azᶜᶠᶠ_δvᶜᶠᶠ_τ₂₃ᶜᶠᶠ, p.closure, closure_fields, p.clock, fields, p.buoyancy) + # C, F, F  → C, C, C

     ℑxzᶜᵃᶜ(i, j, k, grid, Axᶠᶜᶠ_δwᶠᶜᶠ_τ₃₁ᶠᶜᶠ, p.closure, closure_fields, p.clock, fields, p.buoyancy) + # F, C, F  → C, C, C
     ℑyzᵃᶜᶜ(i, j, k, grid, Ayᶜᶠᶠ_δwᶜᶠᶠ_τ₃₂ᶜᶠᶠ, p.closure, closure_fields, p.clock, fields, p.buoyancy) + # C, F, F  → C, C, C
     Azᶜᶜᶜ_δwᶜᶜᶜ_τ₃₃ᶜᶜᶜ(i, j, k, grid,         p.closure, closure_fields, p.clock, fields, p.buoyancy)   # C, C, C
     ) / Vᶜᶜᶜ(i, j, k, grid) # This division by volume, coupled with the call to A*δuᵢ above, ensures a derivative operation

const KineticEnergyDissipationRate = CustomKFO{<:typeof(viscous_dissipation_rate_ccc)}
const DissipationRate = KineticEnergyDissipationRate

"""
    $(SIGNATURES)

Calculate the Kinetic Energy Dissipation Rate, defined as

    ε = ∂ⱼuᵢ ⋅ τᵢⱼ

where ∂ⱼuᵢ is the velocity gradient tensor and τᵢⱼ is the stress tensor.

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid; closure=ScalarDiffusivity(ν=1e-4));

julia> ε = KineticEnergyEquation.DissipationRate(model)
KineticEnergyDissipationRate KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: viscous_dissipation_rate_ccc (generic function with 1 method)
└── arguments: ("Nothing", "NamedTuple", "NamedTuple")
└── computes: kinetic energy dissipation rate  ε = ∂ⱼuᵢ·τᵢⱼ
```
"""
function DissipationRate(model; U=ZeroField(), V=ZeroField(), W=ZeroField(),
                                               location = (Center, Center, Center))
    validate_location(location, "DissipationRate")
    mean_velocities = (u=U, v=V, w=W)
    model_fields = perturbation_fields(model; mean_velocities...)
    parameters = (; model.closure,
                  model.clock,
                  model.buoyancy)

    return KernelFunctionOperation{Center, Center, Center}(viscous_dissipation_rate_ccc, model.grid,
                                                           model.closure_fields, model_fields, parameters)
end
#---

#+++ KineticEnergyIsotropicDissipationRate
@inline function isotropic_viscous_dissipation_rate_ccc(i, j, k, grid, u, v, w, p)

    Σˣˣ² = ∂xᶜᶜᶜ(i, j, k, grid, u)^2
    Σʸʸ² = ∂yᶜᶜᶜ(i, j, k, grid, v)^2
    Σᶻᶻ² = ∂zᶜᶜᶜ(i, j, k, grid, w)^2

    Σˣʸ² = ℑxyᶜᶜᵃ(i, j, k, grid, fψ_plus_gφ², ∂yᶠᶠᶜ, u, ∂xᶠᶠᶜ, v) / 4
    Σˣᶻ² = ℑxzᶜᵃᶜ(i, j, k, grid, fψ_plus_gφ², ∂zᶠᶜᶠ, u, ∂xᶠᶜᶠ, w) / 4
    Σʸᶻ² = ℑyzᵃᶜᶜ(i, j, k, grid, fψ_plus_gφ², ∂zᶜᶠᶠ, v, ∂yᶜᶠᶠ, w) / 4

    ν = _νᶜᶜᶜ(i, j, k, grid, p.closure, p.closure_fields, p.clock, p.model_fields)

    return 2ν * (Σˣˣ² + Σʸʸ² + Σᶻᶻ² + 2 * (Σˣʸ² + Σˣᶻ² + Σʸᶻ²))
end

const KineticEnergyIsotropicDissipationRate = CustomKFO{<:typeof(isotropic_viscous_dissipation_rate_ccc)}
const IsotropicDissipationRate = KineticEnergyIsotropicDissipationRate

"""
    $(SIGNATURES)

Calculate the Viscous Dissipation Rate as

    ε = 2 ν SᵢⱼSᵢⱼ,

where Sᵢⱼ is the strain rate tensor, for a fluid with an isotropic turbulence closure (i.e., a
turbulence closure where ν (eddy or not) is the same for all directions).

```jldoctest
julia> using Oceananigans, Oceanostics

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid; closure=ScalarDiffusivity(ν=1e-4));

julia> ε = KineticEnergyEquation.KineticEnergyIsotropicDissipationRate(model)
KineticEnergyIsotropicDissipationRate KernelFunctionOperation at (Center, Center, Center)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: isotropic_viscous_dissipation_rate_ccc (generic function with 1 method)
└── arguments: ("Field", "Field", "Field", "NamedTuple")
└── computes: isotropic kinetic energy dissipation rate  ε = 2νSᵢⱼSᵢⱼ
```
"""
function KineticEnergyIsotropicDissipationRate(u, v, w, closure, closure_fields, model_fields, clock; location = (Center, Center, Center))
    validate_location(location, "KineticEnergyIsotropicDissipationRate")
    validate_dissipative_closure(closure)

    parameters = (; closure, closure_fields, clock, model_fields)
    return KernelFunctionOperation{Center, Center, Center}(isotropic_viscous_dissipation_rate_ccc, u.grid,
                                                           u, v, w, parameters)
end

@inline KineticEnergyIsotropicDissipationRate(model; location = (Center, Center, Center)) =
    KineticEnergyIsotropicDissipationRate(model.velocities..., model.closure, model.closure_fields, fields(model), model.clock; location = location)
#---

end # module 
