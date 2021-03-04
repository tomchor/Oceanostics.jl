# Dedicated to a collection of general-use useful operators
using Oceananigans.Operators
using KernelAbstractions: @index, @kernel
using Oceananigans.Fields: location
using Oceananigans.AbstractOperations: AbstractOperation




#++++ Multiplication times a simple derivate
@kernel function ϕ∂xᶜ!(result, grid, operand, ϕ)
    i, j, k = @index(Global, NTuple)
    @inbounds result[i, j, k] = φ[i, j, k] * ∂xᶜᵃᵃ(i, j, k, grid, operand) # F, A, A  → C, A, A
end

@kernel function ϕ∂xᶠ!(result, grid, operand, ϕ)
    i, j, k = @index(Global, NTuple)
    @inbounds result[i, j, k] = φ[i, j, k] * ∂xᶠᵃᵃ(i, j, k, grid, operand) # C, A, A  → F, A, A
end



@kernel function ϕ∂yᶜ!(result, grid, operand, ϕ)
    i, j, k = @index(Global, NTuple)
    @inbounds result[i, j, k] = φ[i, j, k] * ∂yᵃᶜᵃ(i, j, k, grid, operand) # A, F, A  → A, C, A
end

@kernel function ϕ∂yᶠ!(result, grid, operand, ϕ)
    i, j, k = @index(Global, NTuple)
    @inbounds result[i, j, k] = φ[i, j, k] * ∂yᵃᶠᵃ(i, j, k, grid, operand) # A, C, A  → A, F, A
end



@kernel function ϕ∂zᶜ!(result, grid, operand, ϕ)
    i, j, k = @index(Global, NTuple)
    @inbounds result[i, j, k] = φ[i, j, k] * ∂zᵃᵃᶜ(i, j, k, grid, operand) # A, A, F  → A, A, C
end

@kernel function ϕ∂zᶠ!(result, grid, operand, ϕ)
    i, j, k = @index(Global, NTuple)
    @inbounds result[i, j, k] = φ[i, j, k] * ∂zᵃᵃᶠ(i, j, k, grid, operand) # A, A, C  → A, A, F
end
#-----



