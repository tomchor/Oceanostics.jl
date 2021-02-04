# KCF_diagnostics

`KernelComputedField`s diagnostics for use with Oceananigans.

## Caveats

- Not every kernel has been thoroughly tested.
- Kernels are written very generally since most uses of averages, etc. do not assume any
  specific kind of averaging procedure. Chances are it "wastes" computations for a given
specific application.
- For now this isn't meant to be a module, but simply a collection of Kernels that can be
  adapted for specific uses.


## Notes on notation and usage

For now I'm assuming that lowercase variables are pertubations around a mean and uppercase
variables are the mean (any kind of mean or even background fields). So, for example,
turbulent kinetic energy is calculated as (the following is a pseudo-code):

```julia
tke(u, v, w) = 1/2*(u^2 + v^2 + w^2)
```

And it is up to the user to make sure that the function is called with the perturbations
(to actually get turbulent kinetic energy), or the full velocity fields if the desired
output is total kinetic energy. So for turbulent kinetic energy one might call the
function as

```julia
U = AveragedField(model.velocities.u, dims=(1, 2))
V = AveragedField(model.velocities.v, dims=(1, 2))
TKE = tke(model.velocities.u-U, model.velocities.v-V, model.velocities.w)
```
