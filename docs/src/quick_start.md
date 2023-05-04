# Quick example

## Notes on notation and usage

For now I'm assuming that lowercase variables are pertubations around a mean and uppercase
variables are the mean (any kind of mean or even background fields). So, for example,
kinetic energy is calculated as (the following is a pseudo-code):

```julia
ke(u, v, w) = 1/2*(u^2 + v^2 + w^2)
```

And it is up to the user to make sure that the function is called with the perturbations
(to actually get turbulent kinetic energy), or the full velocity fields if the desired
output is total kinetic energy. So for turbulent kinetic energy one might call the
function as

```julia
U = Field(Average(model.velocities.u, dims=(1, 2)))
V = Field(Average(model.velocities.v, dims=(1, 2)))
TKE = ke(model.velocities.u-U, model.velocities.v-V, model.velocities.w)
```


## Simple example


