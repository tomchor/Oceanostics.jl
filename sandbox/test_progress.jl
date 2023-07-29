using Revise
using Oceananigans
using Oceanostics.ProgressMessengers

grid = RectilinearGrid(size=(4, 5, 6), extent=(1, 1, 1));
model = NonhydrostaticModel(; grid);
simulation = Simulation(model, Δt=1, stop_time=10);

max_u = MaxUVelocity();
max_v = MaxVVelocity();
max_w = MaxWVelocity();

@show max_u(simulation) max_v(simulation) max_w(simulation)

max_u = MaxUVelocity(prefix=false, with_units=false);
max_v = MaxVVelocity(prefix=false, with_units=false);
max_w = MaxWVelocity(prefix=false, with_units=false);

max_vels = "|u⃗|ₘₐₓ = (" * max_u + max_v + max_w * ") m/s"
@show max_vels(simulation)


pause


function test(; with_units=false)
    max_u = MaxUVelocity(with_units = false)
    max_v = MaxVVelocity(with_units = false)
    max_w = MaxWVelocity(with_units = false)
    return test(max_u + max_v + max_w, with_units)
end

function (muvw::test)(sim)
    message = muvw.func(sim)
    muvw.with_units && (message = message * " m/s")
    return message
end


mv = test(with_units=true)

pm = mv

pm(simulation)

