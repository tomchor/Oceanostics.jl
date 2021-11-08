

function test_progress_messenger(messenger; LES=false, model_kwargs...)
    model = create_model(; model_kwargs...)
    test_progress_messenger(model, messenger)
end

function test_progress_messenger(model, messenger; LES=false)
    simulation = Simulation(model; Δt=1e-2, 
                            stop_iteration=10,
                            )
    simulation.callbacks[:progress] = Callback(messenger, IterationInterval(1))

    run!(simulation)

    return nothing
end





