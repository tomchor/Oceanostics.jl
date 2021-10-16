

function test_progress_messenger(messenger; LES=false, model_kwargs...)
    model = create_model(; model_kwargs...)
    test_progress_messenger(model, messenger)
end

function test_progress_messenger(model, messenger; LES=false)
    simulation = Simulation(model; Δt=1e-2, 
                            stop_iteration=10,
                            iteration_interval=1,
                            progress=messenger,
                            )

    run!(simulation)

    return nothing
end





