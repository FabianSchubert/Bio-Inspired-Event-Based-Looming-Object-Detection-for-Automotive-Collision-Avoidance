import os

base_fold_input_data = os.path.join(
    os.path.dirname(__file__), "../../../data/carla_sim/random_spawn/"
)

base_fold_input_data_noisy = os.path.join(
    os.path.dirname(__file__), "../../../data/carla_sim/random_spawn_noisy/"
)

base_fold_results = os.path.join(
    os.path.dirname(__file__),
    "../../../data/experiments/carla_sim/random_spawn/on_off_cells/"
)

base_fold_results_noisy = os.path.join(
    os.path.dirname(__file__),
    "../../../data/experiments/carla_sim/random_spawn_noisy/on_off_cells/"
)