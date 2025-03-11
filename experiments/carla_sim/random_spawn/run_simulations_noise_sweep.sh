#!/bin/bash

# Define the list of numbers
noise_rates=(2.0 4.0 6.0 8.0)

# Path to the Python script
python_script="./process_number.py"

cd ../../../

# Iterate over the numbers
for rate in "${noise_rates[@]}"; do
    python3 -m experiments.carla_sim.random_spawn.simulate "$rate"
done