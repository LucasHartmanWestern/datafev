from dqn_custom import train
from ev_simulation_environment import EVSimEnvironment
from geolocation.visualize import generate_interactive_plot, read_csv_data, generate_average_reward_plot, generate_charger_only_plot
from baseline_algorithm import baseline
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

############

train_model = True
generate_baseline = True
generate_plots = True

############

num_of_chargers = 10
make = 0
model = 0
starting_charge = 50000 # 50kW
max_charge = 100000 # 100kW
org_lat = 42.98904084
org_long = -81.22821493
dest_lat = 43.006137960450104
dest_long = -81.27651959525788
num_episodes = 5000

max_attempts = 10

env = EVSimEnvironment(num_episodes, num_of_chargers, make, model, starting_charge, max_charge, org_lat, org_long, dest_lat, dest_long)

if generate_baseline:
    baseline(env, max_attempts)

if train_model:
    epsilon = 0.05
    discount_factor = 0.999
    batch_size = 100
    buffer_limit = 250
    max_num_timesteps = 50
    start_from_previous_session = True
    layers = [64, 128, 526, 128, 64]

    state_dimension, action_dimension = env.get_state_action_dimension()
    train(env, epsilon, discount_factor, num_episodes, batch_size, buffer_limit, max_num_timesteps, state_dimension, action_dimension - 1, start_from_previous_session, layers)

if generate_plots:
    env.write_path_to_csv('outputs/routes.csv')
    env.write_chargers_to_csv('outputs/chargers.csv')
    env.write_reward_graph_to_csv('outputs/rewards.csv')

    route_data = read_csv_data('outputs/routes.csv')
    charger_data = read_csv_data('outputs/chargers.csv')
    reward_data = read_csv_data('outputs/rewards.csv')

    route_datasets = []
    for id_value, group in route_data.groupby('Episode Num'):
        route_datasets.append(group)

    if train_model:
        generate_average_reward_plot(reward_data)

    generate_interactive_plot(route_datasets, charger_data, (org_lat, org_long), (dest_lat, dest_long))