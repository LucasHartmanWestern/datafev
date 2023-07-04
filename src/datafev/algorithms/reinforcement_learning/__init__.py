from dqn_custom import train_dqn
from sarsa_custom import train_sarsa
from ev_simulation_environment import EVSimEnvironment
from geolocation.visualize import generate_interactive_plot, read_csv_data, generate_average_reward_plot, generate_charger_only_plot
from baseline_algorithm import BaselineGenerator
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

############

algorithm = "SARSA"

train_model = True
generate_baseline = False
start_from_previous_session = False
save_data = True
generate_plots = True
use_simple = True

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
max_num_timesteps = 20
num_episodes = 5000

max_attempts = 10

env = EVSimEnvironment(num_episodes, num_of_chargers, make, model, starting_charge, max_charge, org_lat, org_long, dest_lat, dest_long)
baselineGen = BaselineGenerator()

generated_baseline = None
if generate_baseline:
    baselineGen.generate_baseline(use_simple, env, max_attempts, max_num_timesteps)
    generated_baseline = baselineGen.path

if train_model:
    epsilon = 0.60
    discount_factor = 0.99999
    batch_size = 50
    buffer_limit = 125
    max_num_timesteps = 50
    layers = [32, 64, 64, 32]

    state_dimension, action_dimension = env.get_state_action_dimension()

    if algorithm == "DQN":
        print("Training using Deep-Q Learning")
        train_dqn(use_simple, env, epsilon, discount_factor, num_episodes, batch_size, buffer_limit, max_num_timesteps, state_dimension, action_dimension - 1, start_from_previous_session, layers, generated_baseline)
    else:
        print("Training using Expected SARSA")
        train_sarsa(use_simple, env, epsilon, discount_factor, num_episodes, batch_size, buffer_limit, max_num_timesteps, state_dimension, action_dimension - 1, start_from_previous_session, layers, generated_baseline)

if save_data:
    env.write_path_to_csv('outputs/routes.csv')
    env.write_chargers_to_csv('outputs/chargers.csv')
    env.write_reward_graph_to_csv('outputs/rewards.csv')

if generate_plots:
    route_data = read_csv_data('outputs/routes.csv')
    charger_data = read_csv_data('outputs/chargers.csv')
    reward_data = read_csv_data('outputs/rewards.csv')

    route_datasets = []
    for id_value, group in route_data.groupby('Episode Num'):
        route_datasets.append(group)

    if train_model or start_from_previous_session:
        generate_average_reward_plot(algorithm, reward_data)

    generate_interactive_plot(algorithm, route_datasets, charger_data, (org_lat, org_long), (dest_lat, dest_long))
