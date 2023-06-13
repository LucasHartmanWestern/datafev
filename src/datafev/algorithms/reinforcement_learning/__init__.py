from dqn_custom import train
from ev_simulation_environment import EVSimEnvironment

num_of_chargers = 10
make = 0
model = 0
starting_charge = 50000 # 50kW
max_charge = 100000 # 100kW
org_lat = 42.98904084
org_long = -81.22821493
dest_lat = 42.98375799
dest_long = -81.29324100

env = EVSimEnvironment(num_of_chargers, make, model, starting_charge, max_charge, org_lat, org_long, dest_lat, dest_long)

epsilon = 0.01
discount_factor = 0.99
num_episodes = 10
batch_size = 1000
max_num_timesteps = 100

state_dimension, action_dimension = env.get_state_action_dimension()
train(env, epsilon, discount_factor, num_episodes, batch_size, max_num_timesteps, state_dimension, action_dimension)

env.write_path_to_csv()