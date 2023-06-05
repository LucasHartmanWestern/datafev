import time
from collections import deque, namedtuple
import numpy as np
import tensorflow as tf
import utils
import env
from rl_functions import compute_loss, agent_learn
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam

def get_Q_function(
    environment,
    MEMORY_SIZE = 100_000,          # size of memory buffer
    GAMMA = 0.995,                  # discount factor
    ALPHA = 1e-3,                   # learning rate
    NUM_STEPS_FOR_UPDATE = 4,       # perform a learning update every C time steps
    num_iters = 2000,               # amount of times to iterate during training process
    max_num_timesteps = 1000,       # amount of steps to take during a training iteration
    num_p_av = 100,                 # number of total points to use for averaging
    epsilon = 1.0,                  # initial ε value for ε-greedy policy
    hidden_layers = [64, 64]        # hidden layer architecture
):
    if not isinstance(environment, env):
        raise TypeError('environment must be an instance of env or its subclass')

    state_size = len(environment.states)
    num_actions = len(environment.actions)

    environment.reset()

    tf.random.set_seed(1337) # Set the random seed for TensorFlow
    q_network, target_q_network, optimizer = buildNetworks(state_size, num_actions, hidden_layers, ALPHA)
    trainNetwork(environment,
                 MEMORY_SIZE,
                 q_network,
                 target_q_network,
                 optimizer,
                 num_iters,
                 max_num_timesteps,
                 NUM_STEPS_FOR_UPDATE,
                 GAMMA,
                 num_p_av)

#### Create network architectures ####
def buildNetworks(
        state_size,     # Number of inputs
        num_actions,    # Number of outputs
        hidden_layers,  # Array representing hidden layers
        learning_rate   # starter learning rate for ADAM
):
    # Create the Q-Network
    q_network = Sequential()
    q_network.add(Input(state_size))
    for units in hidden_layers:
        q_network.add(Dense(units=units, activation='relu'))
    q_network.add(Dense(units=num_actions, activation='linear'))

    # Create the target Q^-Network
    target_q_network = Sequential()
    target_q_network.add(Input(state_size))
    for units in hidden_layers:
        target_q_network.add(Dense(units=units, activation='relu'))
    target_q_network.add(Dense(units=num_actions, activation='linear'))

    optimizer = Adam(learning_rate)

    return q_network, target_q_network, optimizer

#### Train the network ####
def trainNetwork(
    environment,
    MEMORY_SIZE,
    q_network,
    target_q_network,
    optimizer,
    num_iters,
    max_num_timesteps,
    NUM_STEPS_FOR_UPDATE,
    GAMMA,
    num_p_av
):
    start = time.time()
    total_point_history = []

    # Create a memory buffer D with capacity N
    memory_buffer = deque(maxlen=MEMORY_SIZE)

    # Set the target network weights equal to the Q-Network weights
    target_q_network.set_weights(q_network.get_weights())

    # Store experiences as named tuples
    experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    for i in range(num_iters):
        # Reset the environment to the initial state and get the initial state
        state = environment.reset()
        total_points = 0
        for t in range(max_num_timesteps):
            # From the current state S choose an action A using an ε-greedy policy
            state_qn = np.expand_dims(state, axis=0)  # state needs to be the right shape for the q_network
            q_values = q_network(state_qn)
            action = utils.get_action(q_values, epsilon)
            # Take action A and receive reward R and the next state S'
            next_state, reward, done = environment.step(action)
            # Store experience tuple (S,A,R,S') in the memory buffer.
            # We store the done variable as well for convenience.
            memory_buffer.append(experience(state, action, reward, next_state, done))
            # Only update the network every NUM_STEPS_FOR_UPDATE time steps.
            update = utils.check_update_conditions(t, NUM_STEPS_FOR_UPDATE, memory_buffer)
            if update:
                # Sample random mini-batch of experience tuples (S,A,R,S') from D
                experiences = utils.get_experiences(memory_buffer)
                # Set the y targets, perform a gradient descent step, and update the network weights.
                agent_learn(experiences, GAMMA, q_network, target_q_network, optimizer)
            state = next_state.copy()
            total_points += reward
            if done:
                break
        total_point_history.append(total_points)
        av_latest_points = np.mean(total_point_history[-num_p_av:])
        # Update the ε value
        epsilon = utils.get_new_eps(epsilon)
        print(f"\rEpisode {i+1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}", end="")
        if (i+1) % num_p_av == 0:
            print(f"\rEpisode {i+1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}")
        # We will consider that the environment is solved if we get an average of 200 points in the last 100 episodes.
        if av_latest_points >= 200.0:
            print(f"\n\nEnvironment solved in {i+1} episodes!")
            q_network.save('lunar_lander_model.h5')
            break

    tot_time = time.time() - start
    print(f"\nTotal Runtime: {tot_time:.2f} s ({(tot_time/60):.2f} min)")