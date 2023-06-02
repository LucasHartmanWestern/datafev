import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import MSE

def compute_loss(experiences, gamma, q_network, target_q_network):
    states, actions, rewards, next_states, done_vals = experiences  # Unpack the mini-batch of experience tuples
    max_qsa = tf.reduce_max(target_q_network(next_states), axis=-1)  # Compute max Q^(s,a)
    y_targets = rewards + (
                1 - done_vals) * gamma * max_qsa  # y = R if episode terminates, otherwise y = R + γ max Q^(s,a)
    q_values = q_network(states)  # Get the q_values and reshape to match y_targets
    q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]), tf.cast(actions, tf.int32)], axis=1))
    loss = MSE(y_targets, q_values)  # Compute the loss
    return loss


@tf.function
def agent_learn(experiences, gamma, q_network, target_q_network, optimizer):
    # Calculate the loss
    with tf.GradientTape() as tape: loss = compute_loss(experiences, gamma, q_network, target_q_network)
    # Get the gradients of the loss with respect to the weights
    gradients = tape.gradient(loss, q_network.trainable_variables)
    optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))  # Update the weights of the q_network.
    utils.update_target_network(q_network, target_q_network)  # Update the weights of target q_network