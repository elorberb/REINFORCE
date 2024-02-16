import gymnasium as gym
import numpy as np
import tensorflow.compat.v1 as tf
import collections
import time
from tqdm import tqdm
from code.policy_gradients import PolicyNetwork
# optimized for Tf2
tf.disable_v2_behavior()
print("tf_ver:{}".format(tf.__version__))

env = gym.make('CartPole-v1')
np.random.seed(1)

class ValueNetwork:
    def __init__(self, state_size, learning_rate=0.001, name='value_network'):
        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, state_size], name="state")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")

            # Simple one-layer network
            self.W1 = tf.get_variable("W1", [state_size, 20], initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.b1 = tf.get_variable("b1", [20], initializer=tf.zeros_initializer())
            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)

            # Output layer
            self.W2 = tf.get_variable("W2", [20, 1], initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.b2 = tf.get_variable("b2", [1], initializer=tf.zeros_initializer())
            self.value = tf.add(tf.matmul(self.A1, self.W2), self.b2)

            # Loss
            self.loss = tf.reduce_mean(tf.square(self.R_t - self.value))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)


def run_with_baseline():

    results = {'Episode': [], 'Reward':[], 
               "Average_100":[], 'Solved': -1, 
               'Duration': 0, 'Loss': [], 'LossV':[]}
    start = time.time()
    # Define hyperparameters
    state_size = 4
    action_size = env.action_space.n

    max_episodes = 5000
    max_steps = 501
    discount_factor = 0.99
    learning_rate = 0.0004

    render = False

    # Initialize the policy network
    tf.reset_default_graph()
    policy = PolicyNetwork(state_size, action_size, learning_rate)
    value_network = ValueNetwork(state_size, learning_rate=learning_rate)

    # Start training the agent with REINFORCE algorithm
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        solved = False
        Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
        episode_rewards = np.zeros(max_episodes)
        average_rewards = 0.0

        for episode in tqdm(range(max_episodes)):
            state = env.reset()[0]
            state = state.reshape([1, state_size])
            episode_transitions = []

            for step in range(max_steps):
                actions_distribution = sess.run(policy.actions_distribution, {policy.state: state})
                action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)
                next_state, reward, done, _, _ = env.step(action)
                next_state = next_state.reshape([1, state_size])

                if render:
                    env.render()

                action_one_hot = np.zeros(action_size)
                action_one_hot[action] = 1
                episode_transitions.append(Transition(state=state, action=action_one_hot, reward=reward, next_state=next_state, done=done))
                episode_rewards[episode] += reward

                if done:
                    if episode > 98:
                        # Check if solved
                        average_rewards = np.mean(episode_rewards[(episode - 99):episode+1])

                    results['Episode'].append(episode)
                    results['Reward'].append(episode_rewards[episode])
                    results['Average_100'].append(round(average_rewards, 2))
                    if average_rewards > 475:
                        results['Solved'] = episode
                        solved = True
                    break
                state = next_state

            if solved:
                break

            # Compute Rt-baseline for each time-step t and update the network's weights
            for t, transition in enumerate(episode_transitions):
                total_discounted_return = sum(discount_factor ** i * t.reward for i, t in enumerate(episode_transitions[t:]))
                current_value = sess.run(value_network.value, {value_network.state: transition.state})
                advantage = total_discounted_return - current_value
                # Update policy network
                feed_dict_policy = {policy.state: transition.state, policy.R_t: advantage, policy.action: transition.action}
                _, loss_policy = sess.run([policy.optimizer, policy.loss], feed_dict_policy)
                results['Loss'].append(loss_policy)
                # Update value network
                feed_dict_value = {value_network.state: transition.state, value_network.R_t: total_discounted_return}
                _, loss_value = sess.run([value_network.optimizer, value_network.loss], feed_dict_value)
                results['LossV'].append(loss_value)
    
    results['Duration'] = time.time() - start

    return results



if __name__ == '__main__':
    run_with_baseline()