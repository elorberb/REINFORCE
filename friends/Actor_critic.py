import gym
import numpy as np
import tensorflow.compat.v1 as tf
import collections
from policy_gradients import PolicyNetwork

# optimized for Tf2
tf.disable_v2_behavior()
print("tf_ver:{}".format(tf.__version__))

env = gym.make('CartPole-v1')
np.random.seed(1)


class StateValueNetwork:
    def __init__(self, state_size, learning_rate, name='state_value_network'):
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.loss_object = tf.keras.losses.MeanSquaredError()

        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.baseline = tf.placeholder(tf.float32, [1], name="baseline")

            tf2_initializer = tf.keras.initializers.he_uniform()
            self.W1 = tf.get_variable("W1", [self.state_size, 64], initializer=tf2_initializer)
            self.b1 = tf.get_variable("b1", [64], initializer=tf2_initializer)
            self.W2 = tf.get_variable("W2", [64, 32], initializer=tf2_initializer)
            self.b2 = tf.get_variable("b2", [32], initializer=tf2_initializer)
            self.W3 = tf.get_variable("W3", [32, 1], initializer=tf2_initializer)
            self.b3 = tf.get_variable("b3", [1], initializer=tf2_initializer)

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.Z2 = tf.add(tf.matmul(self.A1, self.W2), self.b2)
            self.A2 = tf.nn.relu(self.Z2)
            self.output = tf.add(tf.matmul(self.A2, self.W3), self.b3)

            self.loss = self.loss_object(self.baseline, self.output)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


def run(verbose=False):
    # Define hyperparameters
    state_size = 4
    action_size = env.action_space.n

    max_episodes = 5000
    max_steps = 501
    discount_factor = 0.99
    learning_rate = 0.0001

    render = False

    # Initialize the policy network
    tf.reset_default_graph()
    policy = PolicyNetwork(state_size, action_size, learning_rate)
    state_value = StateValueNetwork(state_size, 0.0005)
    # Start training the agent with REINFORCE algorithm
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        solved = False
        Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
        episode_rewards = np.zeros(max_episodes)
        average_rewards = 0.0
        mean_rewards = []

        for episode in range(max_episodes):
            state, _ = env.reset()
            state = state.reshape([1, state_size])
            episode_transitions = []
            L = 1
            for step in range(max_steps):
                actions_distribution = sess.run(policy.actions_distribution, {policy.state: state})
                action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)
                next_state, reward, done, _, _ = env.step(action)
                next_state = next_state.reshape([1, state_size])

                if render:
                    env.render()

                action_one_hot = np.zeros(action_size)
                action_one_hot[action] = 1
                episode_transitions.append(
                    Transition(state=state, action=action_one_hot, reward=reward, next_state=next_state, done=done))
                episode_rewards[episode] += reward

                # Compute Rt for each time-step t and update the network's weights
                delta = reward + discount_factor * sess.run(state_value.output, {state_value.state: next_state}) * (
                            1 - done)
                delta_error = delta - sess.run(state_value.output, {state_value.state: state})

                feed_dict_baseline = {state_value.state: state, state_value.baseline: np.array(delta).reshape(1)}
                _, baseline_loss = sess.run([state_value.optimizer, state_value.loss], feed_dict_baseline)

                feed_dict = {policy.state: state, policy.R_t: delta_error, policy.action: action_one_hot}
                _, loss = sess.run([policy.optimizer, policy.loss], feed_dict)

                L *= discount_factor
                if done:
                    if episode > 98:
                        # Check if solved
                        average_rewards = np.mean(episode_rewards[(episode - 99):episode + 1])
                    mean_rewards.append(average_rewards)
                    if verbose:
                        print("Episode {} Reward: {} Average over 100 episodes: {}".format(episode,
                                                                                           episode_rewards[episode],
                                                                                           round(average_rewards, 2)))
                    if average_rewards > 475:
                        if verbose:
                            print(' Solved at episode: ' + str(episode))
                        solved = True
                    break
                state = next_state

            if solved:
                break

        return list(range(episode)), episode_rewards, mean_rewards


if __name__ == '__main__':
    run(verbose=True)