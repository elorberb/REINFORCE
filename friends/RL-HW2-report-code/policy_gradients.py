import gym
import numpy as np
import tensorflow as tf
import collections
from datetime import datetime
import time
# from logger import Logger


class PolicyNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='policy_network'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.compat.v1.variable_scope(name):

            self.state = tf.compat.v1.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.compat.v1.placeholder(tf.int32, [self.action_size], name="action")
            self.R_t = tf.compat.v1.placeholder(tf.float32, name="total_rewards")

            self.W1 = tf.compat.v1.get_variable("W1", [self.state_size, 12], initializer=tf.initializers.GlorotUniform(seed=0))
            self.b1 = tf.compat.v1.get_variable("b1", [12], initializer=tf.zeros_initializer())
            self.W2 = tf.compat.v1.get_variable("W2", [12, self.action_size], initializer=tf.initializers.GlorotUniform(seed=0))
            self.b2 = tf.compat.v1.get_variable("b2", [self.action_size], initializer=tf.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)

            # Softmax probability distribution over actions
            self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output))
            # Loss with negative log probability
            self.neg_log_prob = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.action)
            self.loss = tf.reduce_mean(self.neg_log_prob * self.R_t)
            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


def run():
    np.random.seed(1)
    tf.compat.v1.disable_eager_execution()
    # logger = Logger("logs/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-log.csv")

    env = gym.make('CartPole-v1')

    # Define hyperparameters
    state_size = 4
    action_size = env.action_space.n

    max_episodes = 5000
    max_steps = 501
    discount_factor = 0.99
    learning_rate = 0.0004

    render = False

    # Initialize the policy network
    tf.compat.v1.reset_default_graph()
    policy = PolicyNetwork(state_size, action_size, learning_rate)

    tic = time.perf_counter()
    # Start training the agent with REINFORCE algorithm
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        solved = False
        Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
        episode_rewards = np.zeros(max_episodes)
        average_rewards = 0.0

        for episode in range(max_episodes):
            state = env.reset()
            state = state.reshape([1, state_size])
            episode_transitions = []

            for step in range(max_steps):
                actions_distribution = sess.run(policy.actions_distribution, {policy.state: state})
                action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)
                next_state, reward, done, _ = env.step(action)
                next_state = next_state.reshape([1, state_size])

                if render:
                    env.render()

                action_one_hot = np.zeros(action_size)
                action_one_hot[action] = 1
                episode_transitions.append(
                    Transition(state=state, action=action_one_hot, reward=reward, next_state=next_state, done=done))
                episode_rewards[episode] += reward

                if done:
                    if episode > 98:
                        average_rewards = np.round(np.mean(episode_rewards[(episode - 99):episode + 1]), 2)
                    print("Episode {} Reward: {} Average over 100 episodes: {}".format(episode, episode_rewards[episode], average_rewards))
                    # Check if solved
                    if average_rewards > 475:
                        print(' Solved at episode: ' + str(episode))
                        solved = True
                    break
                state = next_state

            if solved:
                break

            # Compute Rt for each time-step t and update the network's weights
            for t, transition in enumerate(episode_transitions):
                total_discounted_return = sum(discount_factor ** i * t.reward for i, t in enumerate(episode_transitions[t:]))  # Rt
                feed_dict = {policy.state: transition.state, policy.R_t: total_discounted_return, policy.action: transition.action}
                _, loss = sess.run([policy.optimizer, policy.loss], feed_dict)
                # logger.write([episode, episode_rewards[episode], average_rewards, loss,  time.perf_counter() - tic])


if __name__ == '__main__':
    run()

