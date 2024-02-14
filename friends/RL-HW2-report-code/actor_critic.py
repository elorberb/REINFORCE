import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
import collections
from policy_gradients import PolicyNetwork
from datetime import datetime
import time
# from logger import Logger


class ValueNetwork:

    def __init__(self, state_size, learning_rate):
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.v = self.init_v()

    def init_v(self):
        v = keras.Sequential([
            keras.layers.InputLayer(input_shape=(self.state_size,)),
            keras.layers.Dense(units=64, activation='elu', kernel_initializer=tf.keras.initializers.HeUniform()),
            keras.layers.Dense(units=16, activation='elu', kernel_initializer=tf.keras.initializers.HeUniform()),
            keras.layers.Dense(units=1, activation='linear')
        ])
        v.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss="mse")
        return v


def run():
    np.random.seed(1)
    tf.compat.v1.disable_eager_execution()
    # episode_logger = Logger("logs/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-ac-episode-log.csv")
    # step_logger = Logger("logs/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-ac-step-log.csv")
    env = gym.make('CartPole-v1')

    # Define hyperparameters
    state_size = 4
    action_size = env.action_space.n

    max_episodes = 5000
    max_steps = 501
    discount_factor = 0.99

    render = False

    # Initialize the policy network
    tf.compat.v1.reset_default_graph()
    policy = PolicyNetwork(state_size, action_size, 0.0004)
    value_network = ValueNetwork(state_size, 0.0007)

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

                current_value = value_network.v.predict(state)
                next_value = value_network.v.predict(next_state)
                td_target = reward + (1 - done) * discount_factor * next_value
                td_error = td_target - current_value

                v_loss = value_network.v.fit(state, np.atleast_2d(td_target), verbose=0)
                v_loss = v_loss.history['loss'][0]

                feed_dict = {policy.state: state, policy.R_t: td_error, policy.action: action_one_hot}
                _, loss = sess.run([policy.optimizer, policy.loss], feed_dict)

                # step_logger.write([step, loss, v_loss])

                if done:
                    if episode > 98:
                        # Check if solved
                        average_rewards = np.mean(episode_rewards[(episode - 99):episode + 1])
                    print("Episode {} Reward: {} Average over 100 episodes: {}".format(episode, episode_rewards[episode], round(average_rewards, 2)))
                    # episode_logger.write([episode, episode_rewards[episode], average_rewards, time.perf_counter() - tic])
                    if average_rewards > 475:
                        print(' Solved at episode: ' + str(episode))
                        solved = True
                    break
                state = next_state

            if solved:
                break


if __name__ == '__main__':
    run()
