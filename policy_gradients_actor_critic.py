import gymnasium as gym
import numpy as np
import tensorflow.compat.v1 as tf
import collections
import time
from tqdm import tqdm
# optimized for Tf2
tf.disable_v2_behavior()
print("tf_ver:{}".format(tf.__version__))

env = gym.make('CartPole-v1')
np.random.seed(1)


class PolicyNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='policy_network'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):

            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.placeholder(tf.int32, [self.action_size], name="action")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")

            tf2_initializer = tf.keras.initializers.glorot_normal(seed=0)
            self.W1 = tf.get_variable("W1", [self.state_size, 12], initializer=tf2_initializer)
            self.b1 = tf.get_variable("b1", [12], initializer=tf2_initializer)
            self.W2 = tf.get_variable("W2", [12, self.action_size], initializer=tf2_initializer)
            self.b2 = tf.get_variable("b2", [self.action_size], initializer=tf2_initializer)

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)

            # Softmax probability distribution over actions
            self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output))
            # Loss with negative log probability
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.action)
            self.loss = tf.reduce_mean(self.neg_log_prob * self.R_t)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)




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

def run_actor_critic():
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
    learning_rate = 0.0007

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
                    #print("Episode {} Reward: {} Average over 100 episodes: {}".format(episode, episode_rewards[episode], round(average_rewards, 2)))
                    if average_rewards > 475:
                        results['Solved'] = episode
                        #print(' Solved at episode: ' + str(episode))
                        solved = True
                    break
                state = next_state

            if solved:
                break

            # Update the networks using the collected transitions
            for t, transition in enumerate(episode_transitions):
                if transition.done:
                    next_state_value = 0
                else:
                    next_state_value = sess.run(value_network.value, {value_network.state: transition.next_state})
                
                td_target = transition.reward + discount_factor * next_state_value
                td_error = td_target - sess.run(value_network.value, {value_network.state: transition.state})
                
                # Update the critic
                feed_dict_value = {value_network.state: transition.state, value_network.R_t: td_target}
                _, loss_value = sess.run([value_network.optimizer, value_network.loss], feed_dict_value)
                results['LossV'].append(loss_value)
                
                # Update the actor
                feed_dict_policy = {
                    policy.state: transition.state,
                    policy.R_t: td_error,
                    policy.action: transition.action
                }
                _, loss_policy = sess.run([policy.optimizer, policy.loss], feed_dict_policy)
                results['Loss'].append(loss_policy)
    
    results['Duration'] = time.time() - start

    return results



if __name__ == '__main__':
    run_actor_critic()
