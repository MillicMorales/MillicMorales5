import numpy as np
import tensorflow as tf

class DeepQNetwork:
    def __init__(self, state_size, action_size, learning_rate=0.001, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

    def build_model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(tf.keras.layers.Dense(24, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def train(self, state, action, reward, next_state, done):
        target = self.model.predict(state)
        if done:
            target[0][action] = reward
        else:
            q_future = np.max(self.target_model.predict(next_state)[0])
            target[0][action] = reward + self.discount_factor * q_future
        self.model.fit(state, target, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

if __name__ == '__main__':
    # Example usage: Training a Deep Q-Network (DQN) on a simple environment

    env = gym.make('CartPole-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    dqn = DeepQNetwork(state_size, action_size)

    num_episodes = 1000
    for episode in range(num_episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        for time_step in range(500):
            action = dqn.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            dqn.train(state, action, reward, next_state, done)
            state = next_state

            if done:
                break

        if episode % 10 == 0:
            dqn.update_target_model()

    # Evaluate the trained agent
    total_reward = 0
    num_eval_episodes = 100
    for _ in range(num_eval_episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        done = False

        while not done:
            action = dqn.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            state = next_state
            total_reward += reward

    average_reward = total_reward / num_eval_episodes
    print("Average reward:", average_reward)
