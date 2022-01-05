"""
Implements all FFN based models
TODO: Port to pytorch
"""
import tensorflow as tf
from tensorflow.keras import layers


class ActorCritic(tf.keras.Model):
    def __init__(self, hidden_units, action_size, init):
        super(ActorCritic, self).__init__()
        if init == 'xavier':
            self.init = tf.initializers.GlorotNormal()
        elif init == 'gauss':
            self.init = tf.initializers.RandomNormal(mean=0.0, stddev=0.01)
        self.action_size = action_size
        self.fc_layers = [layers.Dense(units, activation="relu",
                                       kernel_initializer=self.init) for units in hidden_units]
        self.actor = layers.Dense(action_size, kernel_initializer=self.init)
        self.critic = layers.Dense(1, kernel_initializer=self.init)

    def call(self, x):
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
        return self.actor(x), self.critic(x)


class Agent(tf.keras.Model):
    def __init__(self, hidden_units, action_size, dueling=True, init='xavier'):
        super(Agent, self).__init__()

        if init == 'xavier':
            self.init = tf.initializers.GlorotNormal()
        elif init == 'gauss':
            self.init = tf.initializers.RandomNormal(mean=0.0, stddev=0.01)

        self.action_size = action_size
        self.fc_layers = [layers.Dense(units, activation="relu",
                                       kernel_initializer=self.init) for units in hidden_units]
        self.value_head = layers.Dense(1, kernel_initializer=self.init)
        self.advantage_head = layers.Dense(action_size, kernel_initializer=self.init)
        self.dueling = dueling

    def call(self, x):
        for fc_layer in self.fc_layers:
            x = fc_layer(x)

        if not self.dueling:
            return self.advantage_head(x)  # Treat advantage head as q output

        # Value and Advantage for dueling network
        value = self.value_head(x)
        advantage = self.advantage_head(x)
        # Process advantage to be zero mean
        advantage -= tf.math.reduce_mean(advantage, axis=-1, keepdims=True)
        value_tiled = tf.tile(value, tf.constant([1, self.action_size]))
        q_values = value_tiled + advantage

        return q_values
