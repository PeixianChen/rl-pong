# coding:utf-8

import os
import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque
# from skimage.color import rgb2gray
# from skimage.transform import resize
import cv2
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers.convolutional import Conv2D
import time
import psutil

# CUDA_VISIBLE_DEVICES = 0 # GPU can be seen
ENV_NAME = 'Pong-v0'  # Environment name
FRAME_WIDTH = 84  # Resized frame width
FRAME_HEIGHT = 84  # Resized frame height
NUM_EPISODES = 12000  # Number of episodes the agent plays
STATE_LENGTH = 4  # Number of most recent frames to produce the input to the network
GAMMA = 0.99  # Discount factor
EXPLORATION_STEPS = 10000000  # Number of steps over which the initial value of epsilon is linearly annealed to its final value
INITIAL_EPSILON = 0.1  # Initial value of epsilon in epsilon-greedy
FINAL_EPSILON = 0.1  # Final value of epsilon in epsilon-greedy
INITIAL_REPLAY_SIZE = 15260000 + 2000  # Number of steps to populate the replay memory before training starts
NUM_REPLAY_MEMORY = 400000  # Number of replay memory the agent uses for training
BATCH_SIZE = 32  # Mini batch size
TARGET_UPDATE_INTERVAL = 10000  # The frequency with which the target network is updated
TRAIN_INTERVAL = 4  # The agent selects 4 actions between successive updates
LEARNING_RATE = 0.00025  # Learning rate used by RMSProp
MOMENTUM = 0.95  # Momentum used by RMSProp
MIN_GRAD = 0.01  # Constant added to the squared gradient in the denominator of the RMSProp update
SAVE_INTERVAL = 4000  # The frequency with which the network is saved
NO_OP_STEPS = 30  # Maximum number of "do nothing" actions to be performed by the agent at the start of an episode
LOAD_NETWORK = True
TRAIN = False
SAVE_NETWORK_PATH = 'saved_networks/' + ENV_NAME
SAVE_SUMMARY_PATH = 'summary/' + ENV_NAME
NUM_EPISODES_AT_TEST = 30  # Number of episodes the agent plays at test time
OUTPUT_GRAPH = True


class Agent():
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.epsilon = INITIAL_EPSILON
        self.epsilon_step = (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORATION_STEPS
        self.t = 15260000

        # Parameters used for summary
        self.total_reward = 0
        self.total_q_max = 0
        self.total_loss = 0
        self.duration = 0
        self.episode = 5807

        # Create replay memory
        self.replay_memory = deque()

        # Create q network
        # self.s, self.q_values, q_network = self.build_network("q_network")
        # q_network_weights = q_network.trainable_weights
        self.s, self.q_values= self.build_network("q_network")
        q_network_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="q_network")

        # print (u'内存使用1：',psutil.Process(os.getpid()).memory_info().rss)

        # Create target network
        # self.st, self.target_q_values, target_network = self.build_network("target_network")
        # target_network_weights = target_network.trainable_weights
        self.st, self.target_q_values = self.build_network("target_network")
        target_network_weights = trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="target_network")
        # print (u'内存使用2：',psutil.Process(os.getpid()).memory_info().rss)

        # Define target network update operation
        self.update_target_network = [target_network_weights[i].assign(q_network_weights[i]) for i in range(len(target_network_weights))]

        # Define loss and gradient update operation
        self.a, self.y, self.loss, self.grads_update = self.build_training_op(q_network_weights)

        # self.sess = tf.InteractiveSession(config=tf.ConfigProto(device_count={'gpu':0}))
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        # self.sess = tf.InteractiveSession(config=config)
        self.sess = tf.Session(config=config)
        self.summary_writer = tf.summary.FileWriter(SAVE_SUMMARY_PATH, self.sess.graph)
        self.saver = tf.train.Saver(q_network_weights)
        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()

        if not os.path.exists(SAVE_NETWORK_PATH):
            os.makedirs(SAVE_NETWORK_PATH)

        self.sess.run(tf.global_variables_initializer())
        self.sess.graph.finalize()
        # print (u'内存使用3：',psutil.Process(os.getpid()).memory_info().rss)




        # Load network
        if LOAD_NETWORK:
            self.load_network()

        # Initialize target network
        self.sess.run(self.update_target_network)


    def build_network(self, name):
        with tf.variable_scope(name):
            s = tf.placeholder(tf.float32, [None, FRAME_WIDTH, FRAME_HEIGHT, STATE_LENGTH])
            conv1 = tf.layers.conv2d(s, filters=32, kernel_size=(8,8), strides=4, activation=tf.nn.relu)
            conv2 = tf.layers.conv2d(conv1, filters=64, kernel_size=(4,4), strides=2, activation=tf.nn.relu)
            conv3 = tf.layers.conv2d(conv2, filters=64, kernel_size=(3,3), strides=1, activation=tf.nn.relu)
            conv_flattened = tf.layers.flatten(conv3)
            fc1 = tf.layers.dense(conv_flattened, units=512, activation=tf.nn.relu)
            fc2 = tf.layers.dense(fc1, units=self.num_actions, activation=None)
        return s, fc2

        # model = Sequential()
        # model.add(Convolution2D(32, 8, 8, subsample=(4, 4), activation='relu', input_shape=(STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT)))
        # model.add(Convolution2D(64, 4, 4, subsample=(2, 2), activation='relu'))
        # model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu'))
        # model.add(Flatten())
        # model.add(Dense(512, activation='relu'))
        # model.add(Dense(self.num_actions))

        # s = tf.placeholder(tf.float32, [None, STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT])
        # q_values = model(s)

        # return s, q_values, model

    def build_training_op(self, q_network_weights):
        a = tf.placeholder(tf.int64, [None])
        y = tf.placeholder(tf.float32, [None])

        # Convert action to one hot vector
        a_one_hot = tf.one_hot(a, self.num_actions, 1.0, 0.0)
        q_value = tf.reduce_sum(tf.multiply(self.q_values, a_one_hot), reduction_indices=1)

        # Clip the error, the loss is quadratic when the error is in (-1, 1), and linear outside of that region
        error = tf.abs(y - q_value)
        quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)

        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, momentum=MOMENTUM, epsilon=MIN_GRAD)
        grads_update = optimizer.minimize(loss, var_list=q_network_weights)

        return a, y, loss, grads_update

    def get_initial_state(self, observation, last_observation):
        processed_observation = np.maximum(observation, last_observation)
        img = cv2.cvtColor(processed_observation,cv2.COLOR_BGR2GRAY) 
        img = cv2.resize(img, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_CUBIC)
        processed_observation = np.uint8(img * 255)
        state = [processed_observation for _ in range(STATE_LENGTH)]
        return np.stack(state, axis=0)

    def get_action(self, state):
        state = state.transpose(1, 2, 0)
        if self.epsilon >= random.random() or self.t < INITIAL_REPLAY_SIZE:
            action = random.randrange(self.num_actions)
            # print ("train random:", action)
        else:
            # action = np.argmax(self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]}))
            action = np.argmax(self.sess.run(self.q_values, feed_dict={self.s: [np.float32(state / 255.0)]}))

            # print ("train action:", action)

        # Anneal epsilon linearly over time
        if self.epsilon > FINAL_EPSILON and self.t >= INITIAL_REPLAY_SIZE:
            self.epsilon -= self.epsilon_step

        return action

    def run(self, state, action, reward, terminal, observation):
        # state = 4,84,84
        # observation = 1, 84,84
        next_state = np.append(state[1:, :, :], observation, axis=0)

        # Clip all positive rewards at 1 and all negative rewards at -1, leaving 0 rewards unchanged
        # reward = np.clip(reward, -1, 1)

        # Store transition in replay memory
        self.replay_memory.append((state, action, reward, next_state, terminal))
        if len(self.replay_memory) > NUM_REPLAY_MEMORY:
            self.replay_memory.popleft()# 移出队列

        # print (u'内存使用4：',psutil.Process(os.getpid()).memory_info().rss)

        if self.t >= INITIAL_REPLAY_SIZE:
            # Train network
            if self.t % TRAIN_INTERVAL == 0:
                self.train_network()
                # print (u'内存使用6：',psutil.Process(os.getpid()).memory_info().rss)


            # Update target network
            if self.t % TARGET_UPDATE_INTERVAL == 0:
                self.sess.run(self.update_target_network)
                # print (u'内存使用7：',psutil.Process(os.getpid()).memory_info().rss)


            # Save network
            if self.t % SAVE_INTERVAL == 0:
                save_path = self.saver.save(self.sess, SAVE_NETWORK_PATH + '/' + ENV_NAME, global_step=self.t)
                print('Successfully saved: ' + save_path)

        self.total_reward += reward
        # state = np.expand_dims(state,axis=0)
        state = state.transpose(1, 2, 0)
        # self.total_q_max += np.max(self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]}))
        self.total_q_max += np.max(self.sess.run(self.q_values, feed_dict={self.s: [np.float32(state / 255.0)]}))

        self.duration += 1

        if terminal:
            # Write summary
            if self.t >= INITIAL_REPLAY_SIZE:
                stats = [self.total_reward, self.total_q_max / float(self.duration),
                        self.duration, self.total_loss / (float(self.duration) / float(TRAIN_INTERVAL))]
                for i in range(len(stats)):
                    self.sess.run(self.update_ops[i], feed_dict={
                        self.summary_placeholders[i]: float(stats[i])
                    })
                summary_str = self.sess.run(self.summary_op)
                self.summary_writer.add_summary(summary_str, self.episode + 1)

                # print (u'内存使用8：',psutil.Process(os.getpid()).memory_info().rss)


            # Debug
            if self.t < INITIAL_REPLAY_SIZE:
                mode = 'random'
            elif INITIAL_REPLAY_SIZE <= self.t < INITIAL_REPLAY_SIZE + EXPLORATION_STEPS:
                mode = 'explore'
            else:
                mode = 'exploit'
            print('EPISODE: {0:6d} / TIMESTEP: {1:8d} / DURATION: {2:5d} / EPSILON: {3:.5f} / TOTAL_REWARD: {4:3.0f} / AVG_MAX_Q: {5:2.4f} / AVG_LOSS: {6:.5f} / MODE: {7}'.format(
                self.episode + 1, self.t, self.duration, self.epsilon,
                self.total_reward, self.total_q_max / float(self.duration),
                self.total_loss / (float(self.duration) / float(TRAIN_INTERVAL)), mode))

            self.total_reward = 0
            self.total_q_max = 0
            self.total_loss = 0
            self.duration = 0
            self.episode += 1

        self.t += 1

        return next_state

    def train_network(self):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        terminal_batch = []
        y_batch = []

        # Sample random minibatch of transition from replay memory
        minibatch = random.sample(self.replay_memory, BATCH_SIZE)
        for data in minibatch:
            state_batch.append(data[0])
            action_batch.append(data[1])
            reward_batch.append(data[2])
            next_state_batch.append(data[3])
            terminal_batch.append(data[4])

        # Convert True to 1, False to 0
        terminal_batch = np.array(terminal_batch) + 0

        next_state_batch = np.array(next_state_batch).transpose(0, 2, 3, 1)
        # target_q_values_batch = self.target_q_values.eval(feed_dict={self.st: np.float32(next_state_batch / 255.0)})
        xx = psutil.Process(os.getpid()).memory_info().rss
        # print (u'内存使用6.1：', xx)
        target_q_values_batch = self.sess.run(self.target_q_values, feed_dict={self.st: np.float32(next_state_batch / 255.0)})
        # print (u'内存使用6.2：', psutil.Process(os.getpid()).memory_info().rss - xx)
        xx = psutil.Process(os.getpid()).memory_info().rss
        y_batch = reward_batch + (1 - terminal_batch) * GAMMA * np.max(target_q_values_batch, axis=1)
        # print (len(reward_batch))
        # print (np.max(target_q_values_batch, axis=1).shape)
        # print (y_batch.shape)

        state_batch = np.array(state_batch).transpose(0, 2, 3, 1)
        loss, _ = self.sess.run([self.loss, self.grads_update], feed_dict={
            self.s: np.float32(state_batch / 255.0),
            self.a: action_batch,
            self.y: y_batch
        })

        # print (u'内存使用6.3：',psutil.Process(os.getpid()).memory_info().rss - xx)
        self.total_loss += loss
        del loss
        del _

    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + '/Total Reward/Episode', episode_total_reward)
        episode_avg_max_q = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + '/Average Max Q/Episode', episode_avg_max_q)
        episode_duration = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + '/Duration/Episode', episode_duration)
        episode_avg_loss = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + '/Average Loss/Episode', episode_avg_loss)
        summary_vars = [episode_total_reward, episode_avg_max_q, episode_duration, episode_avg_loss]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op

    def load_network(self):
        print ("---------------------------------------------------")
        checkpoint = tf.train.get_checkpoint_state(SAVE_NETWORK_PATH)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print('Successfully loaded: ' + checkpoint.model_checkpoint_path)
        else:
            print('Training new network...')

    def get_action_at_test(self, state):
        state = state.transpose(1, 2, 0)
        if random.random() <= 0:
            # print ("random random random")
            action = random.randrange(self.num_actions)
        else:
            # action = np.argmax(self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]}))
            action = np.argmax(self.sess.run(self.q_values, feed_dict={self.s: [np.float32(state / 255.0)]}))

        self.t += 1

        return action




def preprocess(observation, last_observation):
    processed_observation = np.maximum(observation, last_observation)
    img = cv2.cvtColor(processed_observation,cv2.COLOR_BGR2GRAY) 
    img = cv2.resize(img, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_CUBIC)
    processed_observation = np.uint8(img * 255)
    # processed_observation = np.uint8(resize(rgb2gray(processed_observation), (FRAME_WIDTH, FRAME_HEIGHT)) * 255)
    return np.reshape(processed_observation, (1, FRAME_WIDTH, FRAME_HEIGHT))


def main():
    env = gym.make(ENV_NAME)
    agent = Agent(num_actions=env.action_space.n)
    f = open("./dqn_score.txt","w")
    win = 0
    if TRAIN:  # Train mode
        for _ in range(NUM_EPISODES):
            terminal = False 
            observation = env.reset()
            for _ in range(random.randint(1, NO_OP_STEPS)):
                last_observation = observation
                observation, _, _, _ = env.step(0)  # Do nothing
            state = agent.get_initial_state(observation, last_observation)
            while not terminal:
                last_observation = observation
                action = agent.get_action(state)
                observation, reward, terminal, _ = env.step(action)
                # env.render()
                processed_observation = preprocess(observation, last_observation)
                state = agent.run(state, action, reward, terminal, processed_observation)
    else:  # Test mode
        # env.monitor.start(ENV_NAME + '-test')
        for _ in range(100):
            terminal = False
            observation = env.reset()
            for _ in range(random.randint(1, NO_OP_STEPS)):
                last_observation = observation
                observation, _, _, _ = env.step(0)  # Do nothing
            state = agent.get_initial_state(observation, last_observation)
            # pre_state = state
            t = 0
            score = 0
            while not terminal:
                t += 1
                last_observation = observation
                action = agent.get_action_at_test(state)
                observation, rew, terminal, _ = env.step(action)
                
                score += rew
                env.render()
                processed_observation = preprocess(observation, last_observation)
                state = np.append(state[1:, :, :], processed_observation, axis=0)
            f.write("score:" + str(score) + " " + str(t) + "\n")
            if score > 0:win += 1
            print ("TIMESTEP:", t, " score:", score)
            # time.sleep(2)
                # if(pre_state == state).all():
                #     print ("same same same")
                # else:
                #     print ("different different")
                # pre_state = state
        # env.monitor.close()
        f.close()
        print ("win:", win)


if __name__ == '__main__':
    main()
