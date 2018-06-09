from __future__ import print_function
from __future__ import division

import os
import argparse
import gym
import numpy as np
import tensorflow as tf

from agent import ActorCritic
from utils import *

def preprocess(obs):
    obs = obs[35:195]
    obs = obs[::2, ::2, 0]
    obs[obs == 144] = 0
    obs[obs == 109] = 0
    obs[obs != 0] = 1

    return obs.astype(np.float).ravel()

def setup_summary():
    episode_total_reward_episode = tf.Variable(0.)
    tf.summary.scalar("AC_Total_Reward/episode", episode_total_reward_episode)
    episode_duration_episode = tf.Variable(0.)
    tf.summary.scalar("AC_Duration/episode", episode_duration_episode)
    episode_ave_max_q_episode = tf.Variable(0.)
    tf.summary.scalar("AC_MAX_Reward/episode", episode_ave_max_q_episode)

    summary_vars = [episode_total_reward_episode, episode_duration_episode, episode_ave_max_q_episode]
    summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
    update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
    summary_op = tf.summary.merge_all()
    return summary_placeholders, update_ops, summary_op

def main(args):
    INPUT_DIM = 80 * 80
    HIDDEN_UNITS = 200
    ACTION_DIM = 6
    MAX_EPISODES = 10000

    # load agent
    agent = ActorCritic(INPUT_DIM, HIDDEN_UNITS, ACTION_DIM)
    agent.construct_model(args.gpu)

    # load model or init a new
    saver = tf.train.Saver(max_to_keep=1)
    #if args.model_path is not None:
    if True:
        # reuse saved model
        saver.restore(agent.sess,"/home/sugon/Peixian/atari-pong/reinforce_py/algorithms/Actor-Critic/model/summary/1.13_9901")
        #ep_base = int(args.save_path.split('_')[-1])
        #mean_rewards = float(args.save_path.split('/')[-1].split('_')[0])
    else:
        # build a new model
        agent.sess.run(tf.global_variables_initializer())
        ep_base = 0
        mean_rewards = None

    summary_writer = tf.summary.FileWriter("./summary/", agent.sess.graph)
    summary_placeholders, update_ops, summary_op = setup_summary()

    # load env
    env = gym.make('Pong-v0')

    win = 0
    f = open("./ac_score.txt","w")
    # training loop
    for ep in range(100):
        # reset env
        step = 0
        total_rewards = 0
        state = preprocess(env.reset())

        while True:
            # sample actions
            action = agent.sample_action(state[np.newaxis, :])
            # act!
            next_state, reward, done, _ = env.step(action)

            next_state = preprocess(next_state)

            step += 1
            total_rewards += reward

            agent.store_rollout(state, action, reward, next_state, done)
            # state shift
            state = next_state

            if done:
                print (win, ":", str(total_rewards) + " " + str(step))
                f.write("score:" + str(total_rewards) + " " + str(step) + "\n")
                if total_rewards > 0:
                    win += 1
                break


        #if mean_rewards is None:
         #   mean_rewards = total_rewards
        #else:
         #   mean_rewards = 0.99 * mean_rewards + 0.01 * total_rewards

        #stats = [total_rewards, step, mean_rewards]
       # for i in range(len(stats)):
        #    agent.sess.run(update_ops[i], feed_dict={
         #       summary_placeholders[i]: float(stats[i])
          #  })
        #summary_str = agent.sess.run(summary_op)
        #summary_writer.add_summary(summary_str, ep + 1)

#        rounds = (21 - np.abs(total_rewards)) + 21
 #       average_steps = (step + 1) / rounds
  #      print('Ep%s: %d rounds' % (ep_base + ep + 1, rounds))
   #     print('Average_steps: %.2f Reward: %s Average_reward: %.4f' %
    #          (average_steps, total_rewards, mean_rewards))

        # update model per episode
        #agent.update_model()
        # model saving
        #if ep > 0 and ep % args.save_every == 0:
         #   if not os.path.isdir(args.save_path):
          #      os.makedirs(args.save_path)
           # save_name = str(round(mean_rewards, 2)) + '_' + str(ep_base + ep+1)
            #saver.save(agent.sess, args.save_path + save_name)
    print ("win:",win)
    f.close()


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path', default=None,
        help='Whether to use a saved model. (*None|model path)')
    parser.add_argument(
        '--save_path', default='./model/summary/',
        help='Path to save a model during training.')
    parser.add_argument(
        '--save_every', default=100, help='Save model every x episodes')
    parser.add_argument(
        '--gpu', default=-1,
        help='running on a specify gpu, -1 indicates using cpu')
    return parser.parse_args()


if __name__ == '__main__':
    main(args_parse())
