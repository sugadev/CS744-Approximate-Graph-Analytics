# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import os
import tensorflow as tf
import numpy as np
import gym
import gym_sampler
import random
from builtins import input
import pdb
import networkx as nx
import copy

from trainer.helpers import discount_rewards, prepro
from agents.tools.wrappers import AutoReset, FrameHistory
from collections import deque

# Open AI gym Atari env: 0: 'Keep edge', 1: 'Remove edge'
ACTIONS = [0, 1]
OBSERVATION_DIM = 16
MEMORY_CAPACITY = 100
ROLLOUT_SIZE = 10

# MEMORY stores tuples:
# (observation, label, reward)
MEMORY = deque([], maxlen=MEMORY_CAPACITY)
def gen():
    for m in list(MEMORY):
        yield m
def get_triangle_count(graph):
        triangles = nx.triangles(graph).values()
        res =0
        for t in triangles:
            res+=t
        return int(res/3)
def build_graph(observations):
    """Calculates logits from the input observations tensor.
    This function will be called twice: rollout and train.
    The weights will be shared.
    """
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
        hidden = tf.layers.dense(observations, args.hidden_dim, use_bias=False, activation=tf.nn.relu)
        logits = tf.layers.dense(hidden, len(ACTIONS), use_bias=False)
        logits = tf.nn.softmax(logits)

    return logits

def main(args):
    args_dict = vars(args)
    print('args: {}'.format(args_dict))
    
    with tf.Graph().as_default() as g:
        # rollout subgraph
        with tf.name_scope('rollout'):
            observations = tf.placeholder(shape=(None, OBSERVATION_DIM), dtype=tf.float32)
            
            logits = build_graph(observations)

            logits_for_sampling = tf.reshape(logits, shape=(1, len(ACTIONS)))

            # # Sample the action to be played during rollout.
            sample_action = tf.squeeze(tf.multinomial(logits=logits_for_sampling, num_samples=1))
        
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate=args.learning_rate,
            decay=args.decay
        )

        # dataset subgraph for experience replay
        with tf.name_scope('dataset'):
            # the dataset reads from MEMORY
            ds = tf.data.Dataset.from_generator(gen, output_types=(tf.float32, tf.int32, tf.float32))
            ds = ds.shuffle(MEMORY_CAPACITY).repeat().batch(args.batch_size)
            iterator = ds.make_one_shot_iterator()

        # training subgraph
        with tf.name_scope('train'):
            # the train_op includes getting a batch of data from the dataset, so we do not need to use a feed_dict when running the train_op.
            next_batch = iterator.get_next()
            train_observations, labels, processed_rewards = next_batch

            # This reuses the same weights in the rollout phase.
            train_observations.set_shape((args.batch_size, OBSERVATION_DIM))
            train_logits = build_graph(train_observations)

            cross_entropies = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=train_logits,
                labels=labels
            )

            # Extra loss when the paddle is moved, to encourage more natural moves.
            # probs = tf.nn.softmax(logits=train_logits)
            # move_cost = tf.reduce_sum(probs * [1.0, 1.0], axis=1)

            loss = tf.reduce_sum(processed_rewards * cross_entropies)

            global_step = tf.train.get_or_create_global_step()

            train_op = optimizer.minimize(loss, global_step=global_step)

        init = tf.global_variables_initializer()

        print('Number of trainable variables: {}'.format(len(tf.trainable_variables())))


    env = gym.make('sampler-v0')

    with tf.Session(graph=g) as sess:
        sess.run(init)

        # lowest possible score after an episode as the
        # starting value of the running reward
        _rollout_reward = 0
        epo =0
        for i in range(1):
            # print('>>>>>>> epoch {}'.format(i+1))

            # print('>>> Rollout phase')
            epoch_memory = []
            episode_memory = []

            # The loop for actions/stepss
            env.reset()
            _observation = env.getFirstState()
            while True:
                # sample one action with the given probability distribution
                # print(_observation)
                _label = sess.run(sample_action, feed_dict={observations: [_observation]})
                
                # rFloat = random.random()
                # print("here", _label)
                # if rFloat < 0.5:
                #     _label = 0
                # else:
                #     _label = 1

                # print(_label)
                _action = ACTIONS[_label]

                cur_state, _reward, _done, graph_done = env.step(_action)

                # record experience
                episode_memory.append((_observation, _action, _reward))

                # Get processed frame delta for the next step
                _observation = cur_state

                if _done:
                    obs, lbl, rwd = zip(*episode_memory)

                    # processed rewards
                    prwd = discount_rewards(rwd, args.gamma)
                    prwd -= np.mean(prwd)
                    prwd /= np.std(prwd)

                    # store the processed experience to memory
                    epoch_memory.extend(zip(obs, lbl, prwd))
                    
                    # calculate the running rollout reward
                    _rollout_reward = 0.9 * _rollout_reward + 0.1 * sum(rwd)
                    episode_memory = []

                if graph_done:
                    epo+=1
                    env.reset()
                    MEMORY.extend(epoch_memory)
                    _observation = env.getFirstState()

                if epo ==10:
                    for i in [5,10,15,20,50,100]:
                        train_graph = "/Users/daravinds/Documents/Projects/CS-744/gym-sampler/gym_sampler/envs/test/graph_"+str(i)+".embeddings"
                        infer = nx.read_edgelist(train_graph, nodetype=int, data=(('f1',float),('f2',float),('f3',float),('f4',float),('f5',float),('f6',float),('f7',float),('f8',float),('f9',float),('f10',float),))
                        # infer = nx.complete_graph(20)
                        print("nodes: ",i)
                        num_triangles = get_triangle_count(infer)
                        edges = copy.deepcopy(infer.edges())
                        removed_count =0
                        num_egdes = infer.number_of_edges()
                        sampling_factor = 0.7
                        print("edges",num_egdes)
                        for e in edges:
                            features = list(infer.get_edge_data(e[0],e[1]).values())
                            #graph feartures
                            features.append(infer.number_of_edges())
                            features.append(infer.number_of_nodes())
                            features.append(infer.number_of_edges()*2.0/infer.number_of_nodes())
                            #algo features
                            features.append(0)
                            features.append(0)
                            features.append(1)

                            label = sess.run(sample_action, feed_dict={observations: [features]})
                            # print(label)
                            action = ACTIONS[label]
                            if action == 1:
                                infer.remove_edge(e[0],e[1])
                                removed_count+=1;
                                if(removed_count >= num_egdes*sampling_factor):
                                    break
                        edges_remaining = infer.number_of_edges()
                        print("Sampling: ",1.0*edges_remaining/num_egdes)
                        print("edges_remaining",edges_remaining)
                        print("Accuracy end",1.0*get_triangle_count(infer)/num_triangles)
                        print("updated\n\n\n")
                    break

            # add to the global memory
           

            print('>>> done')
            print('rollout reward: {}'.format(_rollout_reward))

            # Here we train only once.
            _, _global_step = sess.run([train_op, global_step])

        
        # cur_state, _reward, _done, graph_done = env.step(_action)

    print("end")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('pong trainer')
    parser.add_argument('--n-epoch', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=1000)
    parser.add_argument('--output-dir', type=str, default='/tmp/pong_output')
    parser.add_argument('--job-dir', type=str, default='/tmp/pong_output')
    parser.add_argument('--render', default=False, action='store_true')
    parser.add_argument('--learning-rate', type=float, default=5e-3)
    parser.add_argument('--decay', type=float, default=0.99)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--hidden-dim', type=int, default=200)
    args = parser.parse_args()
    main(args)
    print("end")
