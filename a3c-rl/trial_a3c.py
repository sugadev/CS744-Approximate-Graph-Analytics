import os
import time
import networkx as nx
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import copy
import threading
import gym
import multiprocessing
import numpy as np
from queue import Queue
import argparse
import matplotlib
matplotlib.use('Agg') # no UI backend
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers
import random
tf.enable_eager_execution()
from multiprocessing.pool import ThreadPool as Pool

pool_size = 10
parser = argparse.ArgumentParser(description='Run A3C algorithm on the game '
                                             'Cartpole.')
parser.add_argument('--algorithm', default='a3c', type=str,
                    help='Choose between \'a3c\' and \'random\'.')
parser.add_argument('--train', dest='train', action='store_true',
                    help='Train our model.')
parser.add_argument('--lr', default=0.001,
                    help='Learning rate for the shared optimizer.')
parser.add_argument('--update-freq', default=20, type=int,
                    help='How often to update the global model.')
parser.add_argument('--max-eps', default=1000, type=int,
                    help='Global maximum number of episodes to run.')
parser.add_argument('--gamma', default=0.99,
                    help='Discount factor of rewards.')
parser.add_argument('--save-dir', default='/tmp/', type=str,
                    help='Directory in which you desire to save the model.')
args = parser.parse_args()

Graph=nx.read_edgelist('/home/krishraj95/Big_Data_Project/CS744-Approximate-Graph-Analytics/Featurization/final_features/graph_119.embeddings', nodetype=int, data=(('f1',float),('f2',float),('f3',float),('f4',float),('f5',float),('f6',float),('f7',float),('f8',float),('f9',float),('f10',float)))
G = copy.deepcopy(Graph)
E = len(G.edges(data=True))
sr = 0.7

def getTrianglesCount(G):
  triangles = nx.triangles(G).values()
  res = 0

  for t in triangles:
    res+=t

  return int(res/3);
orgValue = getTrianglesCount(G)

def get_reward(newG):
  t2 = getTrianglesCount(newG)

  # if diff is more discourage it
  diff = orgValue - t2
  # return diff
  
  reward = diff * 1.0/orgValue
  # return 1
  # return random.random()
  return reward
def take_action(graph, edge, action):
  if action == 1:
    graph.remove_edge(edge[0], edge[1])
    return graph, 1

  else:
    return graph, 0

def convert_to_state(graph, edgevec):
  graph_vec = []
  graph_vec.append(nx.number_of_nodes(graph))
  graph_vec.append(nx.density(graph))
  graph_vec.append(nx.number_of_edges(graph))

  for i in edgevec[2]:
    graph_vec.append(edgevec[2][i])

  return graph_vec

org_triangle = getTrianglesCount(G)

class ActorCriticModel(keras.Model):
  def __init__(self, state_size, action_size):
    super(ActorCriticModel, self).__init__()
    self.state_size = state_size
    self.action_size = action_size
    self.dense1 = layers.Dense(50, activation='relu')
    self.policy_logits = layers.Dense(2)
    self.dense2 = layers.Dense(50, activation='relu')
    self.values = layers.Dense(1)

  def call(self, inputs):
    # Forward pass
    x = self.dense1(inputs)
    logits = self.policy_logits(x)
    v1 = self.dense2(inputs)
    values = self.values(v1)
    return logits, values

def record(episode,
           episode_reward,
           worker_idx,
           global_ep_reward,
           result_queue,
           total_loss,
           num_steps):
  """Helper function to store score and print statistics.
  Arguments:
    episode: Current episode
    episode_reward: Reward accumulated over the current episode
    worker_idx: Which thread (worker)
    global_ep_reward: The moving average of the global reward
    result_queue: Queue storing the moving average of the scores
    total_loss: The total loss accumualted over the current episode
    num_steps: The number of steps the episode took to complete
  """
  if global_ep_reward == 0:
    global_ep_reward = episode_reward
  else:
    global_ep_reward = global_ep_reward * 0.99 + episode_reward * 0.01
  print(
      "Episode: " + str(episode) + 
      "| Moving Average Reward: " + str(global_ep_reward) +
      "| Episode Reward: " + str(episode_reward) +
      "| Loss: " + str(int(total_loss / float(num_steps) * 1000) / 1000) +
      "| Steps: " + str(num_steps) +
      "| Worker: " + str(worker_idx)
  )
  
  result_queue.put(global_ep_reward)
  return global_ep_reward


class RandomAgent:
  """Random Agent that will play the specified game
    Arguments:
      env_name: Name of the environment to be played
      max_eps: Maximum number of episodes to run agent for.
  """
  def __init__(self, max_eps):
    # self.env = gym.make(env_name)
    self.max_episodes = max_eps
    self.global_moving_average_reward = 0
    self.res_queue = Queue()

  def run(self):
    reward_avg = 0
    for episode in range(self.max_episodes):
      # done = False
      curr_graph = G 
      # self.env.reset()
      reward_sum = 0.0
      steps = 0
      sampling_ratio = 1
      while not sampling_ratio < sr:
        # Sample randomly from the action space and step
        # _, reward, done, _ = self.env.step(self.env.action_space.sample())
        val1 = get_reward(curr_graph) 
        edges = curr_graph.edges(data = True)
        edge = list(edges)[random.randint(0,len(edges)-1)]
  
        new_graph, action_taken = take_action(copy.deepcopy(curr_graph), edge, random.randint(0,1)) 
        val2 = get_reward(new_graph) 
        reward = val2 - val1
 
        steps += 1
        reward_sum += reward
        curr_graph = new_graph
        new_edges = new_graph.edges(data = True)
        sampling_ratio = len(new_edges) * 1.0 / len(edges)
        # print(sampling_ratio)
     
      # Record statistics
      self.global_moving_average_reward = record(episode,
                                                 reward_sum,
                                                 0,
                                                 self.global_moving_average_reward,
                                                 self.res_queue, 0, steps)

    reward_avg += reward_sum
    final_avg = reward_avg / float(self.max_episodes)
    print("Average score across {} episodes: {}".format(self.max_episodes, final_avg))
    return final_avg


class MasterAgent():
  def __init__(self):
    # self.game_name = 'CartPole-v0'
    save_dir = args.save_dir
    self.save_dir = save_dir
    flag = False
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)

    else:
      flag = True
    # env = gym.make(self.game_name)
    # self.state_size = env.observation_space.shape[0]
    self.state_size = pow(2, E) - 2
    # self.action_size = env.action_space.n
    self.action_size = 2
    self.opt = tf.train.AdamOptimizer(float(args.lr), use_locking=True)
    print(self.state_size, self.action_size)

    self.global_model = ActorCriticModel(self.state_size, self.action_size)  # global network
    edges = G.edges(data=True)
    
    if flag:
      print("Loading from saved file")
      model_path = os.path.join(self.save_dir, 'model.h5') 
      self.global_model.load_weights(model_path)

    self.global_model(tf.convert_to_tensor(np.reshape(np.asarray(convert_to_state(G, list(edges)[random.randint(0, len(edges)-1)])), (-1, 13)), dtype=tf.float32))

  def train(self):
    if args.algorithm == 'random':
      # random_agent = RandomAgent(self.game_name, args.max_eps)
      random_agent = RandomAgent(args.max_eps)
      random_agent.run()
      return

    res_queue = Queue()

    # print(multiprocessing.cpu_count())
    workers = [Worker(self.state_size,
                      self.action_size,
                      self.global_model,
                      self.opt, res_queue,
                      i, 
                      # game_name=self.game_name,
                      save_dir=self.save_dir) for i in range(multiprocessing.cpu_count())]

    for i, worker in enumerate(workers):
      print("Starting worker {}".format(i))
      worker.start()

    moving_average_rewards = []  # record episode reward to plot
    while True:
      reward = res_queue.get()
      if reward is not None:
        moving_average_rewards.append(reward)
      else:
        break
    [w.join() for w in workers]

    plt.plot(moving_average_rewards)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.savefig(os.path.join(self.save_dir,
                             'Moving Average.png'))
    # plt.show()

  def play(self):
    # env = gym.make(self.game_name).unwrapped
    # state = env.reset()
    curr_graph = G
    model = self.global_model
    model_path = os.path.join(self.save_dir, 'model.h5')
    print('Loading model from: {}'.format(model_path))
    model.load_weights(model_path)
    # done = False
    step_counter = 0
    reward_sum = 0

    try:
      sampling_ratio = 1
      while not sampling_ratio < sr:
        # env.render(mode='rgb_array')
        curr_edges = curr_graph.edges(data = True)
        policy, value = model(tf.convert_to_tensor(np.reshape(np.asarray(convert_to_state(curr_graph, list(curr_edges)[random.randint(0, len(curr_edges)-1)])), (-1, 13)), dtype=tf.float32))
        policy = tf.nn.softmax(policy)
        action = np.argmax(policy)
        # state, reward, done, _ = env.step(action)
  
        val1 = get_reward(curr_graph) 
        edges = curr_graph.edges(data = True)
        edge = list(edges)[random.randint(0,len(edges)-1)]
 
        new_graph, action = take_action(copy.deepcopy(curr_graph), edge, action) 
        val2 = get_reward(new_graph) 
        reward = val2 - val1
  
        reward_sum += reward
        curr_graph = new_graph
        new_edges = new_graph.edges(data = True)
        sampling_ratio = len(new_edges) * 1.0 / len(edges)
 
        # print("Considering edge {} and sampling ratio of {}".format(edge, sampling_ratio))
        # print("{}. Reward: {}, action: {}".format(step_counter, reward_sum, action))
        step_counter += 1
      print(curr_graph.edges())
    except KeyboardInterrupt:
      print("Received Keyboard Interrupt. Shutting down.")


class Memory:
  def __init__(self):
    self.states = []
    self.actions = []
    self.rewards = []

  def store(self, state, action, reward):
    self.states.append(state)
    self.actions.append(action)
    self.rewards.append(reward)

  def clear(self):
    self.states = []
    self.actions = []
    self.rewards = []


class Worker(threading.Thread):
  # Set up global variables across different threads
  global_episode = 0
  # Moving average reward
  global_moving_average_reward = 0
  best_score = 0
  save_lock = threading.Lock()

  def __init__(self,
               state_size,
               action_size,
               global_model,
               opt,
               result_queue,
               idx,
               # game_name='CartPole-v0',
               save_dir='/tmp'):
    super(Worker, self).__init__()
    self.state_size = state_size
    self.action_size = action_size
    self.result_queue = result_queue
    self.global_model = global_model
    self.opt = opt
    self.local_model = ActorCriticModel(self.state_size, self.action_size)
    self.worker_idx = idx
    # self.game_name = game_name
    # self.env = gym.make(self.game_name).unwrapped
    self.save_dir = save_dir
    self.ep_loss = 0.0

  def run(self):
    total_step = 1
    mem = Memory()
    while Worker.global_episode < args.max_eps:
      # current_state = self.env.reset()
      curr_graph = G 
      mem.clear()
      ep_reward = 0.
      ep_steps = 0
      self.ep_loss = 0

      time_count = 0
      # done = False
      sampling_ratio = 1
      while not sampling_ratio < sr:
        curr_edges = curr_graph.edges(data=True)
        logits, _ = self.local_model(tf.convert_to_tensor(np.reshape(np.asarray(convert_to_state(curr_graph, list(curr_edges)[random.randint(0, len(curr_edges)-1)])), (-1, 13)), dtype=tf.float32))
        probs = tf.nn.softmax(logits)
        # sess = tf.Session()
        # print("Probabilities")
        # print(probs) 
        # print(type(probs.eval(session=sess)))
        action = np.random.choice(self.action_size, p=probs.numpy()[0])
        
        # new_state, reward, done, _ = self.env.step(action)
        val1 = get_reward(curr_graph) 
        edges = curr_graph.edges(data = True)
        edge = list(edges)[random.randint(0,len(edges)-1)]
  
        new_graph, action = take_action(copy.deepcopy(curr_graph), edge, action)
        val2 = get_reward(new_graph) 
        reward = val2 - val1

        new_edges = new_graph.edges(data = True)
        sampling_ratio = len(new_edges) * 1.0 / len(edges)
        # print(str(self.worker_idx) + " has sampling ratio " + str(sampling_ratio) + " in " + str(Worker.global_episode))
       
        if sampling_ratio < sr:
          reward = 0
        ep_reward += reward
  
        curr_state = convert_to_state(curr_graph, edge)
        mem.store(curr_state, action, reward)

        new_state = convert_to_state(new_graph, edge)
        if time_count == args.update_freq or sampling_ratio < sr:
          # Calculate gradient wrt to local model. We do so by tracking the
          # variables involved in computing the loss by using tf.GradientTape
          with tf.GradientTape() as tape:
            total_loss = self.compute_loss(sampling_ratio < sr,
                                           new_state,
                                           mem,
                                           float(args.gamma))
          self.ep_loss += total_loss
          # Calculate local gradients
          grads = tape.gradient(total_loss, self.local_model.trainable_weights)
           
          # print(grads)
          # Push local gradients to global model
          self.opt.apply_gradients(zip(grads,
                                       self.global_model.trainable_weights))
          # Update local model with new weights
          self.local_model.set_weights(self.global_model.get_weights())

          mem.clear()
          time_count = 0

          if sampling_ratio < sr:  # done and print information
            Worker.global_moving_average_reward = \
              record(Worker.global_episode, ep_reward, self.worker_idx,
                     Worker.global_moving_average_reward, self.result_queue,
                     self.ep_loss, ep_steps)
            # We must use a lock to save our model and to print to prevent data races.
            # if ep_reward >= Worker.best_score:
            with Worker.save_lock:
              print("Saving best model to {}, "
                      "episode score: {}".format(self.save_dir, ep_reward))
              self.global_model.save_weights(
                    os.path.join(self.save_dir,
                                 'model.h5')
                )
              Worker.best_score = ep_reward
            Worker.global_episode += 1
        ep_steps += 1

        # print("Episode steps " + str(ep_steps))
        time_count += 1
        # current_state = new_state
        curr_graph = new_graph
        total_step += 1
    self.result_queue.put(None)

  def compute_loss(self,
                   done,
                   new_state,
                   memory,
                   gamma=0.99):
    # print("New state")
    # print(new_state)
    if done:
      reward_sum = 0.0  # terminal
    else:
      reward_sum_arr = self.local_model(tf.convert_to_tensor(np.reshape(np.asarray(new_state), (-1,13)), dtype=tf.float32))[-1].numpy()[0].tolist()
      reward_sum = reward_sum_arr[0]

    # print(reward_sum)
    # Get discounted rewards
    discounted_rewards = []
    # print("Rewards")
    for reward in memory.rewards[::-1]:  # reverse buffer r
      # print("1")
      # import pdb
      # pdb.set_trace()
      # print(reward)
      # print("2")
      # print(reward_sum)
      # print("3")
      # print(gamma)
      reward_sum = reward + gamma * reward_sum
      discounted_rewards.append(reward_sum)
    discounted_rewards.reverse()

    logits, values = self.local_model(tf.convert_to_tensor(np.reshape(np.vstack(memory.states), (-1, 13)), dtype=tf.float32))

    # Get our advantages
    advantage = tf.convert_to_tensor(np.array(discounted_rewards)[:, None],dtype=tf.float32) - values
    # Value loss
    value_loss = advantage ** 2

    # Calculate our policy loss
    policy = tf.nn.softmax(logits)
    entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=policy, logits=logits)

    policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=memory.actions,logits=logits)
    policy_loss *= tf.stop_gradient(advantage)
    policy_loss -= 0.01 * entropy
    total_loss = tf.reduce_mean((0.5 * value_loss + policy_loss))
    return total_loss


if __name__ == '__main__':
  g = 119
  master = MasterAgent()
  if args.train:
    # global G
    # global E

    # Graph=nx.read_edgelist('/home/krishraj95/Big_Data_Project/CS744-Approximate-Graph-Analytics/Featurization/final_features/graph_' + str(g) + '.embeddings', nodetype=int, data=(('f1',float),('f2',float),('f3',float),('f4',float),('f5',float),('f6',float),('f7',float),('f8',float),('f9',float),('f10',float)))

    # G = copy.deepcopy(Graph)
    # E = len(G.edges(data=True))
    master.train()

  else:
    master.play()
