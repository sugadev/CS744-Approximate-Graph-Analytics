import gym
from gym import error, spaces, utils
from gym.utils import seeding
import networkx as nx
import os, copy, random
import pdb
class SamplerEnv(gym.Env):
	metadata = {'render.modes': ['human']}
	train_folder = "/Users/daravinds/Documents/Projects/CS-744/gym-sampler/gym_sampler/envs/train/"
	graphs = dict()
	graph_name_list =[]
	current_graph = None
	next_graph_index = 0
	current_edge_index = None
	current_edge_list = None
	first_state = None
	prev_v1_v2 = (None, None)
	previous_graph = None
	
	def __init__(self):
		self.resetState()
	def calcAvgDegree(self,graph):
		return graph.number_of_edges()*2/graph.number_of_nodes()
	def resetState(self):
		self.graphs = dict()
		self.graph_name_list =[]
		self.current_graph = None
		self.next_graph_index = 0
		self.current_edge_index = None
		self.current_edge_list = None
		self.first_state = None
		self.prev_v1_v2 = (None, None)
		self.previous_graph = None
		for file in os.listdir(self.train_folder):
			G = nx.read_edgelist(self.train_folder+file, nodetype=int, data=(('f1',float),('f2',float),('f3',float),('f4',float),('f5',float),('f6',float),('f7',float),('f8',float),('f9',float),('f10',float),))
			self.graphs[file] = G
			self.graph_name_list.append(file)
		self.current_graph=self.graphs[self.graph_name_list[self.next_graph_index]]
		self.current_edge_list = list(self.current_graph.edges(data=True))
		self.next_graph_index = self.next_graph_index+1
		self.current_edge_index=0
		self.previous_graph = self.current_graph
		self.prev_v1_v2 = (self.current_edge_list[self.current_edge_index][0],self.current_edge_list[self.current_edge_index][1])
		self.first_state = list(self.current_graph.get_edge_data(self.current_edge_list[self.current_edge_index][0],self.current_edge_list[self.current_edge_index][1]).values())
		#graph features
		self.first_state.append(self.current_graph.number_of_edges())
		self.first_state.append(self.current_graph.number_of_nodes())
		self.first_state.append(self.calcAvgDegree(self.current_graph))
		#algo features
		self.first_state.append(0)#Does it perform a graph traversal?
		self.first_state.append(0)# Is the order of updates important?
		self.first_state.append(1)# Does it compute / update a local quantity?

		self.current_edge_index = self.current_edge_index +1
		if(self.current_edge_index == len(self.current_edge_list)):
			self.current_edge_index =0;
			self.current_graph = copy.deepcopy(self.graphs[self.graph_name_list[self.next_graph_index]])
			self.next_graph_index = self.next_graph_index +1
			self.current_edge_list = list(self.current_graph.edges(data=True))

	def getFirstState(self):
		return self.first_state

	def step(self, action):
		# print(self.current_graph.edges())
		# print(self.prev_v1_v2)
		self.cur_edge = self.current_edge_list[self.current_edge_index]
		v1 = self.prev_v1_v2[0]
		v2 = self.prev_v1_v2[1]
		# print("hjh",self.current_graph.edges(), self.current_edge_list, self.current_edge_list[self.current_edge_index][0], self.current_edge_list[self.current_edge_index][1])
		if self.current_graph.edges() is None:
			pdb.set_trace()
		self.cur_state = list(self.current_graph.get_edge_data(self.current_edge_list[self.current_edge_index][0],self.current_edge_list[self.current_edge_index][1]).values())
		# graph features
		self.cur_state.append(self.current_graph.number_of_edges())
		self.cur_state.append(self.current_graph.number_of_nodes())
		self.cur_state.append(self.calcAvgDegree(self.current_graph))
		#algo features
		self.cur_state.append(0)#Does it perform a graph traversal?
		self.cur_state.append(0)# Is the order of updates important?
		self.cur_state.append(1)# Does it compute / update a local quantity?

		self.prev_v1_v2 = (self.current_edge_list[self.current_edge_index][0],self.current_edge_list[self.current_edge_index][1])
		self.current_edge_index = self.current_edge_index+1
		#prev state features
		done = False
		graphs_done = False

		if(action == 1):
			self.previous_graph.remove_edge(v1,v2)
		rewards = self.get_rewards(self.graphs[self.graph_name_list[self.next_graph_index-1]],self.previous_graph)
		self.previous_graph = self.current_graph
		if(self.current_edge_index == len(self.current_edge_list)):
			self.current_edge_index =0;
			if(self.next_graph_index == len(self.graph_name_list)):
				graphs_done = True
				return 0,0,True,graphs_done
			self.current_graph = copy.deepcopy(self.graphs[self.graph_name_list[self.next_graph_index]])
			self.next_graph_index = self.next_graph_index +1
			self.current_edge_list = list(self.current_graph.edges(data=True))
			done = True
		return self.cur_state, rewards, done, graphs_done

	def get_rewards(self, orginal_graph, calcgraph):
		t1= self.get_triangle_count(orginal_graph)
		t2 = self.get_triangle_count(calcgraph)
		error = ((t1-t2)*1.0)/(t1 + 0.01)
		return -10000 * error

	def get_triangle_count(self, graph):
		triangles = nx.triangles(graph).values()
		res =0
		for t in triangles:
			res+=t
		return int(res/3)

	def reset(self):
		self.resetState()

	def render(self):
		print(self.current_edge_index)
