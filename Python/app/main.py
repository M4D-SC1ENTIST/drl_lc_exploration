"""
Documentation:

request: https://fastapi.tiangolo.com/tutorial/body/ 
uvicorn-gunicorn-fastapi-docker: https://github.com/tiangolo/uvicorn-gunicorn-fastapi-docker 
"""

from glob import glob
from operator import truediv
import gudhi
import time
import scipy
from pathlib import Path
from nsopy.methods.subgradient import SubgradientMethod
from nsopy.loggers import GenericMethodLogger

import pandas as pd
import numpy as np
import networkx as nx

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from itertools import chain

import random

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# The prefix of the exported file
exploration_method = 'DRL'

exploration_df = pd.DataFrame(columns = ['observations_count', 'nodes_count', 'edges_count', 'triangles_count'])

G = nx.Graph()

class Agent():
    def __init__(self):
        self.current_left_observed_landmark = []
        self.current_right_observed_landmark = []
        self.current_straight_observed_landmark = []
        # x, y, z
        self.location = []
        # w, x, y, z
        self.rotation = []
        self.voronoi_partition = set()
        self.isw_goal_landmark = None
        self.isw_path = []

        self.current_dubins_radius = 0
        self.current_dubins_arcLength = 0
        self.current_dubins_direction = 0

        self.isw_counter = 0


        self.initial_rw_count = 0

# Global landmark complex
landmark_complex = gudhi.SimplexTree()

# 2-simplices
triangles = []

# 1-simplices
edges = []

# 0-simplices
nodes = []

# Agents dictionary
agents = {}

# Number of time a landmark has been seen
landmark_seen_times = {}

# Start time
start_time = None

# End time
end_time = None

# Ground Truth Number of Nodes
gt_num_of_nodes = 999

# All nodes discovered
all_node_discovered = False

# Observations Count
observations_count = 0

previous_num_simplices = 0

g_b2 = None
g_x = None

agents[0] = Agent()
agents[1] = Agent()
agents[2] = Agent()
agents[3] = Agent()

# RW count with respect to observations count settings 
record_rw = True
rw_count = 0
rw_df = pd.DataFrame(columns = ['observations_count', 'rw_count'])



class Observation(BaseModel):
    """
    landmark_id: a list of id of observed landmarks
    agent_location: [x, y, z]
    agent_rotation: [w, x, y, z]
    """ 
    agent_id: int = None
    left_observed_landmark_id: List[int] = None
    right_observed_landmark_id: List[int] = None
    straight_observed_landmark_id: List[int] = None
    agent_location: List[float] = None
    agent_rotation: List[float] = None

class ObservationResponse(BaseModel):
    """
    not_in_complex: true if the simplex was not yet in the complex, false otherwise (whatever its original filtration value)
    num_vertices_added: the simplicial complex number of vertices
    """
    not_in_complex: bool = False
    num_triangles_added: int = 0
    triangles_added: List[int] = None

class DubinsCurveParameters(BaseModel):
    radiusCMD: float = 0
    arcLengthCMD: float = 0
    directionCMD: int = 0




app = FastAPI()

@app.get("/")
def read_root():
    global landmark_complex
    global triangles
    global edges
    global nodes
    global agents
    global start_time
    global end_time

    global exploration_df

    return {"Landmark Complex System": "Global Server",
            "Complex Basic Information": 
                {
                    "Number of Simplices": landmark_complex.num_simplices(),
                    "Number of Vertices": landmark_complex.num_vertices() 
                },
            "Simplex Information": {
                    "Number of Triangles": len(triangles),
                    "Number of Edges": len(edges),
                    "Number of Nodes": len(nodes)
                },
            "Simulation Information": {
                    "Start Time": start_time,
                    "End Time": end_time,
                }
            }

# TODO: Initialization API (after implementing multi robot system)

@app.get("/initialize")
def initialize():
    global landmark_complex
    global triangles
    global edges
    global nodes
    global agents
    global start_time
    global end_time
    global gt_num_of_nodes
    global all_node_discovered
    global observations_count
    global exploration_df

    global G
    global landmark_seen_times

    global previous_num_simplices

    global rw_count
    global rw_df

    landmark_complex = gudhi.SimplexTree()
    triangles = []
    edges = []
    nodes = []

    gt_num_of_nodes = 999  

    all_node_discovered = False 

    start_time = time.time()
    end_time = None

    observations_count = 0
    exploration_df = pd.DataFrame(columns = ['observations_count', 'nodes_count', 'edges_count', 'triangles_count'])

    rw_count = 0
    rw_df = pd.DataFrame(columns = ['observations_count', 'rw_count'])

    G = nx.Graph()
    landmark_seen_times = {}

    previous_num_simplices = 0

    return "Server initialized"

@app.get("/terminate")
def terminate():
    global start_time
    global end_time
    global agents
    global triangles
    global exploration_df
    global record_rw
    global rw_df

    agents[0] = Agent()
    agents[1] = Agent()
    agents[2] = Agent()
    agents[3] = Agent()

    # print("triangles: ", triangles)

    end_time = time.time()

    filepath = Path('data_output/'+exploration_method+'/out.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    # print('The file path is: ', filepath)
    exploration_df.to_csv(filepath, index=False)

    if record_rw:
        rw_path = Path('data_output/'+exploration_method+'/rw.csv')
        rw_path.parent.mkdir(parents=True, exist_ok=True)
        # rw_df.to_csv(rw_path, index=False)

    return "Simulation terminated"

@app.post("/observation_receiver_0")
async def process_observation(observation: Observation):
    global landmark_complex
    global triangles
    global edges
    global nodes
    global agents
    global observations_count
    global exploration_df

    global G

    global landmark_seen_times

    new_triangles = []
    new_edges = []
    new_nodes = []

    agent_id = observation.agent_id
    agent = agents[agent_id]
    agent.current_left_observed_landmark = observation.left_observed_landmark_id
    agent.current_right_observed_landmark = observation.right_observed_landmark_id
    agent.current_straight_observed_landmark = observation.straight_observed_landmark_id
    
    agent.location = observation.agent_location
    agent.rotation = observation.agent_rotation

    # print("Agent 0 current location: ", agent.location)
    # print("Agent 0 current rotation: ", agent.rotation)

    # print("Agent ", agent_id, " current left observed landmark: ", agent.current_left_observed_landmark)
    # print("Agent ", agent_id, " current right observed landmark: ", agent.current_right_observed_landmark)
    # print("Agent ", agent_id, " current straight observed landmark: ", agent.current_straight_observed_landmark)

    concat_landmark_id = observation.left_observed_landmark_id + observation.right_observed_landmark_id + observation.straight_observed_landmark_id

    for i in concat_landmark_id:
        if i in landmark_seen_times.keys():
            landmark_seen_times[i] += 1
        else:
            landmark_seen_times[i] = 1

    not_in_complex = landmark_complex.insert(concat_landmark_id)

    if not_in_complex:
        temp_complex = gudhi.SimplexTree()
        temp_complex.insert(concat_landmark_id)
        for i in temp_complex.get_skeleton(2):
            current_simplex = i[0]
            current_simplex_length = len(current_simplex)
            if (current_simplex_length == 3) and not(current_simplex in triangles):
                new_triangles.append(current_simplex)
            elif (current_simplex_length == 2) and not(current_simplex in edges):
                new_edges.append(current_simplex)
                G.add_edges_from([tuple(current_simplex)])
            elif (current_simplex_length == 1) and not(current_simplex in nodes):
                new_nodes.append(current_simplex)
                G.add_nodes_from(current_simplex)

    triangles = triangles + new_triangles
    edges = edges + new_edges
    nodes = nodes + new_nodes
        

    # print(G.nodes())

    num_triangle_added = len(new_triangles)

    observations_count += 1
    exploration_df = exploration_df.append({'observations_count': observations_count, 'nodes_count': len(nodes), 'edges_count': len(edges), 'triangles_count': len(triangles)},
                                            ignore_index = True)
    print("observations count: ", len(exploration_df.index))

    # Return observation response
    obs_res = ObservationResponse()
    obs_res.not_in_complex = not_in_complex
    obs_res.num_triangles_added = num_triangle_added
    obs_res.triangles_added = list(chain.from_iterable(new_triangles))

    return obs_res

@app.post("/observation_receiver_1")
async def process_observation_1(observation: Observation):
    global landmark_complex
    global triangles
    global edges
    global nodes
    global agents
    global observations_count
    global exploration_df

    global G
    global landmark_seen_times

    new_triangles = []
    new_edges = []
    new_nodes = []

    agent_id = observation.agent_id
    agent = agents[agent_id]
    agent.current_left_observed_landmark = observation.left_observed_landmark_id
    agent.current_right_observed_landmark = observation.right_observed_landmark_id
    agent.current_straight_observed_landmark = observation.straight_observed_landmark_id

    agent.location = observation.agent_location
    agent.rotation = observation.agent_rotation

    # print("Agent ", agent_id, " current left observed landmark: ", agent.current_left_observed_landmark)
    # print("Agent ", agent_id, " current right observed landmark: ", agent.current_right_observed_landmark)
    # print("Agent ", agent_id, " current straight observed landmark: ", agent.current_straight_observed_landmark)

    concat_landmark_id = observation.left_observed_landmark_id + observation.right_observed_landmark_id + observation.straight_observed_landmark_id

    for i in concat_landmark_id:
        if i in landmark_seen_times.keys():
            landmark_seen_times[i] += 1
        else:
            landmark_seen_times[i] = 1

    not_in_complex = landmark_complex.insert(concat_landmark_id)

    if not_in_complex:
        temp_complex = gudhi.SimplexTree()
        temp_complex.insert(concat_landmark_id)
        for i in temp_complex.get_skeleton(2):
            current_simplex = i[0]
            current_simplex_length = len(current_simplex)
            if (current_simplex_length == 3) and not(current_simplex in triangles):
                new_triangles.append(current_simplex)
            elif (current_simplex_length == 2) and not(current_simplex in edges):
                new_edges.append(current_simplex)
                G.add_edges_from([tuple(current_simplex)])
            elif (current_simplex_length == 1) and not(current_simplex in nodes):
                new_nodes.append(current_simplex)
                G.add_nodes_from(current_simplex)

    triangles = triangles + new_triangles
    edges = edges + new_edges
    nodes = nodes + new_nodes

        

    num_triangle_added = len(new_triangles)

    observations_count += 1
    exploration_df = exploration_df.append({'observations_count': observations_count, 'nodes_count': len(nodes), 'edges_count': len(edges), 'triangles_count': len(triangles)},
                                            ignore_index = True)

    print("observations count: ", len(exploration_df.index))

    # Return observation response
    obs_res = ObservationResponse()
    obs_res.not_in_complex = not_in_complex
    obs_res.num_triangles_added = num_triangle_added
    obs_res.triangles_added = list(chain.from_iterable(new_triangles))

    

    return obs_res

@app.post("/observation_receiver_2")
async def process_observation_2(observation: Observation):
    global landmark_complex
    global triangles
    global edges
    global nodes
    global agents
    global observations_count
    global exploration_df

    global G
    global landmark_seen_times

    new_triangles = []
    new_edges = []
    new_nodes = []

    agent_id = observation.agent_id
    agent = agents[agent_id]
    agent.current_left_observed_landmark = observation.left_observed_landmark_id
    agent.current_right_observed_landmark = observation.right_observed_landmark_id
    agent.current_straight_observed_landmark = observation.straight_observed_landmark_id

    agent.location = observation.agent_location
    agent.rotation = observation.agent_rotation

    # print("Agent ", agent_id, " current left observed landmark: ", agent.current_left_observed_landmark)
    # print("Agent ", agent_id, " current right observed landmark: ", agent.current_right_observed_landmark)
    # print("Agent ", agent_id, " current straight observed landmark: ", agent.current_straight_observed_landmark)

    concat_landmark_id = observation.left_observed_landmark_id + observation.right_observed_landmark_id + observation.straight_observed_landmark_id

    for i in concat_landmark_id:
        if i in landmark_seen_times.keys():
            landmark_seen_times[i] += 1
        else:
            landmark_seen_times[i] = 1

    not_in_complex = landmark_complex.insert(concat_landmark_id)

    if not_in_complex:
        temp_complex = gudhi.SimplexTree()
        temp_complex.insert(concat_landmark_id)
        for i in temp_complex.get_skeleton(2):
            current_simplex = i[0]
            current_simplex_length = len(current_simplex)
            if (current_simplex_length == 3) and not(current_simplex in triangles):
                new_triangles.append(current_simplex)
            elif (current_simplex_length == 2) and not(current_simplex in edges):
                new_edges.append(current_simplex)
                G.add_edges_from([tuple(current_simplex)])
            elif (current_simplex_length == 1) and not(current_simplex in nodes):
                new_nodes.append(current_simplex)
                G.add_nodes_from(current_simplex)

    triangles = triangles + new_triangles
    edges = edges + new_edges
    nodes = nodes + new_nodes

        

        

    num_triangle_added = len(new_triangles)

    observations_count += 1
    exploration_df = exploration_df.append({'observations_count': observations_count, 'nodes_count': len(nodes), 'edges_count': len(edges), 'triangles_count': len(triangles)},
                                            ignore_index = True)
    print("observations count: ", len(exploration_df.index))
    # Return observation response
    obs_res = ObservationResponse()
    obs_res.not_in_complex = not_in_complex
    obs_res.num_triangles_added = num_triangle_added
    obs_res.triangles_added = list(chain.from_iterable(new_triangles))

    

    return obs_res

@app.post("/observation_receiver_3")
async def process_observation_3(observation: Observation):
    global landmark_complex
    global triangles
    global edges
    global nodes
    global agents
    global observations_count
    global exploration_df

    global G
    global landmark_seen_times

    new_triangles = []
    new_edges = []
    new_nodes = []

    agent_id = observation.agent_id
    agent = agents[agent_id]
    agent.current_left_observed_landmark = observation.left_observed_landmark_id
    agent.current_right_observed_landmark = observation.right_observed_landmark_id
    agent.current_straight_observed_landmark = observation.straight_observed_landmark_id

    agent.location = observation.agent_location
    agent.rotation = observation.agent_rotation

    # print("Agent ", agent_id, " current left observed landmark: ", agent.current_left_observed_landmark)
    # print("Agent ", agent_id, " current right observed landmark: ", agent.current_right_observed_landmark)
    # print("Agent ", agent_id, " current straight observed landmark: ", agent.current_straight_observed_landmark)

    concat_landmark_id = observation.left_observed_landmark_id + observation.right_observed_landmark_id + observation.straight_observed_landmark_id

    for i in concat_landmark_id:
        if i in landmark_seen_times.keys():
            landmark_seen_times[i] += 1
        else:
            landmark_seen_times[i] = 1
    
    # print("landmark seen times: ", landmark_seen_times)

    not_in_complex = landmark_complex.insert(concat_landmark_id)

    if not_in_complex:
        temp_complex = gudhi.SimplexTree()
        temp_complex.insert(concat_landmark_id)
        for i in temp_complex.get_skeleton(2):
            current_simplex = i[0]
            current_simplex_length = len(current_simplex)
            if (current_simplex_length == 3) and not(current_simplex in triangles):
                new_triangles.append(current_simplex)
            elif (current_simplex_length == 2) and not(current_simplex in edges):
                new_edges.append(current_simplex)
                G.add_edges_from([tuple(current_simplex)])
            elif (current_simplex_length == 1) and not(current_simplex in nodes):
                new_nodes.append(current_simplex)
                G.add_nodes_from(current_simplex)

    triangles = triangles + new_triangles
    edges = edges + new_edges
    nodes = nodes + new_nodes

        

    num_triangle_added = len(new_triangles)

    observations_count += 1
    exploration_df = exploration_df.append({'observations_count': observations_count, 'nodes_count': len(nodes), 'edges_count': len(edges), 'triangles_count': len(triangles)},
                                            ignore_index = True)
    print("observations count: ", len(exploration_df.index))
    # Return observation response
    obs_res = ObservationResponse()
    obs_res.not_in_complex = not_in_complex
    obs_res.num_triangles_added = num_triangle_added
    obs_res.triangles_added = list(chain.from_iterable(new_triangles))

    return obs_res


def rw_observe():
    global observations_count
    global record_rw
    global rw_count
    global rw_df
    

    random_radius = random.randrange(200)
    random_arcLength = random.randrange(100)
    random_direction = random.randrange(2)

    if record_rw:
        rw_count += 1
        rw_df = rw_df.append({'observations_count': observations_count, 'rw_count': rw_count},
                                                ignore_index = True)
        print("observations count: ", len(rw_df.index))

    return random_radius, random_arcLength, random_direction


def voronoi():
    global agents

    global G

    agent0_observed_landmarks = agents[0].current_left_observed_landmark + agents[0].current_right_observed_landmark + agents[0].current_straight_observed_landmark
    agent1_observed_landmarks = agents[1].current_left_observed_landmark + agents[1].current_right_observed_landmark + agents[1].current_straight_observed_landmark
    agent2_observed_landmarks = agents[2].current_left_observed_landmark + agents[2].current_right_observed_landmark + agents[2].current_straight_observed_landmark
    agent3_observed_landmarks = agents[3].current_left_observed_landmark + agents[3].current_right_observed_landmark + agents[3].current_straight_observed_landmark

    # agent0_observed_landmarks_len = len(agent0_observed_landmarks)
    # agent1_observed_landmarks_len = len(agent1_observed_landmarks)
    # agent2_observed_landmarks_len = len(agent2_observed_landmarks)
    # agent3_observed_landmarks_len = len(agent3_observed_landmarks)

    center_nodes_list = agent0_observed_landmarks
    center_nodes_list = center_nodes_list + agent1_observed_landmarks
    center_nodes_list = center_nodes_list + agent2_observed_landmarks
    center_nodes_list = center_nodes_list + agent3_observed_landmarks

    center_nodes_set = set(center_nodes_list)

    cells = nx.voronoi_cells(G, center_nodes_set)

    # print("center nodes: ", center_nodes)
    # print("agent 0 observed landmarks: ", agent0_observed_landmarks)
    # print("agent 1 observed landmarks: ", agent1_observed_landmarks)
    # print("agent 2 observed landmarks: ", agent2_observed_landmarks)
    # print("agent 3 observed landmarks: ", agent3_observed_landmarks)
    # print("center nodes set: ", center_nodes_set)

    agent0_partition = set()
    agent1_partition = set()
    agent2_partition = set()
    agent3_partition = set()
    
    for key, value in cells.items():
        if key in agent0_observed_landmarks:
            agent0_partition = set.union(agent0_partition, value)
        elif key in agent1_observed_landmarks:
            agent1_partition = set.union(agent1_partition, value)
        elif key in agent2_observed_landmarks:
            agent2_partition = set.union(agent2_partition, value)  
        elif key in agent3_observed_landmarks:
            agent3_partition = set.union(agent3_partition, value)
    
    agents[0].voronoi_partition = agent0_partition
    agents[1].voronoi_partition = agent1_partition
    agents[2].voronoi_partition = agent2_partition
    agents[3].voronoi_partition = agent3_partition

    # print("agent 0 partition: ", agents[0].voronoi_partition)
    # print("agent 1 partition: ", agents[1].voronoi_partition)
    # print("agent 2 partition: ", agents[2].voronoi_partition)
    # print("agent 3 partition: ", agents[3].voronoi_partition)


def least_observed_landmark():
    global agents
    global landmark_seen_times
    global G

    for key, value in agents.items():
        current_voronoi_partition = agents[key].voronoi_partition
        least_observed_landmark = None
        least_observed_times = 99999

        for landmark in current_voronoi_partition:
            if landmark_seen_times[landmark] < least_observed_times:
                least_observed_times = landmark_seen_times[landmark]
                least_observed_landmark = landmark
        
        agents[key].isw_goal_landmark = least_observed_landmark
    
    # print("agent 0 least observed landmark: ", agents[0].isw_goal_landmark)
    # print("agent 1 least observed landmark: ", agents[1].isw_goal_landmark)
    # print("agent 2 least observed landmark: ", agents[2].isw_goal_landmark)
    # print("agent 3 least observed landmark: ", agents[3].isw_goal_landmark)

def dijkstra_search():
    global agents
    global G

    for key, value in agents.items():
        concat_curr_agent_observation = agents[key].current_left_observed_landmark + agents[key].current_right_observed_landmark + agents[key].current_straight_observed_landmark
        if concat_curr_agent_observation:
            source_node = random.choice(concat_curr_agent_observation)
        target_node = agents[key].isw_goal_landmark

        if target_node is not None:
            agents[key].isw_path = nx.dijkstra_path(G, source_node, target_node)
    
    # print("agent 0 path: ", agents[0].isw_path)
    # print("agent 1 path: ", agents[1].isw_path)
    # print("agent 2 path: ", agents[2].isw_path)
    # print("agent 3 path: ", agents[3].isw_path)

def left_right(agent, landmark):
    if landmark in agent.left_observed_landmark_id:
        return 0
    elif landmark in agent.right_observed_landmark_id:
        return 1
    else:
        return 2

def estt_observe(agent, landmark):
    estt_radius = random.randrange(200)
    estt_arcLength = random.randrange(100)
    estt_direction = left_right(agent, landmark)

    return estt_radius, estt_arcLength, estt_direction


def isw():
    voronoi()
    least_observed_landmark()
    dijkstra_search()

def shortest_path(agent):
    global G

    res = []

    concat_agent_observation = agent.current_left_observed_landmark + agent.current_right_observed_landmark + agent.current_straight_observed_landmark
    
    if source_node:
        source_node = random.choice(concat_agent_observation)
    target_node = agent.isw_goal_landmark

    if target_node is not None:
        res = nx.dijkstra_path(G, source_node, target_node)
    
    return res


def navigate(agent):
    successful = False
    while not successful:
        executed_rw = False
        landmarks = shortest_path(agent)

        while landmarks:
            visible = False

            landmarks_len = len(landmarks)
            for idx in range(landmarks_len):
                reverse_idx = landmarks_len - 1 - idx
                concat_agent_observation = agent.current_left_observed_landmark + agent.current_right_observed_landmark + agent.current_straight_observed_landmark
                if landmarks[reverse_idx] in concat_agent_observation:
                    agent.current_dubins_radius, agent.current_dubins_arcLength, agent.current_dubins_direction = estt_observe(agent, landmarks[reverse_idx])
                    landmarks = landmarks[reverse_idx + 1:]
                visible = True
                break
            
            if not visible:
                if executed_rw:
                    break
                else:
                    for q in range(10):
                        agent.current_dubins_radius, agent.current_dubins_arcLength, agent.current_dubins_direction = rw_observe()
                    executed_rw = True
        
        if not landmarks:
            successful = True

def boundary_matrices_and_higher_order_laplacian():
    global nodes
    global edges
    global triangles

    n = len(nodes)
    m = len(edges)
    p = len(triangles)

    b1 = np.zeros((n, m))
    b2 = np.zeros((m, p))

    # Populate B1 and B2 matrices
    for e_idx in range(edges):
        point_toward_existed = False
        for n_idx in range(nodes):
            if nodes[n_idx] in edges[e_idx]:
                if not point_toward_existed:
                    b1[n_idx, e_idx] = 1
                else:
                    b1[n_idx, e_idx] = -1
            else:
                b1[n_idx, e_idx] = 0

    for t_idx in range(triangles):
        point_forward_count = 0
        for e_idx in range(edges):
            if edges[e_idx] in triangles[t_idx]:
                if point_forward_count < 2:
                    b2[e_idx, t_idx] = 1
                    point_forward_count += 1
                else:
                    b2[e_idx, t_idx] = -1
            else:
                b2[e_idx, t_idx] = 0
    
    l1 = np.dot(b1.T, b1) + np.dot(b2, b2.T)

    return b1, b2, l1

def oracle(z_k):
    global g_b2
    global g_x

    f_z_k = [np.dot(g_b2, z_k) + g_x]

    diff_f_zk = g_b2

    return 0, f_z_k, diff_f_zk

def projection_function(z_k):
    global edges

    m = len(edges)

    zero_mat = np.zeros((m, m))

    if z_k == zero_mat:
        return zero_mat
    else:
        z_k

def laplacian_dynamics_and_l1_norm_minimization():
    global nodes
    global edges
    global triangles

    global g_b2
    global g_x

    b1, b2, l1 = boundary_matrices_and_higher_order_laplacian()
    x = scipy.sparse.csgraph.laplacian(l1)
    g_b2 = b2
    g_x = x

    if g_b2 != None and g_x != None:
        pass

    method = SubgradientMethod(oracle, projection_function, stepsize_0=0.1, stepsize_rule='constant', sense='min')
    logger = GenericMethodLogger(method)

    for iteration in range(50):
        method.step()

    z = logger.z_k_iterates[-1]

    return x, z, b2


def hiw(agent):
    global edges
    global G

    m = len(edges)
    x_star, z_star, b2 = laplacian_dynamics_and_l1_norm_minimization()
    y = x_star + np.dot(b2, z_star)
    
    thres = np.std(y)
    q = []
    for m_idx in range(m):
        if np.abs(y[m_idx] > thres):
            q.append(m_idx)
    
    cc = nx.connected_components(G)

    for i in q:
        while cc != None:
            cost_mat = []
    
            concat_curr_agent_observation = agent.current_left_observed_landmark + agent.current_right_observed_landmark + agent.current_straight_observed_landmark
            if source_node:
                source_node = random.choice(concat_curr_agent_observation)
            target_node = q

            if target_node is not None:
                temp = nx.dijkstra_path(G, source_node, target_node)
                cost = len(temp)
                cost_mat.append(cost)
            row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_mat)

            navigate(agent)
            cc.pop()





def LCCA(agent):
    global landmark_complex
    global previous_num_simplices

    # print("Execute LCCA")

    if agent.initial_rw_count < 10:
        agent.current_dubins_radius, agent.current_dubins_arcLength, agent.current_dubins_direction = rw_observe()
        agent.initial_rw_count += 1

    current_num_simplices = landmark_complex.num_simplices()

    if previous_num_simplices == 0:
        previous_num_simplices = current_num_simplices
        rate_of_growth = 1
    else:
        num_simplices_added = current_num_simplices - previous_num_simplices
        previous_num_simplices = current_num_simplices
        rate_of_growth = num_simplices_added / previous_num_simplices
    
    if rate_of_growth > 0.01:
        isw()
        g_score = len(agent.isw_path)
        if g_score <= 10 or agent.isw_counter > 10:
            for k in range(10):
                agent.current_dubins_radius, agent.current_dubins_arcLength, agent.current_dubins_direction = rw_observe()
            agent.isw_counter = 0
        else:
            navigate(agent)
            agent.isw_counter += 1
    # else:
    #     hiw(agent)    


@app.post("/dubins_command_0")
async def dubins_command_0():
    global agents
    agent_idx = 0
    # voronoi()
    # isw()

    LCCA(agents[agent_idx])

    dubins_params = DubinsCurveParameters()

    dubins_params.radiusCMD, dubins_params.arcLengthCMD, dubins_params.directionCMD = agents[agent_idx].current_dubins_radius, agents[agent_idx].current_dubins_arcLength, agents[agent_idx].current_dubins_direction
    print("dubins params: ", dubins_params)
    return dubins_params


@app.post("/dubins_command_1")
async def dubins_command_1():
    global agents
    agent_idx = 1
    # voronoi()
    # isw()

    LCCA(agents[agent_idx])

    dubins_params = DubinsCurveParameters()

    dubins_params.radiusCMD, dubins_params.arcLengthCMD, dubins_params.directionCMD = agents[agent_idx].current_dubins_radius, agents[agent_idx].current_dubins_arcLength, agents[agent_idx].current_dubins_direction

    return dubins_params


@app.post("/dubins_command_2")
async def dubins_command_2():
    global agents
    agent_idx = 2
    # voronoi()
    # isw()

    LCCA(agents[agent_idx])

    dubins_params = DubinsCurveParameters()

    dubins_params.radiusCMD, dubins_params.arcLengthCMD, dubins_params.directionCMD = agents[agent_idx].current_dubins_radius, agents[agent_idx].current_dubins_arcLength, agents[agent_idx].current_dubins_direction

    return dubins_params


@app.post("/dubins_command_3")
async def dubins_command_3():
    global agents
    agent_idx = 3
    # voronoi()
    # isw()

    LCCA(agents[agent_idx])

    dubins_params = DubinsCurveParameters()

    dubins_params.radiusCMD, dubins_params.arcLengthCMD, dubins_params.directionCMD = agents[agent_idx].current_dubins_radius, agents[agent_idx].current_dubins_arcLength, agents[agent_idx].current_dubins_direction

    return dubins_params