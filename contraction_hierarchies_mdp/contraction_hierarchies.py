"""
Reinforcement learning for contraction hierarchies algorithm.
Author:     Hongzheng Bai
Date:       2022/05/03

"""
import math
import networkx as nx
import numpy as np
import time
import random
from datetime import datetime
import tkinter as tk
from networkx.algorithms.shortest_paths.weighted import single_source_dijkstra
import heapq

class CH(object):
    def __init__(self, road_network, weight=None):
        self.G = nx.Graph()
        self.G_original = nx.Graph()
        self.G_contrast = nx.Graph()
        self.graph = open(road_network,"r")
        self.imp_pq = dict()
        self.order = 0
        self.weight = weight
        self.counter = 0
        self.Connect(G=self.G, weight= self.weight)
        self.Connect(G=self.G_original, weight= self.weight)
        # hypermeter for top-k uncontracted minimum features matrix
        self.k = 10
        # define state -> [[node number, ed of node, cn of node, level of node] * k]
        self.n_features = 3 * self.k
        # define action
        self.action_space = np.arange(self.k)
        self.n_action = len(self.action_space)
        
        
    def Connect(self, G, weight=None):
        self.graph.seek(0)
        # get number of nodes and edges
        graph_parameters = self.graph.readline().split()
        vertices_number = int(graph_parameters[0],base=10)
        edges_number = int(graph_parameters[1],base=10)
        print("vertices number:",vertices_number)
        print("edge number:",edges_number)
        # add nodes to networkx graph
        for i in range(vertices_number):
            G.add_node(i+1, contracted = False,ed= 0,imp=0,level=0,contr_neighbours=0)
        # add edges to networkx graph
        edge = self.graph.readline()
        while edge:
            edge_parameters = edge.split()
            source_node = int(edge_parameters[0], base=10)
            target_node = int(edge_parameters[1], base=10)
            if(weight == True):
                edge_weight = int(edge_parameters[2], base=10)
            else:
                edge_weight = 1
            found_exist = False
            for i in G[source_node]:
                # already store the edge with different weight, choose the min weight
                if i == target_node:
                    G[source_node][target_node]['weight'] = min(G[source_node][target_node]['weight'],edge_weight)
                    found_exist = True
                    break
            if not found_exist:
                G.add_edge(source_node,target_node,weight=edge_weight)
            edge = self.graph.readline()
    
    def SetOrder(self):
        n = len(self.G.nodes)
        self.imp_pq.clear()
        # initialize imp_pq
        for i in range(n):
            self.imp_pq[i+1] = self.GetImportance(i+1)
            # self.imp_pq.append((self.GetImportance(i+1),i+1))
            # print("settled node %d"%(i+1))
        self.order = 1
        # for i in range(1000):
        #       print(self.imp_pq[i+1])
        # lazy_update_counter = 0
        contracted_num = 0
        completed_rate = 0
        k = 0
        print("initializing importance queue settled.")
        while (len(self.imp_pq)>0):
            # find current lowest importance node in imp_pq
            curr_node = min(self.imp_pq, key= self.imp_pq.get)
            # curr_node_imp_pair = min(self.imp_pq, key= lambda pair:pair[0])
            # curr_node  = curr_node_imp_pair[1]   
            self.imp_pq.pop(curr_node)
            self.G.nodes[curr_node]['imp'] = self.order
            self.order +=1
            # contract node
            self.G.nodes[curr_node]['contracted'] = True
            if completed_rate < 0.9:
                self.ContractNode(G=self.G, x=curr_node)
            # print("already contracted node %d"%curr_node)
            # print(len(self.imp_pq))
            contracted_num += 1
            k +=1
            if(contracted_num == len(self.G.nodes)):
                print("contracted completed!")
                break
            if(k == 1000):
                completed_rate = contracted_num / n
                print(completed_rate)
                k = 0
            if completed_rate >= 0.9:
                # print("run here")
                # print("already contracted node %d"%curr_node)
                continue
            # get new importance for current lowest importance node
            for i in self.G[curr_node]:
                if self.G.nodes[i]['contracted'] == False:
                    new_imp = self.GetImportance(i)
                    if self.imp_pq.get(i) == None:
                        continue
                    else:
                        self.imp_pq[i] = new_imp
            # for i in self.G[curr_node]:
            #     new_imp = self.GetImportance(i)
            # self.imp_pq[i] = new_imp
            # new_imp = self.GetImportance(curr_node)
            # # print("get new importance for node %d"%curr_node)
            # # lazy update
            # if((len(self.imp_pq) == 0) or (new_imp - min(self.imp_pq, key= self.imp_pq.get) <= 10) or (lazy_update_counter >= 5)):
            #     # print("update for node %d"%curr_node)
            #     lazy_update_counter = 0
            #     self.G.nodes[curr_node]['imp'] = self.order
            #     self.order +=1
            #     # contract node
            #     self.G.nodes[curr_node]['contracted'] = True
            #     self.ContractNode(G=self.G, x=curr_node)
            #     # print("already contracted node %d"%curr_node)
            #     contracted_num += 1
            #     k +=1
            #     if(contracted_num == len(self.G.nodes)):
            #         print("contracted completed!")
            #     if(k == 100):
            #         completed_rate = contracted_num / n
            #         print(completed_rate)
            #         k = 0
            # else:
            #     self.imp_pq[curr_node] = new_imp
            #     # self.imp_pq.append((new_imp,curr_node))
            #     lazy_update_counter +=1
            #     # print("recalculated prority of node %d" %curr_node)
    
    def GetImportance(self, x):
        # get number of incident edges of x
        edges_incident = len(self.G[x])
        # get number of added shortcut when simulate contracting node x
        shortcuts = 0
        seenBefore = list()
        for i in self.G[x]:
            for j in self.G[x]:
                pair = sorted((i,j))
                if (i==j or (pair in seenBefore)):continue
                seenBefore.append(pair)
                if((self.G.nodes[i]['contracted'] == False) and (self.G.nodes[j]['contracted'] == False)):
                    shortcuts +=1
        edge_difference = shortcuts - edges_incident
        self.G.nodes[x]['ed'] = edge_difference
        return edge_difference + 2*self.G.nodes[x]['contr_neighbours'] + self.G.nodes[x]['level']
    
    def ContractNode(self, G, x):
        mx = self.GetMaxEdge(G, x)
        seenBefore = list()
        for i in G[x]:
            for j in G[x]:
                if ((G.nodes[i]['contracted'] == True) or (G.nodes[j]['contracted'] == True)):
                    continue
                pair = sorted((i,j))
                if (i==j or (pair in seenBefore)):continue
                seenBefore.append(pair)
                self.Check_Witness(G, i, x, mx)
                # print("check witness completed")
        # update importance term in incident node
        for i in G[x]:
            G.nodes[i]['contr_neighbours'] +=1
            G.nodes[i]['level'] = max(G.nodes[i]['level'], G.nodes[x]['level'] + 1)
    
    def GetMaxEdge(self, G, x):
        ret = 0
        for i in G[x]:
            for j in G[x]:
                if((i != j) and (G.nodes[i]['contracted'] == False) and (G.nodes[j]['contracted'] == False)):
                    ret = max(ret, G[x][i]['weight'] + G[x][j]['weight'])
        return ret 
    
    def Check_Witness(self, G, u, x, mx, type=None):
        n = len(G.nodes)
        # dijkstra priority queue for search witness path from u to v, excludes x
        # v is incident edge of x, excludes u
        D_pq = list()
        # initialize D_pq
        D_pq.append((0, u))
        # distance dictionary from u to any node in search tree
        D_dist = dict()
        # initialize D_dist
        D_dist[u] = 0
        # maximum iteration round for dijkstra search
        iter = int(250 * (n - self.order) / n)
        while((len(D_pq) > 0) and (iter > 0)):
            iter -=1
            curr_dist_pair = min(D_pq, key= lambda pair:pair[0])
            curr_dist = curr_dist_pair[0]
            a = curr_dist_pair[1]
            D_pq.remove(curr_dist_pair)
            if(curr_dist > D_dist[a]):
                continue
            for p in G[a]:
                new_dist = curr_dist + G[a][p]['weight']
                # p must not be x and not be contracted
                if(p != x and (G.nodes[p]['contracted'] == False)):
                    # p must not be settled node or distance greater than new_dist
                    if((p not in D_dist) or (D_dist[p] > new_dist)):
                        # prune when witness path greater than mx
                        if(p not in D_dist):
                            if new_dist < mx:
                                D_dist[p] = new_dist
                                D_pq.append((new_dist,p))
                        else:
                            if(D_dist[p] < mx):
                                D_dist[p] = new_dist
                                D_pq.append((new_dist,p))
        for v in G[x]:
            # v can not be u and not be contracted
            if ((v!=u) and (G.nodes[v]['contracted'] == False)):
                new_w = G[u][x]['weight'] + G[x][v]['weight']
                # print("%d %d %d"%(u,v,new_w))
                if((v not in D_dist) or (D_dist[v] > new_w)):
                    # add shortcut
                    # try:
                    #     if(u,v) in G.edges:
                    #         print("run here: no more add_edge")
                    #         continue
                    # except:
                    G.add_edge(u,v,weight=new_w)
                    # print("run here: add_edge:%d %d"%(u,v))
                    
    def GetDistance(self, G, s, t):
        # search with bi-dijkstra with ordering rank
        # initializing dijkstra from source node s
        SP_s = dict()
        parent_s = dict()
        unrelaxed_s = list()
        for node in G.nodes():
            SP_s[node] = math.inf
            parent_s[node] = None
            unrelaxed_s.append(node)
        SP_s[s] = 0
        
        # dijkstra forward
        while unrelaxed_s:
            node = min(unrelaxed_s, key= lambda node:SP_s[node])
            unrelaxed_s.remove(node)
            if SP_s[node] == math.inf:
                break
            # G[node] are the incident edges of node
            for child in G[node]:
                # skip unqualified edges
                if G.nodes[child]['imp'] < G.nodes[node]['imp']:
                    continue
                distance = SP_s[node] + G[node][child]['weight']
                # relax edge
                if distance < SP_s[child]:
                    SP_s[child] = distance
                    parent_s[child] = node
        # initializing dijkstra from target node t
        SP_t = dict()
        parent_t = dict()
        unrelaxed_t = list()
        for node in G.nodes():
            SP_t[node] = math.inf
            parent_t[node] = None
            unrelaxed_t.append(node)
        SP_t[t] = 0
        # dijkstra backward
        while unrelaxed_t:
            node = min(unrelaxed_t, key= lambda node: SP_t[node])
            unrelaxed_t.remove(node)
            if SP_t[node] == math.inf:
                break
            # G[node] are the incident edges of node
            for child in G[node]:
                # skip unqualified edges
                if G.nodes[child]['imp'] < G.nodes[node]['imp']:
                    continue
                distance = SP_t[node] + G[node][child]['weight']
                if distance < SP_t[child]:
                    SP_t[child] = distance
                    parent_t[child] = node
        minimum = math.inf
        merge_node = None
        for i in SP_s:
            if SP_t[i] == math.inf:
                continue
            if SP_t[i] + SP_s[i] < minimum:
                minimum = SP_s[i] + SP_t[i]
                merge_node = i
        return minimum, merge_node, SP_s, SP_t, parent_s, parent_t
    
    # see the route from origin of dijkstra to a given node
    def Route_dijkstra(self, parent, node):
        route = []
        while node != None:
            route.append(node)
            node = parent[node]
        return route[::-1]

    def See_full_route(self, s, t):
        minimum, merge_node, SP_s, SP_t, parent_s, parent_t = self.GetDistance(self.G, s, t)
        print("shortest distance between source node %d to target node %d:"%(s,t))
        if minimum == math.inf:
            print("no path between source node %d to target node %d"%(s,t))
            return
        print(minimum)
        route_from_source = self.Route_dijkstra(parent_s, merge_node)
        # show route
        print("route from source node %d:"%s)
        print(route_from_source)
        route_from_target = self.Route_dijkstra(parent_t, merge_node)
        # show route
        print("route from target node %d:"%t)
        print(route_from_target)
        route = route_from_source + route_from_target[::-1][1:]
        # show route
        print("entire route:")
        print(route)
        unvisited = 0
        for s_node, s_dist in SP_s.items():
            for t_node, t_dist in SP_t.items():
                if s_node == t_node and s_dist == t_dist == math.inf:
                    unvisited += 1
        print(f"""we have skipped {unvisited} nodes from a graph with {len(self.G)}, 
        so we have skipped {unvisited/len(self.G)*100}% of the nodes in our search space.""")
        print("\n\n")
        
    def query_test(self, test_num):
        self.query_list = list()
        i = 0
        while i < test_num:
            a = random.randint(1,len(self.G.nodes))
            b = random.randint(1,len(self.G.nodes))
            pair = sorted((a,b))
            if (a==b or (pair in self.query_list)):
                continue
            self.query_list.append(pair)
            i +=1
        # for i in range(test_num):
        #     self.See_full_route(self.query_list[i][0],self.query_list[i][1])
    
    def See_full_route_constract(self, G, s, t):
        minimum, merge_node, SP_s, SP_t, parent_s, parent_t = self.GetDistance(G, s, t)
        # print("shortest distance between source node %d to target node %d:"%(s,t))
        if minimum == math.inf:
            # print("no path between source node %d to target node %d"%(s,t))
            return
        # print(minimum)
        # route_from_source = self.Route_dijkstra(parent_s, merge_node)
        # show route
        # print("route from source node %d:"%s)
        # print(route_from_source)
        # route_from_target = self.Route_dijkstra(parent_t, merge_node)
        # show route
        # print("route from target node %d:"%t)
        # print(route_from_target)
        # route = route_from_source + route_from_target[::-1][1:]
        # show route
        # print("entire route:")
        # print(route)
        # unvisited = 0
        # for s_node, s_dist in SP_s.items():
            # for t_node, t_dist in SP_t.items():
                # if s_node == t_node and s_dist == t_dist == math.inf:
                    # unvisited += 1
        # print(f"""we have skipped {unvisited} nodes from a graph with {len(self.G)}, 
        # so we have skipped {unvisited/len(self.G)*100}% of the nodes in our search space.""")
        # print("\n\n")
        
    def dijkstra_with_contraction(self, G, source, destination, contracted = None):
        nx.set_node_attributes(G, {contracted: True}, 'contracted')
    
        
        shortest_path = dict()
        heap = list()
    
        for i in G.nodes():
            if not nx.get_node_attributes(G, 'contracted')[i]:
                shortest_path[i] = math.inf
                heap.append(i)
        shortest_path[source] = 0
    
        while len(heap)>0:
            q = min(heap, key= lambda node : shortest_path[node])
            if q == destination:
                nx.set_node_attributes(G, {contracted: False}, 'contracted')
                return shortest_path[q]
            heap.remove(q)
            # G[q] are incident edges of q
            for v in G[q]:
                # if the node is contracted, skip it
                if not nx.get_node_attributes(G, 'contracted')[v]:
                    distance = shortest_path[q] + G[q][v]['weight']
                    if distance < shortest_path[v]:
                        shortest_path[v] = distance
        nx.set_node_attributes(G, {contracted: False}, 'contracted')
    
        # can not reach the destination
        return math.inf
    
    
    def performance_contrast(self):
        sd_a = datetime.now()
        for i in range(len(self.query_list)):
            distance = self.dijkstra_with_contraction(self.G_original, self.query_list[i][0], self.query_list[i][1])
        sd_b = datetime.now()
        consume_sd = ((sd_b - sd_a).total_seconds())/len(self.query_list) 
        print("average time using for query by dijkstra:",consume_sd)
        ch_a = datetime.now()
        for i in range(len(self.query_list)):
            # print(i)
            self.See_full_route_constract(self.G,self.query_list[i][0], self.query_list[i][1])
        ch_b = datetime.now()
        consume_ch = ((ch_b - ch_a).total_seconds())/len(self.query_list)
        print("average time using for query by ch:",consume_ch)
        print("total speed up rate:",(consume_sd - consume_ch)/consume_ch)
        
    def reset(self):
        self.counter = 0
        self.G = self.G_original
        self.imp_pq.clear()
        # initialize imp_pq
        for i in range(len(self.G.nodes)):
            self.imp_pq[i+1] = self.GetImportance(i+1)
            # self.imp_pq.append((self.GetImportance(i+1),i+1))
        self.order = 1
        # skip 20% by ch-basic
        while (len(self.imp_pq)>0):
            # find current lowest importance node in imp_pq
            curr_node = min(self.imp_pq, key= self.imp_pq.get)
            # curr_node_imp_pair = min(self.imp_pq, key= lambda pair:pair[0])
            # curr_node  = curr_node_imp_pair[1]   
            self.imp_pq.pop(curr_node)
            self.G.nodes[curr_node]['imp'] = self.order
            self.order +=1
            # contract node
            self.G.nodes[curr_node]['contracted'] = True
            self.ContractNode(G=self.G, x=curr_node)
            self.counter+=1
            if(self.counter == 100):
                completed_rate = self.order/len(self.G.nodes)
                print(completed_rate)
                self.counter = 0
            if(self.order/len(self.G.nodes) >= 0.2):
                break

        # return observation
        self.curr_min_imp_pq_k = heapq.nsmallest(self.k, self.imp_pq, self.imp_pq.get)
        # self.curr_min_imp_pq_k = heapq.nsmallest(self.k, self.imp_pq, lambda pair:pair[0])
        init_observation = np.array([], dtype=np.float32)
        for i in range(len(self.curr_min_imp_pq_k)):
            node = self.curr_min_imp_pq_k[i]
            temp = np.array([self.G.nodes[node]['ed'], self.G.nodes[node]['contr_neighbours'], self.G.nodes[node]['level']], dtype= np.float32)
            init_observation = np.concatenate((init_observation,temp), axis = 0)
        # init_observation = init_observation[np.newaxis, :]
        return init_observation
        
    def step(self, action):
        # move agent: contract node
        # action space -> [0,1,2....,k-1]
        # calculate reward: ch choose the minxium proiority node
        # sychronize gragh
        self.G_contrast = self.G.copy()
        ch_curr_node = self.curr_min_imp_pq_k[0]
        # print(ch_curr_node)
        # simulate contract node in G_contrast
        self.G_contrast.nodes[ch_curr_node]['imp'] = self.order
        self.G_contrast.nodes[ch_curr_node]['contracted'] = True
        self.ContractNode(G=self.G_contrast,x=ch_curr_node)
        # contract node in G
        curr_node = self.curr_min_imp_pq_k[action]
        # print(curr_node)
        self.G.nodes[curr_node]['imp'] = self.order
        self.order +=1
        # contract node
        self.G.nodes[curr_node]['contracted'] = True
        self.ContractNode(G= self.G,x=curr_node)
        #reward function
        # print(self.G == self.G_contrast)
        total_degree_contrast = 0
        for i in range(len(self.G_contrast.nodes)):
            total_degree_contrast += self.G_contrast.degree(i+1)
        total_degree_mdp = 0
        for i in range(len(self.G.nodes)):
            total_degree_mdp += self.G.degree(i+1)
        
        # print(total_degree_contrast)
        # print(total_degree_mdp)
        reward = total_degree_contrast - total_degree_mdp
        
        
        # update curr_min_imp_pq_k
        self.imp_pq.pop(curr_node)
        # self.imp_pq.remove(self.curr_min_imp_pq_k[action])
        # update incident node priority
        for i in self.G[curr_node]:
            if self.G.nodes[i]['contracted'] == False:
                new_imp = self.GetImportance(i)
                if self.imp_pq.get(i) == None:
                    continue
                else:
                    self.imp_pq[i] = new_imp
        self.curr_min_imp_pq_k = heapq.nsmallest(self.k, self.imp_pq, self.imp_pq.get)
        # self.curr_min_imp_pq_k = heapq.nsmallest(self.k, self.imp_pq, lambda pair:pair[0])
        
        # next state
        next_observation = np.array([], dtype=np.float32)
        for i in range(len(self.curr_min_imp_pq_k)):
            node = self.curr_min_imp_pq_k[i]
            temp = np.array([self.G.nodes[node]['ed'], self.G.nodes[node]['contr_neighbours'], self.G.nodes[node]['level']], dtype= np.float32)
            next_observation = np.concatenate((next_observation, temp), axis=0)

        self.counter+=1
        if(self.counter == 100):
            completed_rate = self.order/len(self.G.nodes)
            print(completed_rate)
            self.counter = 0
        # next_observation = next_observation[np.newaxis, :]
        # finished state
        if (self.order/len(self.G.nodes) >= 0.9):
            done = True
        else:
            done = False
            
        return next_observation, reward, done
        
    def ch_rest(self):
        while(self.order <= len(self.G.nodes)):
            # contract the rest node by ch-basic
            ch_curr_node = self.curr_min_imp_pq_k[0]
            self.G.nodes[ch_curr_node]['imp'] = self.order
            self.G.nodes[ch_curr_node]['contracted'] = True
            self.ContractNode(G=self.G,x=ch_curr_node)
            self.order +=1
            
    def performance_contrast_mdp(self):
        self.G_contrast = self.G.copy()
        self.G = self.G_original.copy()
        self.SetOrder()
        ch_f_a = datetime.now()
        for i in range(len(self.query_list)):
            self.See_full_route_constract(self.G, self.query_list[i][0], self.query_list[i][1])
        ch_f_b = datetime.now()
        consume_sd = ((ch_f_b - ch_f_a).total_seconds())/len(self.query_list) 
        print("average time using for query by ch-basic:",consume_sd)
        ch_a = datetime.now()
        for i in range(len(self.query_list)):
            self.See_full_route_constract(self.G_contrast,self.query_list[i][0], self.query_list[i][1])
        ch_b = datetime.now()
        consume_ch = ((ch_b - ch_a).total_seconds())/len(self.query_list)
        print("average time using for query by ch-mdp:",consume_ch)
        print("total speed up rate:",(consume_sd - consume_ch)/consume_ch)
        
        