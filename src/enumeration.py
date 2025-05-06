# -*- coding: utf-8 -*-
"""
Created on Sun May  4 16:03:13 2025

@author: sezazip
"""

import networkx as nx
import gurobipy as gp
from gurobipy import GRB
import math

# This is algorithm 1 from Fischetti, Matteo, et al. 
#   "Thinning out Steiner trees: a node-based model for uniform edge costs." 
#   Mathematical Programming Computation 9.2 (2017): 203-229.
#
def find_minimal_separator(DG, component, b):
    neighbors_component = { i : False for i in DG.nodes }
    for i in nx.node_boundary(DG, component, None):
        neighbors_component[i] = True
    
    visited = { i : False for i in DG.nodes }
    child = [b]
    visited[b] = True
    
    while child:
        parent = child
        child = list()
        for i in parent:
            if not neighbors_component[i]:
                for j in DG.neighbors(i):
                    if not visited[j]:
                        child.append(j)
                        visited[j] = True
    
    C = [ i for i in DG.nodes if neighbors_component[i] and visited[i] ]
    return C

def callback_function(m, where):
    if where != GRB.Callback.MIPSOL: 
        return
    xval = m.cbGetSolution(m._x)
    DG = m._DG
    k = m._k
    L = m._L
    U = m._U
    root = m._root
    
    district = [ i for i in DG.nodes if xval[i,0] > 0.5 ]
    assert nx.is_strongly_connected( DG.subgraph(district) )
    complement = [ i for i in DG.nodes if xval[i,1] > 0.5 ]
    if len(complement)==1 or nx.is_strongly_connected( DG.subgraph(complement) ):
        m._districts.append( district )
        m._number_of_districts += 1
        print("found district #", m._number_of_districts, ":", district)
    else:
        assert k==3
        components = list( nx.strongly_connected_components( DG.subgraph(complement) ) )
        populations = [ sum( DG.nodes[i]['TOTPOP'] for i in component ) for component in components ]
        pop0_ok = L <= populations[0] and populations[0] <= U
        pop1_ok = L <= populations[1] and populations[1] <= U
        if len(components)==2 and pop0_ok and pop1_ok:
            m._districts.append( district )
            m._number_of_districts += 1
            print("found district #", m._number_of_districts, ":", district)
        
    # force at least one vertex from complement must move to first_district. 
    # (this is generally not valid, but with 1-person deviation it will only cut off 
    #   this particular solution as all counties have nontrivial population)
    m.cbLazy( gp.quicksum( m._x[i,0] for i in complement ) >= 1 )
    if m._number_of_districts == m._enumeration_limit:
        print("Enumeration limit has been reached. Exiting search.")
        m.cbLazy( m._x[root,1] >= 1 )

def cut_callback (m , where):
    if where != GRB.Callback.MIPSOL: 
        return
        
    xval = m.cbGetSolution(m._x)
    DG = m._DG
    
    second_district = [ i for i in DG.nodes if xval[i,1] > 0.5 ]
    b = None
    max_component_population = -1
    for component in nx.strongly_connected_components( DG.subgraph(second_district) ):
        component_population = sum( DG.nodes[i]['TOTPOP'] for i in component )
        if component_population > max_component_population:
            max_component_population = component_population
            max_vertex_population = max( DG.nodes[i]['TOTPOP'] for i in component )
            max_population_vertices = [ i for i in component if DG.nodes[i]['TOTPOP'] == max_vertex_population ]
            b = max_population_vertices[0]

    for component in nx.strongly_connected_components( DG.subgraph(second_district) ):
        if b in component:
            continue
        max_vertex_population = max( DG.nodes[i]['TOTPOP'] for i in component )
        max_population_vertices = [ i for i in component if DG.nodes[i]['TOTPOP'] == max_vertex_population ]
        a = max_population_vertices[0]
        C = find_minimal_separator(DG, component, b)
        m.cbLazy( m._x[a,1] + m._x[b,1] <= 1 + gp.quicksum( m._x[c,1] for c in C ) )

def to_string(my_list):
    string = str()
    for i in my_list:
        string += str(i) + ","
    return string[:-1]   # remove last character (the comma)

def remove_duplicates(list_of_lists):
    input_len = len(list_of_lists)
    set_of_strings = { to_string(my_list) for my_list in list_of_lists }
    list_of_strings = [ string.split(',') for string in set_of_strings ]
    new_lol = [ [ int(i) for i in my_list ] for my_list in list_of_strings ]
    output_len = len(new_lol)
    print("Removed this many duplicates:", input_len - output_len )
    return new_lol

def enumerate_districts(G, L, U, k, root=None):
    m = gp.Model()
    x = m.addVars(G.nodes, k, vtype=GRB.BINARY)

    m.addConstrs( x[i,0] + x[i,1] == 1 for i in G.nodes )
    m.addConstr( gp.quicksum( G.nodes[i]['TOTPOP'] * x[i,0] for i in G.nodes ) >= L )
    m.addConstr( gp.quicksum( G.nodes[i]['TOTPOP'] * x[i,0] for i in G.nodes ) <= U )
    m.addConstr( gp.quicksum( G.nodes[i]['TOTPOP'] * x[i,1] for i in G.nodes ) >= (k-1)*L )
    m.addConstr( gp.quicksum( G.nodes[i]['TOTPOP'] * x[i,1] for i in G.nodes ) <= (k-1)*U )

    # symmetry breaking: fix root to be in first district
    x[root,0].LB = 1

    M = G.number_of_nodes() - 1
    DG = nx.DiGraph(G)

    # add flow-based contiguity constraints (Shirabe) for first district
    f = m.addVars(DG.edges)
    m.addConstrs( gp.quicksum( f[j,i] - f[i,j] for j in G.neighbors(i) ) == x[i,0] for i in G.nodes if i != root )
    m.addConstrs( gp.quicksum( f[j,i] for j in G.neighbors(i) ) <= M * x[i,0] for i in G.nodes if i != root )
    m.addConstr( gp.quicksum( f[j,root] for j in G.neighbors(root) ) == 0 )

    dist = nx.shortest_path_length(G, source=root)
    m.setObjective( gp.quicksum( dist[i] * dist[i] * math.ceil(G.nodes[i]['TOTPOP']/1000) * x[i,0] for i in G.nodes ), GRB.MINIMIZE )
    
    # add cut-based contiguity constraints for second district (whose root we do not know a priori)
    m.Params.LazyConstraints = 1
    m._callback = callback_function
    m._x = x                    
    m._k = k
    m._L = L
    m._U = U
    m._DG = DG                 
    m._root = root               
    m._enumeration_limit = 5001 
    m._districts = list()      
    m._number_of_districts = 0  

    # MIP tolerances
    m.Params.IntFeasTol = 1e-7
    m.Params.FeasibilityTol = 1e-7

    # solve
    m.optimize( m._callback )
    
    # remove any possible duplicates
    m._districts = remove_duplicates(m._districts)
    
    districts = []
    # report solutions
    print("We found the following",m._number_of_districts,"districts:\n")
    for p in range(m._number_of_districts):
        print("district #",p+1,"is", m._districts[p],"\n")
        districts.append( m._districts[p])
        
    return  districts

def min_cut_districts(G, L, U, k, root=None):
    district_cache = list()
    m = gp.Model()
    x = m.addVars(G.nodes, k, vtype=GRB.BINARY)

    m.addConstrs( x[i,0] + x[i,1] == 1 for i in G.nodes )
    m.addConstr( gp.quicksum( G.nodes[i]['TOTPOP'] * x[i,0] for i in G.nodes ) >= L )
    m.addConstr( gp.quicksum( G.nodes[i]['TOTPOP'] * x[i,0] for i in G.nodes ) <= U )
    m.addConstr( gp.quicksum( G.nodes[i]['TOTPOP'] * x[i,1] for i in G.nodes ) >= (k-1)*L )
    m.addConstr( gp.quicksum( G.nodes[i]['TOTPOP'] * x[i,1] for i in G.nodes ) <= (k-1)*U )

    # symmetry breaking: fix root to be in first district
    if root is None:
        max_population = max( G.nodes[i]['TOTPOP'] for i in G.nodes )
        root = [ i for i in G.nodes if G.nodes[i]['TOTPOP']==max_population ][0]
        
    x[root,0].LB = 1

    M = G.number_of_nodes() - 1
    DG = nx.DiGraph(G)

    # add flow-based contiguity constraints (Shirabe) for first district
    f = m.addVars(DG.edges)
    m.addConstrs( gp.quicksum( f[j,i] - f[i,j] for j in G.neighbors(i) ) == x[i,0] for i in G.nodes if i != root )
    m.addConstrs( gp.quicksum( f[j,i] for j in G.neighbors(i) ) <= M * x[i,0] for i in G.nodes if i != root )
    m.addConstr( gp.quicksum( f[j,root] for j in G.neighbors(root) ) == 0 )
    
  
    z = m.addVars(G.edges, vtype=GRB.BINARY)
    m.addConstrs( x[i,0]-x[j,0] <= z[i,j] for i,j in G.edges )
    m.addConstrs( x[j,0]-x[i,0] <= z[i,j] for i,j in G.edges )
    m.setObjective( gp.quicksum(z), GRB.MINIMIZE)
    m.Params.LazyConstraints = 1
    m._callback = cut_callback
    
    m._x = x                   
    m._k = k
    m._L = L
    m._U = U
    m._DG = DG                  
    m._root = root              
    m.Params.IntFeasTol = 1e-7
    m.Params.FeasibilityTol = 1e-7
    
    # speedups exploiting articulation points
    for v in nx.articulation_points(G):
        
        # check if components of G-v have population < L. 
        # if so, we can merge that component with v
        Vv = [ i for i in G.nodes if i != v ]
        for component in nx.connected_components( G.subgraph(Vv) ):
            population = sum( G.nodes[i]['TOTPOP'] for i in component )
            if population < L:
                print("Component of G -",v,"has insufficient population; merging with",component)
                print("The articulation vertex corresponds to",G.nodes[v]['NAME20'])
                m.addConstrs( m._x[v,0] == m._x[w,0] for w in component )

    # inject solution from cache?
    m.NumStart = 0
    V = set( G.nodes )
    for district in district_cache:
        if set(district).issubset(V):
            complement = [ i for i in G.nodes if i not in district ]
            population = sum( G.nodes[i]['TOTPOP'] for i in complement )
            if nx.is_connected(G.subgraph(complement)) and L <= population and population <= U:
                m.NumStart += 1
                m.params.StartNumber = m.NumStart
                for i in district:
                    m._x[i,0].start = 1
                for i in complement:
                    m._x[i,1].start = 1
    
    m.optimize( m._callback )
    
    if  m.status == 2:    
        min_cut_districts = [ [ i for i in DG.nodes if m._x[i,j].x > 0.5 ] for j in range(k)]
        # report solutions
        print("We found the following districts:\n")
        print(min_cut_districts)
        print("the objective value is:", m.objval)
        district_cache.append( [ i for i in DG.nodes if m._x[i,0].x > 0.5 ] )
    else:
        print('m.status is', m.status) 
        min_cut_districts = list()
        
    return min_cut_districts


def enumeration_with_fixed_roots(G, L, U, k, roots=None):
        
    if roots is None or len(roots) == 0:
        raise ValueError("At least one root must be provided.")

    root_nodes = []
    for name in roots:
        matches = [i for i in G.nodes if G.nodes[i]['NAME20'] == name]
        if not matches:
            raise ValueError(f"Root with NAME20 '{name}' not found in graph.")
        root_nodes.append(matches[0])

    full_plans = []
    
    primary_root = root_nodes[0]
    first_districts = enumerate_districts(G, L, U, k, root=primary_root)

    for idx, district_nodes in enumerate(first_districts):
        print(f"\n*************** Plan {idx+1} *********************")
        
        remaining_nodes = [i for i in G.nodes if i not in district_nodes]
        remaining_graph = G.subgraph(remaining_nodes)

        remaining_k = k - 1
        secondary_root = root_nodes[1] if len(root_nodes) > 1 else None

        other_districts = min_cut_districts(remaining_graph, L, U, remaining_k, root=secondary_root)

        if other_districts:
            plan = [district_nodes] + other_districts
            full_plans.append(plan)

    print("\nWe found the following plans:\n")
    for i, plan in enumerate(full_plans):
        print(f"Plan {i+1}: {plan}")

    return full_plans
   
    