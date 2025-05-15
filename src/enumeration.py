# -*- coding: utf-8 -*-
"""
Created on Sun May  4 16:03:13 2025

@author: sezazip
"""

import networkx as nx
import gurobipy as gp
from gurobipy import GRB
from mip import*
from metrics import check_plan
import math


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


def enumerate_and_solve_k2_subproblems(G, deviation_persons, L, U, k, root=None, obj_type = 'cut_edges'):
    assert obj_type in {
        'inverse_Polsby_Popper', 'cut_edges', 'perimeter',
        'average_Polsby_Popper', 'bottleneck_Polsby_Popper',
        'stay_in_old_districts'
    }, "Invalid objective type."
    
        
    matches = [i for i in G.nodes if G.nodes[i]['NAME10'] == root]
    if not matches:
        raise ValueError(f"Root with NAME10 '{root}' not found in graph.")
    root_node = matches[0]

    full_plans = []
    first_districts = enumerate_districts(G, L, U, k, root= root_node)

    for idx, district_nodes in enumerate(first_districts):
        print(f"\n*************** Plan {idx+1} *********************")
        
        remaining_nodes = [i for i in G.nodes if i not in district_nodes]
        remaining_graph = G.subgraph(remaining_nodes)

        remaining_k = k - 1
        remaining_graph._ideal_population =G._ideal_population
        remaining_graph._k = remaining_k
        
        other_districts, upper_bound, lower_bound, status = labeling_model(
            remaining_graph, deviation_persons=deviation_persons, obj_type=obj_type,
            contiguity='shir', verbose=True)

        if other_districts:
            plan = [district_nodes] + other_districts
            full_plans.append(plan)
            print('plan =', plan)
            check_plan(G, plan,year=2020)
    print("\nWe found the following plans:\n")
    for i, plan in enumerate(full_plans):
        print(f"Plan {i+1}: {plan}")

    return full_plans
   
    