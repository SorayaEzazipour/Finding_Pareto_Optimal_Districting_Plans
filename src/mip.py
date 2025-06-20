# # -*- coding: utf-8 -*-
# """
# Created on Fri Aug 16 12:26:53 2024

# @author: sezazip
# """

import ordering, mip_fixing, mip_objective, mip_contiguity
import gurobipy as gp
from gurobipy import GRB
import math
from metrics import *
import networkx as nx

#############################     gurobi_status    ####################################
#############################     gurobi_status    ####################################
def gurobi_status(s):
    if s == 1:
        print("Model is loaded, but no solution information is available.")
    elif s == 2: 
        print("Model was solved to optimality (subject to tolerances), and an optimal solution is available.")
    elif s == 3: 
        print("Model was proven to be infeasible.")
    elif s == 4: 
        print("Model was proven to be either infeasible or unbounded. To obtain a more definitive conclusion, set the DualReductions parameter to 0 and reoptimize.")
    elif s == 5: 
        print("Model was proven to be unbounded. Important note: an unbounded status indicates the presence of an unbounded ray that allows the objective to improve without limit. It says nothing about whether the model has a feasible solution. If you require information on feasibility, you should set the objective to zero and reoptimize.")
    elif s == 6: 
        print("Optimal objective for model was proven to be worse than the value specified in the Cutoff parameter. No solution information is available.")
    elif s == 7: 
        print("Optimization terminated because the total number of simplex iterations performed exceeded the value specified in the IterationLimit parameter, or because the total number of barrier iterations exceeded the value specified in the BarIterLimit parameter.")
    elif s == 8: 
        print("Optimization terminated because the total number of branch-and-cut nodes explored exceeded the value specified in the NodeLimit parameter.")
    elif s == 9: 
        print("Optimization terminated because the time expended exceeded the value specified in the TimeLimit parameter.")
    elif s == 10: 
        print("Optimization terminated because the number of solutions found reached the value specified in the SolutionLimit parameter.")
    elif s == 11: 
        print("Optimization was terminated by the user.")
    elif s == 12: 
        print("Optimization was terminated due to unrecoverable numerical difficulties.")
    elif s == 13: 
        print("Unable to satisfy optimality tolerances; a sub-optimal solution is available.")
    elif s == 14: 
        print("An asynchronous optimization call was made, but the associated optimization run is not yet complete.")
    elif s == 15:
        print("User specified an objective limit (a bound on either the best objective or the best bound), and that limit has been reached.")
    elif s == 16:
        print("Optimization terminated because the work expended exceeded the value specified in the WorkLimit parameter.")
    else:
        print("No specific error could be recognized.")

#############################     labeling_model    ####################################
#############################     labeling_model    ####################################

def labeling_model(G, deviation_persons, obj_type, contiguity='lcut', cutoff=None, verbose=False,
                   warm_start=None, time_limit=7200, sizes=None, max_B=False,
                   symmetry_breaking=None, similarity=None):
    
    G._L = math.ceil(G._ideal_population - deviation_persons)
    G._U = math.floor(G._ideal_population + deviation_persons)
    k = G._k
    print(f'L = {G._L} and U = {G._U} and k = {k}' )

    m = gp.Model()
    if not verbose:
        m.Params.OutputFlag = 0
    m._G = G
    
    if sizes is None:
        sizes = [1 for _ in range(k)]    
    m._sizes = sizes
    print('sizes = ', m._sizes)
    
    m._x = m.addVars(G.nodes, range(len(m._sizes)), name='x', vtype=GRB.BINARY)
    m._r = m.addVars(G.nodes, range(len(m._sizes)), name='r', vtype=GRB.BINARY)
    
    # add constraints saying that each node i is assigned to one district
    m.addConstrs(sum(m._x[i, j] for j in range(len(m._sizes))) == 1 for i in G.nodes)
    
    # add constraints saying that if node i roots district j, then i should be in district j
    m.addConstrs(m._r[i, j] <= m._x[i, j] for i in G.nodes for j in range(len(m._sizes)))
    
    # add constraints saying that each district has population at least L and at most U
    m.addConstrs(sum(G.nodes[i]['TOTPOP'] * m._x[i, j] for i in G.nodes) >= G._L * m._sizes[j] for j in range(len(m._sizes)))
    m.addConstrs(sum(G.nodes[i]['TOTPOP'] * m._x[i, j] for i in G.nodes) <= G._U * m._sizes[j] for j in range(len(m._sizes)))
    
    DG = nx.DiGraph(G)
    DG._k = G._k
    DG._L = G._L
    DG._U = G._U
    
    m._y = m.addVars(DG.edges, range(len(m._sizes)), name='y', vtype=GRB.BINARY)
    
    # add constraints saying that edge {u,v} is cut if u is assigned to district j but v is not.
    m.addConstrs(m._x[u, j] - m._x[v, j] <= m._y[u, v, j] for u, v in DG.edges for j in range(len(m._sizes)))
    
    m._is_cut = m.addVars(DG.edges, vtype=GRB.BINARY)
    m.addConstrs(m._is_cut[u, v] == sum(m._y[u, v, j] for j in range(len(m._sizes))) for u, v in DG.edges)
    m.addConstrs(m._is_cut[u, v] == sum(m._y[v, u, j] for j in range(len(m._sizes))) for u, v in DG.edges)
    #m.addConstrs( m._is_cut[min(u,v),max(u,v)] == sum( m._y[u,v,j] for j in range(len(m._sizes)))  for u,v in DG.edges )
     
    if obj_type == 'cut_edges':
        mip_objective.add_cut_edges_objective(m, G)
    elif obj_type == 'perimeter':
        mip_objective.add_perimeter_objective(m, DG)
    elif obj_type == 'inverse_Polsby_Popper':
        mip_objective.add_inverse_Polsby_Popper_objective(m, DG)
    elif obj_type == 'average_Polsby_Popper':
        mip_objective.add_average_Polsby_Popper_objective(m, DG)
    elif obj_type == 'bottleneck_Polsby_Popper':      
        mip_objective.add_bottleneck_Polsby_Popper_objective(m, DG)
    else:
        assert False, f"Unsupported objective encountered: {obj_type}"
    
    m._callback = None
    m._numCallbacks = 0
    m._numLazyCuts = 0

    if max_B:
        (B, B_time) = ordering.solve_maxB_problem(G)
        DG._ordering = ordering.find_ordering(G, B)
    
    if symmetry_breaking == 'orbitope' and not max_B:
        raise ValueError("Error: 'orbitope' symmetry breaking requires max_B parameter to be True.")
    elif symmetry_breaking == 'orbitope' and  max_B: 
        add_partitioning_orbitope_constraints(m, DG)
        mip_fixing.do_variable_fixing(m, DG)
    if symmetry_breaking == 'rsum':
        add_symmetry_breaking_constraints(m, DG)          
        
    if contiguity == 'lcut':
        if symmetry_breaking == None:
            #Each district should have a root
            m.addConstrs(sum(m._r[i, j] for i in G.nodes) == 1 for j in range(len(m._sizes)))   
        m._DG = DG
        m.Params.LazyConstraints = 1
        m._callback = mip_contiguity.lcut_callback
    elif contiguity == 'scf':
        add_symmetry_breaking_constraints(m, G)
        mip_contiguity.add_scf_constraints(m, G)
    elif contiguity == 'shir':
        add_symmetry_breaking_constraints(m, G)
        mip_contiguity.add_shir_constraints(m, G)
    else:
        assert False, f"ERROR: contiguity constraint {contiguity} is not supported."

    if warm_start:
        print("Applying warm start!")
        assert len(warm_start) == k
        assert all(i in DG.nodes for j in range(k) for i in warm_start[j])
        labeling = label(warm_start)
        orbitope_friendly_labeling = get_orbitope_friendly_labeling(DG, labeling)
        inject_warm_start(m, DG, orbitope_friendly_labeling)
        
    # speedup that exploits articulation points   
    articulation_pts = list(nx.articulation_points(G)) 
    for j in range(k - 1):
        for v in articulation_pts:
            Vv = [ i for i in G.nodes if i != v ]
            for component in nx.connected_components(G.subgraph(Vv)):
                population = sum( G.nodes[i]['TOTPOP'] for i in component )
                if population < G._L:
                    m.addConstrs( m._x[v, j] == m._x[w, j] for w in component )
            
    #add similarity constraint if needed
    if similarity is not None:
        m.addConstr(sum(G.nodes[i]["TOTPOP"] * m._x[i, j] 
                        for j in range(G._k) for i in similarity[0][j]) >= similarity[1])
  
    if cutoff is not None:
        m.Params.Cutoff = cutoff
    m.Params.FeasibilityTol = 1e-7
    m.Params.IntFeasTol = 1e-7
    m.Params.MIPGap = 1e-7
    m.Params.TimeLimit = time_limit
    m.optimize(m._callback)
    
    if m.solCount > 0:
        solution = [[i for i in G.nodes if m._x[i, j].x > 0.5] for j in range(len(m._sizes))]
        is_valid_clustering = check_county_clustering(G, solution, m._sizes)
    else:
        is_valid_clustering = False
    
    if is_valid_clustering:
        if m.status == GRB.Status.TIME_LIMIT:
            print("Time limit reached! Best feasible solution found:")
        if obj_type in {"inverse_Polsby_Popper", "cut_edges", "perimeter"}:
            return (solution, m.objVal, m.objBound, int(m.status))
        else:
            return (solution, m.objBound, m.objVal, int(m.status))
    else:
        print("No feasible solution found." if m.status != GRB.Status.TIME_LIMIT else "Time limit reached, but no feasible solution found.")
        nontrivial_bound = m.objBound if m.objBound < GRB.INFINITY else None
        if obj_type in {"inverse_Polsby_Popper", "cut_edges", "perimeter"}:
            return (None, None, nontrivial_bound, int(m.status))
        else:
            return (None, nontrivial_bound, None, int(m.status))
        

############################# add_partitioning_orbitope_constraints ####################################
############################# add_partitioning_orbitope_constraints ####################################
def add_partitioning_orbitope_constraints(m, G):

    s = m.addVars(G.nodes, G._k, name='s')
    u = m.addVars(G.nodes, G._k, name='u')
    w = m.addVars(G.nodes, G._k, name='w')

    m.addConstrs(m._x[i,j] == s[i,j]-s[i,j+1] for i in G.nodes for j in range(G._k-1))
    m.addConstrs(m._x[i,G._k-1] == s[i,G._k-1] for i in G.nodes)

    m.addConstrs(m._r[G._ordering[0],j] == w[G._ordering[0],j] for j in range(G._k))
    m.addConstrs(m._r[G._ordering[i],j] == w[G._ordering[i],j] - w[G._ordering[i-1],j] for i in range(1,G.number_of_nodes()) for j in range(G._k))

    m.addConstrs(s[i,j] <= w[i,j] for i in G.nodes for j in range(G._k))

    m.addConstrs(u[G._ordering[i],j]+m._r[G._ordering[i],j] == u[G._ordering[i+1],j] + m._r[G._ordering[i+1],j+1] for i in range(0,G.number_of_nodes()-1) for j in range(G._k-1))
    m.addConstrs(u[G._ordering[i],G._k-1]+m._r[G._ordering[i],G._k-1] == u[G._ordering[i+1],G._k-1] for i in range(0,G.number_of_nodes()-1))
    m.addConstrs(u[G._ordering[G.number_of_nodes()-1],j]+m._r[G._ordering[G.number_of_nodes()-1],j] == 0 for j in range(G._k-1))

    m._r[G._ordering[0],0].LB = 1
    m.addConstr( u[G._ordering[G.number_of_nodes()-1],G._k-1] + m._r[G._ordering[G.number_of_nodes()-1],G._k-1]==1 )  
    m.update()
    return

############################# get_orbitope_friendly_labeling ####################################
############################# get_orbitope_friendly_labeling ####################################

# We need to be careful because of partitioning orbitope constraints. 
#   For each district, find its earliest vertex in the ordering,
#   then sort these earliest vertices to get the district labels
#    
def get_orbitope_friendly_labeling(G, unfriendly_labeling):    
    district_map = { j : -1 for j in range(G._k) }
    labeling = { i : -1 for i in G.nodes }
    count = 0
    for i in G._ordering:
        j = unfriendly_labeling[i]
        # have we found earliest vertex from district j?
        if district_map[j] == -1:
            # if so, then vertex i roots district 'j' in unmapped labeling,
            #   and anything labeled 'j' should instead be relabeled 'count'
            district_map[j] = count
            count += 1
        labeling[i] = district_map[j]
    return labeling

############################# inject_warm_start ####################################
############################# inject_warm_start ####################################
    
def inject_warm_start(m, G, labeling):
    # init vars to 0
    for i in G._ordering:
        for j in range(G._k):
            m._x[i,j].start = 0
            m._r[i,j].start = 0
    for u,v in G.edges:
        for j in range(G._k):
            m._y[u,v,j].start = 0
    
    # next, inject nonzeros of solution
    root_found = { j : False for j in range(G._k) }
    for i in G._ordering:
        j = labeling[i]
        m._x[i,j].start = 1
        if not root_found[j]:
            root_found[j] = True
            m._r[i,j].start = 1
    for u,v in G.edges:
        j = labeling[u]
        if labeling[v] != j:
            m._y[u,v,j].start = 1
    m.update()
    return
    
############################# add_symmetry_breaking_constraints ####################################
############################# add_symmetry_breaking_constraints ####################################

def add_symmetry_breaking_constraints(m, G):
    DG = nx.DiGraph(G)  

    # Symmetry breaking inequalities
    if m._sizes[0] == 1 and len(m._sizes) == 2: 
        rsum = m.addVars(DG.nodes, range(len(m._sizes)), vtype=GRB.BINARY, name="rsum")
        nodes_list = list(DG.nodes)
        m.addConstrs(m._x[i,j] <= rsum[i,j] for i in DG.nodes for j in range(len(m._sizes)))
        for p in range(len(nodes_list)):
            i = nodes_list[p]  
            if p==0:
                m.addConstrs(rsum[i, j]==m._r[i, j] for j in range(len(m._sizes)))
            else:
                previous_node = nodes_list[p-1]
                m.addConstrs(rsum[i, j]==rsum[previous_node, j] + m._r[i, j] for j in range(len(m._sizes)))
    m.update()
    return
