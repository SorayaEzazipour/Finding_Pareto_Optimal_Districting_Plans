import networkx as nx
import gurobipy as gp
from gurobipy import GRB
import math
from util import district_objective

# Finds a minimal vertex subset that separates component from vertex b in digraph DG.
#
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

# Gurobi callback function, 
#   used to find/store the discovered districts
#   or cut off infeasible (disconnected) ones
#
def callback_function(m, where):

    if where == GRB.Callback.MIPNODE and not m._exit:
        best_bound = m.cbGet(GRB.Callback.MIPNODE_OBJBND)
        if best_bound >= m._cutoff:
            DG = m._DG
            m.cbLazy( m._x[DG._root,0] <= -1 )
            m._exit = True
    
    # check if LP relaxation at this branch-and-bound node has an integer solution
    if where != GRB.Callback.MIPSOL: 
        return
        
    # retrieve the LP solution
    xval = m.cbGetSolution(m._x)
    #objval = m.cbGetSolution(m._obj)
    DG = m._DG
    
    district = [ i for i in DG.nodes if xval[i,0] > 0.5 ]
    assert nx.is_strongly_connected( DG.subgraph(district) )
    objval = district_objective(DG, district, m._obj_type)
    
    # is complement connected? if so, we may be able to build a plan from this district
    complement = [ i for i in DG.nodes if xval[i,1] > 0.5 ]
    if len(complement)==1 or nx.is_strongly_connected( DG.subgraph(complement) ):
        
        if m._verbose:
            print("found", district,"with objective =",objval)
        add_no_worse_cut = False
        
        if m._number_of_districts < m._enumeration_limit:
            
            m._districts.append( district )
            m._objectives.append( objval )
            m._number_of_districts += 1
            if m._verbose:
                print("   ^added to list!")
            
            if m._number_of_districts == m._enumeration_limit:
                add_no_worse_cut = True
            
        else:
            worst_j = 0
            for j in range(1, m._enumeration_limit):
                if m._objectives[j] > m._objectives[worst_j]:
                    worst_j = j
            
            worst_obj = m._objectives[worst_j]
            if objval < worst_obj:
                m._districts[worst_j] = district
                m._objectives[worst_j] = objval
                if m._verbose:
                    print("   replaced worse solution with objective =",worst_obj)
                add_no_worse_cut = True
            else:
                if m._verbose:
                    print("   ^tossing it because it's worse than the others")
                
        # hack to disallow worse solutions
        if add_no_worse_cut:
            new_worst_obj = max( m._objectives[j] for j in range(m._enumeration_limit) )
            if m._verbose:
                print("adding cut saying that objective should be less than",new_worst_obj)
            if m._obj_type == 'cut_edges':
                # exploit integrality of cut_edges objective. If worst is 12, then cutoff is 11.00...01
                m._cutoff = min( m._cutoff, new_worst_obj - (1 - 1e-6) ) 
            else:
                m._cutoff = min( m._cutoff, new_worst_obj - 1e-6 )
            m.cbLazy( m._obj <= m._cutoff )
            
        # add no-good cut
        m.cbLazy( gp.quicksum( m._x[i,1] for i in district ) + gp.quicksum( m._x[i,0] for i in complement ) >= 1 )
        
    # if complement is disconnected....
    else:
        # find a maximum population component
        b = None
        max_component_population = -1
        for component in nx.strongly_connected_components( DG.subgraph(complement) ):
            component_population = sum( DG.nodes[i]['TOTPOP'] for i in component )
            if component_population > max_component_population:

                # find a maximum population vertex 'b' from this component
                max_component_population = component_population
                max_vertex_population = max( DG.nodes[i]['TOTPOP'] for i in component )
                max_population_vertices = [ i for i in component if DG.nodes[i]['TOTPOP'] == max_vertex_population ]
                b = max_population_vertices[0]

        # each other component (besides b's), find some vertex 'a' and add cut.
        for component in nx.strongly_connected_components( DG.subgraph(complement) ):
            if b in component:
                continue

            # find a maximum population vertex 'a' from this component
            max_vertex_population = max( DG.nodes[i]['TOTPOP'] for i in component )
            max_population_vertices = [ i for i in component if DG.nodes[i]['TOTPOP'] == max_vertex_population ]
            a = max_population_vertices[0]

            # add a,b-separator inequality
            C = find_minimal_separator(DG, component, b)
            m.cbLazy( m._x[a,1] + m._x[b,1] <= 1 + gp.quicksum( m._x[c,1] for c in C ) )


# key:    ( tuple(sorted(G.nodes)), G._L, G._U, G._k, G._size, obj_type)
# value:  m._districts
etd_cache = dict()


# The optimization function for the enumeration task
#
def enumerate_top_districts(G, obj_type='cut_edges', enumeration_limit=10, forced_names=list(), forbidden_names=list(), cache=False, time_limit=None, verbose=False):
    
    assert obj_type in {'cut_edges', 'perimeter', 'inverse_polsby_popper'}

    # use cache to exit early?
    key = ( tuple(sorted(G.nodes)), G._L, G._U, G._k, G._size, obj_type )
    if cache and key in etd_cache:
        #print( "Used cache!" )
        return etd_cache[key]
    
    # build model
    m = gp.Model()
    if verbose==False:
        m.Params.OutputFlag = 0

    # x[i,j]=1 if vertex i is assigned to district j
    x = m.addVars(G.nodes, 2, vtype=GRB.BINARY)

    # each vertex is assigned to one district
    m.addConstrs( x[i,0] + x[i,1] == 1 for i in G.nodes )

    # add population balance constraints for district and its complement
    m.addConstr( gp.quicksum( G.nodes[i]['TOTPOP'] * x[i,0] for i in G.nodes ) >= G._L*G._size )
    m.addConstr( gp.quicksum( G.nodes[i]['TOTPOP'] * x[i,0] for i in G.nodes ) <= G._U*G._size )
    m.addConstr( gp.quicksum( G.nodes[i]['TOTPOP'] * x[i,1] for i in G.nodes ) >= (G._k-G._size)*G._L )
    m.addConstr( gp.quicksum( G.nodes[i]['TOTPOP'] * x[i,1] for i in G.nodes ) <= (G._k-G._size)*G._U )

    # fix root to be in first district
    if G._root is None:
        max_population = max( G.nodes[i]['TOTPOP'] for i in G.nodes )
        G._root = [ i for i in G.nodes if G.nodes[i]['TOTPOP']==max_population ][0]
        
    x[G._root,0].LB = 1

    # fix forced names
    for i in G.nodes:
        if G.nodes[i]['NAME20'] in forced_names:
            x[i,0].LB = 1

    # fix forced names
    for i in G.nodes:
        if G.nodes[i]['NAME20'] in forbidden_names:
            x[i,0].UB = 0

    M = G.number_of_nodes() - 1
    DG = nx.DiGraph(G)
    DG._root = G._root

    # add flow-based contiguity constraints (Shirabe) for first district
    f = m.addVars(DG.edges)
    m.addConstrs( gp.quicksum( f[j,i] - f[i,j] for j in G.neighbors(i) ) == x[i,0] for i in G.nodes if i != G._root )
    m.addConstrs( gp.quicksum( f[j,i] for j in G.neighbors(i) ) <= M * x[i,0] for i in G.nodes if i != G._root )
    m.addConstr( gp.quicksum( f[j,G._root] for j in G.neighbors(G._root) ) == 0 )
    
    # cut edge vars
    is_cut = m.addVars(G.edges, vtype=GRB.BINARY)
    m.addConstrs( x[i,0]-x[j,0] <= is_cut[i,j] for i,j in G.edges )
    m.addConstrs( x[j,0]-x[i,0] <= is_cut[i,j] for i,j in G.edges )
    
    # add objective
    obj = m.addVar()
    m.setObjective( obj, GRB.MINIMIZE)
    
    if obj_type == 'cut_edges':
        
        m.addConstr( obj == gp.quicksum(is_cut) )
    
    elif obj_type == 'perimeter':
        
        # perimeter_length = within_state_perimeter + state_boundary_perimeter
        m.addConstr( obj == gp.quicksum( G.edges[i,j]['shared_perim'] * is_cut[i,j] for i,j in G.edges ) 
                    + gp.quicksum( G.nodes[i]['boundary_perim'] * x[i,0] for i in G.nodes if G.nodes[i]['boundary_node'] ) )
    elif obj_type == 'inverse_polsby_popper':
        
        # area of district
        A = m.addVar() 
        m.addConstr( A == gp.quicksum( G.nodes[i]['area'] * x[i,0] for i in G.nodes ) )

        # perimeter of district
        P = m.addVar() 
        m.addConstr( P == gp.quicksum( G.edges[u,v]['shared_perim'] * is_cut[u,v] for u,v in G.edges )
                 + gp.quicksum( G.nodes[i]['boundary_perim'] * x[i,0] for i in G.nodes if G.nodes[i]['boundary_node'] ) )

        # relate inverse polsby-popper score 'obj' to A and P
        m.addConstr( P * P <= 4 * math.pi * A * obj )

    else:
        print("Objective type",obj_type,"is not supported.")
        assert False
        
    # add cut-based contiguity constraints for second district (whose root we do not know a priori)
    m.Params.LazyConstraints = 1
    m._callback = callback_function
    
    # additional bookkeeping 
    m._x = x                    # assignment variables
    m._obj = obj
    m._obj_type = obj_type
    m._DG = DG                  # directed graph
    m._exit = False
    m._number_of_districts = 0
    m._cutoff = float('inf')
    m._districts = list()
    m._objectives = list()
    m._enumeration_limit = enumeration_limit
    m._verbose = verbose

    # MIP tolerances
    m.Params.IntFeasTol = 1e-7
    m.Params.FeasibilityTol = 1e-7
    
    # speedup that exploits articulation points
    for v in nx.articulation_points(G):
        Vv = [ i for i in G.nodes if i != v ]
        for component in nx.connected_components( G.subgraph(Vv) ):
            population = sum( G.nodes[i]['TOTPOP'] for i in component )
            case1 = population < G._L*G._size and G._root in component
            case2 = population < G._L*(G._k-G._size) and G._root not in component
            if case1 or case2:
                m.addConstrs( m._x[v,0] == m._x[w,0] for w in component )

    # solve
    if time_limit is not None:
        m.Params.TimeLimit = time_limit
    m.optimize( m._callback )

    # add solution to our cache and return
    if cache:
        etd_cache[key] = m._districts
    return m._districts


def districting_heuristic(G, obj_type='cut_edges', enumeration_limit=10):
    
    G._size = 1
    districts = enumerate_top_districts( G, obj_type=obj_type, enumeration_limit=enumeration_limit )
    partial_plans = [ [district] for district in districts ]
    plans = list()
    
    while partial_plans != list():
        
        partial_plan = partial_plans.pop()
        ndistricts = len(partial_plan)
        
        used = [ i for j in range(ndistricts) for i in partial_plan[j] ]
        unused = [ i for i in G.nodes if i not in used ]
        
        if ndistricts == G._k - 1:
            partial_plan.append( unused )
            plans.append(partial_plan)
            print("Finished plan #",len(plans))
            continue
            
        H = G.subgraph(unused).copy()
        H._L = G._L
        H._U = G._U
        H._k = G._k - ndistricts
        H._root = None
        H._size = 1
        for i in H.nodes:

            # which nodes are boundary in H = G - used?
            if H.nodes[i]['boundary_node'] == False:
                for j in G.neighbors(i):
                    if j in used:
                        H.nodes[i]['boundary_node'] = True
                        H.nodes[i]['boundary_perim'] = 0
                        break # loop over neighbors j

            #  and what is their new exterior boundary length?
            for j in G.neighbors(i):
                if j in used:
                    if (i,j) in G.edges:
                        H.nodes[i]['boundary_perim'] += G.edges[i,j]['shared_perim']
                    else:
                        H.nodes[i]['boundary_perim'] += G.edges[j,i]['shared_perim']
             
            
        print("\n ***Seeking district #", ndistricts+1,"for partial plan",partial_plan)
        districts = enumerate_top_districts( H, obj_type=obj_type, enumeration_limit=enumeration_limit )
        for district in districts:
            new_partial_plan = partial_plan.copy()
            new_partial_plan.append( district )
            partial_plans.append(new_partial_plan)
        
    return plans 


def printif(condition, statement):
    if condition:
        print(statement)


def iterative_refinement(G, L, U, k, enumeration_limit=10, enum_time_limit=600, break_size=1, cache=True, verbose=False):

    # initializations
    trivial_clustering = [ list(G.nodes) ]
    county_clusterings = [ trivial_clustering ]
    list_of_sizes = [ [k] ]
    plans = list()

    while county_clusterings != list():

        # take a county clustering and their sizes from the list
        county_clustering = county_clusterings.pop()
        num_clusters = len(county_clustering)
        sizes = list_of_sizes.pop()
        
        if all( size <= break_size for size in sizes ):
            printif(verbose and break_size==1, "Found plan!")
            plans.append( county_clustering )
            continue

        printif(verbose,f"Currently, sizes = {sizes}.")
        
        # pick a cluster to divide up?
        min_size = min( size for size in sizes if size > break_size )
        p = [ p for p in range(len(sizes)) if sizes[p] == min_size ][0]
        cluster = county_clustering[p]

        GS = G.subgraph(cluster).copy()
        GS._L = L
        GS._U = U
        GS._k = sizes[p]
        printif(verbose,f"Trying to break cluster of size {sizes[p]} into sub-clusters.")

        # try all possible sizes?
        for size in range(1,sizes[p]):

            GS._size = size
            GS._root = None
            printif(verbose,f"Trying sub-cluster sizes: {size} and {sizes[p]-size}.")
            lefts = enumerate_top_districts( GS, obj_type='cut_edges', enumeration_limit=enumeration_limit, cache=cache, time_limit=enum_time_limit, verbose=False )

            for left in lefts:
                right = [ i for i in cluster if i not in left ]
                new_county_clustering = county_clustering[0:p] + [left, right] + county_clustering[p+1:num_clusters+1] 
                new_sizes = sizes[0:p] + [ size, sizes[p] - size ] + sizes[p+1:num_clusters+1]

                county_clusterings.append( new_county_clustering )
                list_of_sizes.append( new_sizes )

    return plans
