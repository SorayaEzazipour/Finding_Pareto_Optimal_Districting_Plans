import gurobipy as gp
from gurobipy import GRB
import networkx as nx

def lcut_callback(m, where):
    if where == GRB.Callback.MIPSOL:
        m._numCallbacks += 1 
        DG = m._DG
        xval = m.cbGetSolution(m._x)
        rval = m.cbGetSolution(m._r)

        for j in range(len(m._sizes)):
            
            # vertices assigned to this district (label j)
            S = [ v for v in DG.nodes if xval[v,j] > 0.5 ]
            
            # what shall we deem as the "root" of this district? call it b
            b = [ i for i in DG.nodes if rval[i,j] > 0.5 ][0]
            
            # for each component that doesn't contain b, add a cut
            for component in nx.strongly_connected_components(DG.subgraph(S)):
                if b in component: 
                    continue
                
                # find some vertex "a" that has largest population in this component
                maxp = max( DG.nodes[v]['TOTPOP'] for v in component)
                maxp_vertices = [ v for v in component if DG.nodes[v]['TOTPOP'] == maxp ]
                a = maxp_vertices[0]  
                    
                # get minimal a,b-separator
                C = find_fischetti_separator(DG, component, b)
                
                # make it a minimal *length-U* a,b-separator
                for (u,v) in DG.edges():
                    DG[u][v]['lcutweight'] = DG.nodes[u]['TOTPOP']  
                    
                # "remove" C from graph
                for c in C:
                    for node in DG.neighbors(c):
                        DG[c][node]['lcutweight'] = m._sizes[j] * DG._U + 1
                
                # is C\{c} a length-U a,b-separator still? If so, remove c from C
                drop_from_C = []
                for c in C:
                    
                    # temporarily add c back to graph (i.e., "remove" c from cut C)
                    for node in DG.neighbors(c):
                        DG[c][node]['lcutweight'] = DG.nodes[c]['TOTPOP']
                    
                    # what is distance from a to b in G-C ?
                    distance_from_a = nx.single_source_dijkstra_path_length(DG, a, weight='lcutweight')
                    
                    if distance_from_a[b] + DG.nodes[b]['TOTPOP'] > m._sizes[j] * DG._U:
                        # c was not needed in the cut C. delete c from C
                        drop_from_C.append(c)
                    else:
                        # keep c in C. revert arc weights back to "infinity"
                        for node in DG.neighbors(c):
                            DG[c][node]['lcutweight'] = m._sizes[j] * DG._U + 1
                    
                # add lazy cut
                minC = [ c for c in C if c not in drop_from_C ]
                m.cbLazy( m._x[a,j] + m._x[b,j] <= 1 + sum( m._x[c,j] for c in minC ) )
                m._numLazyCuts += 1
    return

# This is algorithm 1 from Fischetti, Matteo, et al. 
#   "Thinning out Steiner trees: a node-based model for uniform edge costs." 
#   Mathematical Programming Computation 9.2 (2017): 203-229.
#
def find_fischetti_separator(G, component, b):
    neighbors_component = { i : False for i in G.nodes }
    for i in nx.node_boundary(G, component, None):
        neighbors_component[i] = True
    
    visited = { i : False for i in G.nodes }
    child = [ b ]
    visited[b] = True
    
    while child:
        parent = child
        child = []
        for i in parent:
            if not neighbors_component[i]:
                for j in G.neighbors(i):
                    if not visited[j]:
                        child.append(j)
                        visited[j] = True
    
    C = [ i for i in G.nodes if neighbors_component[i] and visited[i] ]
    return C

def add_shir_constraints(m, G):
    
    DG = nx.DiGraph(G)
    # g[i,j] = amount of flow generated at node i of type j
    g = m.addVars(DG.nodes, range(len(m._sizes)), name='g')
    
    # f[j,u,v] = amount of flow sent across arc uv of type j
    f = m.addVars( range(len(m._sizes)), DG.edges, name='f' )
    
    for j, size in enumerate(m._sizes):

        # compute big-M  
        M = most_possible_nodes_in_one_district(G, size) - 1
    
        # flow can only be generated at roots
        m.addConstrs( g[i,j] <= ( M + 1) * m._r[i,j] for i in DG.nodes )
    
        # flow balance
        m.addConstrs( g[i,j] - m._x[i,j] == sum( f[j,i,u] - f[j,u,i] for u in DG.neighbors(i)) for i in DG.nodes )
    
        # flow type j can enter vertex i only if (i is assigned to district j) and (i is not root of j)
        m.addConstrs( sum( f[j,u,i] for u in DG.neighbors(i) ) <= M * (m._x[i,j] - m._r[i,j]) for i in DG.nodes )
    
    m.update()
    return

def add_scf_constraints(m, G):
    DG = nx.DiGraph(G)
    
    # Add flow variables: f[u,v] = amount of flow sent across arc uv 
    #  Flows are sent across arcs of DG
    m._f = m.addVars( DG.edges, name='f' )
        
    #  the region should have one root
    m.addConstrs(sum(m._r[i,j] for i in DG.nodes) == 1 for j in range(len(m._sizes)))
   
    most = 0
    for size in m._sizes:
       # for big M
       M = most_possible_nodes_in_one_district(G, size) 
       most = max(most, M)
        
    # if not a root, consume some flow.
    # if a root, only send out (so much) flow.
    m.addConstrs( sum( m._f[v,u] - m._f[u,v] for v in DG.neighbors(u) ) 
                         >= 1 - ( most ) * sum( m._r[u,j] for j in range(len(m._sizes)) ) for u in DG.nodes )
        
    # do not send flow across cut edges
    m.addConstrs( m._f[u,v] + m._f[v,u] <= ( most - 1 ) * ( 1 - sum( m._y[u,v,j] for j in range(len(m._sizes)) ) ) for u,v in DG.edges )

    m.update()
    return

def most_possible_nodes_in_one_district(G, size=None):
    cumulative_population = 0
    num_nodes = 0
    population = [ G.nodes[i]['TOTPOP'] for i in G.nodes ]
    
    upper_limit = size * G._U if size is not None else G._U
    for ipopulation in sorted(population):
        cumulative_population += ipopulation
        num_nodes += 1
        
        if cumulative_population > upper_limit:
            return num_nodes - 1
    return len(population)
