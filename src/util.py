import networkx as nx
from networkx.readwrite import json_graph
import json
import math

# Function to read graph from json file
#
def read_graph_from_json(json_file, update_population=True, rescale_distance=True):
    with open(json_file) as f:
        data = json.load(f)
    G = json_graph.adjacency_graph(data) 

    if update_population:
        # total population
        for i in G.nodes:
            G.nodes[i]['TOTPOP'] = G.nodes[i]['P0010001'] 
    
    # change distance units from meters to 100km for better numerics in gurobi
    #  note: 100km roughly corresponds to one degree of latitude
    if rescale_distance:
        ht = 100000  
        for i in G.nodes:
            G.nodes[i]['area'] /= ( ht * ht )
            if G.nodes[i]['boundary_node']:
                G.nodes[i]['boundary_perim'] /= ht
        for i,j in G.edges:
            G.edges[i,j]['shared_perim'] /= ht
        
    return G


def district_objective(G, district, obj_type):
    district_bool = { i : False for i in G.nodes }
    for i in district:
        district_bool[i] = True
    if obj_type == 'cut_edges':
        return sum( 1 for u in district for v in G.neighbors(u) if not district_bool[v] )
    elif obj_type == 'perimeter':
        internal_perim = sum( G.edges[u,v]['shared_perim'] for u in district for v in G.neighbors(u) if not district_bool[v] )
        external_perim = sum( G.nodes[i]['boundary_perim'] for i in district if G.nodes[i]['boundary_node'] ) 
        return internal_perim + external_perim
    elif obj_type == 'shared_perim':
        return sum( G.edges[u,v]['shared_perim'] for u in district for v in G.neighbors(u) if not district_bool[v] )
    elif obj_type =='inverse_polsby_popper':
        internal_perim = sum( G.edges[u,v]['shared_perim'] for u in district for v in G.neighbors(u) if not district_bool[v] )
        external_perim = sum( G.nodes[i]['boundary_perim'] for i in district if G.nodes[i]['boundary_node'] ) 
        P = internal_perim + external_perim
        A = sum( G.nodes[i]['area'] for i in district )
        return P * P / ( 4 * math.pi * A )
    else:
        assert False
        
        
def plan_objective(G, plan, obj_type):
    label = { i : j for j in range(len(plan)) for i in plan[j] }
    if obj_type == 'cut_edges':
        return sum( 1 for u,v in G.edges if label[u] != label[v] )
    elif obj_type == 'perimeter':
        internal_perim = sum( G.edges[u,v]['shared_perim'] for u,v in G.edges if label[u] != label[v] )
        external_perim = sum( G.nodes[i]['boundary_perim'] for i in G.nodes if G.nodes[i]['boundary_node'] ) 
        return internal_perim + external_perim
    elif obj_type =='inverse_polsby_popper':
        average_ipp = 0
        for p in range(len(plan)):
            internal_perim = sum( G.edges[u,v]['shared_perim'] for u in plan[p] for v in G.neighbors(u) if label[u]!=label[v] )
            external_perim = sum( G.nodes[i]['boundary_perim'] for i in plan[p] if G.nodes[i]['boundary_node'] ) 
            P = internal_perim + external_perim
            A = sum( G.nodes[i]['area'] for i in plan[p] )
            average_ipp += P * P / ( 4 * math.pi * A )
        return average_ipp / len(plan)
    else:
        assert False