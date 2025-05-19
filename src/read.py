# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 12:15:50 2024

@author: sezazip
"""
from networkx.readwrite import json_graph
import json

import math

congressional_districts_2020  = {
      'CA': 52, 'TX': 38, 'FL': 28, 'NY': 26, 'PA': 17, 
      'IL': 17, 'OH': 15, 'GA': 14, 'NC': 14, 'MI': 13, 
      'NJ': 12, 'VA': 11, 'WA': 10, 'AZ': 9,  'MA': 9, 
      'TN': 9,  'IN': 9,  'MD': 8,  'MO': 8,  'WI': 8, 
      'CO': 8,  'MN': 8,  'SC': 7,  'AL': 7,  'LA': 6, 
      'KY': 6,  'OR': 6,  'OK': 5,  'CT': 5,  'UT': 4, 
      'IA': 4,  'NV': 4,  'AR': 4,  'MS': 4,  'KS': 4, 
      'NM': 3,  'NE': 3,  'ID': 2,  'WV': 2,  'HI': 2, 
      'NH': 2,  'ME': 2,  'RI': 2,  'MT': 2,  'DE': 1, 
      'SD': 1,  'ND': 1,  'AK': 1,  'VT': 1,  'WY': 1 
}

congressional_districts_2010 = {
    'WA': 10, 'DE': 1, 'WI': 8, 'WV': 3, 'HI': 2,
    'FL': 27, 'WY': 1, 'NJ': 12, 'NM': 3, 'TX': 36,
    'LA': 6, 'NC': 13, 'ND': 1, 'NE': 3, 'TN': 9, 'NY': 27,
    'PA': 18, 'AK': 1, 'NV': 4, 'NH': 2, 'VA': 11, 'CO': 7,
    'CA': 53, 'AL': 7, 'AR': 4, 'VT': 1, 'IL': 18, 'GA': 14,
    'IN': 9, 'IA': 4, 'MA': 9, 'AZ': 9, 'ID': 2, 'CT': 5,
    'ME': 2, 'MD': 8, 'OK': 5, 'OH': 16, 'UT': 4, 'MO': 8,
    'MN': 8, 'MI': 14, 'RI': 2, 'KS': 4, 'MT': 1, 'MS': 4,
    'SC': 7, 'KY': 6, 'OR': 5, 'SD': 1
}

# Function to read graph from json file
#
def read_graph_from_json(state,json_file, update_population=True, rescale_distance=True, year=2020):
    
    if year==2020:
        k = congressional_districts_2020[state]
    elif year==2010:
        k = congressional_districts_2010[state]
        update_population=False
    else: 
        assert False, f"year = {year} is not supported."
    
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
    G._k = k  
    return G

