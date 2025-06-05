# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 12:12:17 2024

@author: sezazip
"""


import math
import networkx as nx
import os
import sys
from pathlib import Path

# Includes functions for calculating metrics like compactness, deviation, etc
def label(plan):
    return {i : j for j in range(len(plan)) for i in plan[j]}

def cut_edges(G, plan):
    labeling = label(plan)
    return sum(1 for i,j in G.edges if labeling[i] != labeling[j])

def observed_deviation_persons(G, plan, ideal_population, year=None):
    if year==2000:
        return max(abs(sum(G.nodes[i]['TOTPOP2000'] for i in district) - ideal_population) for district in plan)
    else:   
        return max(abs(sum(G.nodes[i]['TOTPOP'] for i in district) - ideal_population) for district in plan)

def Polsby_Popper(G, district, labeling):
    area = sum(G.nodes[i]['area'] for i in district)
    perim = sum(G.edges[u,v]['shared_perim'] for u in district for v in G.neighbors(u) if labeling[u]!=labeling[v])
    perim += sum(G.nodes[i]['boundary_perim'] for i in district if G.nodes[i]['boundary_node']) 
    return 4 * math.pi * area / ( perim * perim )

def bottleneck_Polsby_Popper(G, plan, verbose=False):
    labeling = label(plan)
    if verbose:
        print("\nDistrict Polsby-Popper scores:")
        for p in range(len(plan)):
            print(p, round(Polsby_Popper(G, plan[p], labeling),4))
    return min(Polsby_Popper(G, district, labeling) for district in plan) 

def average_Polsby_Popper(G, plan, verbose=False):
    labeling = label(plan)
    if verbose:
        print("\nDistrict Polsby-Popper scores:")
        for p in range(len(plan)):
            print(p, round(Polsby_Popper(G, plan[p], labeling),4))
    return sum(Polsby_Popper(G, district, labeling) for district in plan) / len(plan) 

def average_inverse_Polsby_Popper(G, plan, verbose=False):
    labeling = label(plan)
    if verbose:
        print("\nDistrict Polsby-Popper scores:")
        for p in range(len(plan)):
            print(p, round(1/Polsby_Popper(G, plan[p], labeling),4))
    return sum(1/Polsby_Popper(G, district, labeling) for district in plan) / len(plan)

def perimeter(G, districts, verbose=False):
    total_perimeter = 0
    for district in districts:
        district_bool = {i : False for i in G.nodes}
        for i in district:
            district_bool[i] = True        
        internal_perim = sum(G.edges[u,v]['shared_perim'] for u in district for v in G.neighbors(u) if not district_bool[v])
        external_perim = sum(G.nodes[i]['boundary_perim'] for i in district if G.nodes[i]['boundary_node']) 
        total_perimeter += internal_perim + external_perim
    return total_perimeter
    
def stay_in_old_districts(G, districts, old_districts):
    return sum(
        G.nodes[i]['TOTPOP']
        for j in range(min(len(old_districts), len(districts)))  # Ensure both lists are aligned
        for i in set(old_districts[j]).intersection(set(districts[j]))  # Find common nodes
    )

# Returns the scores for
#   1. compactness, i.e., worst Polsby-Popper score and 
#   2. deviation, i.e., worst deviation (expressed in percentage terms)
def scores(G, plan, ideal_population, objective):
    if objective == 'bottleneck_Polsby_Popper':
        comp = bottleneck_Polsby_Popper(G, plan)
    elif objective == 'average_Polsby_Popper':
        comp = average_Polsby_Popper(G, plan)
    elif objective == 'inverse_Polsby_Popper':
        comp = average_inverse_Polsby_Popper(G, plan)
    elif objective == 'perimeter':
        comp = perimeter(G, plan)
    elif objective == 'cut_edges':
        comp = cut_edges(G, plan)
    dev_abs = observed_deviation_persons(G, plan, ideal_population)
    return [100 * dev_abs / ideal_population, comp]

def check_county_clustering(G, clustering, sizes, year=None):
    all_assigned_nodes = set()
    
    for idx, district in enumerate(clustering):
        # Check if the district is connected
        if not nx.is_connected(G.subgraph(district)):
            print(f"District {idx} is not connected.")
            return False

        # Check if any node is assigned to more than one district
        for node in district:
            if node in all_assigned_nodes:
                print(f"Node {node} in district {idx} is also assigned to a previous district.")
                return False
        
        all_assigned_nodes.update(district)
        
        # Check if the district's population is within the allowed bounds
        if year == 2000:
            population = sum(G.nodes[i]['TOTPOP2000'] for i in district)            
        else:
            population = sum(G.nodes[i]['TOTPOP'] for i in district)
            if not (sizes[idx] * G._L <= population <= sizes[idx] * G._U):
                print(f"Population of district {idx} is {population}, but should be between {sizes[idx] * G._L} and {sizes[idx] * G._U}.")
                return False
            
    # Check if all nodes in G are assigned to some district
    if all_assigned_nodes != set(G.nodes):
        missing_nodes = set(G.nodes) - all_assigned_nodes
        print(f"Not all vertices are assigned to districts. Missing nodes: {missing_nodes}.")
        return False
    return True

def check_plan(G, plan, year=2020):
    sizes = [1 for j in range(len(plan))]
    return check_county_clustering(G, plan, sizes=[1 for j in range(len(plan))], year=year)

def compute_obj(G, districts, obj_type, old_districts=None, year=None):
    DG = nx.DiGraph(G)
    
    assert obj_type in {'inverse_Polsby_Popper', 'cut_edges', 'perimeter', 
                        'average_Polsby_Popper', 'bottleneck_Polsby_Popper','stay_in_old_districts'}, "Invalid objective type"

    if districts is None:
        if obj_type in {'cut_edges', 'inverse_Polsby_Popper', 'perimeter'}:
            return math.inf  
        elif obj_type in {'bottleneck_Polsby_Popper', 'average_Polsby_Popper','stay_in_old_districts'}:
            return -math.inf  
    else:
        if check_plan(G, districts, year):
            if obj_type == 'bottleneck_Polsby_Popper':
                return bottleneck_Polsby_Popper(DG, districts)
            elif obj_type == 'cut_edges':
                return cut_edges(G, districts)
            elif obj_type == 'inverse_Polsby_Popper':
                return average_inverse_Polsby_Popper(G, districts)
            elif obj_type == 'average_Polsby_Popper':
                return average_Polsby_Popper(G, districts)
            elif obj_type == 'perimeter':
                return perimeter(G, districts)
            elif obj_type == 'stay_in_old_districts':
                return stay_in_old_districts(G, districts, old_districts)
        else:
            if obj_type in {'cut_edges', 'inverse_Polsby_Popper', 'perimeter'}:
                return math.inf  
            elif obj_type in {'bottleneck_Polsby_Popper', 'average_Polsby_Popper','stay_in_old_districts'}:
                return -math.inf
    
def get_best_plan(G, obj_type, plans):
    assert obj_type in {'inverse_Polsby_Popper', 'cut_edges', 'perimeter', 
                        'average_Polsby_Popper', 'bottleneck_Polsby_Popper','stay_in_old_districts'}, "Invalid objective type"
     
    # Set initial best objective value and best plan
    if obj_type in {'cut_edges', 'inverse_Polsby_Popper', 'perimeter'}:
        best_obj_value = math.inf
    else:
        best_obj_value = -math.inf
    
    best_plan = None
    for plan in plans:
        obj_value = compute_obj(G, plan, obj_type)

        if obj_type in {'cut_edges', 'inverse_Polsby_Popper', 'perimeter'}:
            if obj_value < best_obj_value:
                best_obj_value = obj_value
                best_plan = plan
        else:
            if obj_value > best_obj_value:
                best_obj_value = obj_value
                best_plan = plan
    
    return best_plan

def district_objective(G, district, obj_type):
    district_bool = {i : False for i in G.nodes}
    for i in district:
        district_bool[i] = True
    if obj_type == 'cut_edges':
        return sum(1 for u in district for v in G.neighbors(u) if not district_bool[v])
    elif obj_type == 'perimeter':
        internal_perim = sum(G.edges[u,v]['shared_perim'] for u in district for v in G.neighbors(u) if not district_bool[v])
        external_perim = sum(G.nodes[i]['boundary_perim'] for i in district if G.nodes[i]['boundary_node']) 
        return internal_perim + external_perim
    elif obj_type == 'shared_perim':
        return sum(G.edges[u,v]['shared_perim'] for u in district for v in G.neighbors(u) if not district_bool[v])
    elif obj_type =='inverse_polsby_popper':
        internal_perim = sum(G.edges[u,v]['shared_perim'] for u in district for v in G.neighbors(u) if not district_bool[v])
        external_perim = sum(G.nodes[i]['boundary_perim'] for i in district if G.nodes[i]['boundary_node']) 
        P = internal_perim + external_perim
        A = sum(G.nodes[i]['area'] for i in district)
        return P * P / (4 * math.pi * A)
    else:
        assert False
        
def save_plans(plans, state, year=2020):
    filename = f"{state}_plans_{year}.py"
    with open(filename, "w") as f:
        f.write(f"plans = {repr(plans)}\n")
