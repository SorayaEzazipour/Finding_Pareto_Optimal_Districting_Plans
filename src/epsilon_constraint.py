#-*- coding: utf-8 -*-
"""
Created on Mon Sep  9 12:15:07 2024

@author: sezazip
"""

from metrics import *
from mip import *
from math import ceil, floor
from optimization import iterative_refinement

#############################  epsilon_constraint_method    ####################################
#############################  epsilon_constraint_method  ####################################

def epsilon_constraint_method(G, obj_type='bottleneck_Polsby_Popper', contiguity='lcut', cutoff=None,
                              verbose=False, warm_start_mode='refinement', warm_starts=None,
                              starting_deviation=0.01, time_limit=7200, sizes=None, max_B=False, 
                              symmetry_breaking=None, similarity=None, state=None, year=2020):
    
    assert obj_type in { 'inverse_Polsby_Popper', 'cut_edges', 'perimeter',
        'average_Polsby_Popper', 'bottleneck_Polsby_Popper' }, "Invalid objective type."
    assert contiguity in {'lcut', 'scf', 'shir'}, "Invalid contiguity type."
    assert symmetry_breaking in {None, 'orbitope', 'rsum'}, "Invalid symmetry_breaking type."
    assert warm_start_mode in {'none', 'user', 'refinement'}, "Invalid warm_start_mode."

    deviation_persons = starting_deviation * G._ideal_population
    G._L = ceil( G._ideal_population - deviation_persons )
    G._U = floor( G._ideal_population + deviation_persons )
    print(f"Initially, L = {G._L} and U = {G._U} and k = {G._k}.")
    
    epsilon = 1 / (2 * G._k)
    (plans, deviations, obj_bounds) = ([], [], [])

    persistent_warm_starts = warm_starts if warm_start_mode == 'user' else None

    while True:
        print(f"\n{'*' * 40}\nTrying deviation = {deviation_persons}\n{'*' * 40}")
        current_warm_starts = list()

        if warm_start_mode == 'refinement':
            print(f"{'*' * 40}\nGenerating warm starts via iterative_refinement...\n{'*' * 40}")
            G._L = ceil( G._ideal_population - deviation_persons )
            G._U = floor( G._ideal_population + deviation_persons )
            current_warm_starts = iterative_refinement(G, G._L, G._U, G._k, state=state, year=year, enumeration_limit=5, verbose=False )

        elif warm_start_mode == 'user' and persistent_warm_starts:
            print("Using user-provided warm starts.")
            current_warm_starts = persistent_warm_starts
            
        warm_start = None    
        best_obj = None
        best_deviation = None

        for candidate in current_warm_starts:
            deviation = observed_deviation_persons(G, candidate, G._ideal_population)
            is_better = False
            if deviation <= deviation_persons :
                obj_value = compute_obj(G, candidate, obj_type)
                if warm_start is None:
                    is_better = True
                else:
                    if obj_type in {"inverse_Polsby_Popper", "cut_edges", "perimeter"}:
                        if obj_value < best_obj or (obj_value == best_obj and deviation < best_deviation):
                            is_better = True
                    else:
                        if obj_value > best_obj or (obj_value == best_obj and deviation < best_deviation):
                            is_better = True
            if is_better:
                warm_start = candidate
                best_obj = obj_value
                best_deviation = deviation
        
        if warm_start:
            print("Selected warm_start =", warm_start)
            print("Objective value:", best_obj)
            print("Deviation:", best_deviation)
        else:
            print("No valid warm start used.")

        print(f"\n{'*' * 40}\nRunning labeling model!\n{'*' * 40}")
        plan, upper_bound, lower_bound, status = labeling_model(
            G, deviation_persons=deviation_persons, obj_type=obj_type,
            contiguity=contiguity, cutoff=cutoff, verbose=verbose,
            warm_start=warm_start, time_limit=time_limit, sizes=sizes, max_B=max_B,
            symmetry_breaking=symmetry_breaking, similarity=similarity)

        if plan is None:
            print(f"\n{'*' * 40}\nNo feasible solution found! Gurobi status: {status}\n{'*' * 40}")
            break

        if status != 2:
            print(f"\n{'*' * 40}\nNo optimal solution found! Gurobi status: {status}\n{'*' * 40}")
        else:
            print(f"\n{'*' * 40}\nOptimal solution found! Gurobi status: {status}\n{'*' * 40}")

        print("plan =",plan)
        plans.append(plan)
        obj_bounds.append([upper_bound, lower_bound])
        dev = observed_deviation_persons(G, plan, G._ideal_population)
        deviations.append(dev)
        deviation_persons = dev - epsilon

        if deviation_persons < epsilon:
            print("Deviation is too small, exiting now.")
            break

    # Filter nondominated plans
    nondominated_plans = filter_dominated_plans(G, plans, obj_type)
    indices = [ plans.index(plan) for plan in nondominated_plans ]
    nondominated_deviations = [ deviations[i] for i in indices ]
    nondominated_obj_bounds = [ obj_bounds[i] for i in indices ]

    return (nondominated_plans, nondominated_obj_bounds, nondominated_deviations)

#############################     filter_dominated_plans    ####################################
#############################     filter_dominated_plans  ####################################
# def filter_dominated_plans(G, plans, obj_type):
#     nondominated_plans = list()
#     senses = ['max' if obj_type in ['bottleneck_Polsby_Popper', 'average_Polsby_Popper'] else 'min', 'min']
    
#     for plan1 in plans:
#         if not any(dominates(scores(G, plan2, G._ideal_population, obj_type), 
#                             scores(G, plan1, G._ideal_population, obj_type), senses) 
#                     for plan2 in plans):
#             nondominated_plans.append(plan1)
    
#     return nondominated_plans


def filter_dominated_plans(G, plans, obj_type):
    nondominated_plans = []
    senses = ['max' if obj_type in ['bottleneck_Polsby_Popper', 'average_Polsby_Popper'] else 'min', 'min']
    scored_plans = [(scores(G, plan, G._ideal_population, obj_type), plan) for plan in plans]
    
    def sort_key(x):
        deviation = x[0][0]
        compactness = x[0][1]
        if senses[1] == 'max':
            compactness = -compactness
        return (deviation, compactness)

    scored_plans.sort(key=sort_key)

    best_compactness = None
    for (deviation, compactness), plan in scored_plans:
        if best_compactness is None or (compactness > best_compactness if senses[1] == 'max' else compactness < best_compactness):
            best_compactness = compactness
            nondominated_plans.append(plan)

    return nondominated_plans