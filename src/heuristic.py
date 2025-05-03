# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 12:26:11 2024

@author: sezazip
"""
import math
from metrics import*
from mip import labeling_model, gurobi_status
       
#############################     carve_heuristic    ####################################
#############################     carve_heuristic    ####################################
def carve_heuristic(G, deviation_persons, tries=100, obj_type='cut_edges', verbose=False, contiguity='shir', warm_starts=None, time_limit=7200, multiplier=1, max_B=False, symmetry_breaking='rsum'):
    
    assert obj_type in {'inverse_Polsby_Popper', 'cut_edges', 'perimeter', 
                        'average_Polsby_Popper', 'bottleneck_Polsby_Popper'}, "Invalid objective type."
     
    assert contiguity in {'lcut', 'scf', 'shir'}, "Invalid contiguity type."
    
    assert symmetry_breaking in {None, 'orbitope', 'rsum'}, "Invalid symmetry_breaking type."
    
    k = G._k
           
    plans = list()   
    for t in range(tries):
        print(f"\n{'*' * 40}\nTry #{t}\n{'*' * 40}\n")

        # districts for this try (list of lists)
        districts = list()
        
        # "remaining" nodes which have not been assigned
        R = {i for i in G.nodes}
        
        for i in range(k-2):  # Carve first k-2 districts
            if verbose:
                print(f"\n{'*' * 40}\nDistrict #{i}\n{'*' * 40}\n")

            RG = G.subgraph(R)    # remaining graph
            RG._k = k-i
            RG._ideal_population = G._ideal_population

            (mip_districts, obj_val, s) = labeling_model(RG, deviation_persons, obj_type=obj_type, contiguity=contiguity,   sizes=[1,k-i-1], verbose=verbose, multiplier=multiplier, max_B=max_B, symmetry_breaking=symmetry_breaking)
                       
            if  mip_districts is None:
                gurobi_status(s)
                print(f"try # {t} -> Not found mip_districts, Gurobi status number is", s)
                break
            districts.append(mip_districts[0])
            R = set(mip_districts[1])

        # If the last carve failed, then start a new try
        if mip_districts is None:
            continue
        
        # For the last two districts, use the MIP model
        RG = G.subgraph(R)
        RG._ideal_population = G._ideal_population
        RG._k = 2
        (last_two_districts, s) = labeling_model(RG, deviation_persons, obj_type=obj_type, contiguity=contiguity, verbose=verbose, max_B=max_B, symmetry_breaking=symmetry_breaking)

        if last_two_districts:
            for district in last_two_districts:
                districts.append(district)
        else:
            gurobi_status(s)
            print("last two districts was not found! Gurobi status number is", s)
            continue
       
        objective = compute_obj(G, districts, obj_type)
        gurobi_status(s)
        print(f"Objective value for this try is: {objective}")
        
        
        plans.append(districts)
        
    return plans

#############################     halving_heuristic    ####################################
#############################     halving_heuristic    ####################################
def halving_heuristic (G, deviation_persons, tries=100, obj_type='cut_edges', verbose=False, first_cluster_contiguity='shir',second_cluster_contiguity='shir', warm_starts=None, time_limit=7200, multiplier=1, max_B=False, symmetry_breaking='rsum'):
   
    assert obj_type in {'inverse_Polsby_Popper', 'cut_edges', 'perimeter', 
                        'average_Polsby_Popper', 'bottleneck_Polsby_Popper'}, "Invalid objective type."
     
    assert first_cluster_contiguity in {'lcut', 'scf', 'shir'}, "Invalid contiguity type."
    assert second_cluster_contiguity in {'lcut', 'scf', 'shir'}, "Invalid contiguity type."
    
    assert symmetry_breaking in {None, 'orbitope', 'rsum'}, "Invalid symmetry_breaking type."
    
    k = G._k
            
    plans = list()    
    for t in range(tries):
        print(f"\n{'*' * 40}\nTry #{t}\n{'*' * 40}\n")
            
        clusters =[list( G.nodes )]
        sizes = [k]
        districts = list()
        num_multidistricts = 0     
         
        while len(sizes) > 0:
            
             #pick a cluster
             cluster = clusters.pop()
             size = sizes.pop()
             
             size1 = math.floor( size / 2 )
             size2 = math.ceil( size / 2 )
             
             H = G.subgraph(cluster)
             H._ideal_population = G._ideal_population
             H._k = k
             
             # For the last two districts, use the exact MIP model without randomization
             if size == 2 and size1 == 1 and size2 == 1:
                print("Using exact MIP model for the last two districts.")
                H._k = 2
                (mip_clusters, obj_val, s) = labeling_model(H, deviation_persons=deviation_persons, obj_type=obj_type, contiguity = first_cluster_contiguity, verbose=verbose, max_B=max_B, symmetry_breaking=symmetry_breaking)
             else:
                (mip_clusters , obj_val, s) = labeling_model(H, deviation_persons, obj_type=obj_type, contiguity=second_cluster_contiguity, sizes=[size1 , size2], verbose=verbose, multiplier=multiplier, max_B=max_B, symmetry_breaking=symmetry_breaking)
             
             if mip_clusters == None:
                print("Unable to bipartition. Keeping as multidistrict of size =",size)
                districts.append(cluster)
                num_multidistricts += 1
                continue
               
             if size1 == 1:
                 districts.append( mip_clusters[0])
             else:
                clusters.append(mip_clusters[0])
                sizes.append(size1)

             if size2 == 1:
                districts.append(mip_clusters[1])
             else:
                clusters.append(mip_clusters[1])
                sizes.append(size2) 
         
        print(f"try # {t}  ->  After halving, we have {len(districts)} districts.",end =' ')
        
        if num_multidistricts > 0:
            print(f"This includes {num_multidistricts} multidistricts.")
        
        else:
            objective = compute_obj(G, districts, obj_type)
            gurobi_status(s)
            print(f"Objective value for this try is: {objective}")
            
            plans.append(districts)
            
    return plans


