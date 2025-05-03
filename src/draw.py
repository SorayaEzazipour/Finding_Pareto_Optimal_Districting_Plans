import geopandas as gpd
import matplotlib.pyplot as plt

# function to draw single districts/county_clusters/multi_districts
#
def draw_single_district( filepath, filename, G, district, zoom=False, title=None ):
    
    df = gpd.read_file( filepath + filename )
    node_with_this_geoid = { G.nodes[i]['GEOID20'] : i for i in G.nodes }
    assignment = [ -1 for i in G.nodes ]

    if zoom:
        picked = { i : None for i in G.nodes }
    else:
        picked = { i : False for i in G.nodes }
    
    for i in district:
        picked[i] = True

    for u in range(G.number_of_nodes()):
        geoid = df['GEOID20'][u]
        i = node_with_this_geoid[geoid]
        assignment[u] = picked[i]

    df['assignment'] = assignment
    my_fig = df.plot(column='assignment').get_figure()
    plt.axis('off')
    plt.title(title)
    return 



def draw_plan(filepath, filename, G, plan, title=None, year=2020):
    # Select GEOID based on the year
    GEOID = 'GEOID20' if year == 2020 else 'GEOID10'


    df = gpd.read_file(filepath + filename)
    
    if GEOID not in G.nodes[next(iter(G.nodes))] or GEOID not in df.columns:
        raise ValueError(f"GEOID '{GEOID}' not found in the graph or geojson file.")

    assignment = [-1 for _ in G.nodes]

    labeling = {i: j for j in range(len(plan)) for i in plan[j]}
    node_with_this_geoid = {G.nodes[i][GEOID]: i for i in G.nodes}

    for u in range(len(df)):
        geoid = df[GEOID][u]
        if geoid in node_with_this_geoid:
            i = node_with_this_geoid[geoid]
            assignment[u] = labeling[i]

    df['assignment'] = assignment

    # Plot the map
    my_fig = df.plot(column='assignment', legend=False, figsize=(10, 10)).get_figure()
    plt.axis('off')
    plt.title(title or "Districting Plan")
    plt.show()

    #return my_fig
