# #import numpy as np

# # # Example usage
# # senses = ['min', 'max']  # Minimize the first objective, maximize the second
# # obj_names = ['deviation', 'bottleneck_polsby_popper']
# # pareto = ParetoFrontier(senses, obj_names)

# # plan1 = [[1,2],[3,4]]
# # pareto.add_plan(plan1,[6000,0.44])

# # plan2 = [[1,3],[2,4]]
# # pareto.add_plan(plan2,[1,0.19])

# # plan3 = [[1,4],[2,3]]
# # pareto.add_plan(plan3,[1000,0.42])

# # plan4 = [[1],[2,3,4]]
# # pareto.add_plan(plan4,[2000,0.41]) # <--- dominated
    
# # print("Pareto front plans:", pareto.plans)
# # print("Pareto front objective values:", pareto.objvals)
# # pareto.plot()



import matplotlib.pyplot as plt
import matplotlib.patches as patches
from draw import draw_plan
from metrics import*
import numpy as np
import math

class ParetoFrontier:
    def __init__(self, senses, obj_names=None, state='IA', level='county'):
        self.state = state
        self.level = level
        self.upper_bounds = list()
        self.lower_bounds = list()
        self.plans = list()
        assert len( senses ) == 2, "Must pick two objective senses"
        assert all( sense in {'min', 'max'} for sense in senses ), "Must pick 'min' and 'max' objective senses"
        self.senses = senses
        if obj_names is None:
            obj_names = [ 'Objective 1', 'Objective 2' ]
        assert len(obj_names)==2, "Must pick two objective names"
        self.obj_names = obj_names

    def add_plan(self, plan, upper_bound=None, lower_bound=None):
        assert len(upper_bound) == 2, "Plan must have two objective values"
        if lower_bound is None:
            lower_bound = upper_bound.copy()
        assert len(lower_bound) == 2  
        self.upper_bounds.append(upper_bound)
        self.lower_bounds.append(lower_bound)
        self.plans.append(plan)
        
        sorted_indices = np.argsort([p[0] for p in self.upper_bounds])
        self.upper_bounds = [self.upper_bounds[i] for i in sorted_indices]
        self.lower_bounds = [self.lower_bounds[i] for i in sorted_indices]
        self.plans = [self.plans[i] for i in sorted_indices]
        self.upper_bounds, self.lower_bounds, self.plans = filter_and_sort_pareto(
           self.plans, self.upper_bounds, self.lower_bounds, self.obj_names[1])
       
    def tighten_bounds(self):
        for i in range(len(self.lower_bounds) - 2, -1, -1):
            if self.lower_bounds[i][1] != self.upper_bounds[i][1]:
                if self.senses[1] == 'min':
                    self.lower_bounds[i][1] = max(self.lower_bounds[i][1], self.lower_bounds[i + 1][1])
                else:
                    self.upper_bounds[i][1] = min(self.upper_bounds[i][1], self.upper_bounds[i + 1][1])
    
    def calculate_limits(self):
        if len(self.upper_bounds) == 0:
            print("No points in the Pareto frontier.")
            return None, None, None, None

        # Extract the first and second objectives from self.objvals
        obj1_vals = [val[0] for val in self.upper_bounds]  # Deviation values
        obj2_vals = [val[1] for val in self.upper_bounds]  # Objective values

        # Calculate max and min deviations and objectives
        max_deviation = max(obj1_vals)
        min_deviation = min(obj1_vals)
        max_objective = max(obj2_vals)
        min_objective = min(obj2_vals)

        return max_deviation, min_deviation, max_objective, min_objective
    
    def plot(self, method = 'epsilon_constraint_method', o1lim=None, 
             o2lim=None, no_solution_region=None, extra_symbols=None, symbol_sizes=None, extra_colors=None):
        
        if len(self.upper_bounds) == 0:
            print("No points in the Pareto frontier.")
            return

        fig, ax = plt.subplots(figsize=(9, 5))

        ax.grid(True, linestyle='--', alpha=0.7, color='lightgray')
        ax.set_axisbelow(False)

        max_deviation, min_deviation, max_objective, min_objective = self.calculate_limits()
        self.upper_bounds.sort(key=lambda x: x[0])
        points = np.array(self.upper_bounds)

        # Plot infeasible region if provided
        if no_solution_region is not None:
            if o1lim is None:
                o1lim = [0, max_deviation + 100]
            if o2lim is not None:
                o2lim = [min_objective - 1, max_objective + 1]
            ax.fill_betweenx(
                y=[o2lim[0], o2lim[1]],
                x1=no_solution_region[0],
                x2=no_solution_region[1],
                color='none',
                alpha=0.3,
                hatch='X',
                edgecolor='red',
                linewidth=0.5)
            
        # Plot any additional points (e.g., enacted map or other specific plans) with specific colors
        if extra_points:
            for i, ep in enumerate(extra_points):
                deviation, objective_value, label = ep
                marker = extra_symbols[i]
                size = symbol_sizes[marker]
                color = extra_color[i]
                ax.plot(
                    deviation,
                    objective_value,
                    marker,
                    color=color,
                    markersize=size,
                    label=label)
                
                # Optionally, add dashed lines for better visualization
                ax.plot([deviation, deviation], [ax.get_ylim()[0], objective_value], color=color, linestyle='--', alpha=0.0, linewidth=1)
                ax.plot([0, deviation], [objective_value, objective_value], color=color, linestyle='--', alpha=0.0, linewidth=1)
            ax.legend(loc='best')

        # Plot the Pareto frontier points
        if len(points) > 1:
            for i in range(len(points) - 1):
                ax.plot([points[i][0], points[i + 1][0]], [points[i][1], points[i][1]], 'b')
            ax.plot(
                [points[-1][0], points[-1][0] + 0.05 * (points[-1][0] - points[0][0])],
                [points[-1][1], points[-1][1]],
                'b' )
            open_points = np.array([[points[i + 1][0], points[i][1]] for i in range(len(points) - 1)])
            ax.plot(open_points[:, 0], open_points[:, 1], 'bo', markerfacecolor='white')
        
        # Special case: only one Pareto point
        elif len(points) == 1:  
            max_x = ax.get_xlim()[1] 
            
            # Add a small horizontal line for better visibility
            ax.plot([points[0, 0] , max_x],[points[0, 1], points[0, 1]],'b-', linewidth=1.5)
            
        ax.plot(points[:, 0], points[:, 1], 'bo')

        plt.xlabel(self.obj_names[0])
        plt.ylabel(self.obj_names[1])
        if method == 'epsilon_constraint_method':
            plt.title('Restricted Value Function')
        else:
            plt.title('Restricted Value Function (estimated)')

        if o1lim is not None:
            ax.set_xlim(*o1lim)
        if o2lim is not None:
            ax.set_ylim(*o2lim)

        plt.tight_layout()
        plt.show()
        
    def plot_with_gap_box(self, method = 'epsilon_constraint_method', 
                          o1lim=None, o2lim=None, no_solution_region=None, extra_points=None, 
                          extra_symbols=None, symbol_sizes=None, extra_colors=None):
        
        if len(self.upper_bounds) == 0:
            print("No points in the Pareto frontier.")
            return
    
        fig, ax = plt.subplots(figsize=(9, 5))
    
        ax.grid(True, linestyle='--', alpha=0.7, color='lightgray')
        
        max_deviation, min_deviation, max_objective, min_objective = self.calculate_limits()
        points = np.array(self.upper_bounds)
        
        if o1lim is None:
                o1lim = [min_deviation - 0.05 * max_deviation, max_deviation + 0.05 * max_deviation]
        if o2lim is None:
                o2lim = [min_objective - 1, max_objective + 1]
    
        # Plot infeasible region if provided
        if no_solution_region is not None:
            ax.fill_betweenx(
                y=[o2lim[0], o2lim[1]],
                x1=no_solution_region[0],
                x2=no_solution_region[1],
                color='none',
                alpha=0.3,
                hatch='X',
                edgecolor='red',
                linewidth=0.5)
            
        # Plot any additional points
        if extra_points:
            for i, ep in enumerate(extra_points):
                deviation, objective_value, label = ep
                marker = extra_symbols[i]
                size = symbol_sizes[marker]
                color = extra_colors[i]
                ax.plot(
                    deviation,
                    objective_value,
                    marker,
                    color=color,
                    markersize=size,
                    label=label)
            ax.legend(loc='best')
        
        sorted_indices = np.argsort([p[0] for p in self.upper_bounds])
        sorted_points = [self.upper_bounds[i] for i in sorted_indices]
        sorted_lower_bounds = [self.lower_bounds[i] for i in sorted_indices]
        rightmost_x = o1lim[1]
            
        for i in range(len(sorted_points)):
            current_dev = sorted_points[i][0]
            current_upper = sorted_points[i][1]
            current_lower = current_upper if sorted_lower_bounds[i] is None else sorted_lower_bounds[i][1]       
            next_dev = sorted_points[i + 1][0] if i < len(sorted_points) - 1 else rightmost_x
            
            rect_width = next_dev - current_dev
            rect_height = abs(current_upper - current_lower)
            
            if rect_height > 1e-7:  
                # Plot gap box only
                rect = patches.Rectangle(
                    (current_dev, current_lower),
                    rect_width,
                    rect_height,
                    color='r',
                    alpha=0.3,
                    linewidth=0)
                ax.add_patch(rect)
        
            else:
                # No uncertainty(draw horizontal segment with open and closed circles)
                ax.plot(current_dev, current_upper, 'bo', markersize=6)  
        
                if i < len(sorted_points) - 1:
                    ax.plot([current_dev, next_dev], [current_upper, current_upper], 'b-', linewidth=1.5)
                    ax.plot(next_dev, current_upper, 'bo', markerfacecolor='white', markersize=6)  # Open circle
                else:
                    ax.plot([current_dev, rightmost_x], [current_upper, current_upper], 'b-', linewidth=1.5)

    
        plt.xlabel(self.obj_names[0])
        plt.ylabel(self.obj_names[1])
        if method == 'epsilon_constraint_method':
            plt.title('Restricted Value Function') 
        if method =='heuristic':
            plt.title('Restricted Value Function (estimated)')
         
        if o1lim is not None:
            ax.set_xlim(*o1lim)
        if o2lim is not None:
            ax.set_ylim(*o2lim)
    
        plt.tight_layout()
        plt.show()  
        
    def plot_with_custom_x_ranges(self, method='epsilon_constraint_method', splits=None, 
                                  o1lim=None, o2lim=None, no_solution_region=None, extra_points=None,
                                  extra_symbols=None, symbol_sizes=None, extra_colors=None):
        if len(self.upper_bounds) == 0:
            print("No points in the Pareto frontier.")
            return

        if splits is None:
            self.plot_with_gap_box(method=method, o1lim=o1lim, o2lim=o2lim,
                                   no_solution_region=no_solution_region, extra_points=extra_points,
                                   extra_symbols=extra_symbols, symbol_sizes=symbol_sizes, extra_colors=extra_colors)
            return

        splits = sorted(splits)
        split_bounds = [-float("inf")] + splits + [float("inf")]
        figsize = (5 * (len(splits) + 1), 5)
        num_panels = len(splits) + 1
        fig, axes = plt.subplots(1, num_panels, figsize=figsize, sharey=True)

        fig.suptitle("Restricted Value Function" if method == 'epsilon_constraint_method' else "Restricted Value Function (estimated)")

        if num_panels == 1:
            axes = [axes]

        rightmost_x = max(
            max((p[0] for p in self.upper_bounds), default=0),
            max((pt[0] for pt in extra_points), default=0) if extra_points else 0,
            o1lim[1] if o1lim else 0
        )
        

        last_points = [None] * num_panels
        first_points = [None] * num_panels

        self.upper_bounds.sort(key=lambda x: x[0])
        self.lower_bounds.sort(key=lambda x: x[0] if x is not None else float('inf'))

        for i in range(len(self.upper_bounds)):
            current_dev = self.upper_bounds[i][0]
            current_upper = self.upper_bounds[i][1]
            current_lower = self.lower_bounds[i][1] if self.lower_bounds[i] is not None else current_upper

            for panel_index in range(num_panels):
                lo = split_bounds[panel_index]
                hi = split_bounds[panel_index + 1]
                if lo < current_dev <= hi:
                    break

            next_dev = self.upper_bounds[i + 1][0] if i < len(self.upper_bounds) - 1 else rightmost_x
            
            rect_width = round(next_dev - current_dev, 2)
            rect_height = abs(current_upper - current_lower)
            ax = axes[panel_index]

            if first_points[panel_index] is None:
                first_points[panel_index] = (current_dev, current_lower, rect_height)
            last_points[panel_index] = (current_dev, current_upper, rect_height)

            if rect_height > 1e-7:
                rect = patches.Rectangle((current_dev, current_lower), rect_width, rect_height,
                                         color='r', alpha=0.3, linewidth=0)
                ax.add_patch(rect)
            else:
                ax.plot(current_dev, current_upper, 'bo', markersize=6)
                if i < len(self.upper_bounds) - 1:
                    ax.plot([current_dev, next_dev], [current_upper, current_upper], 'b-', linewidth=1.5)
                    ax.plot(next_dev, current_upper, 'bo', markerfacecolor='white', markersize=6)
                else:
                     ax.plot([current_dev, rightmost_x], [current_upper, current_upper], 'b-', linewidth=1.5)
        
        for i in range(num_panels - 1):
            if last_points[i] and first_points[i + 1]:
                x1, y1, h1 = last_points[i]
                x2, y2, h2 = first_points[i + 1]

                if abs(h1) > 1e-7 and abs(h2) > 1e-7:
                    # Red rectangle bridge
                    height = abs(y1 - y2)
                    lower = min(y1, y2)
                    width = x2 - x1
                    if width > 0 and height > 1e-7:
                        rect = patches.Rectangle((x1, lower), width, height, color='r', alpha=0.3, linewidth=0)
                        axes[i + 1].add_patch(rect)
                   
                else:
                    # Blue horizontal line bridge
                    axes[i + 1].plot([x1, x2], [y1, y1], 'b-', linewidth=1.2)
                    axes[i + 1].plot(x2, y1, 'bo', markerfacecolor='white', markersize=6)

        y_min = min(min(lb[1] for lb in self.lower_bounds if lb), min(ub[1] for ub in self.upper_bounds)) * 0.95
        y_max = max(max(ub[1] for ub in self.upper_bounds), max(lb[1] for lb in self.lower_bounds if lb)) * 1.05

        for i, ax in enumerate(axes):
            if i == 0:
                x_lo = -1
                x_hi = splits[0]
                if no_solution_region is not None:
                    ax.fill_betweenx(
                        y=[o2lim[0], o2lim[1]],
                        x1=no_solution_region[0],
                        x2=no_solution_region[1],
                        color='none',
                        alpha=0.3,
                        hatch='X',
                        edgecolor='red',
                        linewidth=0.5)
            elif i == len(axes) - 1:
                x_lo = splits[-1]
                x_hi = o1lim[1]
            else:
                x_lo = splits[i - 1]
                x_hi = splits[i]

            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(o2lim if o2lim else (y_min, y_max))

            if no_solution_region and i == 0:
                ax.fill_betweenx(
                    y=[o2lim[0], o2lim[1]],
                    x1=no_solution_region[0],
                    x2=no_solution_region[1],
                    color='none',
                    alpha=0.3,
                    hatch='X',
                    edgecolor='red',
                    linewidth=0.5)

            if extra_points:
                for j, ep in enumerate(extra_points):
                    deviation, objective_value, label = ep
                    if x_lo <= deviation <= x_hi:
                        marker = extra_symbols[j]
                        size = symbol_sizes[marker]
                        color = extra_colors[j]
                        ax.plot(deviation, objective_value, marker, color=color, markersize=size)

            ax.set_xlabel(self.obj_names[0])
            ax.grid(True, linestyle='--', alpha=0.7, color='lightgray')

            yticks = ax.get_yticks()
            if len(yticks) >= 2:
                y_min_tick, y_max_tick = ax.get_ylim()
                y_range = (y_max_tick - y_min_tick) / len(yticks)
                new_last_tick = y_max_tick + y_range
                yticks = np.append(yticks[:-1], new_last_tick)
                labels = [f"{y:.2f}" if abs(y) < 1 else str(int(y)) for y in yticks]
                ax.set_yticks(yticks)
                ax.set_yticklabels(labels)

        axes[0].set_ylabel(self.obj_names[1])
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.1)
        plt.show()



    def plot_with_dominance(self, frontiers, labels, colors, markers):
        if len(self.upper_bounds) == 0:
            print("No points in the Pareto frontier.")
            return
    
        import matplotlib.pyplot as plt
    
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.grid(True, linestyle='--', alpha=0.7, color='lightgray')
        ax.set_axisbelow(False)
    
        seen_points = set()
        all_points = list()
    
        for frontier, label, color, marker in zip(frontiers, labels, colors, markers):
            for point in frontier.upper_bounds:
                key = (point[0], point[1], label)
                if key not in seen_points:
                    seen_points.add(key)
                    all_points.append((point[0], point[1], label, color, marker))
    
        non_dominated = list()
        dominated = list()
    
        for i, (x1, y1, method1, color1, marker1) in enumerate(all_points):
            dominated_flag = False
            dominating_point = None
            dominating_method = None
    
            for j, (x2, y2, method2, _, _) in enumerate(all_points):
                if i != j and self._dominates((x2, y2), (x1, y1)):
                    dominated_flag = True
                    dominating_point = (x2, y2)
                    dominating_method = method2
                    break
    
            if dominated_flag:
                dominated.append((x1, y1, method1, color1, marker1, dominating_point, dominating_method))
            else:
                non_dominated.append((x1, y1, method1, color1, marker1))
    
        print("\nDominated Points and Their Dominators:")
        printed_coords = set()
        for x, y, method, _, _, dom_point, dom_method in dominated:
            if (x, y) not in printed_coords:
                printed_coords.add((x, y))
                print(f"Point ({x}, {y}) from {method} is dominated by Point {dom_point} from {dom_method}")
    
        legend_entries = {}
    
        for x, y, method, color, marker in non_dominated:
            facecolors = 'none' if marker == 'o' else color
            if method not in legend_entries:
                legend_entries[method] = ax.scatter(x, y, marker=marker, color=color, 
                                                    facecolors=facecolors, s=70, label=f"{method} (Non-Dominated)")
            else:
                ax.scatter(x, y, marker=marker, color=color, facecolors=facecolors, s=70)
    
        for x, y, method, color, marker, _, _ in dominated:
            facecolors = 'none' if marker == 'o' else color
            label_key = f"{method} (Dominated)"
            if label_key not in legend_entries:
                legend_entries[label_key] = ax.scatter(x, y, marker=marker, color=color, 
                                                       facecolors=facecolors, alpha=0.4, s=70, label=label_key)
            else:
                ax.scatter(x, y, marker=marker, color=color, facecolors=facecolors, alpha=0.4, s=70)
    
        ax.legend(loc='best')
        plt.xlabel(self.obj_names[0])
        plt.ylabel(self.obj_names[1])
        plt.title('Pareto Points with Dominated Points Highlighted')
        plt.tight_layout()
        plt.show()
    
    def draw_plans(self, G, filepath, filename, year=2020, similarity=None):
        ideal_population = sum(G.nodes[i]['TOTPOP'] for i in G.nodes) / len(self.plans[0])
        for (plan, upper_bound, lower_bound) in zip(self.plans, self.upper_bounds, self.lower_bounds):
            
            if self.senses[1] == 'min':
                obj_val = upper_bound[1]
            else:
                obj_val = lower_bound[1]
            
            title = f"{round(upper_bound[0],2)}-person deviation ({round(100 * upper_bound[0] / ideal_population, 4)}%), {int(obj_val) if self.obj_names[1] == 'cut_edges' else round(obj_val, 4)} {self.obj_names[1]}"
            
            if similarity:
                similarity_score = stay_in_old_districts(G, plan, similarity[0])
                title += f", similarity score {similarity_score}"
                
            draw_plan(filepath=filepath, filename=filename, G=G, plan=plan, title=title, year=year)
         
        
def filter_and_sort_pareto(plans, upper_bounds=None, lower_bounds=None, obj_type='cut_edges'):
    if upper_bounds is None and lower_bounds is None:
        raise ValueError("At least one of 'upper_bounds' or 'lower_bounds' must be provided.") 
        
    # Fill in dummy values if only one set of bounds is provided
    if upper_bounds is None:
        upper_bounds = lower_bounds.copy()
    if lower_bounds is None:
        lower_bounds = upper_bounds.copy()
        
    pareto_upper_bounds = list()
    pareto_lower_bounds = list()
    pareto_plans = list()
    compactness_sense = 'max' if obj_type in ['bottleneck_Polsby_Popper', 'average_Polsby_Popper'] else 'min'
    scored_plans = list()
    
    # Sort plans by deviation and then compactness (descending if 'max')
    if compactness_sense == 'max': 
        scored_plans = [((lower_bounds[i][0], lower_bounds[i][1]), plans[i], upper_bounds[i]) for i in range(len(plans))]
    else:
        scored_plans = [((upper_bounds[i][0], upper_bounds[i][1]), plans[i], lower_bounds[i]) for i in range(len(plans))]
    
    scored_plans.sort(key=lambda x: (x[0][0], -x[0][1] if compactness_sense == 'max' else x[0][1])) 
   
    best_compactness = None
    
    for (deviation, compactness), plan, b in scored_plans:
        if best_compactness is None or (
            (compactness > best_compactness) if compactness_sense == 'max' else (compactness < best_compactness)):
            best_compactness = compactness
            if compactness_sense == 'min':
                pareto_upper_bounds.append([deviation, compactness])
                pareto_lower_bounds.append(b)
            else:
                pareto_upper_bounds.append(b)
                pareto_lower_bounds.append([deviation, compactness])
            pareto_plans.append(plan)

    sorted_tuples = sorted(zip(pareto_upper_bounds, pareto_lower_bounds, pareto_plans), key=lambda x: x[0][0])
    upper_bounds = [t[0] for t in sorted_tuples]
    lower_bounds = [t[1] for t in sorted_tuples]
    plans = [t[2] for t in sorted_tuples] 
    return (upper_bounds, lower_bounds, plans) 

def plot_pareto_frontiers(G, method='epsilon_constraint_method', plans=None, obj_types='cut_edges', ideal_population=None, state=None, filepath=None, filename2=None, no_solution_region=None, year=None, result=None, special_plans_scores=None, similarity=None):
    max_deviation = 0.01 * ideal_population
    # Determine x-axis limits
    o1lim = [-0.025*max_deviation, max_deviation]
    
    if method=='heuristic':
        G._L = 0
        G._U = G._k * ideal_population

        pareto = dict()
        pareto_plans = dict()
        for obj_type in obj_types:
            print("***************************************")
            print("obj_type =", obj_type)
            print("***************************************")

            senses = ['min', 'max' if obj_type in ['average_Polsby_Popper', 'bottleneck_Polsby_Popper'] else 'min']
            obj_names = ['deviation_persons', obj_type]

            # Initialize Pareto frontier
            pareto[obj_type] = ParetoFrontier(senses, obj_names, state=state, level='county')

            # Add plans to frontier
            for plan in plans:
                dev = observed_deviation_persons(G, plan, ideal_population)
                obj = compute_obj(G, plan, obj_type)
                pareto[obj_type].add_plan(plan, upper_bound=[dev, obj])

            print("Pareto front objective values:", pareto[obj_type].upper_bounds)

            # Determine y-axis limits
            upper_bounds = pareto[obj_type].upper_bounds
            max_obj = max(ub[1] for ub in upper_bounds)
            min_obj = min(ub[1] for ub in upper_bounds)
            if obj_type in {"inverse_Polsby_Popper", "cut_edges", "perimeter"}:
                o2lim = [min_obj * 0.9, max_obj * 1.1]
            else:
                o2lim=[max(min_obj-1,0), max_obj+0.1]

            pareto[obj_type].tighten_bounds()
            pareto[obj_type].plot_with_custom_x_ranges(method=method, splits=None, o1lim=o1lim, o2lim=o2lim,
                                                                                     no_solution_region=no_solution_region)
            pareto[obj_type].draw_plans(G, filepath, filename2, year=year)
            pareto_plans[obj_type]= pareto[obj_type].plans
        return pareto_plans
        
    if method=='epsilon_constraint_method':
        senses = ['min', 'max' if obj_types in ['average_Polsby_Popper','bottleneck_Polsby_Popper'] else 'min']
        obj_names = ['deviation_persons', obj_types]
        
        print(f"\n{'#' * 100}\nPareto Frontier for state {state},  objective {obj_names[1]}\n{'#' * 100}\n")
        pareto = ParetoFrontier(senses, obj_names, state=state, level='county')

        if not result:
            print("No plan found!")
        else:
            for plan, obj_bound, dev in result:
                upper_bound = [dev,  1/obj_bound[0] if  obj_names[1] == 'bottleneck_Polsby_Popper' else obj_bound[0]]
                lower_bound = [dev,  1/obj_bound[1] if  obj_names[1] == 'bottleneck_Polsby_Popper' else obj_bound[1]]
        
                pareto.add_plan(plan, upper_bound, lower_bound)
            
            print("Pareto front plans:", pareto.plans)
            print("Pareto front upper bounds:", pareto.upper_bounds)
            print("Pareto front lower bounds:", pareto.lower_bounds)
    
         #extra_points: list of tuples, each containing (deviation, objective_value, label)
        if  year == 2010 and state == 'WV':
            extra_points = list()
            for p in special_plans_scores.keys(): 
                extra_points.append((
                    special_plans_scores[p][0],                # deviation
                    special_plans_scores[p][1][obj_names[1]],  # objective score
                    p ))                                        # map key (label)
                
            #extra_points_symbols: list of symbols corresponding to the points in extra_points
            extra_symbols = ['h', 'v', 'o', 's', '^']  # plus, star, x, square, upward triangle
            symbol_sizes = {'h': 8, 'o': 8, 'v': 8, 's': 8, '^': 8}
            extra_colors = ['darkgreen', 'red', 'dimgray', 'purple', 'orange']
    
            for ep in extra_points:
                print(f"The {ep[2]} has an objective value of {ep[1]} and a deviation of {ep[0]}.")
        else:
            extra_points = None
            extra_colors = None
        
        pareto.tighten_bounds()
        
        # Determine y-axis limits
        upper_bounds = pareto.upper_bounds
        max_obj = max(upper_bound[1] for upper_bound in upper_bounds)
        min_obj = min(upper_bound[1] for upper_bound in upper_bounds)
        if obj_names[1] in {"inverse_Polsby_Popper", "cut_edges", "perimeter"}:
             o2lim = [min_obj * 0.9, max_obj * 1.1]
        else:
             o2lim=[max(min_obj-1,0), max_obj+0.1]

        pareto.plot_with_custom_x_ranges(method=method, splits=None, 
                                 o1lim=o1lim, o2lim=o2lim, no_solution_region = no_solution_region,
                                 extra_points=extra_points, extra_symbols=extra_symbols, 
                                         symbol_sizes=symbol_sizes, extra_colors=extra_colors)  
        pareto.draw_plans(G, filepath, filename2, year=year, similarity=similarity)
        
