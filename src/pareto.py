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



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class ParetoFrontier:
    def __init__(self, senses, obj_names=None, state='IA', level='county'):
        self.state = state
        self.level = level
        self.upper_bounds = []
        self.lower_bounds = []
        self.plans = []
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
        self._filter_and_sort_pareto(self.upper_bounds, self.plans,  self.lower_bounds)
       
        
    # TODO: make this simpler/faster using sorting ideas 
    # Currently it takes time n*n*log(n), but I think it can be n*log(n).
    def _filter_and_sort_pareto(self, upper_bounds, plans, lower_bounds):
        # remove dominated
        pareto_upper_bounds = list()
        pareto_lower_bounds = list()
        pareto_plans = list()
        for i, upper_bound in enumerate(upper_bounds):
            dominated = any(self._dominates(other_upper_bound, upper_bound) for j, other_upper_bound in enumerate(upper_bounds) if i != j)
            duplicated = any( self._is_same_plan(plans[i], plan) for plan in pareto_plans )
            if not dominated and not duplicated:
                pareto_upper_bounds.append(upper_bound)
                pareto_lower_bounds.append(lower_bounds[i])
                pareto_plans.append(plans[i])

        # sort by smallest objective1 to largest objective1
        plan_tuples = [ (pareto_upper_bounds[i][0], pareto_upper_bounds[i][1],pareto_lower_bounds[i][0], pareto_lower_bounds[i][1], pareto_plans[i]) for i in range(len(pareto_plans)) ]
        sorted_tuples = sorted(plan_tuples, key=lambda tup : tup[0])
        self.pareto_upper_bounds = [ [tup[0], tup[1]] for tup in sorted_tuples ]
        self.lower_bounds = [ [tup[2], tup[3]] for tup in sorted_tuples ]
        self.plans = [ tup[4] for tup in sorted_tuples ]
        
    def tighten_lower_bounds(self):
        for i in range(len(self.lower_bounds) - 1, -1, -1):
            if self.lower_bounds[i][1] != self.upper_bounds[i][1]:
                if self.senses[1] == 'min':
                    self.lower_bounds[i][1] = max(self.lower_bounds[i][1], self.lower_bounds[i + 1][1])
                else:
                    self.lower_bounds[i][1] = min(self.lower_bounds[i][1], self.lower_bounds[i + 1][1])
        
    def _dominates(self, objval1, objval2):
        for i, (o1, o2) in enumerate(zip(objval1, objval2)):
            if self.senses[i] == 'min' and o1 > o2:
                return False
            if self.senses[i] == 'max' and o1 < o2:
                return False
        # now, objval1 is at least as good as objval2 in all objectives. 
        # so, domination amounts to having objval1 != objval2
        return objval1 != objval2

    def _is_same_plan(self, plan1, plan2):
        plan1_set = { frozenset(district) for district in plan1 }
        plan2_set = { frozenset(district) for district in plan2 }
        return plan1_set == plan2_set
    
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
    
    def plot(self, o1lim=None, o2lim=None, infeasible_region=None, extra_points=None, extra_colors=None):
        
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
        if infeasible_region is not None:
            if o1lim is None:
                o1lim = [0, max_deviation + 100]
            if o2lim is not None:
                o2lim = [min_objective - 1, max_objective + 1]
            ax.fill_betweenx(
                y=[o2lim[0], o2lim[1]],
                x1=infeasible_region[0],
                x2=infeasible_region[1],
                color='none',
                alpha=0.3,
                hatch='X',
                edgecolor='red',
                linewidth=0.5,
            )
            
        # Plot any additional points (e.g., enacted map or other specific plans) with specific colors
        if extra_points:
            for i, ep in enumerate(extra_points):
                deviation, objective_value, label = ep
                color = extra_colors[i] if extra_colors else 'g'  # Default to red if no colors are provided
                ax.plot(
                    deviation,
                    objective_value,
                    'o',
                    color=color,
                    markersize=10,
                    label=label,
                )
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
                'b',
            )
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
        plt.title('Restricted Value Function (estimated)')

        if o1lim is not None:
            ax.set_xlim(*o1lim)
        if o2lim is not None:
            ax.set_ylim(*o2lim)

        plt.tight_layout()
        plt.show()
        
    def plot_with_gap_box(self, o1lim=None, o2lim=None, infeasible_region=None, extra_points=None, extra_colors=None):
        
        if len(self.upper_bounds) == 0:
            print("No points in the Pareto frontier.")
            return
    
        fig, ax = plt.subplots(figsize=(9, 5))
    
        ax.grid(True, linestyle='--', alpha=0.7, color='lightgray')
        
        max_deviation, min_deviation, max_objective, min_objective = self.calculate_limits()
        points = np.array(self.upper_bounds)
    
        # Plot infeasible region if provided
        if infeasible_region is not None:
            if o1lim is None:
                o1lim = [min_deviation - 0.5, max_deviation + 0.5]
            if o2lim is None:
                o2lim = [min_objective - 1, max_objective + 1]
            ax.fill_betweenx(
                y=[o2lim[0], o2lim[1]],
                x1=infeasible_region[0],
                x2=infeasible_region[1],
                color='none',
                alpha=0.3,
                hatch='X',
                edgecolor='red',
                linewidth=0.5,
            )
            
        # Plot any additional points
        if extra_points:
            for i, ep in enumerate(extra_points):
                deviation, objective_value, label = ep
                color = extra_colors[i] if extra_colors and i < len(extra_colors) else 'g'
                ax.plot(
                    deviation,
                    objective_value,
                    'o',
                    color=color,
                    markersize=10,
                    label=label,
                )
            ax.legend(loc='best')
        
        
        # sorted_indices = np.argsort([p[0] for p in self.upper_bounds])
        # sorted_points = [self.upper_bounds[i] for i in sorted_indices]
        # sorted_lower_bounds = [self.lower_bounds[i] for i in sorted_indices]
        
        # Extension amount for the last point's horizontal line
        extension = 0.2
        
        # Get the rightmost x-coordinate to determine the end of the last horizontal line
        if o1lim is not None:
            rightmost_x = o1lim[1]
        else:
            rightmost_x = self.upper_bounds[-1][0] + extension
        for i in range(len(self.upper_bounds)):
            current_dev = self.upper_bounds[i][0]
            current_upper = self.upper_bounds[i][1]
        
            if self.lower_bounds[i] is None:
                current_lower = current_upper
            else:
                current_lower = self.lower_bounds[i][1]
        
            if i < len(self.upper_bounds) - 1:
                next_dev = self.upper_bounds[i + 1][0]
            else:
                next_dev = current_dev + 0.2
        
            rect_width = next_dev - current_dev
            rect_height = current_upper - current_lower
        
            if rect_height > 0: 
                
                # Plot gap box only
                rect = patches.Rectangle(
                    (current_dev, current_lower),
                    rect_width,
                    rect_height,
                    color='r',
                    alpha=0.3,
                    linewidth=0
                )
                ax.add_patch(rect)
        
            else:
                # No uncertainty(draw horizontal segment with open and closed circles)
                ax.plot(current_dev, current_upper, 'bo', markersize=6)  
        
                if i < len(self.upper_bounds) - 1:
                    ax.plot([current_dev, next_dev], [current_upper, current_upper], 'b-', linewidth=1.5)
                    ax.plot(next_dev, current_upper, 'bo', markerfacecolor='white', markersize=6)  # Open circle
                else:
                    ax.plot([current_dev, rightmost_x], [current_upper, current_upper], 'b-', linewidth=1.5)

    
        plt.xlabel(self.obj_names[0])
        plt.ylabel(self.obj_names[1])
        plt.title('Restricted Value Function (estimated)')
    
        if o1lim is not None:
            ax.set_xlim(*o1lim)
        if o2lim is not None:
            ax.set_ylim(*o2lim)
    
        plt.tight_layout()
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
        all_points = []
    
        for frontier, label, color, marker in zip(frontiers, labels, colors, markers):
            for point in frontier.upper_bounds:
                key = (point[0], point[1], label)
                if key not in seen_points:
                    seen_points.add(key)
                    all_points.append((point[0], point[1], label, color, marker))
    
        non_dominated = []
        dominated = []
    
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
                legend_entries[method] = ax.scatter(x, y, marker=marker, color=color, facecolors=facecolors, s=70, label=f"{method} (Non-Dominated)")
            else:
                ax.scatter(x, y, marker=marker, color=color, facecolors=facecolors, s=70)
    
        for x, y, method, color, marker, _, _ in dominated:
            facecolors = 'none' if marker == 'o' else color
            label_key = f"{method} (Dominated)"
            if label_key not in legend_entries:
                legend_entries[label_key] = ax.scatter(x, y, marker=marker, color=color, facecolors=facecolors, alpha=0.4, s=70, label=label_key)
            else:
                ax.scatter(x, y, marker=marker, color=color, facecolors=facecolors, alpha=0.4, s=70)
    
        ax.legend(loc='best')
        plt.xlabel(self.obj_names[0])
        plt.ylabel(self.obj_names[1])
        plt.title('Pareto Points with Dominated Points Highlighted')
        plt.tight_layout()
        plt.show()


    def draw_plans(self):
        
        assert self.state is not None, 'draw_plans requires to specify a state at init'
        assert self.level is not None, 'draw_plans requires to specify a level (e.g., county, tract, vtd) at init'

        try:
            self.graph
        except:
            from util import read_graph_from_json
            filepath1 = 'C:\\districting-data-2020-reprojection\\'
            filename1 = self.state + '_' + self.level + '.json'
            self.graph = read_graph_from_json(filepath1+filename1)
        
        from draw import draw_plan
        filepath2 = 'C:\\districting-data-2020\\'
        filename2 = self.state + '_' + self.level + '.shp'
        ideal_population = sum( self.graph.nodes[i]['TOTPOP'] for i in self.graph.nodes ) / len( self.plans[0] )
        for (plan,upper_bound) in zip(self.plans,self.upper_bounds):
            title = f"{upper_bound[0]}-person deviation ({round(100*upper_bound[0]/ideal_population,4)}%), {round(upper_bound[1],4)} {self.obj_names[1]}"
            draw_plan( filepath2, filename2, self.graph, plan, title=title)
    
    
   
