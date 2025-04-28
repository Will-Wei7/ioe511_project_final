import matplotlib.pyplot as plt
import numpy as np

# Convergence plot without success/failure criteria
def plot_convergence(detailed_info, use_raw_values=False):
    """
    Plot convergence of function values across iterations.
    
    Args:
        detailed_info: Dictionary with problem/method results
        use_raw_values: If True, always plots raw f_values instead of f_gaps
    """
    for problem_name in detailed_info:
        plt.figure(figsize=(10, 6))
        plt.title(f"Convergence Plot for {problem_name} f(x_k)")
        
        # Track min/max values for proper y-axis scaling
        min_y = float('inf')
        max_y = float('-inf')
        
        # Check if this is a quadratic problem or if user requested raw values
        use_raw = use_raw_values or 'quad' in problem_name.lower()
        
        for method_name, info in detailed_info[problem_name].items():
            # Skip LBFGS method
            if 'lbfgs' in method_name.lower():
                continue
                
            # Plot all methods using f_values if requested or for quadratic problems
            if use_raw and 'f_values' in info and info['f_values']:
                iterations = range(len(info['f_values']))
                f_values_plot = np.array(info['f_values'])
                
                plt.plot(iterations, f_values_plot, label=f"{method_name}", markersize=3)
                
                # Update min/max for y-axis limits
                min_y = min(min_y, np.min(f_values_plot))
                max_y = max(max_y, np.max(f_values_plot))
                
            # Otherwise use f_gaps if available 
            elif not use_raw and 'f_gaps' in info and info['f_gaps']:
                iterations = range(len(info['f_gaps']))
                # Add a small epsilon to avoid log(0) issues
                f_gaps_plot = np.maximum(np.array(info['f_gaps']), 1e-16) 
                
                plt.plot(iterations, f_gaps_plot, label=method_name, markersize=3)
                
                # Update min/max for y-axis limits
                min_y = min(min_y, np.min(f_gaps_plot))
                max_y = max(max_y, np.max(f_gaps_plot))
                
            # Fallback to f_values if neither of the above conditions are met
            elif 'f_values' in info and info['f_values']:
                iterations = range(len(info['f_values']))
                f_values_plot = np.array(info['f_values'])
                
                plt.plot(iterations, f_values_plot, label=f"{method_name}", markersize=3)
                
                # Update min/max for y-axis limits
                min_y = min(min_y, np.min(f_values_plot))
                max_y = max(max_y, np.max(f_values_plot))

        # Use appropriate y-axis label based on what we're plotting
        plt.ylabel('f(x_k)')

            
        plt.xlabel('Iterations')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend(fontsize='small')
        
        # Set custom x-axis range based on problem name
        if problem_name in ['quad_10_10', 'P5_quartic_1', 'P6_quartic_2', 'Rosenbrock_2']:
            plt.xlim(0, 100)  # Specific range for quad_10_10
        elif problem_name in ['quad_10_1000', 'quad_1000_10', 'quad_1000_1000']:
            plt.xlim(0, 1000)  # Keep [0, 1000] for large quadratic problems
        else:
            plt.xlim(0, 200)  # Default range [0, 200] for all other problems
        
        # Set y-axis limits with proper padding to show full range
        if min_y != float('inf') and max_y != float('-inf'):
            y_range = max_y - min_y
            plt.ylim(min_y - 0.05 * y_range, max_y + 0.05 * y_range)
            
        plt.tight_layout()
        plt.show()

# Performance Profiles without success/failure criteria
def plot_performance_profile(metric_data, metric_name, ax):
    "Helper to plot a single performance profile."
    max_ratio = 0  # Track max ratio for plot limits
    num_problems = 0
    if metric_data:
        num_problems = len(next(iter(metric_data.values())))  # Get number of problems from first method

    for method, ratios in metric_data.items():
        # Skip LBFGS method
        if 'lbfgs' in method.lower():
            continue
            
        valid_ratios = [r for r in ratios if np.isfinite(r)]
        if not valid_ratios: continue
        
        sorted_ratios = np.sort(valid_ratios)
        y = np.arange(1, len(sorted_ratios) + 1) / num_problems  # Normalize by total problems
        
        # Extend the line for step plot visualization
        plot_x = np.concatenate(([1], sorted_ratios, [sorted_ratios[-1]*1.1]))  # Start at 1, include points, extend slightly
        plot_y = np.concatenate(([0], y, [y[-1]]))  # Step plot y-coordinates
        
        ax.step(plot_x, plot_y, where='post', label=method)
        max_ratio = max(max_ratio, sorted_ratios[-1] if sorted_ratios.size > 0 else 1)
        
    ax.set_title(f'Performance Profile - {metric_name}')
    ax.set_xlabel('Performance Ratio τ')
    ax.set_ylabel('P(ratio ≤ τ)')
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(fontsize='small')
    # Set x-limits dynamically, but with a reasonable upper bound
    ax.set_xlim(1, max(2, min(10, max_ratio * 1.1))) 
    ax.set_ylim(0, 1.05)

def calculate_performance_ratios(results, method_specs):
    perf_ratios = {'Iterations': {}, 'CPU Time (s)': {}}
    problems_list = list(results.keys())
    
    # Filter out LBFGS from method_specs
    methods_list = [m['name'] for m in method_specs if 'lbfgs' not in m['name'].lower()]

    for metric in perf_ratios.keys():
        min_perf = {}
        for prob_name in problems_list:
            min_val = float('inf')
            for meth_name in methods_list:
                # Check if results exist for this problem/method combo
                res = results.get(prob_name, {}).get(meth_name, {})
                val = res.get(metric, float('inf'))
                # Consider all runs for minimum performance, regardless of success
                if isinstance(val, (int, float)) and np.isfinite(val):
                    min_val = min(min_val, val)
            min_perf[prob_name] = min_val if np.isfinite(min_val) else 1.0  # Avoid division by zero/inf
            if min_perf[prob_name] == 0: min_perf[prob_name] = 1e-10  # Avoid division by exactly zero

        for meth_name in methods_list:
            ratios_list = []
            for prob_name in problems_list:
                res = results.get(prob_name, {}).get(meth_name, {})
                val = res.get(metric, float('inf'))
                
                # Include all methods in the plot with the same treatment
                if isinstance(val, (int, float)) and np.isfinite(val):
                    ratio = val / min_perf[prob_name]
                    ratios_list.append(min(ratio, 20))  # Cap ratio at 20 for plotting
                else:
                    ratios_list.append(20)  # Assign highest ratio for non-finite values
                    
            perf_ratios[metric][meth_name] = ratios_list
    
    return perf_ratios

def plot_profiles(perf_ratios):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    if perf_ratios['Iterations']:
        plot_performance_profile(perf_ratios['Iterations'], 'Iterations', axes[0])
    else:
        axes[0].set_title('Performance Profile - Iterations (No Data)')

    if perf_ratios['CPU Time (s)']:
        plot_performance_profile(perf_ratios['CPU Time (s)'], 'CPU Time (s)', axes[1])
    else:
        axes[1].set_title('Performance Profile - CPU Time (s) (No Data)')

    plt.tight_layout()
    plt.show() 