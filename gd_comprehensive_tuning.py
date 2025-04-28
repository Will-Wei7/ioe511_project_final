"""
IOE 511/MATH 562, University of Michigan
Comprehensive Gradient Descent Hyperparameter Tuning

This script performs detailed hyperparameter tuning for both Gradient Descent variants
(with backtracking and Wolfe line search) on all available test problems.
"""

import numpy as np
import itertools
import time
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
from datetime import datetime
import multiprocessing as mp
from functools import partial

# Import project modules
from solver import optSolver_WHY
from problems import get_problem, list_available_problems
from algorithms import RemovalStrategy

def run_gd_with_params(problem_name, method_name, options):
    """
    Run a single gradient descent optimization with specified parameters
    
    Args:
        problem_name: Name of the problem to solve
        method_name: Method name ("GradientDescent" or "GradientDescentW")
        options: Dictionary of optimization options
        
    Returns:
        Dictionary with results
    """
    try:
        # Load problem
        problem = get_problem(problem_name)
        
        # Convert problem to dictionary if needed
        if hasattr(problem, 'to_dict'):
            problem_dict = problem.to_dict()
        else:
            problem_dict = problem
        
        method = {'name': method_name}
        start_time = time.time()
        
        # Run optimization
        x_final, f_final, info = optSolver_WHY(problem_dict, method, options)
        runtime = time.time() - start_time
        
        # Extract detailed metrics
        result = {
            'problem': problem_name,
            'method': method_name,
            'f_final': f_final,
            'f_opt_gap': abs(f_final - problem_dict.get('f_opt', f_final)) if problem_dict.get('f_opt') is not None else np.nan,
            'iterations': info['iterations'],
            'runtime': runtime,
            'function_evals': info.get('func_calls', 0),
            'gradient_evals': info.get('grad_calls', 0),
            'final_grad_norm': info['grad_norms'][-1] if 'grad_norms' in info else np.nan,
            'success': info.get('success', True),  # Use success from info, default to True for backward compatibility
            **options  # Include parameters
        }
        
        # Add analysis of convergence rate
        if 'f_vals' in info and len(info['f_vals']) > 1:
            # Calculate average reduction per iteration
            f_values = np.array(info['f_vals'])
            if f_values[0] != f_values[-1]:  # Avoid division by zero
                reductions = np.diff(f_values)
                avg_reduction = np.mean(reductions[reductions < 0])  # Only count iterations with progress
                result['avg_reduction_per_iter'] = avg_reduction
                
    except Exception as e:
        print(f"Error with {method_name} on {problem_name}, parameters: {options}: {str(e)}")
        result = {
            'problem': problem_name,
            'method': method_name,
            'success': False,
            'error': str(e),
            **options
        }
    
    return result

def tune_gd_for_problem(problem_name, verbose=True):
    """
    Perform comprehensive GD parameter tuning for a single problem
    
    Args:
        problem_name: Name of the problem to tune for
        verbose: Whether to print detailed progress
        
    Returns:
        DataFrame with all results
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Tuning gradient descent methods for problem: {problem_name}")
        print(f"{'='*70}")
    
    # Fixed parameters as specified
    fixed_options = {
        'max_iterations': 1000,
        'term_tol': 1e-6,
        'verbose': False
    }
    
    # Parameter grids - comprehensive for detailed analysis
    param_grids = {
        # Backtracking line search parameters
        'GradientDescent': {
            'c1_ls': [1e-5, 1e-4, 1e-3, 1e-2],  # Armijo constant
            'c': [0.1, 0.3, 0.5, 0.7, 0.9],     # Backtracking reduction factor
            'alpha_init': [0.1, 0.5, 1.0, 2.0, 5.0]  # Initial step size
        },
        # Wolfe line search parameters
        'GradientDescentW': {
            'c1_ls': [1e-5, 1e-4, 1e-3, 1e-2],  # Armijo constant
            'c2_ls': [0.5, 0.7, 0.9, 0.99],     # Wolfe curvature constant
            'alpha_init': [0.1, 0.5, 1.0, 2.0, 5.0]  # Initial step size
        }
    }
    
    # Store all results
    all_results = []
    
    # Run all parameter combinations for both methods
    for method_name, param_grid in param_grids.items():
        if verbose:
            print(f"\n{'-'*50}")
            print(f"Tuning {method_name} on {problem_name}")
            print(f"{'-'*50}")
        
        # Generate all combinations
        keys = list(param_grid.keys())
        values = [param_grid[key] for key in keys]
        combinations = list(itertools.product(*values))
        
        total_combos = len(combinations)
        if verbose:
            print(f"Testing {total_combos} parameter combinations")
        
        for i, combo in enumerate(combinations):
            # Create parameter dictionary
            options = fixed_options.copy()
            for k, v in zip(keys, combo):
                options[k] = v
            
            # Display progress
            if verbose:
                param_str = ", ".join([f"{k}={options[k]}" for k in keys])
                print(f"Running {i+1}/{total_combos}: {param_str}")
            
            # Run optimization
            result = run_gd_with_params(problem_name, method_name, options)
            all_results.append(result)
    
    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(all_results)
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"gd_tuning_{problem_name}_{timestamp}.csv"
    results_df.to_csv(filename, index=False)
    if verbose:
        print(f"\nResults saved to {filename}")
    
    return results_df

def analyze_results(results_df, problem_name):
    """
    Generate detailed analysis and visualizations of the tuning results for a single problem
    
    Args:
        results_df: DataFrame with tuning results
        problem_name: Name of the problem for report titles
        
    Returns:
        Dictionary with best parameters for each method
    """
    # Filter out failed runs
    success_df = results_df[results_df['success'] == True].copy()
    
    if len(success_df) == 0:
        print(f"No successful runs to analyze for {problem_name}")
        return {}
    
    print(f"\n{'='*60}")
    print(f"Analysis of Gradient Descent Performance on {problem_name}")
    print(f"{'='*60}")
    
    # 1. Overall best performers
    print("\n--- Best Performing Configurations ---")
    
    best_params = {}
    for method in success_df['method'].unique():
        method_df = success_df[success_df['method'] == method]
        
        # Best by iterations
        best_iter_idx = method_df['iterations'].idxmin()
        best_iter = method_df.loc[best_iter_idx]
        
        # Best by runtime
        best_time_idx = method_df['runtime'].idxmin()
        best_time = method_df.loc[best_time_idx]
        
        # Best by final function value (closest to optimum)
        if 'f_opt_gap' in method_df.columns and not method_df['f_opt_gap'].isna().all():
            best_opt_idx = method_df['f_opt_gap'].idxmin()
            best_opt = method_df.loc[best_opt_idx]
            
            print(f"\n{method}:")
            print(f"  Fastest convergence (iterations): {best_iter['iterations']} iterations")
            print(f"    Parameters: c1_ls={best_iter['c1_ls']}, " + 
                  (f"c2_ls={best_iter['c2_ls']}, " if 'c2_ls' in best_iter else f"c={best_iter['c']}, ") +
                  f"alpha_init={best_iter['alpha_init']}")
            
            print(f"  Fastest runtime: {best_time['runtime']:.4f} seconds")
            print(f"    Parameters: c1_ls={best_time['c1_ls']}, " + 
                  (f"c2_ls={best_time['c2_ls']}, " if 'c2_ls' in best_time else f"c={best_time['c']}, ") +
                  f"alpha_init={best_time['alpha_init']}")
            
            print(f"  Most accurate solution: f_opt_gap={best_opt['f_opt_gap']:.8e}")
            print(f"    Parameters: c1_ls={best_opt['c1_ls']}, " + 
                  (f"c2_ls={best_opt['c2_ls']}, " if 'c2_ls' in best_opt else f"c={best_opt['c']}, ") +
                  f"alpha_init={best_opt['alpha_init']}")
        
        # Store best parameters (by iterations)
        param_dict = {}
        for param in ['c1_ls', 'alpha_init']:
            param_dict[param] = float(best_iter[param])
            
        if method == 'GradientDescent':
            param_dict['c'] = float(best_iter['c'])
        else:
            param_dict['c2_ls'] = float(best_iter['c2_ls'])
            
        best_params[method] = param_dict
    
    # 2. Compare GD vs GDW (backtracking vs Wolfe)
    print("\n--- GradientDescent vs GradientDescentW Comparison ---")
    method_stats = success_df.groupby('method').agg({
        'iterations': ['mean', 'min', 'max', 'std'],
        'runtime': ['mean', 'min', 'max', 'std'],
        'function_evals': ['mean', 'sum'] if 'function_evals' in success_df.columns else [],
        'gradient_evals': ['mean', 'sum'] if 'gradient_evals' in success_df.columns else []
    })
    print(method_stats)
    
    # 3. Parameter Influence Analysis
    print("\n--- Parameter Influence Analysis ---")
    
    # Create visualizations
    fig_iterations = plt.figure(figsize=(15, 10))
    fig_iterations.suptitle(f"Effect of Parameters on Iteration Count ({problem_name})", fontsize=16)
    
    # Analysis for each method
    for i, method in enumerate(['GradientDescent', 'GradientDescentW']):
        method_df = success_df[success_df['method'] == method]
        if len(method_df) == 0:
            continue
            
        # Parameter columns for this method
        param_cols = ['c1_ls', 'alpha_init']
        if method == 'GradientDescent':
            param_cols.append('c')
        else:
            param_cols.append('c2_ls')
            
        # Plot for iterations
        for j, param in enumerate(param_cols):
            ax_iter = fig_iterations.add_subplot(2, 3, i*3 + j + 1)
            
            # Group by parameter
            grouped = method_df.groupby(param)['iterations'].agg(['mean', 'min', 'std']).reset_index()
            ax_iter.bar(grouped[param].astype(str), grouped['mean'], yerr=grouped['std'], capsize=5)
            ax_iter.plot(grouped[param].astype(str), grouped['min'], 'ro-', label='Min iterations')
            ax_iter.set_title(f"{method}: {param} vs Iterations")
            ax_iter.set_xlabel(param)
            ax_iter.set_ylabel('Iterations')
            ax_iter.legend()
            
            # Print analysis
            print(f"\n{method} - Effect of {param} on iterations:")
            print(grouped.sort_values('mean'))
    
    # Save figures
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_iterations.savefig(f"gd_iterations_{problem_name}_{timestamp}.png")
    
    # 4. Detailed analysis of best combinations
    print("\n--- Top-5 Parameter Combinations ---")
    for method in success_df['method'].unique():
        method_df = success_df[success_df['method'] == method]
        
        # Get top 5 by iterations
        top5_iter = method_df.nsmallest(5, 'iterations')
        print(f"\n{method} - Top 5 by iterations:")
        print(top5_iter[['iterations', 'runtime', 'f_final', 'c1_ls', 
                         'c2_ls' if method == 'GradientDescentW' else 'c', 'alpha_init']])
    
    # Save best parameters to JSON
    with open(f"gd_best_params_{problem_name}.json", 'w') as f:
        json.dump(best_params, f, indent=4)
    
    print(f"\nBest parameters saved to gd_best_params_{problem_name}.json")
    return best_params

def worker_function(problem_name):
    """Worker function for parallel processing"""
    try:
        results = tune_gd_for_problem(problem_name)
        best_params = analyze_results(results, problem_name)
        return {problem_name: best_params}
    except Exception as e:
        print(f"Error processing {problem_name}: {str(e)}")
        return {problem_name: {}}

def run_comprehensive_tuning():
    """Run comprehensive tuning on all available problems"""
    # Get all available problems
    problem_names = list_available_problems()
    
    print(f"Starting comprehensive gradient descent tuning for {len(problem_names)} problems:")
    for i, name in enumerate(problem_names):
        print(f"  {i+1}. {name}")
    
    # Ask user if they want to proceed with all problems or select specific ones
    print("\nOptions:")
    print("1. Tune all problems")
    print("2. Select specific problems")
    print("3. Select problem categories")
    
    choice = input("Enter your choice (1-3): ")
    
    selected_problems = []
    if choice == "1":
        selected_problems = problem_names
    elif choice == "2":
        print("\nAvailable problems:")
        for i, name in enumerate(problem_names):
            print(f"  {i+1}. {name}")
        indices = input("Enter problem numbers separated by commas (e.g., 1,3,5): ")
        try:
            selected_indices = [int(idx.strip()) - 1 for idx in indices.split(",")]
            selected_problems = [problem_names[idx] for idx in selected_indices if 0 <= idx < len(problem_names)]
        except:
            print("Invalid input. Using all problems.")
            selected_problems = problem_names
    elif choice == "3":
        categories = {
            "1": [p for p in problem_names if "quad" in p.lower()],
            "2": [p for p in problem_names if "rosenbrock" in p.lower()],
            "3": [p for p in problem_names if "quartic" in p.lower()],
            "4": [p for p in problem_names if any(x in p.lower() for x in ["datafit", "exponential", "genhumps", "powell", "illconditioned"])]
        }
        
        print("\nProblem categories:")
        print("1. Quadratic problems")
        print("2. Rosenbrock problems")
        print("3. Quartic problems")
        print("4. Other problems")
        
        cat_choice = input("Enter category numbers separated by commas: ")
        try:
            selected_categories = [cat.strip() for cat in cat_choice.split(",")]
            for cat in selected_categories:
                if cat in categories:
                    selected_problems.extend(categories[cat])
        except:
            print("Invalid input. Using all problems.")
            selected_problems = problem_names
    else:
        print("Invalid choice. Using all problems.")
        selected_problems = problem_names
    
    # Remove duplicates
    selected_problems = list(dict.fromkeys(selected_problems))
    
    print(f"\nRunning tuning for {len(selected_problems)} problems: {', '.join(selected_problems)}")
    
    # Ask whether to run in parallel
    use_parallel = input("\nRun in parallel? This will be faster but produce less detailed output (y/n): ")
    
    all_best_params = {}
    
    if use_parallel.lower().startswith('y'):
        # Set up parallel processing
        num_cores = min(mp.cpu_count(), len(selected_problems))
        print(f"Using {num_cores} cores for parallel processing")
        
        with mp.Pool(processes=num_cores) as pool:
            results = pool.map(worker_function, selected_problems)
            
        # Combine results
        for result in results:
            all_best_params.update(result)
    else:
        # Sequential processing
        for problem in selected_problems:
            results = tune_gd_for_problem(problem)
            best_params = analyze_results(results, problem)
            all_best_params[problem] = best_params
    
    # Save comprehensive results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"gd_all_best_params_{timestamp}.json", 'w') as f:
        json.dump(all_best_params, f, indent=4)
    
    print(f"\nAll best parameters saved to gd_all_best_params_{timestamp}.json")
    
    # Generate code for updating solver options
    generate_solver_options_code(all_best_params)
    
    return all_best_params

def generate_solver_options_code(all_best_params):
    """Generate code for updating solver options with the best parameters"""
    
    # Create problem-specific options dictionary
    code = "# Updated default options based on hyperparameter tuning for Gradient Descent methods\n"
    code += "gd_problem_specific_options = {\n"
    
    for problem, methods in all_best_params.items():
        if not methods:  # Skip if no methods were successful
            continue
            
        code += f"    '{problem}': {{\n"
        for method, params in methods.items():
            code += f"        '{method}': {{\n"
            for param, value in params.items():
                code += f"            '{param}': {value},\n"
            code += "        },\n"
        code += "    },\n"
    code += "}\n\n"
    
    # Add code for using these options in the solver
    code += "# Code to add to solver.py:\n"
    code += "'''\n"
    code += "# In optSolver_WHY function, before merging user options:\n"
    code += "\n"
    code += "# Apply problem-specific optimal parameters for GD methods\n"
    code += "if problem_dict['name'] in gd_problem_specific_options:\n"
    code += "    problem_options = gd_problem_specific_options[problem_dict['name']]\n"
    code += "    if method_name in problem_options:\n"
    code += "        # Get optimal parameters for this problem-method combination\n"
    code += "        optimal_params = problem_options[method_name]\n"
    code += "        # Update default options with optimal parameters\n"
    code += "        for param, value in optimal_params.items():\n"
    code += "            default_options[param] = value\n"
    code += "'''\n"
    
    # Save the code to a file
    with open("update_gd_solver_options.py", 'w') as f:
        f.write(code)
    
    print("Code for updating solver options with optimal GD parameters saved to update_gd_solver_options.py")

def print_summary_table(all_best_params):
    """Print a summary table of best parameters for all problems"""
    print("\n" + "="*100)
    print("SUMMARY OF OPTIMAL GRADIENT DESCENT PARAMETERS FOR ALL PROBLEMS")
    print("="*100)
    
    print("\nGradientDescent (Backtracking Line Search):")
    print(f"{'Problem':<20} | {'c1_ls':<10} | {'c':<10} | {'alpha_init':<10}")
    print("-" * 55)
    
    for problem, methods in sorted(all_best_params.items()):
        if not methods or 'GradientDescent' not in methods:
            continue
            
        params = methods['GradientDescent']
        print(f"{problem:<20} | {params.get('c1_ls', '-'):<10.0e} | {params.get('c', '-'):<10.2f} | {params.get('alpha_init', '-'):<10.2f}")
    
    print("\nGradientDescentW (Wolfe Line Search):")
    print(f"{'Problem':<20} | {'c1_ls':<10} | {'c2_ls':<10} | {'alpha_init':<10}")
    print("-" * 55)
    
    for problem, methods in sorted(all_best_params.items()):
        if not methods or 'GradientDescentW' not in methods:
            continue
            
        params = methods['GradientDescentW']
        print(f"{problem:<20} | {params.get('c1_ls', '-'):<10.0e} | {params.get('c2_ls', '-'):<10.2f} | {params.get('alpha_init', '-'):<10.2f}")
    
    # Print overall recommendations
    print("\n" + "="*100)
    print("OVERALL RECOMMENDATIONS")
    print("="*100)
    
    # For each method, calculate the most common best values
    for method in ['GradientDescent', 'GradientDescentW']:
        param_values = {}
        valid_problems = 0
        
        for problem, methods in all_best_params.items():
            if method in methods:
                valid_problems += 1
                for param, value in methods[method].items():
                    if param not in param_values:
                        param_values[param] = []
                    param_values[param].append(value)
        
        if valid_problems > 0:
            print(f"\nRecommended default parameters for {method}:")
            for param, values in param_values.items():
                # Find the most common value
                unique_values = {}
                for val in values:
                    if val not in unique_values:
                        unique_values[val] = 0
                    unique_values[val] += 1
                
                # Sort by frequency
                sorted_values = sorted(unique_values.items(), key=lambda x: x[1], reverse=True)
                
                # Calculate mean value as well
                mean_val = sum(values) / len(values)
                
                # Print recommendations
                print(f"  {param}: {sorted_values[0][0]} (used in {sorted_values[0][1]}/{valid_problems} problems, mean: {mean_val:.4g})")

if __name__ == "__main__":
    print("Starting comprehensive Gradient Descent hyperparameter tuning...")
    
    # Run tuning on all problems
    all_best_params = run_comprehensive_tuning()
    
    # Print summary table
    print_summary_table(all_best_params)
    
    print("\nTuning complete!")
    plt.show()  # Display all figures 