"""
IOE 511/MATH 562, University of Michigan
Summary of Gradient Descent results across all problems with final metrics
"""

import numpy as np
import time
import pandas as pd
from tabulate import tabulate

# Import project modules
from solver import optSolver_WHY
from problems import get_problem, list_available_problems

def run_gd_on_problem(problem_name, method_name="GradientDescent"):
    """
    Run gradient descent on a specified problem and return key metrics
    
    Args:
        problem_name: Name of the problem to solve
        method_name: Method name (default: "GradientDescent" with backtracking)
        
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
        
        # Standard options that work reasonably well across problems
        options = {
            'max_iterations': 1000,
            'term_tol': 1e-6,
            'verbose': False,
            'c1_ls': 1e-4,      # Armijo constant
            'c': 0.5,           # Backtracking reduction factor
            'alpha_init': 1.0   # Initial step size
        }
        
        # If using Wolfe line search
        if method_name == "GradientDescentW":
            options['c2_ls'] = 0.9  # Wolfe curvature constant
            del options['c']        # Remove backtracking parameter
        
        method = {'name': method_name}
        start_time = time.time()
        
        # Run optimization
        x_final, f_final, info = optSolver_WHY(problem_dict, method, options)
        runtime = time.time() - start_time
        
        # Extract key metrics
        result = {
            'problem': problem_name,
            'method': method_name,
            'iterations': info['iterations'],
            'f_final': f_final,
            'final_grad_norm': info['grad_norms'][-1] if 'grad_norms' in info else np.nan,
            'runtime': runtime,
            'f_opt': problem_dict.get('f_opt', np.nan),
            'success': info.get('success', False) or info['grad_norms'][-1] < options['term_tol']
        }
        
        # Add f_gap if f_opt is known
        if problem_dict.get('f_opt') is not None:
            result['f_gap'] = abs(f_final - problem_dict['f_opt'])
        
        return result
    
    except Exception as e:
        print(f"Error running {method_name} on {problem_name}: {str(e)}")
        return {
            'problem': problem_name,
            'method': method_name,
            'iterations': np.nan,
            'f_final': np.nan,
            'final_grad_norm': np.nan,
            'runtime': np.nan,
            'success': False,
            'error': str(e)
        }

def run_all_problems():
    """Run GD on all problems and print summary table"""
    
    # Get all problems
    all_problems = list_available_problems()
    print(f"Found {len(all_problems)} problems to test:")
    for i, name in enumerate(all_problems):
        print(f"  {i+1}. {name}")
    
    # Methods to test
    methods = ["GradientDescent", "GradientDescentW"]
    
    # Results container
    all_results = []
    
    # Run each method on each problem
    for method in methods:
        print(f"\nRunning {method} on all problems...")
        
        for i, problem_name in enumerate(all_problems):
            print(f"  [{i+1}/{len(all_problems)}] Running {method} on {problem_name}...")
            result = run_gd_on_problem(problem_name, method)
            all_results.append(result)
            
            # Print immediate result
            status = "✓" if result.get('success', False) else "✗"
            print(f"    {status} Iterations: {result.get('iterations', 'N/A')}, " +
                  f"Final gradient norm: {result.get('final_grad_norm', 'N/A'):.2e}, " +
                  f"Final f(x): {result.get('f_final', 'N/A'):.2e}")
    
    # Create DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results to CSV
    results_df.to_csv("gd_all_problems_summary.csv", index=False)
    print("\nResults saved to gd_all_problems_summary.csv")
    
    # Create and print summary table
    print("\n" + "="*100)
    print("SUMMARY OF GRADIENT DESCENT RESULTS ON ALL PROBLEMS")
    print("="*100)
    
    summary_data = []
    for method in methods:
        for problem in all_problems:
            result = next((r for r in all_results if r['problem'] == problem and r['method'] == method), None)
            if result:
                # Format values nicely for display
                row = [
                    problem,
                    method,
                    f"{result.get('iterations', 'N/A')}",
                    f"{result.get('final_grad_norm', np.nan):.2e}",
                    f"{result.get('f_final', np.nan):.4e}",
                    "Yes" if result.get('success', False) else "No",
                    f"{result.get('runtime', np.nan):.2f}s"
                ]
                
                # Add f_gap if f_opt is known
                if 'f_gap' in result:
                    row.append(f"{result['f_gap']:.2e}")
                else:
                    row.append("N/A")
                    
                summary_data.append(row)
    
    # Create formatted table
    headers = ["Problem", "Method", "Iterations", "Final ‖∇f(x)‖", "Final f(x)", "Success", "Runtime", "f_gap"]
    table = tabulate(summary_data, headers=headers, tablefmt="grid", numalign="right", stralign="left")
    
    print(table)
    
    # Create comparison of GD vs GDW
    print("\n" + "="*100)
    print("COMPARISON OF GD vs GDW (BACKTRACKING vs WOLFE)")
    print("="*100)
    
    comparison_data = []
    for problem in all_problems:
        gd_result = next((r for r in all_results if r['problem'] == problem and r['method'] == "GradientDescent"), None)
        gdw_result = next((r for r in all_results if r['problem'] == problem and r['method'] == "GradientDescentW"), None)
        
        if gd_result and gdw_result:
            # Compare iterations
            if gd_result.get('iterations', np.nan) < gdw_result.get('iterations', np.nan):
                iter_winner = "GD"
            elif gd_result.get('iterations', np.nan) > gdw_result.get('iterations', np.nan):
                iter_winner = "GDW"
            else:
                iter_winner = "Tie"
            
            # Compare gradient norm
            if gd_result.get('final_grad_norm', np.inf) < gdw_result.get('final_grad_norm', np.inf):
                grad_winner = "GD"
            elif gd_result.get('final_grad_norm', np.inf) > gdw_result.get('final_grad_norm', np.inf):
                grad_winner = "GDW"
            else:
                grad_winner = "Tie"
            
            # Compare function value
            if abs(gd_result.get('f_final', np.inf)) < abs(gdw_result.get('f_final', np.inf)):
                func_winner = "GD"
            elif abs(gd_result.get('f_final', np.inf)) > abs(gdw_result.get('f_final', np.inf)):
                func_winner = "GDW"
            else:
                func_winner = "Tie"
            
            comparison_data.append([
                problem,
                gd_result.get('iterations', 'N/A'),
                gdw_result.get('iterations', 'N/A'),
                iter_winner,
                f"{gd_result.get('final_grad_norm', np.nan):.2e}",
                f"{gdw_result.get('final_grad_norm', np.nan):.2e}",
                grad_winner,
                f"{gd_result.get('f_final', np.nan):.4e}",
                f"{gdw_result.get('f_final', np.nan):.4e}",
                func_winner
            ])
    
    # Create formatted comparison table
    comparison_headers = [
        "Problem", 
        "GD Iters", "GDW Iters", "Winner",
        "GD ‖∇f(x)‖", "GDW ‖∇f(x)‖", "Winner",
        "GD f(x)", "GDW f(x)", "Winner"
    ]
    comparison_table = tabulate(comparison_data, headers=comparison_headers, tablefmt="grid", numalign="right", stralign="left")
    
    print(comparison_table)

if __name__ == "__main__":
    run_all_problems() 