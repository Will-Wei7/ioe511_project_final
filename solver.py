# IOE 511/MATH 562, University of Michigan
# Main optimization solver function

import numpy as np
import time
import algorithms 

def optSolver_WHY(problem, method, options):
    """
    Function that runs a chosen algorithm on a chosen problem.
    Matches the required signature: [x,f]=optSolver_WHY(problem, method, options) [cite: 266]

    Inputs:
        problem (dict): Problem definition struct/dict [cite: 268]
                        Requires 'x0', 'name', 'func', 'grad'
                        Optional: 'Hess', 'f_opt'
        method (dict): Method specification struct/dict [cite: 270]
                       Requires 'name' (e.g., 'GradientDescent', 'NewtonW', 'BFGS', 'LBFGS-M5-FIFO')
        options (dict): Algorithm options struct/dict [cite: 276]
                        Optional: 'term_tol', 'max_iterations', line search params, TR params, etc.

    Outputs:
        x (ndarray): Final iterate (solution found) [cite: 267]
        f (float): Final function value [cite: 267]
        # Note: The description asks only for x, f. Returning info for analysis.
        # info (dict): Dictionary with performance details (iterations, f_values, etc.)
    """
    method_name = method['name']
    problem_dict = problem # problem is a dict or compatible object

    # --- Set Default Options ---
    default_options = {
        'term_tol': 1e-6,              
        'max_iterations': 1000,        
        'c1_ls': 1e-4,                 # Armijo constant 
        'c2_ls': 0.9,                  # Wolfe curvature constant 
        'alpha_init': 1.0,             # Initial step size for line search
        'c': 0.5,                      # Backtracking reduction factor / Wolfe interpolation factor
        'alpha_high': 10.0,            # Max step size for Wolfe
        'max_ls_iter': 20,             # Max line search iterations
        'eta': 0.1,                    # TR step acceptance threshold 
        'delta_init': 1.0,             # Initial TR radius 
        'delta_max': 10.0,             # Max TR radius
        'c1_tr': 0.25,                 # TR radius decrease factor 
        'c2_tr': 2.0,                  # TR radius increase factor 
        'term_tol_CG': 1e-8,           # CG tolerance 
        'max_iterations_CG': max(10, len(problem_dict['x0'])), # CG iteration limit 
        'sr1_tol': 1e-8,               # SR1 update tolerance 
        'verbose': False               # Print iteration details
    }
    # Merge user options with defaults
    current_options = default_options.copy()
    current_options.update(options)

    # --- Algorithm Selection ---
    x_final, f_final, info = None, None, None

    # Line Search Methods 
    if method_name == "GradientDescent":
        x_final, f_final, info = algorithms.gradient_descent(problem_dict, current_options, backtracking=True)
    elif method_name == "GradientDescentW": # W indicates Wolfe
        x_final, f_final, info = algorithms.gradient_descent(problem_dict, current_options, backtracking=False)
    elif method_name == "Newton":
        x_final, f_final, info = algorithms.newton(problem_dict, current_options, backtracking=True)
    elif method_name == "NewtonW": # W indicates Wolfe 
        x_final, f_final, info = algorithms.newton(problem_dict, current_options, backtracking=False)
    elif method_name == "BFGS":
        x_final, f_final, info = algorithms.bfgs(problem_dict, current_options, backtracking=True)
    elif method_name == "BFGSW": # W indicates Wolfe
        x_final, f_final, info = algorithms.bfgs(problem_dict, current_options, backtracking=False)
    elif method_name == "DFP":
        x_final, f_final, info = algorithms.dfp(problem_dict, current_options, backtracking=True)
    elif method_name == "DFPW": # W indicates Wolfe 
        x_final, f_final, info = algorithms.dfp(problem_dict, current_options, backtracking=False)

    # Trust Region Methods 
    elif method_name == "TRNewtonCG":
        x_final, f_final, info = algorithms.trust_region_newton(problem_dict, current_options)
    elif method_name == "TRSR1CG":
        x_final, f_final, info = algorithms.trust_region_sr1(problem_dict, current_options)

    # L-BFGS Methods
    elif method_name.startswith("LBFGS"):
        # Parse memory size and strategy from name "LBFGS-M{size}-{strategy}" 
        parts = method_name.split('-')
        memory_size = 5 # Default
        strategy = algorithms.RemovalStrategy.FIFO # Default

        if len(parts) >= 2 and parts[1].startswith('M'):
            try:
                memory_size = int(parts[1][1:])
            except ValueError:
                print(f"Warning: Could not parse memory size from {method_name}. Using default {memory_size}.")

        if len(parts) >= 3:
            strategy_name = parts[2].upper()
            if strategy_name == "FIFO":
                strategy = algorithms.RemovalStrategy.FIFO
            elif strategy_name == "MIN_CURV" or strategy_name == "MIN_CURVATURE": # Allow abbreviation
                strategy = algorithms.RemovalStrategy.MIN_CURVATURE
            elif strategy_name == "ADAPTIVE":
                strategy = algorithms.RemovalStrategy.ADAPTIVE
            else:
                 print(f"Warning: Unknown L-BFGS strategy '{strategy_name}'. Using FIFO.")


        backtracking = not ('W' in method_name) # Check if Wolfe line search requested
        x_final, f_final, info = algorithms.run_lbfgs_with_strategy(
            problem_dict, current_options, memory_size, strategy, backtracking)

    else:
        raise ValueError(f"Method '{method_name}' is not implemented yet.")

    return x_final, f_final 