# IOE 511/MATH 562, University of Michigan
# Main optimization solver function

import numpy as np
import time
import algorithms

def optSolver_WHY(problem, method, options):
    """
    Function that runs a chosen algorithm on a chosen problem.
    Matches the required signature: [x,f]=optSolver_WHY(problem, method, options)

    Inputs:
        problem (dict): Problem definition struct/dict 
                        Requires 'x0', 'name', 'func', 'grad'
                        Optional: 'Hess', 'f_opt'
        method (dict): Method specification struct/dict 
                       Requires 'name' (e.g., 'GradientDescent', 'NewtonW', 'BFGS', 'LBFGS-M5-FIFO')
        options (dict): Algorithm options struct/dict 
                        Optional: 'term_tol', 'max_iterations', line search params, TR params, etc.

    Outputs:
        x (ndarray): Final iterate (solution found) 
        f (float): Final function value
        # Note: The description asks only for x, f. Returning info for analysis.
        # info (dict): Dictionary with performance details (iterations, f_values, etc.)
    """
    method_name = method['name']
    problem_dict = problem
    
    # Check if we should skip Newton-based methods for Exponential_1000
    problem_name = problem_dict.get('name', '')
    if problem_name == 'Exponential_1000' and any(x in method_name for x in ['Newton', 'TRNewtonCG', 'TRSR1CG']):
        print(f"Skipping {method_name} for Exponential_1000 as requested.")
        # Return dummy results to indicate it was skipped
        x_final = problem_dict['x0'].copy()
        f_final = problem_dict['func'](x_final)
        info = {
            'iterations': 0,
            'f_values': [f_final],
            'grad_norms': [np.linalg.norm(problem_dict['grad'](x_final))],
            'times': [0.0],
            'success': False,
            'termination_reason': "Skipped for Exponential_1000"
        }
        return x_final, f_final, info

    # Construct the options
    default_options = {
        'term_tol': 1e-6,            # Optimality tolerance (gradient norm)
        'max_iterations': 1000,       # Maximum number of iterations
        
        # Line Search Parameters
        'alpha_init': 1.0,            # Initial step size
        'c1_ls': 1e-4,                # Armijo constant - normally between 1e-4 and 1e-1
        'c2_ls': 0.9,                 # Wolfe curvature constant - normally between 0.1 and 0.9
        'c': 0.5,                     # Backtracking line search reduction factor (rho)
        'max_ls_iter': 20,            # Maximum line search iterations
        'alpha_max': 10.0,            # Maximum step size for Wolfe line search
        
        # Trust Region Parameters
        'delta_init': 1.0,            # Initial trust region radius
        'delta_max': 10.0,            # Maximum trust region radius
        'eta': 0.15,                  # Trust region step acceptance threshold
        
        # CG Solver for Trust Region
        'cg_max_iter': 100,           # Maximum CG iterations
        'cg_tol': 0.1,                # CG tolerance for TR subproblem
        
        # SR1 parameters
        'sr1_threshold': 0.0,         # Threshold for SR1 update to maintain stability
        'max_consecutive_rejects': 5,  # Maximum consecutive rejected steps before regularization
        'sr1_reg_init': 1e-6,         # Initial regularization value for SR1
        'sr1_reg_update': 1.2,        # Regularization increase factor
        
        'verbose': False              # Print iterations
    }
    
    # Apply optimal parameters for Gradient Descent methods based on hyperparameter tuning
    if method_name == "GradientDescent":
        gd_optimal_params = {
            'c1_ls': 1e-5,            # Optimal Armijo constant from tuning
            'alpha_init': 5.0,         # Optimal initial step size
            'c': 0.9                   # Optimal backtracking reduction factor
        }
        for param, value in gd_optimal_params.items():
            default_options[param] = value
    elif method_name == "GradientDescentW":
        gdw_optimal_params = {
            'c1_ls': 1e-5,            # Optimal Armijo constant from tuning
            'alpha_init': 0.1,         # Optimal initial step size
            'c2_ls': 0.5               # Optimal Wolfe curvature constant
        }
        for param, value in gdw_optimal_params.items():
            default_options[param] = value
    # Apply optimal parameters for Newton methods based on hyperparameter tuning
    elif method_name == "Newton":
        newton_optimal_params = {
            'c1_ls': 1e-5,            # Optimal Armijo constant from tuning
            'alpha_init': 1.0,         # Optimal initial step size
            'c': 0.1                   # Optimal backtracking reduction factor
        }
        for param, value in newton_optimal_params.items():
            default_options[param] = value
    elif method_name == "NewtonW":
        newtonw_optimal_params = {
            'c1_ls': 1e-5,            # Optimal Armijo constant from tuning
            'alpha_init': 0.5,         # Optimal initial step size
            'c2_ls': 0.5               # Optimal Wolfe curvature constant
        }
        for param, value in newtonw_optimal_params.items():
            default_options[param] = value
    # Apply optimal parameters for BFGS methods based on hyperparameter tuning
    elif method_name == "BFGS":
        bfgs_optimal_params = {
            'c1_ls': 1e-4,            # Optimal Armijo constant from tuning
            'alpha_init': 1.0,         # Optimal initial step size
            'c': 0.5                   # Optimal backtracking reduction factor
        }
        for param, value in bfgs_optimal_params.items():
            default_options[param] = value
    elif method_name == "BFGSW":
        bfgsw_optimal_params = {
            'c1_ls': 1e-4,            # Optimal Armijo constant from tuning
            'alpha_init': 1.0,         # Optimal initial step size
            'c2_ls': 0.9              # Optimal Wolfe curvature constant
        }
        for param, value in bfgsw_optimal_params.items():
            default_options[param] = value

    elif method_name == "BFGSW+":
        # Enhanced BFGS with Wolfe line search and robust improvements
        bfgsw_plus_params = {
            'c1_ls': 1e-6,            # Armijo parameter (more lenient)
            'c2_ls': 0.7,             # Wolfe parameter (less strict)
            'alpha_init': 0.5,        # Initial step size
            'alpha_max': 20.0,        # Maximum step size
            'max_ls_iter': 30,        # More line search iterations
            'hessian_reset_threshold': 1e-10,  # Threshold for Hessian resets
            'regularization_factor': 1e-6,    # Regularization for positive definiteness
            'max_small_steps': 5,     # Max consecutive small steps before jittering
            'min_step_size': 1e-10,   # Minimum effective step size
            'max_condition_number': 1e8  # Maximum condition number before regularization
        }
        for param, value in bfgsw_plus_params.items():
            default_options[param] = value
    
    # Apply any user-specified options
    current_options = default_options.copy()
    if options:
        for key, value in options.items():
            current_options[key] = value

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
    elif method_name == "BFGSW+": # Enhanced BFGS with Wolfe line search
        # Use the regular BFGS implementation with the special parameters set above
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

    return x_final, f_final, info