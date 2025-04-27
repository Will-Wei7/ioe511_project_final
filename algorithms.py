# IOE 511/MATH 562, University of Michigan
# Optimization algorithms and line search methods

import numpy as np
from numpy import linalg as LA
from collections import deque
import time
from enum import Enum # <<< Added this line to import Enum

class RemovalStrategy(Enum):
    FIFO = 1               # First-in-first-out (oldest pair)
    MIN_CURVATURE = 2      # Smallest curvature update magnitude
    ADAPTIVE = 3           # Based on historical performance (estimated contribution)

# --- Line Search Methods ---

def backtracking_line_search(func, grad, x, d, f_x, grad_x, options):
    """
    Backtracking line search to find a step size that satisfies the Armijo condition.

    """
    alpha = options.get('alpha_init', 1.0) # Initial step size
    c1 = options.get('c1_ls', 1e-4)        # Armijo condition parameter
    rho = options.get('c', 0.5)          # Step size reduction factor

    dir_deriv = np.dot(grad_x, d) # Directional derivative

    # Ensure search direction is a descent direction
    # Check needed for stability, especially with quasi-Newton
    if dir_deriv >= 0:
        print(f"Warning: Search direction is not descent (grad_x.T @ d = {dir_deriv:.3e}). Using gradient norm.")
        # If not descent, return very small alpha or handle error
        # For simplicity, returning a small alpha to avoid halting
        return 1e-8, func(x + 1e-8 * d), grad(x + 1e-8 * d)


    max_ls_iter = options.get('max_ls_iter', 20) # Max iterations for line search
    ls_iter = 0

    # Backtracking loop
    while ls_iter < max_ls_iter:
        x_new = x + alpha * d
        f_new = func(x_new)

        # Check Armijo condition (sufficient decrease)
        if f_new <= f_x + c1 * alpha * dir_deriv:
            grad_new = grad(x_new)
            return alpha, f_new, grad_new

        # Reduce step size
        alpha *= rho
        ls_iter += 1

    # If max iterations reached, return last computed step or handle failure
    print("Warning: Backtracking line search reached max iterations.")
    grad_new = grad(x + alpha * d) # Recalculate grad just in case
    return alpha, func(x + alpha * d), grad_new

def wolfe_line_search(func, grad, x, d, f_x, grad_x, options):
    """
    Line search satisfying strong Wolfe conditions using bracketing and interpolation.
    Based on Nocedal & Wright, Numerical Optimization (2006), Algorithm 3.5/3.6

    """
    alpha = options.get('alpha_init', 1.0) # Initial step size
    c1 = options.get('c1_ls', 1e-4)        # Armijo parameter
    c2 = options.get('c2_ls', 0.9)        # Curvature parameter
    alpha_max = options.get('alpha_high', 10.0) # Max step size
    max_ls_iter = options.get('max_ls_iter', 20) # Max line search iterations

    phi_0 = f_x
    dphi_0 = np.dot(grad_x, d)

    # Check if initial direction is descent
    if dphi_0 >= 0:
         print(f"Warning: Search direction is not descent (grad_x.T @ d = {dphi_0:.3e}). Using gradient norm.")
         return 1e-8, func(x + 1e-8 * d), grad(x + 1e-8 * d)


    alpha_prev = 0.0
    phi_prev = phi_0
    dphi_prev = dphi_0
    alpha_i = alpha

    ls_iter = 0
    while ls_iter < max_ls_iter:
        x_i = x + alpha_i * d
        phi_i = func(x_i)
        grad_i = grad(x_i)
        dphi_i = np.dot(grad_i, d)

        # Check Armijo condition
        if (phi_i > phi_0 + c1 * alpha_i * dphi_0) or (phi_i >= phi_prev and ls_iter > 0):
            alpha, f_new, grad_new = zoom(func, grad, x, d, phi_0, dphi_0, alpha_prev, alpha_i, phi_prev, phi_i, c1, c2, options)
            return alpha, f_new, grad_new

        # Check strong curvature condition
        if abs(dphi_i) <= -c2 * dphi_0:
            return alpha_i, phi_i, grad_i

        # Check if derivative is positive (overshot minimum)
        if dphi_i >= 0:
            alpha, f_new, grad_new = zoom(func, grad, x, d, phi_0, dphi_0, alpha_i, alpha_prev, phi_i, phi_prev, c1, c2, options)
            return alpha, f_new, grad_new

        # If conditions not met, update and try larger alpha
        alpha_prev = alpha_i
        phi_prev = phi_i
        dphi_prev = dphi_i
        alpha_i = min(2.0 * alpha_i, alpha_max) # Increase alpha, respecting max

        ls_iter += 1

    print("Warning: Wolfe line search reached max iterations.")
    # Return last valid values or potentially re-evaluate at alpha_i
    return alpha_i, phi_i, grad_i

def zoom(func, grad, x, d, phi_0, dphi_0, alpha_lo, alpha_hi, phi_lo, phi_hi, c1, c2, options):
    """Zoom phase for Wolfe line search."""
    max_zoom_iter = options.get('max_zoom_iter', 10)
    zoom_iter = 0

    while zoom_iter < max_zoom_iter:
        # Use bisection or interpolation to find alpha_j between alpha_lo and alpha_hi
        # Simple bisection for now:
        alpha_j = 0.5 * (alpha_lo + alpha_hi)

        x_j = x + alpha_j * d
        phi_j = func(x_j)
        grad_j = grad(x_j)
        dphi_j = np.dot(grad_j, d)

        # Check Armijo condition for alpha_j
        if (phi_j > phi_0 + c1 * alpha_j * dphi_0) or (phi_j >= phi_lo):
            alpha_hi = alpha_j # New upper bound
            phi_hi = phi_j
        else:
            # Check curvature condition for alpha_j
            if abs(dphi_j) <= -c2 * dphi_0:
                return alpha_j, phi_j, grad_j # Found suitable point

            # Update bounds based on derivative sign
            if dphi_j * (alpha_hi - alpha_lo) >= 0:
                alpha_hi = alpha_lo # Move upper bound closer
                phi_hi = phi_lo
            alpha_lo = alpha_j # New lower bound
            phi_lo = phi_j

        zoom_iter += 1
        # Check interval width
        if abs(alpha_hi - alpha_lo) < 1e-10 * max(alpha_lo, alpha_hi):
             break


    print("Warning: Zoom phase reached max iterations.")
    # Return best point found so far (usually alpha_lo corresponds to lowest function value)
    x_lo = x + alpha_lo * d
    return alpha_lo, phi_lo, grad(x_lo)

# --- Optimization Algorithms ---

def gradient_descent(problem, options, backtracking=True):
    """
    Gradient descent with specified line search.
    """
    x = problem['x0'].copy()
    func = problem['func']
    grad = problem['grad']
    f_opt = problem.get('f_opt') # Get optimal value if provided

    max_iter = options['max_iterations']
    tol = options['term_tol']
    verbose = options.get('verbose', False)

    line_search = backtracking_line_search if backtracking else wolfe_line_search

    f_x = func(x)
    grad_x = grad(x)
    grad_norm = LA.norm(grad_x)

    # History tracking
    iter_count = 0
    f_history = [f_x]
    grad_norms = [grad_norm]
    times = [0.0]
    step_sizes = []
    start_time = time.time()

    if verbose:
        print(f"{'Iter':^5} | {'f(x)':^15} | {'||∇f(x)||':^15} | {'α':^10}")
        print("-" * 50)
        print(f"{iter_count:^5} | {f_x:^15.6e} | {grad_norm:^15.6e} | {'---':^10}")

    while iter_count < max_iter and grad_norm >= tol:
        d = -grad_x # Search direction

        # Perform line search
        alpha, f_new, grad_new = line_search(func, grad, x, d, f_x, grad_x, options)

        # Update iterate
        x = x + alpha * d
        f_x = f_new
        grad_x = grad_new
        grad_norm = LA.norm(grad_x)

        iter_count += 1

        # Store history
        f_history.append(f_x)
        grad_norms.append(grad_norm)
        times.append(time.time() - start_time)
        step_sizes.append(alpha)

        if verbose:
            print(f"{iter_count:^5} | {f_x:^15.6e} | {grad_norm:^15.6e} | {alpha:^10.6e}")

    success = grad_norm < tol
    termination_reason = "Gradient norm tolerance reached" if success else "Max iterations reached"

    info = {
        'iterations': iter_count,
        'f_values': f_history,
        'grad_norms': grad_norms,
        'times': times,
        'step_sizes': step_sizes,
        'success': success,
        'termination_reason': termination_reason
    }
    # Add f_gaps if f_opt is known
    if f_opt is not None:
        info['f_gaps'] = [f - f_opt for f in f_history]


    return x, f_x, info

def newton(problem, options, backtracking=True):
    """
    Modified Newton's method with specified line search.
    """
    x = problem['x0'].copy()
    func = problem['func']
    grad = problem['grad']
    hess = problem['hess']
    f_opt = problem.get('f_opt')

    max_iter = options['max_iterations']
    tol = options['term_tol']
    verbose = options.get('verbose', False)

    line_search = backtracking_line_search if backtracking else wolfe_line_search

    f_x = func(x)
    grad_x = grad(x)
    grad_norm = LA.norm(grad_x)

    iter_count = 0
    f_history = [f_x]
    grad_norms = [grad_norm]
    times = [0.0]
    step_sizes = []
    start_time = time.time()

    if verbose:
        print(f"{'Iter':^5} | {'f(x)':^15} | {'||∇f(x)||':^15} | {'α':^10}")
        print("-" * 50)
        print(f"{iter_count:^5} | {f_x:^15.6e} | {grad_norm:^15.6e} | {'---':^10}")

    while iter_count < max_iter and grad_norm >= tol:
        H = hess(x)

        # Modify Hessian if not positive definite
        min_eig = min(np.real(LA.eigvals(H))) if len(x) > 0 else 1.0 # Handle 0-dim case
        tau = 0.0
        if min_eig <= 1e-8: # Add small tolerance
            tau = abs(min_eig) + 1e-6 # Amount to add to diagonal
            H = H + tau * np.eye(len(x)) # Modified Hessian

        # Solve Newton system H*d = -grad_x
        try:
            d = LA.solve(H, -grad_x)
        except LA.LinAlgError:
             print("Warning: Hessian solve failed even after modification. Using steepest descent direction.")
             d = -grad_x


        # Perform line search
        alpha, f_new, grad_new = line_search(func, grad, x, d, f_x, grad_x, options)

        x = x + alpha * d
        f_x = f_new
        grad_x = grad_new
        grad_norm = LA.norm(grad_x)

        iter_count += 1

        f_history.append(f_x)
        grad_norms.append(grad_norm)
        times.append(time.time() - start_time)
        step_sizes.append(alpha)

        if verbose:
            print(f"{iter_count:^5} | {f_x:^15.6e} | {grad_norm:^15.6e} | {alpha:^10.6e}")

    success = grad_norm < tol
    termination_reason = "Gradient norm tolerance reached" if success else "Max iterations reached"

    info = {
        'iterations': iter_count,
        'f_values': f_history,
        'grad_norms': grad_norms,
        'times': times,
        'step_sizes': step_sizes,
        'success': success,
        'termination_reason': termination_reason
    }
    if f_opt is not None:
         info['f_gaps'] = [f - f_opt for f in f_history]

    return x, f_x, info

def conjugate_gradient_subproblem(B, g, delta, options):
    """
    Steihaug Conjugate Gradient method for solving the trust region subproblem:
    min m(p) = f + g.T*p + 0.5*p.T*B*p   s.t. ||p|| <= delta

    """
    n = len(g)
    p = np.zeros(n)
    r = -g.copy() # Residual starts as -g
    d = r.copy()  # Search direction
    g_norm = LA.norm(g)

    max_iter_cg = options.get('max_iterations_CG', max(10, n)) # Default max CG iterations
    # Relative tolerance based on Nocedal & Wright p. 171
    tol_cg_rel = options.get('term_tol_CG_rel', min(0.5, np.sqrt(g_norm)))
    tol_cg = tol_cg_rel * g_norm

    if g_norm < 1e-10: # If gradient is near zero, return zero step
        return p

    for j in range(max_iter_cg):
        Bd = B @ d
        dBd = d @ Bd # Curvature d.T * B * d

        # Check for non-positive curvature
        if dBd <= 1e-10:
            # Find positive tau such that ||p + tau*d|| = delta (hit boundary along d)
            a = d @ d
            b = 2 * (p @ d)
            c = (p @ p) - delta**2
            discriminant = b**2 - 4*a*c
            # We need tau > 0
            if discriminant >= 0:
                 tau1 = (-b + np.sqrt(discriminant)) / (2*a)
                 tau2 = (-b - np.sqrt(discriminant)) / (2*a)
                 tau = max(tau1, tau2) # Take the larger root for positive tau
                 if tau < 0: # If both roots are negative or zero, something is wrong or p is already on boundary
                      # This case implies we should have stopped earlier or d is orthogonal to p on boundary
                      # For simplicity, return current p (or p=p, tau=0)
                      return p
            else:
                # Should not happen if p is strictly inside the region.
                # If p is on boundary, might indicate d points outwards or tangentially.
                # Return current p as the boundary point.
                return p

            p = p + tau * d
            return p # Terminate due to non-positive curvature

        alpha = (r @ r) / dBd
        p_next = p + alpha * d

        # Check if step leaves the trust region boundary
        if LA.norm(p_next) >= delta:
            # Find positive tau such that ||p + tau*d|| = delta (intersect boundary)
            a = d @ d
            b = 2 * (p @ d)
            c = (p @ p) - delta**2
            discriminant = b**2 - 4*a*c
            if discriminant >= 0:
                 tau1 = (-b + np.sqrt(discriminant)) / (2*a)
                 tau2 = (-b - np.sqrt(discriminant)) / (2*a)
                 tau = max(tau1, tau2) # Need tau > 0
                 if tau < 0:
                     return p # Should not happen if p is inside, return p
            else:
                return p # No intersection, return p

            p = p + tau * d
            return p # Terminate: Hit boundary

        # Update iterate, residual, and direction
        p = p_next
        r_new = r - alpha * Bd

        # Check for convergence
        if LA.norm(r_new) < tol_cg:
            return p # Converged within tolerance

        beta = (r_new @ r_new) / (r @ r)
        d = r_new + beta * d
        r = r_new

    return p # Max CG iterations reached


def trust_region_newton(problem, options):
    """
    Trust region Newton method with CG subproblem solver.
    """
    # Determine problem dimension
    n = len(problem['x0'])
    
    # Use more conservative initial radius for small dimension problems
    delta_init_default = 1.0 if n >= 100 else 0.5
    delta = options.get('delta_init', delta_init_default)
    max_delta = options.get('delta_max', 10.0 * delta_init_default)
    
    # Adjusted parameters for better stability
    eta = options.get('eta', 0.2)          # Increased acceptance threshold
    c1_tr = options.get('c1_tr', 0.5)      # Less aggressive shrink
    c2_tr = options.get('c2_tr', 1.5)      # Less aggressive expansion
    
    # Add damping factor to slow down radius updates
    radius_damping = options.get('radius_damping', 0.8)
    shrink_thresh  = options.get('shrink_thresh', 0.10)
    expand_thresh  = options.get('expand_thresh', 0.90)
    radius_damping = options.get('radius_damping', 0.95)

    x = problem['x0'].copy()
    func = problem['func']
    grad = problem['grad']
    hess = problem['hess']
    f_opt = problem.get('f_opt')

    max_iter = options['max_iterations']
    tol = options['term_tol']
    verbose = options.get('verbose', False)

    f_x = func(x)
    grad_x = grad(x)
    grad_norm = LA.norm(grad_x)

    iter_count = 0
    f_history = [f_x]
    grad_norms = [grad_norm]
    times = [0.0]
    tr_radii = [delta]
    rho_values = []
    start_time = time.time()

    if verbose:
        print(f"{'Iter':^5} | {'f(x)':^15} | {'||∇f(x)||':^15} | {'Δ':^10} | {'ρ':^10}")
        print("-" * 65)
        print(f"{iter_count:^5} | {f_x:^15.6e} | {grad_norm:^15.6e} | {delta:^10.6e} | {'---':^10}")

    while iter_count < max_iter and grad_norm >= tol:
        H = hess(x)

        # Solve the trust region subproblem using Steihaug CG
        p = conjugate_gradient_subproblem(H, grad_x, delta, options)

        # Calculate actual reduction and predicted reduction
        f_new = func(x + p)
        actual_reduction = f_x - f_new
        # Predicted reduction from quadratic model m(p) - m(0)
        pred_reduction = -(grad_x @ p) - 0.5 * (p @ H @ p)

        # Calculate agreement ratio rho
        if abs(pred_reduction) < 1e-12:
            # If predicted reduction is tiny, rho depends on actual reduction sign
            rho = 1.0 if actual_reduction >= -1e-12 else -1.0
            if verbose and actual_reduction < -1e-12:
                 print(f"Warning: Actual reduction negative ({actual_reduction:.2e}) while predicted reduction near zero ({pred_reduction:.2e})")
        else:
            rho = actual_reduction / pred_reduction

        # Calculate new trust region radius with damping
        old_delta = delta
        
        # Update trust region radius based on rho with damping
        if rho < shrink_thresh:
            new_delta = c1_tr * delta
            delta = old_delta*radius_damping + new_delta*(1-radius_damping)
        elif rho > expand_thresh and LA.norm(p) > (1-1e-6)*delta:
            new_delta = min(c2_tr*delta, max_delta)
            delta = old_delta*radius_damping + new_delta*(1-radius_damping)

        # Otherwise (0.25 <= rho <= 0.75), delta remains unchanged

        # Ensure delta doesn't become excessively small
        delta = max(delta, 1e-8)

        # Update iterate if step is acceptable
        if rho > eta: # Using the adjusted eta threshold
            x = x + p
            f_x = f_new
            grad_x = grad(x) # Recompute gradient
            grad_norm = LA.norm(grad_x)
        # Else: x remains the same, f_x, grad_x remain the same

        iter_count += 1

        # Store history
        f_history.append(f_x)
        grad_norms.append(grad_norm)
        times.append(time.time() - start_time)
        tr_radii.append(delta)
        rho_values.append(rho)

        if verbose:
            print(f"{iter_count:^5} | {f_x:^15.6e} | {grad_norm:^15.6e} | {delta:^10.6e} | {rho:^10.6e}")

    success = grad_norm < tol
    termination_reason = "Gradient norm tolerance reached" if success else "Max iterations reached"

    info = {
        'iterations': iter_count,
        'f_values': f_history,
        'grad_norms': grad_norms,
        'times': times,
        'tr_radii': tr_radii,
        'rho_values': rho_values,
        'success': success,
        'termination_reason': termination_reason
    }
    if f_opt is not None:
         info['f_gaps'] = [f - f_opt for f in f_history]


    return x, f_x, info

def trust_region_sr1(problem, options):
    """
    Trust region SR1 quasi-Newton method with CG subproblem solver.
    """
    x = problem['x0'].copy()
    func = problem['func']
    grad = problem['grad']
    f_opt = problem.get('f_opt')
    n = len(x)

    # Initialize SR1 Hessian approximation
    B = np.eye(n) # Start with identity matrix

    # Use more conservative initial radius for small dimension problems
    delta_init_default = 1.0 if n >= 100 else 0.5
    delta = options.get('delta_init', delta_init_default)
    max_delta = options.get('delta_max', 10.0 * delta_init_default)
    
    # Adjusted parameters for better stability
    eta = options.get('eta', 0.2)          # Increased acceptance threshold
    c1_tr = options.get('c1_tr', 0.5)      # Less aggressive shrink
    c2_tr = options.get('c2_tr', 1.5)      # Less aggressive expansion
    
    # Add damping factor to slow down radius updates
    radius_damping = options.get('radius_damping', 0.8)
    
    # Regularization parameters for SR1 update
    sr1_threshold = options.get('sr1_tol', 1e-8) # Threshold for SR1 update
    reg_init = options.get('sr1_reg_init', 1e-6) # Initial regularization parameter
    reg_update = options.get('sr1_reg_update', 1.2) # Regularization increase factor
    reg_param = reg_init  # Initialize regularization parameter
    
    # Careful step acceptance parameters
    strict_acceptance = options.get('strict_acceptance', True) # Whether to use stricter acceptance criteria
    rho_accept_threshold = options.get('rho_accept', 0.1)  # Default is 0.1 (eta)
    consecutive_rejects = 0
    max_consecutive_rejects = options.get('max_rejects', 3)
    
    # Adaptive trust region parameters
    use_adaptive_tr = options.get('adaptive_tr', True)
    tr_history_size = options.get('tr_history', 5)
    rho_history = []

    max_iter = options['max_iterations']
    tol = options['term_tol']
    verbose = options.get('verbose', False)

    f_x = func(x)
    grad_x = grad(x)
    grad_norm = LA.norm(grad_x)

    iter_count = 0
    f_history = [f_x]
    grad_norms = [grad_norm]
    times = [0.0]
    tr_radii = [delta]
    rho_values = []
    start_time = time.time()

    if verbose:
        print(f"{'Iter':^5} | {'f(x)':^15} | {'||∇f(x)||':^15} | {'Δ':^10} | {'ρ':^10}")
        print("-" * 65)
        print(f"{iter_count:^5} | {f_x:^15.6e} | {grad_norm:^15.6e} | {delta:^10.6e} | {'---':^10}")

    while iter_count < max_iter and grad_norm >= tol:
        # Solve the trust region subproblem using CG
        p = conjugate_gradient_subproblem(B, grad_x, delta, options)

        # Evaluate trial point
        x_new = x + p
        f_new = func(x_new)
        grad_new = grad(x_new)

        # Calculate actual and predicted reduction
        actual_reduction = f_x - f_new
        # Predicted reduction from quadratic model m(p) = f_k + g_k.T*p + 0.5*p.T*B_k*p
        pred_reduction = -(grad_x @ p) - 0.5 * (p @ B @ p)

        if abs(pred_reduction) < 1e-12:
            rho = 1.0 if actual_reduction >= -1e-12 else -1.0
        else:
            rho = actual_reduction / pred_reduction

        # Calculate new trust region radius with damping
        old_delta = delta
        
        # Update trust region radius based on rho with damping
        if rho < 0.25:
            # Apply damping when decreasing radius
            new_delta = c1_tr * delta
            delta = old_delta * radius_damping + new_delta * (1 - radius_damping)
        elif rho > 0.75 and LA.norm(p) > (1.0 - 1e-6) * delta:
            # Apply damping when increasing radius
            new_delta = min(c2_tr * delta, max_delta)
            delta = old_delta * radius_damping + new_delta * (1 - radius_damping)
        # Otherwise (0.25 <= rho <= 0.75), delta remains unchanged

        # Ensure delta doesn't become excessively small
        delta = max(delta, 1e-8)
        
        # Save rho for adaptive trust region management
        if use_adaptive_tr:
            rho_history.append(rho)
            if len(rho_history) > tr_history_size:
                rho_history.pop(0)
                
            # Adjust parameters based on history
            if len(rho_history) >= 3:
                recent_rhos = np.array(rho_history[-3:])
                # If we're seeing oscillating behavior
                if np.std(recent_rhos) > 0.5:
                    # Increase damping to stabilize
                    radius_damping = min(0.95, radius_damping * 1.05)
                else:
                    # Gradually reduce damping when stable
                    radius_damping = max(0.7, radius_damping * 0.99)

        # Update iterate if step is acceptable
        if rho > eta:
            # Actually move to new point
            s = p  # Step vector
            y = grad_new - grad_x  # Gradient difference
            x = x_new
            f_x = f_new
            grad_x = grad_new
            grad_norm = LA.norm(grad_x)
            
            # SR1 update of Hessian approximation
            Bs = B @ s
            diff = y - Bs
            denom = diff @ s
            
            # Skip update if denominator is too small or curvature is negative
            if abs(denom) >= sr1_threshold * LA.norm(diff) * LA.norm(s):
                # Standard SR1 update formula
                B = B + np.outer(diff, diff) / denom
            
            # Reset rejection counter
            consecutive_rejects = 0
        else:
            # Step rejected, increment counter
            consecutive_rejects += 1
            
            # If too many consecutive rejections, try increasing regularization
            if consecutive_rejects >= max_consecutive_rejects:
                reg_param *= reg_update
                B = B + reg_param * np.eye(n)
                consecutive_rejects = 0
        
        iter_count += 1
        
        # Store history
        f_history.append(f_x)
        grad_norms.append(grad_norm)
        times.append(time.time() - start_time)
        tr_radii.append(delta)
        rho_values.append(rho)
        
        if verbose:
            print(f"{iter_count:^5} | {f_x:^15.6e} | {grad_norm:^15.6e} | {delta:^10.6e} | {rho:^10.6e}")
    
    success = grad_norm < tol
    termination_reason = "Gradient norm tolerance reached" if success else "Max iterations reached"
    
    info = {
        'iterations': iter_count,
        'f_values': f_history,
        'grad_norms': grad_norms,
        'times': times,
        'tr_radii': tr_radii,
        'rho_values': rho_values,
        'success': success,
        'termination_reason': termination_reason
    }
    if f_opt is not None:
        info['f_gaps'] = [f - f_opt for f in f_history]
    
    return x, f_x, info

def bfgs(problem, options, backtracking=True):
    """
    BFGS quasi-Newton method with specified line search.
    """
    x = problem['x0'].copy()
    func = problem['func']
    grad = problem['grad']
    f_opt = problem.get('f_opt')
    n = len(x)

    # Initialize BFGS Hessian approximation
    B = np.eye(n) # Start with identity

    line_search = backtracking_line_search if backtracking else wolfe_line_search

    max_iter = options['max_iterations']
    tol = options['term_tol']
    verbose = options.get('verbose', False)

    f_x = func(x)
    grad_x = grad(x)
    grad_norm = LA.norm(grad_x)

    iter_count = 0
    f_history = [f_x]
    grad_norms = [grad_norm]
    times = [0.0]
    step_sizes = []
    start_time = time.time()

    if verbose:
        print(f"{'Iter':^5} | {'f(x)':^15} | {'||∇f(x)||':^15} | {'α':^10}")
        print("-" * 50)
        print(f"{iter_count:^5} | {f_x:^15.6e} | {grad_norm:^15.6e} | {'---':^10}")

    while iter_count < max_iter and grad_norm >= tol:
        # Solve B*d = -grad_x for search direction d
        try:
             # Check if B is positive definite enough before solving
             # Add a small regularization if needed (more robust than just try-except)
             min_eig_B = min(np.real(LA.eigvals(B))) if n > 0 else 1.0
             if min_eig_B <= 1e-8:
                  print(f"Warning: BFGS matrix B nearly singular/indefinite at iter {iter_count}. Resetting.")
                  B = np.eye(n)
                  d = -grad_x # Use steepest descent if reset
             else:
                  d = LA.solve(B, -grad_x)

             # Ensure descent direction
             if np.dot(grad_x, d) >= 0:
                  print(f"Warning: BFGS direction not descent at iter {iter_count}. Resetting.")
                  B = np.eye(n)
                  d = -grad_x


        except LA.LinAlgError:
             # Fallback if solve fails even after potential reset
             print(f"Warning: BFGS matrix solve failed at iteration {iter_count}. Resetting.")
             B = np.eye(n)
             d = -grad_x

        # Perform line search
        alpha, f_new, grad_new = line_search(func, grad, x, d, f_x, grad_x, options)

        # Update iterate
        x_new = x + alpha * d
        s = x_new - x
        y = grad_new - grad_x

        # Update BFGS matrix if curvature condition holds
        ys = y @ s
        if ys > 1e-10: # Check curvature condition
            Bs = B @ s
            sBs = s @ Bs
            if abs(sBs) > 1e-12: # Avoid division by zero
                 B = B - np.outer(Bs, Bs)/sBs + np.outer(y, y)/ys # BFGS update formula
            else:
                 print(f"Warning: Denominator sBs near zero at iteration {iter_count}. Skipping BFGS update.")

        else:
            print(f"Warning: Curvature condition y.T*s <= 0 ({ys:.3e}) at iteration {iter_count}. Skipping BFGS update.")


        x = x_new
        f_x = f_new
        grad_x = grad_new
        grad_norm = LA.norm(grad_x)

        iter_count += 1

        f_history.append(f_x)
        grad_norms.append(grad_norm)
        times.append(time.time() - start_time)
        step_sizes.append(alpha)

        if verbose:
            print(f"{iter_count:^5} | {f_x:^15.6e} | {grad_norm:^15.6e} | {alpha:^10.6e}")

    success = grad_norm < tol
    termination_reason = "Gradient norm tolerance reached" if success else "Max iterations reached"

    info = {
        'iterations': iter_count,
        'f_values': f_history,
        'grad_norms': grad_norms,
        'times': times,
        'step_sizes': step_sizes,
        'success': success,
        'termination_reason': termination_reason
    }
    if f_opt is not None:
         info['f_gaps'] = [f - f_opt for f in f_history]


    return x, f_x, info

def dfp(problem, options, backtracking=True):
    """
    DFP quasi-Newton method with specified line search.
    Updates the inverse Hessian approximation H_inv.
    """
    x = problem['x0'].copy()
    func = problem['func']
    grad = problem['grad']
    f_opt = problem.get('f_opt')
    n = len(x)

    # Initialize DFP inverse Hessian approximation
    H_inv = np.eye(n) # Start with identity

    line_search = backtracking_line_search if backtracking else wolfe_line_search

    max_iter = options['max_iterations']
    tol = options['term_tol']
    verbose = options.get('verbose', False)

    f_x = func(x)
    grad_x = grad(x)
    grad_norm = LA.norm(grad_x)

    iter_count = 0
    f_history = [f_x]
    grad_norms = [grad_norm]
    times = [0.0]
    step_sizes = []
    start_time = time.time()

    if verbose:
        print(f"{'Iter':^5} | {'f(x)':^15} | {'||∇f(x)||':^15} | {'α':^10}")
        print("-" * 50)
        print(f"{iter_count:^5} | {f_x:^15.6e} | {grad_norm:^15.6e} | {'---':^10}")

    while iter_count < max_iter and grad_norm >= tol:
        # Compute search direction d = -H_inv * grad_x
        d = -H_inv @ grad_x

        # Ensure descent direction
        if np.dot(grad_x, d) >= 0:
             print(f"Warning: DFP search direction not descent at iteration {iter_count}. Resetting.")
             d = -grad_x
             H_inv = np.eye(n) # Reset H_inv


        # Perform line search
        alpha, f_new, grad_new = line_search(func, grad, x, d, f_x, grad_x, options)

        x_new = x + alpha * d
        s = x_new - x
        y = grad_new - grad_x

        # Update DFP inverse Hessian if curvature condition holds
        ys = y @ s
        if ys > 1e-10: # Check curvature condition
            Hy = H_inv @ y
            yHy = y @ Hy
            if abs(yHy) > 1e-12: # Avoid division by zero
                 # DFP update formula: H = H - (Hyy'H)/(y'Hy) + (ss')/(y's)
                 H_inv = H_inv - np.outer(Hy, Hy)/yHy + np.outer(s, s)/ys
            else:
                 print(f"Warning: Denominator yHy near zero at iteration {iter_count}. Skipping DFP update.")
        else:
            print(f"Warning: Curvature condition y.T*s <= 0 ({ys:.3e}) at iteration {iter_count}. Skipping DFP update.")


        x = x_new
        f_x = f_new
        grad_x = grad_new
        grad_norm = LA.norm(grad_x)

        iter_count += 1

        f_history.append(f_x)
        grad_norms.append(grad_norm)
        times.append(time.time() - start_time)
        step_sizes.append(alpha)

        if verbose:
            print(f"{iter_count:^5} | {f_x:^15.6e} | {grad_norm:^15.6e} | {alpha:^10.6e}")

    success = grad_norm < tol
    termination_reason = "Gradient norm tolerance reached" if success else "Max iterations reached"

    info = {
        'iterations': iter_count,
        'f_values': f_history,
        'grad_norms': grad_norms,
        'times': times,
        'step_sizes': step_sizes,
        'success': success,
        'termination_reason': termination_reason
    }
    if f_opt is not None:
         info['f_gaps'] = [f - f_opt for f in f_history]


    return x, f_x, info

# --- L-BFGS Algorithm ---
def run_lbfgs_with_strategy(problem, options, memory_size=5, removal_strategy=RemovalStrategy.FIFO, backtracking=True):
    """
    L-BFGS implementation with different removal strategies.
    """
    x = problem['x0'].copy()
    func = problem['func']
    grad = problem['grad']
    f_opt = problem.get('f_opt')
    n = len(x)

    max_iter = options['max_iterations']
    tol = options['term_tol']
    verbose = options.get('verbose', False)

    line_search = backtracking_line_search if backtracking else wolfe_line_search

    # L-BFGS storage using deques
    s_vectors = deque(maxlen=memory_size)
    y_vectors = deque(maxlen=memory_size)
    rho_values = deque(maxlen=memory_size)
    # Store curvature values only if needed for strategy
    curvature_values = deque(maxlen=memory_size) if removal_strategy != RemovalStrategy.FIFO else None

    f_x = func(x)
    grad_x = grad(x)
    grad_norm = LA.norm(grad_x)

    iter_count = 0
    f_history = [f_x]
    grad_norms = [grad_norm]
    times = [0.0]
    step_sizes = []
    start_time = time.time()

    if verbose:
        print(f"{'Iter':^5} | {'f(x)':^15} | {'||∇f(x)||':^15} | {'α':^10}")
        print("-" * 50)
        print(f"{iter_count:^5} | {f_x:^15.6e} | {grad_norm:^15.6e} | {'---':^10}")

    while iter_count < max_iter and grad_norm >= tol:
        # L-BFGS two-loop recursion
        q = grad_x.copy()
        alpha_storage = {} # Use dict for storing alphas by index

        # First loop (backward) - Use indexes 0 to k-1 for storage
        for i in range(len(s_vectors) - 1, -1, -1):
            s_i = s_vectors[i]
            y_i = y_vectors[i]
            rho_i = rho_values[i]
            alpha_i = rho_i * np.dot(s_i, q)
            alpha_storage[i] = alpha_i
            q = q - alpha_i * y_i

        # Initial Hessian approximation (scaling factor gamma)
        gamma = 1.0 # Default scaling
        if len(s_vectors) > 0:
            s_last = s_vectors[-1]
            y_last = y_vectors[-1]
            ys = np.dot(y_last, s_last)
            if ys > 1e-10:
                gamma = ys / np.dot(y_last, y_last)

        r = gamma * q

        # Second loop (forward) - Use indexes 0 to k-1 for storage
        for i in range(len(s_vectors)):
            s_i = s_vectors[i]
            y_i = y_vectors[i]
            rho_i = rho_values[i]
            alpha_i = alpha_storage[i]
            beta = rho_i * np.dot(y_i, r)
            r = r + s_i * (alpha_i - beta)

        d = -r # Search direction

        # Perform line search
        alpha, f_new, grad_new = line_search(func, grad, x, d, f_x, grad_x, options)

        x_new = x + alpha * d
        s = x_new - x
        y = grad_new - grad_x

        # Check curvature condition
        ys = np.dot(y, s)

        if ys > 1e-10:
             # Check if memory is full and if FIFO is not the strategy
             if len(s_vectors) == memory_size and removal_strategy != RemovalStrategy.FIFO:
                 # --- Removal Strategies ---
                 if removal_strategy == RemovalStrategy.MIN_CURVATURE:
                     # Find index of pair with minimum curvature y^T*s
                     if curvature_values: # Check if curvature_values is populated
                         min_idx = np.argmin(list(curvature_values))
                     else:
                         min_idx = 0 # Default to removing oldest if curvatures not stored
                 elif removal_strategy == RemovalStrategy.ADAPTIVE:
                      # Estimate contribution (simple heuristic: based on magnitude of rho*s.dot(g))
                      if s_vectors: # Check if deques are populated
                           contributions = [abs(rho_values[i] * np.dot(s_vectors[i], grad_x)) for i in range(len(s_vectors))]
                           min_idx = np.argmin(contributions)
                      else:
                           min_idx = 0
                 else:
                      min_idx = 0 # Should not happen, default to FIFO behavior


                 # Efficiently remove element at min_idx from deque without full conversion
                 if 0 <= min_idx < len(s_vectors):
                      s_vectors_list = list(s_vectors)
                      y_vectors_list = list(y_vectors)
                      rho_values_list = list(rho_values)

                      s_vectors_list.pop(min_idx) 
                      y_vectors_list.pop(min_idx) 
                      rho_values_list.pop(min_idx) 

                      if curvature_values is not None:
                           curv_list = list(curvature_values)
                           curv_list.pop(min_idx)
                           curvature_values = deque(curv_list, maxlen=memory_size)

                      # Rebuild deques using the corrected list names
                      s_vectors = deque(s_vectors_list, maxlen=memory_size)
                      y_vectors = deque(y_vectors_list, maxlen=memory_size)
                      rho_values = deque(rho_values_list, maxlen=memory_size)



             # Append new pair (deque handles maxlen automatically for FIFO)
             # If non-FIFO and memory was full, space was already made
             s_vectors.append(s)
             y_vectors.append(y)
             rho_values.append(1.0 / ys)
             if curvature_values is not None:
                  curvature_values.append(ys)
        else:
             print(f"Warning: Curvature condition y.T*s <= 0 ({ys:.3e}) at iteration {iter_count}. Skipping L-BFGS update.")


        x = x_new
        f_x = f_new
        grad_x = grad_new
        grad_norm = LA.norm(grad_x)

        iter_count += 1

        f_history.append(f_x)
        grad_norms.append(grad_norm)
        times.append(time.time() - start_time)
        step_sizes.append(alpha)

        if verbose:
            print(f"{iter_count:^5} | {f_x:^15.6e} | {grad_norm:^15.6e} | {alpha:^10.6e}")

    success = grad_norm < tol
    termination_reason = "Gradient norm tolerance reached" if success else "Max iterations reached"

    info = {
        'iterations': iter_count,
        'f_values': f_history,
        'grad_norms': grad_norms,
        'times': times,
        'step_sizes': step_sizes,
        'success': success,
        'termination_reason': termination_reason,
        'memory_size': memory_size,
        'removal_strategy': removal_strategy.name
    }
    if f_opt is not None:
         info['f_gaps'] = [f - f_opt for f in f_history]

    return x, f_x, info