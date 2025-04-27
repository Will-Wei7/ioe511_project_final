# IOE 511/MATH 562, University of Michigan
# Problem definitions for optimization project

import numpy as np
import scipy.io
import os

class OptimizationProblem:
    """Container class for optimization problems."""
    def __init__(self, name, x0, func, grad, hess=None, f_opt=None):
        self.name = name
        self.x0 = np.array(x0, dtype=float)
        self.func = func
        self.grad = grad
        # Provide a default Hessian approximation if none is given
        self.hess = hess if hess is not None else lambda x: finite_diff_hess(func, x)
        self.f_opt = f_opt # Optimal function value if known

    def to_dict(self):
        """Convert to dictionary format for compatibility."""
        return {
            'name': self.name,
            'x0': self.x0,
            'func': self.func,
            'grad': self.grad,
            'hess': self.hess,
            'f_opt': self.f_opt
        }

def finite_diff_hess(func, x, eps=1e-5):
    """Approximate Hessian using finite differences."""
    n = len(x)
    hess = np.zeros((n, n))
    fx = func(x)
    eps_vec = np.eye(n) * eps

    for i in range(n):
        for j in range(i, n):
            # Central difference approximation
            f_pp = func(x + eps_vec[i] + eps_vec[j])
            f_pm = func(x + eps_vec[i] - eps_vec[j])
            f_mp = func(x - eps_vec[i] + eps_vec[j])
            f_mm = func(x - eps_vec[i] - eps_vec[j])

            if i == j:
                # Diagonal elements using (f(x+h) - 2f(x) + f(x-h)) / eps^2
                f_p = func(x + eps_vec[i])
                f_m = func(x - eps_vec[i])
                hess[i,i] = (f_p - 2*fx + f_m) / (eps**2)
            else:
                # Off-diagonal elements
                hess[i,j] = (f_pp - f_pm - f_mp + f_mm) / (4 * eps**2)
                hess[j,i] = hess[i,j] # Symmetric
    return hess


# --- Quadratic Problems ---
# Define path relative to the script file
script_dir = os.path.dirname(os.path.abspath(__file__))
# Assuming .mat files are in 'Project_Problems_PYTHON/Project_Problems_PYTHON'
# relative to the directory containing this script
# *** Adjust this relative path if your .mat files are elsewhere ***
mat_files_subdir = 'Project_Problems_PYTHON/Project_Problems_PYTHON'
QUAD_DATA_PATH = os.path.join(script_dir, mat_files_subdir)

# Basic check if the primary assumed path exists
if not os.path.isdir(QUAD_DATA_PATH):
     print(f"Warning: Default QUAD_DATA_PATH not found: {QUAD_DATA_PATH}")
     print(f"Attempting fallback relative to current working directory: {os.getcwd()}")
     # Fallback to current directory - adjust if needed
     QUAD_DATA_PATH = '.'


def get_quad_matrix(filename):
    """Loads Q matrix from .mat file from the defined QUAD_DATA_PATH."""
    filepath = os.path.join(QUAD_DATA_PATH, filename)
    if not os.path.exists(filepath):
        # If not found in primary path, try searching relative to the CWD as a fallback
        potential_paths = [
             os.path.join('./', filename),
             os.path.join('./Project_Problems_PYTHON/Project_Problems_PYTHON', filename),
             os.path.join('./511 Proj/Project_Problems_PYTHON/Project_Problems_PYTHON', filename),
             os.path.join('./511 Proj', filename)
        ]
        found_fallback = False
        for p in potential_paths:
             if os.path.exists(p):
                  filepath = p
                  found_fallback = True
                  print(f"Note: Found {filename} via fallback path relative to CWD: {p}")
                  break
        if not found_fallback:
             # Try one level up from script_dir as last resort before failing
             script_parent_dir = os.path.dirname(script_dir)
             filepath_alt = os.path.join(script_parent_dir, mat_files_subdir, filename)
             if os.path.exists(filepath_alt):
                  print(f"Note: Found {filename} via alternative script parent path: {filepath_alt}")
                  filepath = filepath_alt
             else:
                  raise FileNotFoundError(f"Required data file {filename} not found in primary path ({os.path.join(QUAD_DATA_PATH, filename)}), CWD fallbacks, or parent path ({filepath_alt}).")

    mat = scipy.io.loadmat(filepath)
    Q = mat['Q']
    if hasattr(Q, "toarray"): # Handle sparse matrices
        Q = Q.toarray()
    return Q


def create_quad_problem(name_suffix, dim, cond_num):
    """Factory function for quadratic problems."""
    filename = f'quad_{dim}_{cond_num}_Q.mat'
    Q = get_quad_matrix(filename)
    np.random.seed(0) # For reproducibility
    q = np.random.normal(size=(dim,1))

    # Specific starting points based on dimension
    if dim == 10:
        np.random.seed(0) # Re-seed for consistency if needed
        x0_val = 20 * np.random.rand(dim, 1) - 10
        x0_val = x0_val.flatten()
    elif dim == 1000:
        np.random.seed(0) # Re-seed for consistency if needed
        x0_val = 20 * np.random.rand(dim, 1) - 10
        x0_val = x0_val.flatten()
    else:
        x0_val = np.zeros(dim) # Default starting point

    def func(x):
        x_col = x.reshape(-1, 1)
        val = 0.5 * x_col.T @ Q @ x_col + q.T @ x_col
        return float(val.item())

    def grad(x):
        x_col = x.reshape(-1, 1)
        return (Q @ x_col + q).flatten()

    def quad_hess(x): # Actual Hessian
        return Q

    # Calculate f_opt if possible
    f_opt = None
    try:
        x_opt = -np.linalg.solve(Q, q)
        f_opt = func(x_opt)
    except np.linalg.LinAlgError:
        problem_name = f'quad_{dim}_{cond_num}' # Define for warning msg
        print(f"Warning: Matrix Q for {problem_name} is singular or ill-conditioned. Cannot compute exact f_opt via solve.")
        try:
            x_opt_pinv = -np.linalg.pinv(Q) @ q
            f_opt = func(x_opt_pinv)
            print(f"Using pseudo-inverse for f_opt calculation for {problem_name}.")
        except Exception as e:
             print(f"Could not compute f_opt using pseudo-inverse either for {problem_name}: {e}")


    return OptimizationProblem(
        name=f'quad_{dim}_{cond_num}',
        x0=x0_val,
        func=func,
        grad=grad,
        hess=quad_hess,
        f_opt=f_opt
    )

# --- Quartic Problems ---
def create_quartic_problem(name, sigma):
    """Factory function for quartic problems."""
    Q = np.array([
        [5, 1, 0, 0.5],
        [1, 4, 0.5, 0],
        [0, 0.5, 3, 0],
        [0.5, 0, 0, 2]
    ])
    n = 4

    # Specific starting point
    x0_val = np.array([np.cos(np.radians(70)), np.sin(np.radians(70)),
                       np.cos(np.radians(70)), np.sin(np.radians(70))])

    def func(x):
        if x.shape != (n,): x = x.flatten()
        return 0.5 * (x.T @ x) + (sigma / 4.0) * (x.T @ Q @ x)**2

    def grad(x):
        if x.shape != (n,): x = x.flatten()
        q_term_scalar = x.T @ Q @ x
        Qx = Q @ x
        return x + sigma * q_term_scalar * Qx

    def quartic_hess(x): # Actual Hessian
        if x.shape != (n,): x = x.flatten()
        q_term_scalar = x.T @ Q @ x
        Qx = Q @ x
        Qx_col = Qx.reshape(-1, 1)
        return np.eye(n) + sigma * (Qx_col @ Qx_col.T + q_term_scalar * Q)

    # Check if f_opt is likely 0
    eigvals = np.linalg.eigvalsh(Q)
    f_opt_val = 0.0 if np.all(eigvals > 1e-10) and sigma >= 0 else None # Added tolerance

    return OptimizationProblem(
        name=name,
        x0=x0_val,
        func=func,
        grad=grad,
        hess=quartic_hess,
        f_opt=f_opt_val
    )

# --- Rosenbrock Problems ---
def create_rosenbrock_problem(n=2):
    """Factory function for Rosenbrock problems."""
    if n < 2: raise ValueError("Rosenbrock requires n >= 2")

    def func(x):
        x = x.flatten()
        if n == 2:
            return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
        else: # Generalized Rosenbrock
            term1 = (1 - x[:-1])**2
            term2 = 100 * (x[1:] - x[:-1]**2)**2
            return np.sum(term1 + term2)

    def grad(x):
        x = x.flatten()
        g = np.zeros_like(x)
        if n == 2:
            g[0] = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
            g[1] = 200 * (x[1] - x[0]**2)
        else: # Generalized gradient
            term_in_paren = x[1:] - x[:-1]**2
            g[:-1] = -2 * (1 - x[:-1]) - 400 * x[:-1] * term_in_paren
            g[1:] = 200 * term_in_paren
        return g

    # Specific starting points
    if n == 2:
        x0_val = np.array([-1.2, 1.0])
        problem_name = 'Rosenbrock_2'
    elif n == 100:
        x0_val = np.full(n, 1.0)
        x0_val[0] = -1.2
        problem_name = 'Rosenbrock_100'
    else: # Default starting point for other n
        x0_val = np.zeros(n)
        x0_val[0] = -1.2
        x0_val[1::2] = 1.0
        problem_name = f'Rosenbrock_{n}'

    f_opt_val = 0.0 # Known optimal value

    return OptimizationProblem(
        name=problem_name,
        x0=x0_val,
        func=func,
        grad=grad,
        hess=None, # Use default finite difference Hessian
        f_opt=f_opt_val
    )


# --- DataFit Problem ---
def create_datafit_problem():
    """Creates the DataFit_2 problem."""
    y = np.array([1.5, 2.25, 2.625])
    n = 2

    def func(x):
        x = x.flatten()
        if len(x) != n: raise ValueError(f"DataFit requires n={n}")
        i_vals = np.arange(1, len(y) + 1)
        residuals = y - x[0] * (1 - x[1]**i_vals)
        return np.sum(residuals**2)

    def grad(x):
        x = x.flatten()
        if len(x) != n: raise ValueError(f"DataFit requires n={n}")
        g = np.zeros(n)
        i_vals = np.arange(1, len(y) + 1)
        residuals = y - x[0] * (1 - x[1]**i_vals)

        # Gradient w.r.t x[0]
        dr_dx1 = -(1 - x[1]**i_vals)
        g[0] = np.sum(2 * residuals * dr_dx1)

        # Gradient w.r.t x[1]
        # d(r_i)/dx1 = x1 * i * x2^(i-1)
        # Handle x2=0 case
        if np.abs(x[1]) < 1e-12 and 1 in i_vals:
             dr_dx2 = np.zeros_like(i_vals, dtype=float)
             for idx, i in enumerate(i_vals):
                  dr_dx2[idx] = x[0] * i * (1.0 if i==1 else 0.0)
        else:
            dr_dx2 = x[0] * i_vals * x[1]**(i_vals - 1)
        g[1] = np.sum(2 * residuals * dr_dx2)

        return g

    # Specific starting point
    x0_val = np.array([1.0, 1.0])
    f_opt_val = None # Optimal value unknown

    return OptimizationProblem(
        name='DataFit_2',
        x0=x0_val,
        func=func,
        grad=grad,
        hess=None, # Use default finite difference Hessian
        f_opt=f_opt_val
    )


# --- Exponential Problems ---
def create_exponential_problem(n=10):
    """Factory function for Exponential problems."""
    if n < 2: raise ValueError("Exponential problem requires n >= 2")

    def func(x):
        x = x.flatten()
        if len(x) != n: raise ValueError(f"Exponential problem {n} requires n={n}")

        exp_x0 = np.exp(x[0])
        term1 = (exp_x0 - 1) / (exp_x0 + 1) + 0.1 * np.exp(-x[0])
        term2 = np.sum((x[1:] - 1)**4) if n > 1 else 0
        return term1 + term2

    def grad(x):
        x = x.flatten()
        if len(x) != n: raise ValueError(f"Exponential problem {n} requires n={n}")
        g = np.zeros(n)

        # Gradient for the first term (affects g[0])
        exp_x0 = np.exp(x[0])
        exp_neg_x0 = np.exp(-x[0])
        g[0] = (2 * exp_x0 / ((exp_x0 + 1)**2)) - (0.1 * exp_neg_x0)

        # Gradient for the second term (affects g[1:])
        if n > 1:
            g[1:] = 4 * (x[1:] - 1)**3
        return g

    # Specific starting point: [1, 0, ..., 0]
    x0_val = np.zeros(n)
    x0_val[0] = 1.0
    f_opt_val = None # Optimal value unknown
    problem_name = f'Exponential_{n}'

    return OptimizationProblem(
        name=problem_name,
        x0=x0_val,
        func=func,
        grad=grad,
        hess=None, # Use default finite difference Hessian
        f_opt=f_opt_val
    )


# --- Genhumps Problem ---
def create_genhumps_problem(n=5):
    """Factory function for genhumps problem."""
    if n < 2: raise ValueError("Genhumps requires n >= 2")
    def func(x):
        x = x.flatten()
        if len(x) != n: raise ValueError(f"Genhumps problem requires n={n}")
        f = 0.0
        for i in range(n - 1): # Sum i=1 to n-1
            sin_2xi = np.sin(2*x[i])
            sin_2xip1 = np.sin(2*x[i+1])
            f += sin_2xi**2 * sin_2xip1**2 + 0.05*(x[i]**2 + x[i+1]**2)
        return f

    def grad(x):
        x = x.flatten()
        if len(x) != n: raise ValueError(f"Genhumps problem requires n={n}")
        g = np.zeros(n)
        # Loop i from 0 to n-2 (for terms involving i and i+1)
        for i in range(n - 1):
            xi, xip1 = x[i], x[i+1]
            sin_2xi, cos_2xi = np.sin(2*xi), np.cos(2*xi)
            sin_2xip1, cos_2xip1 = np.sin(2*xip1), np.cos(2*xip1)
            # Derivative of term i w.r.t. x[i]
            g[i] += 4 * sin_2xi * cos_2xi * sin_2xip1**2 + 0.1 * xi
            # Derivative of term i w.r.t. x[i+1]
            g[i+1] += sin_2xi**2 * 4 * sin_2xip1 * cos_2xip1 + 0.1 * xip1
        return g

    # Specific starting point: [-506.2, 506.2, ..., 506.2]
    x0_val = np.full(n, 506.2)
    x0_val[0] = -506.2
    f_opt_val = None # Optimal value unknown
    hess_func = None # Use default finite difference Hessian

    return OptimizationProblem(
        name=f'Genhumps_{n}',
        x0=x0_val,
        func=func,
        grad=grad,
        hess=hess_func,
        f_opt=f_opt_val
    )

# --- Powell Problem ---
def create_powell_problem():
    """Creates the Powell singular function problem (n=4)."""
    n=4
    def func(x):
        x=x.flatten()
        if len(x) != n: raise ValueError("Powell requires n=4")
        t1 = x[0] + 10*x[1]
        t2 = x[2] - x[3]
        t3 = x[1] - 2*x[2]
        t4 = x[0] - x[3]
        return t1**2 + 5*t2**2 + t3**4 + 10*t4**4

    def grad(x):
        x=x.flatten()
        if len(x) != n: raise ValueError("Powell requires n=4")
        g = np.zeros(n)
        t1 = x[0] + 10*x[1]
        t2 = x[2] - x[3]
        t3 = x[1] - 2*x[2]
        t4 = x[0] - x[3]
        g[0] = 2*t1 + 40*t4**3
        g[1] = 20*t1 + 4*t3**3
        g[2] = 10*t2 - 8*t3**3
        g[3] = -10*t2 - 40*t4**3
        return g

    x0_val=np.array([3.0, -1.0, 0.0, 1.0]) # Standard starting point
    f_opt_val=0.0 # Known minimum value

    return OptimizationProblem(
        name='Powell',
        x0=x0_val,
        func=func,
        grad=grad,
        hess=None, # Use default finite diff
        f_opt=f_opt_val
    )

# --- Ill-Conditioned Quadratic ---
def create_ill_conditioned_problem(n=10, condition_number=1e6):
    """Creates an ill-conditioned quadratic problem."""
    np.random.seed(42) # For reproducibility
    A = np.random.randn(n, n)
    Q_orth, _ = np.linalg.qr(A)
    log_cond = np.log10(condition_number)
    diag_vals = np.logspace(0, log_cond, n)
    H_mat = Q_orth @ np.diag(diag_vals) @ Q_orth.T # Hessian matrix
    b = np.random.randn(n)
    problem_name = f'IllConditioned_{n}_{int(log_cond)}'

    def func(x):
         x_col = x.reshape(-1, 1)
         b_col = b.reshape(-1, 1)
         val = 0.5 * x_col.T @ H_mat @ x_col - b_col.T @ x_col
         return float(val.item())

    def grad(x):
         x_col = x.reshape(-1, 1)
         b_col = b.reshape(-1, 1)
         return (H_mat @ x_col - b_col).flatten() # Gradient is Hx - b

    def ill_cond_hess(x): # Actual Hessian
         return H_mat

    # Calculate f_opt
    f_opt = None
    try:
        x_opt = np.linalg.solve(H_mat, b)
        f_opt = -0.5 * b.T @ x_opt # Simplified calculation
    except np.linalg.LinAlgError:
        print(f"Warning: Matrix H for {problem_name} is singular or ill-conditioned. Cannot compute exact f_opt via solve.")
        try:
            x_opt_pinv = np.linalg.pinv(H_mat) @ b
            f_opt = -0.5 * b.T @ x_opt_pinv
            print(f"Using pseudo-inverse for f_opt calculation for {problem_name}.")
        except Exception as e:
             print(f"Could not compute f_opt using pseudo-inverse either for {problem_name}: {e}")

    x0_val=np.ones(n) # Standard starting point

    return OptimizationProblem(
        name=problem_name,
        x0=x0_val,
        func=func,
        grad=grad,
        hess=ill_cond_hess,
        f_opt=f_opt
    )


# --- Problem dictionary ---
# Uses lambda functions to delay instantiation until requested
PROBLEMS_DICT = {
    # Course-defined problems
    'quad_10_10': lambda: create_quad_problem('quad_10_10', 10, 10),
    'quad_10_1000': lambda: create_quad_problem('quad_10_1000', 10, 1000),
    'quad_1000_10': lambda: create_quad_problem('quad_1000_10', 1000, 10),
    'quad_1000_1000': lambda: create_quad_problem('quad_1000_1000', 1000, 1000),
    'quartic_1': lambda: create_quartic_problem('P5_quartic_1', 1e-4),
    'quartic_2': lambda: create_quartic_problem('P6_quartic_2', 1e4),
    'Rosenbrock_2': lambda: create_rosenbrock_problem(n=2),
    'Rosenbrock_100': lambda: create_rosenbrock_problem(n=100),
    'DataFit_2': create_datafit_problem,
    'Exponential_10': lambda: create_exponential_problem(n=10),
    # Note: Discrepancy in original problem P11 name vs dimension.
    # Offering both based on interpretation.
    'Exponential_1000': lambda: create_exponential_problem(n=1000), # Based on P11 name
    'Exponential_100': lambda: create_exponential_problem(n=100),  # Based on P11 text
    'Genhumps_5': lambda: create_genhumps_problem(n=5),

    # Other standard test problems
    'Powell': create_powell_problem,
    'IllConditioned_10_6': lambda: create_ill_conditioned_problem(n=10, condition_number=1e6)
}

# --- Utility functions ---
def get_problem(name):
    """Retrieves a problem object by name using the factory dictionary."""
    if name not in PROBLEMS_DICT:
        available = list(PROBLEMS_DICT.keys())
        raise ValueError(f"Unknown problem name: {name}. Available problems: {available}")
    problem_factory = PROBLEMS_DICT[name]
    problem_obj = problem_factory()
    return problem_obj

def list_available_problems():
    """Return a list of available problem names"""
    return list(PROBLEMS_DICT.keys())

# --- Example usage ---
if __name__ == '__main__':
    print("Available problems:", list_available_problems())

    test_problems = ['Exponential_10', 'Rosenbrock_100', 'DataFit_2', 'quad_10_10']
    print("\n--- Testing selected problems ---")

    for problem_name in test_problems:
        print(f"\n* Testing: {problem_name}")
        try:
            problem = get_problem(problem_name)
            x0 = problem.x0
            print(f"  x0 ({x0.shape}): {x0[:5]}...{x0[-5:]}" if len(x0)>10 else f"x0 ({x0.shape}): {x0}")
            f0 = problem.func(x0)
            print(f"  f(x0): {f0}")
            g0 = problem.grad(x0)
            print(f"  grad(x0) ({g0.shape}): {g0[:5]}...{g0[-5:]}" if len(g0)>10 else f"grad(x0) ({g0.shape}): {g0}")
            # Optionally test Hessian calculation by uncommenting below
            # try:
            #      H0 = problem.hess(x0)
            #      print(f"  hess(x0) ({H0.shape}) calculated.") # Avoid printing large Hessians
            # except Exception as e:
            #      print(f"  Could not compute Hessian: {e}")

        except FileNotFoundError as e:
            print(f"  Error getting problem: {e}")
            print("  Ensure .mat files are accessible (check QUAD_DATA_PATH definition and file locations).")
        except ValueError as e:
            print(f"  Error getting problem: {e}")
        except KeyError:
            print(f"  Error: Problem '{problem_name}' not found in PROBLEMS_DICT.")
        except Exception as e:
            print(f"  An unexpected error occurred: {e}")