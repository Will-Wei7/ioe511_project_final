# IOE 511/MATH 562, University of Michigan
# Problem definitions for optimization project

import numpy as np
import scipy.io
import os

class OptimizationProblem:
    """Container class for optimization problems."""
    def __init__(self, name, x0, func, grad, hess=None, f_opt=None): # Expects lowercase 'hess'
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
            'hess': self.hess, # Returns lowercase 'hess'
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
            f1 = func(x + eps_vec[i] + eps_vec[j])
            f2 = func(x + eps_vec[i] - eps_vec[j])
            f3 = func(x - eps_vec[i] + eps_vec[j])
            f4 = func(x - eps_vec[i] - eps_vec[j])

            hess[i,j] = (f1 - f2 - f3 + f4) / (4 * eps**2)
            if i != j:
                hess[j,i] = hess[i,j]
    return hess

# --- Quadratic Problems ---
QUAD_DATA_PATH = './' 

def get_quad_matrix(filename):
    """Loads Q matrix from .mat file."""
    filepath = os.path.join(QUAD_DATA_PATH, filename)
    # Add checks for potential subdirectories where the .mat files might be located
    if not os.path.exists(filepath):
         sub_dir_path = os.path.join(QUAD_DATA_PATH, 'Project_Problems_PYTHON', 'Project_Problems_PYTHON', filename)
         if os.path.exists(sub_dir_path):
             filepath = sub_dir_path
         else:
              sub_dir_path_alt = os.path.join(QUAD_DATA_PATH, '511 Proj', 'Project_Problems_PYTHON', 'Project_Problems_PYTHON', filename)
              if os.path.exists(sub_dir_path_alt):
                  filepath = sub_dir_path_alt
              else:
                  sub_dir_path_notebook = os.path.join(QUAD_DATA_PATH, '511 Proj', filename)
                  if os.path.exists(sub_dir_path_notebook):
                       filepath = sub_dir_path_notebook
                  else:
                       raise FileNotFoundError(f"Required data file {filename} not found in checked paths.")


    mat = scipy.io.loadmat(filepath)
    Q = mat['Q']
    if hasattr(Q, "toarray"):
        Q = Q.toarray()
    return Q

def create_quad_problem(name, dim, cond_num):
    """Factory function for quadratic problems."""
    filename = f'quad_{dim}_{cond_num}_Q.mat'
    Q = get_quad_matrix(filename)
    np.random.seed(0)
    q = np.random.normal(size=(dim,1))

    def func(x):
        x_col = x.reshape(-1, 1)
        val = 0.5 * x_col.T @ Q @ x_col + q.T @ x_col
        return float(val.item())

    def grad(x):
        x_col = x.reshape(-1, 1)
        return (Q @ x_col + q).flatten()

    def quad_hess(x): # Renamed local function to avoid conflict
        return Q

    try:
        x_opt = -np.linalg.solve(Q, q)
        f_opt = (0.5 * x_opt.T @ Q @ x_opt + q.T @ x_opt).item()
    except np.linalg.LinAlgError:
        print(f"Warning: Matrix Q for {name} is singular. Cannot compute exact f_opt.")
        f_opt = None

    return OptimizationProblem(
        name=f'quad_{dim}_{cond_num}',
        x0=np.zeros(dim),
        func=func,
        grad=grad,
        hess=quad_hess, # Fixed: Use lowercase 'hess' keyword argument
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

    def func(x):
        return 0.5 * (x.T @ x) + sigma/4 * (x.T @ Q @ x)**2

    def grad(x):
        return x + sigma * (x.T @ Q @ x) * (Q @ x)

    def quartic_hess(x): # Renamed local function
        q_term = x.T @ Q @ x
        Qx = Q @ x
        return np.eye(4) + sigma * (np.outer(Qx, Qx) + q_term * Q)

    return OptimizationProblem(
        name=name,
        x0=np.array([1, 1, 1, 1], dtype=float),
        func=func,
        grad=grad,
        hess=quartic_hess, # Fixed: Use lowercase 'hess' keyword argument
        f_opt=0.0
    )

# --- Rosenbrock Problem ---
def rosenbrock_func(x):
    if len(x) < 2: raise ValueError("Rosenbrock requires n >= 2")
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def rosenbrock_grad(x):
    if len(x) < 2: raise ValueError("Rosenbrock requires n >= 2")
    return np.array([
        -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2),
        200 * (x[1] - x[0]**2)
    ])

def rosenbrock_hess(x):
    if len(x) < 2: raise ValueError("Rosenbrock requires n >= 2")
    return np.array([
        [2 - 400 * x[1] + 1200 * x[0]**2, -400 * x[0]],
        [-400 * x[0], 200]
    ])

def create_rosenbrock_problem():
    return OptimizationProblem(
        name='Rosenbrock',
        x0=np.array([-1.2, 1.0]),
        func=rosenbrock_func,
        grad=rosenbrock_grad,
        hess=rosenbrock_hess, # Fixed: Use lowercase 'hess' keyword argument
        f_opt=0.0
    )

# --- Powell Problem ---
def powell_func(x):
    if len(x) < 4: raise ValueError("Powell requires n >= 4")
    return (x[0] + 10*x[1])**2 + 5*(x[2] - x[3])**2 + (x[1] - 2*x[2])**4 + 10*(x[0] - x[3])**4

def powell_grad(x):
    if len(x) < 4: raise ValueError("Powell requires n >= 4")
    g = np.zeros(4)
    g[0] = 2*(x[0] + 10*x[1]) + 40*(x[0] - x[3])**3
    g[1] = 20*(x[0] + 10*x[1]) + 4*(x[1] - 2*x[2])**3
    g[2] = 10*(x[2] - x[3]) - 8*(x[1] - 2*x[2])**3
    g[3] = -10*(x[2] - x[3]) - 40*(x[0] - x[3])**3
    return g

def create_powell_problem():
     return OptimizationProblem(
        name='Powell',
        x0=np.array([3.0, -1.0, 0.0, 1.0]),
        func=powell_func,
        grad=powell_grad,
        hess=None, # Fixed: Use lowercase 'hess'. Uses default finite diff.
        f_opt=0.0
    )

# --- Genhumps Problem ---
def create_genhumps_problem(n=5):
    """Factory function for genhumps problem."""
    if n < 2: raise ValueError("Genhumps requires n >= 2")
    def func(x):
        f = 0.0
        for i in range(n - 1):
            f += np.sin(2*x[i])**2 * np.sin(2*x[i+1])**2 + 0.05*(x[i]**2 + x[i+1]**2)
        return f

    def grad(x):
        g = np.zeros(n)
        if n >= 2:
             g[0] = 4*np.sin(2*x[0])*np.cos(2*x[0])*np.sin(2*x[1])**2 + 0.1*x[0]
             g[n-1] = 4*np.sin(2*x[n-1])*np.cos(2*x[n-1])*np.sin(2*x[n-2])**2 + 0.1*x[n-1]
        for i in range(1, n - 1):
             g[i] = (4*np.sin(2*x[i])*np.cos(2*x[i])*(np.sin(2*x[i-1])**2 + np.sin(2*x[i+1])**2)
                     + 0.1*x[i] + 0.1*x[i])
        return g

    def genhumps_hess(x): # Use finite difference for genhumps hessian
         return finite_diff_hess(func, x)

    if n == 5:
        x0_val = np.array([-506.2, -506.0, -506.5, -506.1, -506.0])
    else:
        x0_val = np.full(n, -506.0)

    return OptimizationProblem(
        name=f'genhumps_{n}',
        x0=x0_val,
        func=func,
        grad=grad,
        hess=genhumps_hess, # Fixed: Use lowercase 'hess' keyword argument
        f_opt=0.0
    )

# --- Ill-Conditioned Quadratic ---
def create_ill_conditioned_problem(n=10, condition_number=1e6):
     """Creates an ill-conditioned quadratic problem."""
     np.random.seed(42)
     A = np.random.randn(n, n)
     Q_orth, _ = np.linalg.qr(A)
     diag_vals = np.logspace(0, np.log10(condition_number), n)
     H_mat = Q_orth @ np.diag(diag_vals) @ Q_orth.T # Renamed local H matrix
     b = np.random.randn(n)
     
     # Define problem name here
     problem_name = f'IllConditioned_{n}_{int(np.log10(condition_number))}'

     def func(x):
          return 0.5 * x.T @ H_mat @ x - b.T @ x

     def grad(x):
          return H_mat @ x - b

     def ill_cond_hess(x): # Renamed local function
          return H_mat

     try:
        x_opt = np.linalg.solve(H_mat, b)
        f_opt = -0.5 * b.T @ x_opt
     except np.linalg.LinAlgError:
        print(f"Warning: Matrix H for {problem_name} is singular. Cannot compute exact f_opt.")
        f_opt = None

     return OptimizationProblem(
        name=problem_name,
        x0=np.ones(n),
        func=func,
        grad=grad,
        hess=ill_cond_hess, # Fixed: Use lowercase 'hess' keyword argument
        f_opt=f_opt
    )

# --- Problem dictionary ---
PROBLEMS_DICT = {
    'quad_10_10': lambda: create_quad_problem('quad_10_10', 10, 10),
    'quad_10_1000': lambda: create_quad_problem('quad_10_1000', 10, 1000),
    'quad_1000_10': lambda: create_quad_problem('quad_1000_10', 1000, 10),
    'quad_1000_1000': lambda: create_quad_problem('quad_1000_1000', 1000, 1000),
    'quartic_1': lambda: create_quartic_problem('quartic_1', 1e-4),
    'quartic_2': lambda: create_quartic_problem('quartic_2', 1e4),
    'rosenbrock': create_rosenbrock_problem,
    'powell': create_powell_problem,
    'genhumps_5': lambda: create_genhumps_problem(n=5),
    'ill_conditioned_10_6': lambda: create_ill_conditioned_problem(n=10, condition_number=1e6)
}

# --- get_problem function ---
def get_problem(name):
    """Retrieves a problem object by name using the factory dictionary."""
    if name not in PROBLEMS_DICT:
        available = list(PROBLEMS_DICT.keys())
        raise ValueError(f"Unknown problem name: {name}. Available problems: {available}")
    problem_obj = PROBLEMS_DICT[name]()
    return problem_obj

def list_available_problems():
    """Return a list of available problem names"""
    return list(PROBLEMS_DICT.keys())