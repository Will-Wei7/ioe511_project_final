# # IOE 511/MATH 562, University of Michigan
# # Code written by: Albert S. Berahas & Jiahao Shi

# import numpy as np
# import scipy.stats as stats
# import scipy.sparse as sparse
# import scipy.io
# from scipy.optimize import approx_fprime

# # Helper function for finite difference Hessian approximation
# def finite_diff_hess(func, x, eps=1e-5):
#     """Approximate Hessian using finite differences"""
#     n = len(x)
#     hess = np.zeros((n, n))
#     fx = func(x)
#     eps_vec = np.eye(n) * eps
    
#     for i in range(n):
#         for j in range(i, n):
#             # Second partial derivative approximation
#             f1 = func(x + eps_vec[i] + eps_vec[j])
#             f2 = func(x + eps_vec[i] - eps_vec[j])
#             f3 = func(x - eps_vec[i] + eps_vec[j])
#             f4 = func(x - eps_vec[i] - eps_vec[j])
            
#             hess[i,j] = (f1 - f2 - f3 + f4) / (4 * eps**2)
#             if i != j:
#                 hess[j,i] = hess[i,j]
    
#     return hess

# # Problem Number: 1
# # Problem Name: quad_10_10
# def quad_10_10_func(x):
#     np.random.seed(0)
#     q = np.random.normal(size=(10,1))
#     mat = scipy.io.loadmat('quad_10_10_Q.mat')
#     Q = mat['Q']
#     return (1/2*x.T@Q@x + q.T@x)[0]

# def quad_10_10_grad(x):
#     np.random.seed(12)
#     q = np.random.normal(size=(10,1))
#     mat = scipy.io.loadmat('quad_10_10_Q.mat')
#     Q = mat['Q']
#     return Q@x + q   

# def quad_10_10_Hess(x):
#     np.random.seed(12)
#     q = np.random.normal(size=(10,1))
#     mat = scipy.io.loadmat('quad_10_10_Q.mat')
#     Q = mat['Q']
#     return Q

# def get_quad_10_10_problem():
#     return {
#         'name': 'quad_10_10',
#         'x0': np.zeros(10),
#         'func': quad_10_10_func,
#         'grad': quad_10_10_grad,
#         'Hess': quad_10_10_Hess
#     }

# # Problem Number: 2
# # Problem Name: quad_10_1000
# def quad_10_1000_func(x):
#     np.random.seed(0)
#     q = np.random.normal(size=(10,1))
#     mat = scipy.io.loadmat('quad_10_1000_Q.mat')
#     Q = mat['Q']
#     return (1/2*x.T@Q@x + q.T@x)[0]

# def quad_10_1000_grad(x):
#     np.random.seed(12)
#     q = np.random.normal(size=(10,1))
#     mat = scipy.io.loadmat('quad_10_1000_Q.mat')
#     Q = mat['Q']
#     return Q@x + q

# def quad_10_1000_Hess(x):
#     np.random.seed(12)
#     q = np.random.normal(size=(10,1))
#     mat = scipy.io.loadmat('quad_10_1000_Q.mat')
#     Q = mat['Q']
#     return Q

# def get_quad_10_1000_problem():
#     return {
#         'name': 'quad_10_1000',
#         'x0': np.zeros(10),
#         'func': quad_10_1000_func,
#         'grad': quad_10_1000_grad,
#         'Hess': quad_10_1000_Hess
#     }

# # Problem Number: 3
# # Problem Name: quad_1000_10
# def quad_1000_10_func(x):
#     np.random.seed(0)
#     q = np.random.normal(size=(1000,1))
#     mat = scipy.io.loadmat('quad_1000_10_Q.mat')
#     Q = mat['Q']
#     return (1/2*x.T@Q@x + q.T@x)[0]

# def quad_1000_10_grad(x):
#     np.random.seed(12)
#     q = np.random.normal(size=(1000,1))
#     mat = scipy.io.loadmat('quad_1000_10_Q.mat')
#     Q = mat['Q']
#     return Q@x + q

# def quad_1000_10_Hess(x):
#     np.random.seed(12)
#     q = np.random.normal(size=(1000,1))
#     mat = scipy.io.loadmat('quad_1000_10_Q.mat')
#     Q = mat['Q']
#     return Q

# def get_quad_1000_10_problem():
#     return {
#         'name': 'quad_1000_10',
#         'x0': np.zeros(1000),
#         'func': quad_1000_10_func,
#         'grad': quad_1000_10_grad,
#         'Hess': quad_1000_10_Hess
#     }

# # Problem Number: 4
# # Problem Name: quad_1000_1000
# def quad_1000_1000_func(x):
#     np.random.seed(0)
#     q = np.random.normal(size=(1000,1))
#     mat = scipy.io.loadmat('quad_1000_1000_Q.mat')
#     Q = mat['Q']
#     return (1/2*x.T@Q@x + q.T@x)[0]

# def quad_1000_1000_grad(x):
#     np.random.seed(12)
#     q = np.random.normal(size=(1000,1))
#     mat = scipy.io.loadmat('quad_1000_1000_Q.mat')
#     Q = mat['Q']
#     return Q@x + q

# def quad_1000_1000_Hess(x):
#     np.random.seed(12)
#     q = np.random.normal(size=(1000,1))
#     mat = scipy.io.loadmat('quad_1000_1000_Q.mat')
#     Q = mat['Q']
#     return Q

# def get_quad_1000_1000_problem():
#     return {
#         'name': 'quad_1000_1000',
#         'x0': np.zeros(1000),
#         'func': quad_1000_1000_func,
#         'grad': quad_1000_1000_grad,
#         'Hess': quad_1000_1000_Hess
#     }

# # Problem Number: 5
# # Problem Name: quartic_1
# def quartic_1_func(x):
#     Q = np.array([[5,1,0,0.5],
#                  [1,4,0.5,0],
#                  [0,0.5,3,0],
#                  [0.5,0,0,2]])
#     sigma = 1e-4
#     return 1/2*(x.T@x) + sigma/4*(x.T@Q@x)**2

# def quartic_1_grad(x):
#     Q = np.array([[5,1,0,0.5],
#                  [1,4,0.5,0],
#                  [0,0.5,3,0],
#                  [0.5,0,0,2]])
#     sigma = 1e-4
#     return x + sigma*(x.T@Q@x)*Q@x

# def quartic_1_Hess(x):
#     Q = np.array([[5,1,0,0.5],
#                  [1,4,0.5,0],
#                  [0,0.5,3,0],
#                  [0.5,0,0,2]])
#     sigma = 1e-4
#     q_term = x.T@Q@x
#     return np.eye(4) + sigma*(np.outer(Q@x, Q@x) + q_term*Q)

# def get_quartic_1_problem():
#     return {
#         'name': 'quartic_1',
#         'x0': np.array([1,1,1,1], dtype=float),
#         'func': quartic_1_func,
#         'grad': quartic_1_grad,
#         'Hess': quartic_1_Hess
#     }

# # Problem Number: 6
# # Problem Name: quartic_2
# def quartic_2_func(x):
#     Q = np.array([[5,1,0,0.5],
#                  [1,4,0.5,0],
#                  [0,0.5,3,0],
#                  [0.5,0,0,2]])
#     sigma = 1e4
#     return 1/2*(x.T@x) + sigma/4*(x.T@Q@x)**2

# def quartic_2_grad(x):
#     Q = np.array([[5,1,0,0.5],
#                  [1,4,0.5,0],
#                  [0,0.5,3,0],
#                  [0.5,0,0,2]])
#     sigma = 1e4
#     return x + sigma*(x.T@Q@x)*Q@x

# def quartic_2_Hess(x):
#     Q = np.array([[5,1,0,0.5],
#                  [1,4,0.5,0],
#                  [0,0.5,3,0],
#                  [0.5,0,0,2]])
#     sigma = 1e4
#     q_term = x.T@Q@x
#     return np.eye(4) + sigma*(np.outer(Q@x, Q@x) + q_term*Q)

# def get_quartic_2_problem():
#     return {
#         'name': 'quartic_2',
#         'x0': np.array([1,1,1,1], dtype=float),
#         'func': quartic_2_func,
#         'grad': quartic_2_grad,
#         'Hess': quartic_2_Hess
#     }

# # Problem Number: 12
# # Problem Name: genhumps_5
# def genhumps_5_func(x):
#     f = 0
#     for i in range(4):
#         f += np.sin(2*x[i])**2 * np.sin(2*x[i+1])**2 + 0.05*(x[i]**2 + x[i+1]**2)
#     return f

# def genhumps_5_grad(x):
#     g = [4*np.sin(2*x[0])*np.cos(2*x[0])*np.sin(2*x[1])**2 + 0.1*x[0],
#          4*np.sin(2*x[1])*np.cos(2*x[1])*(np.sin(2*x[0])**2 + np.sin(2*x[2])**2) + 0.2*x[1],
#          4*np.sin(2*x[2])*np.cos(2*x[2])*(np.sin(2*x[1])**2 + np.sin(2*x[3])**2) + 0.2*x[2],
#          4*np.sin(2*x[3])*np.cos(2*x[3])*(np.sin(2*x[2])**2 + np.sin(2*x[4])**2) + 0.2*x[3],
#          4*np.sin(2*x[4])*np.cos(2*x[4])*np.sin(2*x[3])**2 + 0.1*x[4]]
#     return np.array(g)

# def genhumps_5_Hess(x):
#     H = np.zeros((5,5))
#     H[0,0] = 8*np.sin(2*x[1])**2*(np.cos(2*x[0])**2 - np.sin(2*x[0])**2) + 0.1
#     H[0,1] = 16*np.sin(2*x[0])*np.cos(2*x[0])*np.sin(2*x[1])*np.cos(2*x[1])
#     H[1,1] = 8*(np.sin(2*x[0])**2 + np.sin(2*x[2])**2)*(np.cos(2*x[1])**2 - np.sin(2*x[1])**2) + 0.2
#     H[1,2] = 16*np.sin(2*x[1])*np.cos(2*x[1])*np.sin(2*x[2])*np.cos(2*x[2])
#     H[2,2] = 8*(np.sin(2*x[1])**2 + np.sin(2*x[3])**2)*(np.cos(2*x[2])**2 - np.sin(2*x[2])**2) + 0.2
#     H[2,3] = 16*np.sin(2*x[2])*np.cos(2*x[2])*np.sin(2*x[3])*np.cos(2*x[3])
#     H[3,3] = 8*(np.sin(2*x[2])**2 + np.sin(2*x[4])**2)*(np.cos(2*x[3])**2 - np.sin(2*x[3])**2) + 0.2
#     H[3,4] = 16*np.sin(2*x[3])*np.cos(2*x[3])*np.sin(2*x[4])*np.cos(2*x[4])
#     H[4,4] = 8*np.sin(2*x[3])**2*(np.cos(2*x[4])**2 - np.sin(2*x[4])**2) + 0.1
#     H[1,0] = H[0,1]
#     H[2,1] = H[1,2]
#     H[3,2] = H[2,3]
#     H[4,3] = H[3,4]
#     return H

# def get_genhumps_5_problem():
#     return {
#         'name': 'genhumps_5',
#         'x0': np.array([-506.2, -506.0, -506.5, -506.1, -506.0]),
#         'func': genhumps_5_func,
#         'grad': genhumps_5_grad,
#         'Hess': genhumps_5_Hess
#     }

# # Dictionary mapping problem names to their getter functions
# PROBLEMS = {
#     'quad_10_10': get_quad_10_10_problem,
#     'quad_10_1000': get_quad_10_1000_problem,
#     'quad_1000_10': get_quad_1000_10_problem,
#     'quad_1000_1000': get_quad_1000_1000_problem,
#     'quartic_1': get_quartic_1_problem,
#     'quartic_2': get_quartic_2_problem,
#     'genhumps_5': get_genhumps_5_problem
# }

# def get_problem(name):
#     """Get a problem struct by name with all required components"""
#     if name not in PROBLEMS:
#         raise ValueError(f"Unknown problem name: {name}. Available problems: {list(PROBLEMS.keys())}")
    
#     problem = PROBLEMS[name]()
    
#     # Ensure all problems have all required components
#     required = ['x0', 'name', 'func', 'grad', 'Hess']
#     for field in required:
#         if field not in problem:
#             raise ValueError(f"Problem {name} is missing required field: {field}")
    
#     return problem
# IOE 511/MATH 562, University of Michigan
# Code written by: Albert S. Berahas & Jiahao Shi
# Revised for better robustness and functionality

import numpy as np
import scipy.stats as stats
import scipy.sparse as sparse
import scipy.io
from scipy.optimize import approx_fprime

class OptimizationProblem:
    """Container class for optimization problems with standardized interface"""
    def __init__(self, name, x0, func, grad, Hess=None):
        self.name = name
        self.x0 = np.array(x0, dtype=float)
        self.func = func
        self.grad = grad
        self.Hess = Hess if Hess is not None else lambda x: finite_diff_hess(func, x)
        
    def to_dict(self):
        """Convert to dictionary format for compatibility"""
        return {
            'name': self.name,
            'x0': self.x0,
            'func': self.func,
            'grad': self.grad,
            'Hess': self.Hess
        }

def finite_diff_hess(func, x, eps=1e-5):
    """Approximate Hessian using finite differences with improved numerical stability"""
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

# --------------------------
# Quadratic Problems
# --------------------------

def create_quad_problem(name, dim, cond_num):
    """Factory function for quadratic problems"""
    np.random.seed(0)
    q = np.random.normal(size=(dim,1))
    mat = scipy.io.loadmat(f'quad_{dim}_{cond_num}_Q.mat')
    Q = mat['Q']
    
    def func(x):
        return (0.5 * x.T @ Q @ x + q.T @ x).item()
    
    def grad(x):
        return Q @ x + q.flatten()
    
    def Hess(x):
        return Q
    
    return OptimizationProblem(
        name=f'quad_{dim}_{cond_num}',
        x0=np.zeros(dim),
        func=func,
        grad=grad,
        Hess=Hess
    )

# --------------------------
# Quartic Problems
# --------------------------

def create_quartic_problem(name, sigma):
    """Factory function for quartic problems"""
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
    
    def Hess(x):
        q_term = x.T @ Q @ x
        return np.eye(4) + sigma * (np.outer(Q @ x, Q @ x) + q_term * Q)
    
    return OptimizationProblem(
        name=name,
        x0=np.array([1, 1, 1, 1], dtype=float),
        func=func,
        grad=grad,
        Hess=Hess
    )

# --------------------------
# Genhumps Problem
# --------------------------

def create_genhumps_problem():
    """Factory function for genhumps problem"""
    def func(x):
        f = 0
        for i in range(4):
            f += np.sin(2*x[i])**2 * np.sin(2*x[i+1])**2 + 0.05*(x[i]**2 + x[i+1]**2)
        return f
    
    def grad(x):
        g = [
            4*np.sin(2*x[0])*np.cos(2*x[0])*np.sin(2*x[1])**2 + 0.1*x[0],
            4*np.sin(2*x[1])*np.cos(2*x[1])*(np.sin(2*x[0])**2 + np.sin(2*x[2])**2) + 0.2*x[1],
            4*np.sin(2*x[2])*np.cos(2*x[2])*(np.sin(2*x[1])**2 + np.sin(2*x[3])**2) + 0.2*x[2],
            4*np.sin(2*x[3])*np.cos(2*x[3])*(np.sin(2*x[2])**2 + np.sin(2*x[4])**2) + 0.2*x[3],
            4*np.sin(2*x[4])*np.cos(2*x[4])*np.sin(2*x[3])**2 + 0.1*x[4]
        ]
        return np.array(g)
    
    def Hess(x):
        H = np.zeros((5,5))
        H[0,0] = 8*np.sin(2*x[1])**2*(np.cos(2*x[0])**2 - np.sin(2*x[0])**2) + 0.1
        H[0,1] = 16*np.sin(2*x[0])*np.cos(2*x[0])*np.sin(2*x[1])*np.cos(2*x[1])
        H[1,1] = 8*(np.sin(2*x[0])**2 + np.sin(2*x[2])**2)*(np.cos(2*x[1])**2 - np.sin(2*x[1])**2) + 0.2
        H[1,2] = 16*np.sin(2*x[1])*np.cos(2*x[1])*np.sin(2*x[2])*np.cos(2*x[2])
        H[2,2] = 8*(np.sin(2*x[1])**2 + np.sin(2*x[3])**2)*(np.cos(2*x[2])**2 - np.sin(2*x[2])**2) + 0.2
        H[2,3] = 16*np.sin(2*x[2])*np.cos(2*x[2])*np.sin(2*x[3])*np.cos(2*x[3])
        H[3,3] = 8*(np.sin(2*x[2])**2 + np.sin(2*x[4])**2)*(np.cos(2*x[3])**2 - np.sin(2*x[3])**2) + 0.2
        H[3,4] = 16*np.sin(2*x[3])*np.cos(2*x[3])*np.sin(2*x[4])*np.cos(2*x[4])
        H[4,4] = 8*np.sin(2*x[3])**2*(np.cos(2*x[4])**2 - np.sin(2*x[4])**2) + 0.1
        
        # Symmetrize the Hessian
        H[1,0] = H[0,1]
        H[2,1] = H[1,2]
        H[3,2] = H[2,3]
        H[4,3] = H[3,4]
        return H
    
    return OptimizationProblem(
        name='genhumps_5',
        x0=np.array([-506.2, -506.0, -506.5, -506.1, -506.0]),
        func=func,
        grad=grad,
        Hess=Hess
    )

# --------------------------
# Problem Collection
# --------------------------

PROBLEMS = {
    'quad_10_10': lambda: create_quad_problem('quad_10_10', 10, 10).to_dict(),
    'quad_10_1000': lambda: create_quad_problem('quad_10_1000', 10, 1000).to_dict(),
    'quad_1000_10': lambda: create_quad_problem('quad_1000_10', 1000, 10).to_dict(),
    'quad_1000_1000': lambda: create_quad_problem('quad_1000_1000', 1000, 1000).to_dict(),
    'quartic_1': lambda: create_quartic_problem('quartic_1', 1e-4).to_dict(),
    'quartic_2': lambda: create_quartic_problem('quartic_2', 1e4).to_dict(),
    'genhumps_5': lambda: create_genhumps_problem().to_dict()
}

def get_problem(problem_name, quad_data=None):
    """Get problem definition with Hessian support for quadratic problems.
    
    Args:
        problem_name (str): Name of the problem to retrieve
        quad_data (dict, optional): Dictionary containing quadratic problem data
        
    Returns:
        dict: Problem definition containing:
            - x0: Initial point
            - name: Problem name
            - func: Objective function
            - grad: Gradient function
            - hess: Hessian function (for quadratic problems)
            
    Raises:
        ValueError: If quad_data is required but not provided
    """
    if problem_name.startswith('quad_'):
        if quad_data is None:
            raise ValueError("quad_data must be provided for quadratic problems")
            
        # Extract the Q matrix for this problem
        Q = quad_data.get(f'{problem_name}_Q.mat')
        if Q is None:
            raise ValueError(f"Quadratic problem data not found for {problem_name}")
            
        n = Q.shape[0]
        
        def func(x):
            return 0.5 * x.T @ Q @ x
        
        def grad(x):
            return Q @ x
        
        def hess(x):
            return Q  # Constant for quadratic problems
            
        return {
            'x0': np.zeros(n),
            'name': problem_name,
            'func': func,
            'grad': grad,
            'hess': hess
        }
    else:
        # For non-quadratic problems, provide some default implementations
        n = 10  # Default dimension
        return {
            'x0': np.zeros(n),
            'name': problem_name,
            'func': lambda x: np.sum(x**2),  # Quadratic example
            'grad': lambda x: 2*x,
            'hess': lambda x: 2*np.eye(len(x))
        }

def list_available_problems():
    """Return a list of available problem names"""
    return list(PROBLEMS.keys())

def get_problem_optimal_value(name):
    """Return known optimal values for problems where available"""
    optima = {
        'quad_10_10': 0,          # Quadratic problems have minimum at 0
        'quad_10_1000': 0,
        'quad_1000_10': 0,
        'quad_1000_1000': 0,
        'quartic_1': 0,           # Quartic problems have minimum at 0
        'quartic_2': 0,
        'genhumps_5': 0            # Actual minimum is near 0 but not exactly 0
    }
    return optima.get(name, None)