import pandas as pd

# 1) Read the CSV; the first two rows are the multi‐level header:
df = pd.read_csv('output_summary.csv', header=[0,1])

# 2) Define exactly which methods and metrics to pull
methods = [
    'GradientDescent','GradientDescentW','Newton','NewtonW',
    'BFGS','BFGSW','DFP','DFPW','TRNewtonCG','TRSR1CG'
]
metrics = [
    ('Iterations',      'It.'),
    ('Final f(x)',      r'\(f_{\rm final}\)'),
    ('FinalGradNorm',   r'\(\|\nabla f\|\)'),
    ('CPU Time (s)',    'CPU(s)')
]

# 3) Specify the exact problems in the order you want them
problems = [
    'quad_10_10','quad_10_1000','quad_1000_10','quad_1000_1000',
    'P5_quartic_1','P6_quartic_2',
    'Exponential_10','Exponential_1000',
    'Rosenbrock_2','Rosenbrock_100',
    'DataFit_2','Genhumps_5'
]

# 4) A little helper to format floats in scientific notation
def sci(x, sig=2):
    if pd.isna(x):
        return 'nan'
    return f"{x:.{sig}E}".replace('E+0','E+').replace('E-0','E-')

# 5) Generate the LaTeX
print(r"\begin{table}[h!]")
print(r"\centering")
print(r"\resizebox{\textwidth}{!}{%")
print(r"  \begin{tabular}{|c|c|" + "c|"*10 + r"}")
print(r"  \hline")
print(r"  \textbf{Prob.} & \textbf{M.} & " + " & ".join(r"\texttt{%s}" % m for m in methods) + r" \\")
print(r"  \hline")

for prob in problems:
    # fix the quartic names
    display_name = prob.replace('P5_quartic_1','quartic\_1')\
                       .replace('P6_quartic_2','quartic\_2')
    idx = df[('Unnamed: 0_level_0','Method')]==prob
    row = df[idx].squeeze()
    print(f"  \\multirow{{4}}{{*}}{{\\texttt{{{display_name}}}}}")
    for key, label in metrics:
        vals = [ row[(key, m)] for m in methods ]
        if key == 'Iterations':
            cells = [ str(int(v)) if not pd.isna(v) else 'nan' for v in vals ]
        elif key == 'CPU Time (s)':
            cells = [ f"{v:.4f}" if not pd.isna(v) else 'nan' for v in vals ]
        else:
            cells = [ sci(v) for v in vals ]
        print("    & %-15s & %s \\\\" % (label, " & ".join(cells)))
    print("  \\hline")

print(r"  \end{tabular}%")
print(r"}")
print(r"\caption{Comparison of all ten methods across problems}")
print(r"\end{table}")