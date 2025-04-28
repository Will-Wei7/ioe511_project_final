# Tuning script for BFGSW parameters with early stopping

import numpy as np
import pandas as pd
import itertools
import time
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import sys # Needed for exiting nested loops cleanly

# Import your solver and problems (adjust import paths as needed)
# Assuming they are in the current directory or accessible via PYTHONPATH
try:
    from solver import optSolver_WHY
    from problems import get_problem, list_available_problems
except ImportError as e:
    print(f"Error importing solver or problems: {e}")
    print("Please ensure solver.py and problems.py are in the correct path.")
    sys.exit(1)

def tune_bfgsw_early_stopping(time_limit_minutes=15, success_threshold=80.0):
    """
    Tunes BFGSW parameters with early stopping based on success rate and time limit.

    Args:
        time_limit_minutes (int): Maximum tuning time in minutes.
        success_threshold (float): Success rate (percentage) to trigger early stopping.

    Returns:
        dict: The best parameter combination found, or None if no successful runs.
    """
    print("=" * 70)
    print("BFGSW PARAMETER TUNING (EARLY STOPPING ENABLED)")
    print(f" - Time Limit: {time_limit_minutes} minutes")
    print(f" - Success Rate Threshold: {success_threshold}%")
    print("=" * 70)

    time_limit_seconds = time_limit_minutes * 60

    # Define parameter grid to explore
    param_grid = {
        'c1_ls': [1e-4],         # Armijo constant
        'alpha_init': [1.0], # Initial step size
        'c2_ls': [0.9],         # Wolfe curvature constant
        'alpha_max': [10.0]              # Maximum step size
    }

    # Get all available problems
    try:
        all_problems = list_available_problems()
    except Exception as e:
        print(f"Error listing available problems: {e}")
        sys.exit(1)

    # Remove Powell and IllConditioned_10_6 as specified
    excluded_problems = ['Powell', 'IllConditioned_10_6']
    problem_names = [p for p in all_problems if p not in excluded_problems]

    if not problem_names:
        print("Error: No problems available for testing after exclusions.")
        sys.exit(1)

    print(f"Testing on {len(problem_names)} problems:")
    # Truncate list for display if too long
    display_limit = 10
    for i, name in enumerate(problem_names[:display_limit]):
        print(f"   {i+1}. {name}")
    if len(problem_names) > display_limit:
        print(f"   ... and {len(problem_names) - display_limit} more.")


    # Fixed options that won't be tuned
    fixed_options = {
        'max_iterations': 1000,
        'term_tol': 1e-6,
        'max_ls_iter': 25,  # Increased from default 20
        'verbose': False
    }

    results = []
    total_combinations = (
        len(param_grid['c1_ls']) *
        len(param_grid['alpha_init']) *
        len(param_grid['c2_ls']) *
        len(param_grid['alpha_max'])
    )

    # Adjust total combinations count by removing invalid c1 >= c2 cases
    valid_combinations_count = 0
    param_combinations = []
    for c1 in param_grid['c1_ls']:
      for alpha in param_grid['alpha_init']:
        for c2 in param_grid['c2_ls']:
          for alpha_max in param_grid['alpha_max']:
            if c1 < c2:
              valid_combinations_count += 1
              param_combinations.append({'c1_ls': c1, 'alpha_init': alpha, 'c2_ls': c2, 'alpha_max': alpha_max})

    print(f"Total potential valid combinations: {valid_combinations_count}")
    print(f"Total potential runs: {valid_combinations_count * len(problem_names)}")

    # Track progress and best results found so far
    combination_count = 0
    start_time = time.time()
    best_overall_success_rate = -1.0
    best_overall_params = None
    best_overall_stats = {}
    early_stop_reason = "Completed Full Search"
    tuning_stopped = False

    # Generate and test parameter combinations
    for params_to_tune in param_combinations:

        combination_count += 1
        current_params = {**params_to_tune, **fixed_options}

        # --- Time Limit Check ---
        elapsed_time = time.time() - start_time
        if elapsed_time > time_limit_seconds:
            early_stop_reason = f"Time Limit Reached ({elapsed_time:.1f}s > {time_limit_seconds}s)"
            tuning_stopped = True
            print(f"\nStopping: {early_stop_reason}")
            break # Break from the main parameter loop

        print(f"\nTesting combination {combination_count}/{valid_combinations_count}: {params_to_tune}")
        print(f"Elapsed time: {elapsed_time:.1f}s")

        # Test this parameter combination on all problems
        num_success = 0
        total_runs_for_combo = 0
        combo_iterations = []
        combo_runtimes = []
        combo_grad_norms = []

        for problem_name in problem_names:
            try:
                # Load problem
                problem = get_problem(problem_name)

                # Convert problem to dictionary if needed
                if hasattr(problem, 'to_dict'):
                    problem_dict = problem.to_dict()
                elif isinstance(problem, dict):
                     problem_dict = problem
                else:
                    print(f"Warning: Problem '{problem_name}' is not a dict or convertible object. Skipping.")
                    continue

                # Set method
                method = {'name': 'BFGSW'}

                # Run optimization
                start_run = time.time()
                # Use a copy of params to avoid modification issues if optSolver_WHY modifies its input
                solver_params = current_params.copy()
                x_final, f_final, info = optSolver_WHY(problem_dict, method, solver_params)
                runtime = time.time() - start_run

                # Calculate success based on gradient norm
                # Check if info is a dict and contains 'grad_norms' which is not empty
                final_grad_norm = np.nan
                if isinstance(info, dict) and 'grad_norms' in info and info['grad_norms']:
                   final_grad_norm = info['grad_norms'][-1]

                success = not np.isnan(final_grad_norm) and final_grad_norm < current_params['term_tol']


                # Record results for this run
                run_result = {
                    'problem': problem_name,
                    **params_to_tune, # Only log tunable params
                    'iterations': info.get('iterations', 0) if isinstance(info, dict) else 0,
                    'runtime': runtime,
                    'final_grad_norm': final_grad_norm,
                    'success': success
                }
                results.append(run_result)

                # Track stats for this combination
                total_runs_for_combo += 1
                if success:
                    num_success += 1
                combo_iterations.append(run_result['iterations'])
                combo_runtimes.append(runtime)
                if not np.isnan(final_grad_norm):
                     combo_grad_norms.append(final_grad_norm)


            except Exception as e:
                print(f"Error on {problem_name} with params {params_to_tune}: {str(e)}")
                # Record failure
                results.append({
                    'problem': problem_name,
                    **params_to_tune,
                    'iterations': 0,
                    'runtime': 0,
                    'final_grad_norm': np.nan,
                    'success': False
                })
                total_runs_for_combo += 1
                combo_iterations.append(0)
                combo_runtimes.append(0)


        # Calculate success rate for the current combination
        if total_runs_for_combo > 0:
            current_success_rate = (num_success / total_runs_for_combo) * 100
            print(f"Combination {combination_count} Success Rate: {current_success_rate:.1f}% ({num_success}/{total_runs_for_combo})")

            # --- Update Best Performing Parameters ---
            # Prioritize higher success rate, then lower median iterations
            current_median_iter = np.median(combo_iterations) if combo_iterations else np.inf

            needs_update = False
            # Use a small tolerance for comparing floats
            tolerance = 1e-6
            if current_success_rate > best_overall_success_rate + tolerance:
                needs_update = True
            elif abs(current_success_rate - best_overall_success_rate) < tolerance:
                 # Check if current median iterations is better (lower) than the best found so far
                 if best_overall_params is None or current_median_iter < best_overall_stats.get('iter_median', np.inf):
                      needs_update = True

            if needs_update:
                 print(f"*** New best combination found (Success: {current_success_rate:.1f}%, Median Iter: {current_median_iter:.1f}) ***")
                 best_overall_success_rate = current_success_rate
                 best_overall_params = params_to_tune.copy()
                 best_overall_stats = {
                     'success_rate': current_success_rate,
                     'iter_mean': np.mean(combo_iterations) if combo_iterations else np.nan,
                     'iter_median': current_median_iter,
                     'runtime_mean': np.mean(combo_runtimes) if combo_runtimes else np.nan,
                     'runtime_median': np.median(combo_runtimes) if combo_runtimes else np.nan,
                     'grad_norm_median': np.median(combo_grad_norms) if combo_grad_norms else np.nan
                 }


            # --- Success Threshold Check ---
            if current_success_rate >= success_threshold:
                early_stop_reason = f"Success Rate >= {success_threshold}% ({current_success_rate:.1f}%) Achieved"
                tuning_stopped = True
                print(f"\nStopping: {early_stop_reason}")
                # Make sure the just-found combination is stored as the best before stopping
                if needs_update: # Re-affirm the best params were just updated
                     pass # Already updated above
                elif best_overall_params is None : # Handle case where this is the *first* combo meeting threshold
                     best_overall_success_rate = current_success_rate
                     best_overall_params = params_to_tune.copy()
                     best_overall_stats = {
                         'success_rate': current_success_rate,
                         'iter_mean': np.mean(combo_iterations) if combo_iterations else np.nan,
                         'iter_median': current_median_iter,
                         'runtime_mean': np.mean(combo_runtimes) if combo_runtimes else np.nan,
                         'runtime_median': np.median(combo_runtimes) if combo_runtimes else np.nan,
                         'grad_norm_median': np.median(combo_grad_norms) if combo_grad_norms else np.nan
                     }
                break # Break from the main parameter loop
        else:
             print(f"Warning: No problems successfully run for combination {combination_count}.")


    # --- End of Tuning ---
    end_time = time.time()
    total_tuning_time = end_time - start_time
    print("\n" + "=" * 70)
    print(f"TUNING FINISHED ({early_stop_reason})")
    print(f"Total Tuning Time: {total_tuning_time:.2f} seconds")
    print(f"Combinations Tested: {combination_count}/{valid_combinations_count}")
    print("=" * 70)

    # --- Process and Save Results ---
    if not results:
         print("\nNo results were recorded during tuning.")
         return None

    results_df = pd.DataFrame(results)
    # Save raw results
    results_df.to_csv('bfgsw_tuning_results_earlystop.csv', index=False)
    print(f"\nRaw results for tested combinations saved to bfgsw_tuning_results_earlystop.csv")

    # If we stopped early, the 'best' is already known.
    # If we completed the full search without hitting the threshold,
    # but tracked best_overall_params, it should be correct.
    # Add a check in case the loop finished without *any* valid runs
    if best_overall_params is None and not results_df.empty:
         # This happens if the loop finished without ever finding a 'best' (e.g., all failed)
         # Or if the logic somehow missed updating 'best_overall_params'.
         # Let's recalculate the best from the collected data as a fallback.
         print("\nCalculating best parameters from all collected results as fallback...")
         param_cols = list(param_grid.keys())
         # Ensure param_cols only contains columns actually present in results_df
         param_cols = [col for col in param_cols if col in results_df.columns]
         if not param_cols:
             print("Error: No parameter columns found in results DataFrame.")
             return None

         try:
             param_performance = results_df.groupby(param_cols).agg(
                success_rate=('success', lambda x: x.mean() * 100),
                iter_mean=('iterations', 'mean'),
                iter_median=('iterations', 'median'),
                runtime_mean=('runtime', 'mean'),
                runtime_median=('runtime', 'median')
             ).reset_index()

             param_performance = param_performance.sort_values(
                 ['success_rate', 'iter_median'],
                 ascending=[False, True]
             )

             if not param_performance.empty:
                  best_row = param_performance.iloc[0]
                  best_overall_params = best_row[param_cols].to_dict()
                  # Ensure success rate is directly from the aggregated best row
                  best_overall_success_rate = best_row['success_rate']
                  best_overall_stats = best_row.drop(param_cols).to_dict()
                  # Overwrite success rate in stats for consistency
                  best_overall_stats['success_rate'] = best_overall_success_rate
                  print("Best parameters determined from final aggregation.")
             else:
                  print("Could not determine best parameters from aggregation (param_performance was empty).")
         except KeyError as e:
             print(f"Error during fallback aggregation: Missing column {e}")
             return None
         except Exception as agg_e:
              print(f"Unexpected error during fallback aggregation: {agg_e}")
              return None


    # --- Report Best Found Parameters ---
    if best_overall_params:
        print("\n" + "=" * 70)
        print("BEST PARAMETERS FOUND:")
        # Ensure all expected keys exist before trying to access them
        print(f"c1_ls: {best_overall_params.get('c1_ls', 'N/A')}")
        print(f"alpha_init: {best_overall_params.get('alpha_init', 'N/A')}")
        print(f"c2_ls: {best_overall_params.get('c2_ls', 'N/A')}")
        print(f"alpha_max: {best_overall_params.get('alpha_max', 'N/A')}")
        print("-" * 30)
        # Use the most reliable success rate value
        final_success_rate = best_overall_stats.get('success_rate', best_overall_success_rate)
        print(f"Achieved Success Rate: {final_success_rate:.1f}%" if final_success_rate != -1.0 else "Achieved Success Rate: N/A")
        # Format stats nicely, handling potential missing values or np.nan
        iter_median = best_overall_stats.get('iter_median', np.nan)
        runtime_median = best_overall_stats.get('runtime_median', np.nan)
        grad_norm_median = best_overall_stats.get('grad_norm_median', np.nan)

        print(f"Median Iterations: {iter_median:.1f}" if not np.isnan(iter_median) else "Median Iterations: N/A")
        print(f"Median Runtime: {runtime_median:.3f}s" if not np.isnan(runtime_median) else "Median Runtime: N/A")
        print(f"Median Final Grad Norm: {grad_norm_median:.2e}" if not np.isnan(grad_norm_median) else "Median Final Grad Norm: N/A")

        print("=" * 70)

        # --- Create Visualizations (Optional - Based on available data) ---
        if not results_df.empty and len(results_df['c2_ls'].unique()) > 1 :
             try:
                 # Aggregate success rate per parameter value *for plotting*
                 # Note: This uses ALL results collected, not just the best run
                 param_cols_viz = list(param_grid.keys())
                 param_cols_viz = [col for col in param_cols_viz if col in results_df.columns]

                 if param_cols_viz: # Check if we have parameter columns for grouping
                    agg_results_df = results_df.groupby(param_cols_viz).agg(success_rate=('success', lambda x: x.mean() * 100)).reset_index()

                    plt.figure(figsize=(10, 6))
                    sns.boxplot(x='c2_ls', y='success_rate', data=agg_results_df)
                    plt.title('Effect of c2_ls on Success Rate (Across Tested Combinations)')
                    plt.xlabel('c2_ls (Wolfe Curvature)')
                    plt.ylabel('Success Rate (%)')
                    plt.tight_layout()
                    plt.savefig('bfgsw_c2_vs_success_earlystop.png')
                    print("Saved plot: bfgsw_c2_vs_success_earlystop.png")

                    plt.figure(figsize=(10, 6))
                    sns.boxplot(x='alpha_init', y='success_rate', data=agg_results_df)
                    plt.title('Effect of Initial Step Size on Success Rate (Across Tested Combinations)')
                    plt.xlabel('alpha_init (Initial Step Size)')
                    plt.ylabel('Success Rate (%)')
                    plt.tight_layout()
                    plt.savefig('bfgsw_alpha_vs_success_earlystop.png')
                    print("Saved plot: bfgsw_alpha_vs_success_earlystop.png")
                    plt.close('all') # Close figures
                 else:
                    print("Skipping plots: No parameter columns found for aggregation.")

             except Exception as plot_e:
                 print(f"Could not generate plots: {plot_e}")


        # Return best parameters as a dictionary
        return best_overall_params
    else:
        print("\nNo successful parameter combination found within the constraints.")
        return None


if __name__ == "__main__":
    # You can change the limits here if needed
    TIME_LIMIT_MIN = 30
    SUCCESS_THRESHOLD_PERCENT = 67.0

    # Make sure matplotlib doesn't try to open windows if running non-interactively
    import matplotlib
    matplotlib.use('Agg')

    best_params = tune_bfgsw_early_stopping(
        time_limit_minutes=TIME_LIMIT_MIN,
        success_threshold=SUCCESS_THRESHOLD_PERCENT
    )

    if best_params:
        print("\nAdd these optimal parameters to your solver configuration:")
        print("bfgsw_optimal_params = {")
        # Use .get() for safer access in final printout
        print(f"    'c1_ls': {best_params.get('c1_ls', 'N/A')},")
        print(f"    'alpha_init': {best_params.get('alpha_init', 'N/A')},")
        print(f"    'c2_ls': {best_params.get('c2_ls', 'N/A')},")
        print(f"    'alpha_max': {best_params.get('alpha_max', 'N/A')}")
        print("}")
    else:
        print("\nNo optimal parameters identified.")