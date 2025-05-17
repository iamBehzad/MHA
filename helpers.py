# Helper functions
from imports import *

# Constants
BEST_FIT_FOLDER = 'history/best_fit'
CONVERGENCE_FOLDER = 'history/convergence'
ALGORITHMS = ['ARDE', 'OriginalPSO', 'OriginalWOA', 'BaseGA', 'OriginalDE', 'OriginalAOA', 'OriginalSCSO' ]
PROPOSED_ALGORITHM = 'ARDE'
RESULT_FILENAME = 'results.xlsx'
OUTPUT_FOLDER = 'convergence_curves'

def read_algorithm_data(base_folder, algorithms):
    """Reads CSV files for each algorithm and returns a combined stats summary."""
    stats_summary = []
    for algorithm in algorithms:
        file_path = os.path.join(base_folder, f"{algorithm}_best_fit.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            for name, costs in df.items():
                stats_summary.append({
                    'Function': name,
                    'Algorithm': algorithm,
                    'Costs': costs,
                    'Best Cost': np.min(costs),
                    'Worst Cost': np.max(costs),
                    'Average Cost': np.mean(costs),
                    'STD': np.std(costs) if len(costs) > 1 else 0,
                })
    return pd.DataFrame(stats_summary)

def read_convergence_data(base_folder, benchmarks, algorithms):
    """Reads convergence data for each benchmark and algorithm."""
    convergence_data = {benchmark: {} for benchmark in benchmarks}
    for benchmark in benchmarks:
        for algorithm in algorithms:
            file_path = os.path.join(base_folder, algorithm, f"{benchmark}_convergence.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                convergence_data[benchmark][algorithm] = df.mean(axis=1)  # Average across trials
            else:
                convergence_data[benchmark][algorithm] = None
    return convergence_data


def prepare_performance_data(stats_df):
    """Creates a structured performance dictionary and DataFrame."""
    performance_data = {}
    for f_index in stats_df['Function'].unique():
        f_data = stats_df[stats_df['Function'] == f_index].set_index('Algorithm')
        performance_data[f_index] = {
            'Best Cost': f_data['Best Cost'].to_dict(),
            'Average Cost': f_data['Average Cost'].to_dict(),
            'STD': f_data['STD'].to_dict()
        }
    performance_rows = [
        {'F': f_index, 'Unnamed: 1': metric, **values}
        for f_index, metrics_data in performance_data.items()
        for metric, values in metrics_data.items()
    ]
    return pd.DataFrame(performance_rows).set_index(['F', 'Unnamed: 1'])

def perform_friedman_test(stats_summary):
    """Performs the Friedman test and calculates mean ranks."""
    friedman_data = {}
    for entry in stats_summary:
        friedman_data.setdefault(entry['Algorithm'], []).append(entry['Average Cost'])
    stat, p_value = friedmanchisquare(*friedman_data.values())
    mean_costs = np.mean(list(friedman_data.values()), axis=1)
    mean_ranks = np.argsort(np.argsort(mean_costs)) + 1
    algorithm_ranks = dict(zip(friedman_data.keys(), mean_ranks))
    return stat, p_value, algorithm_ranks

def perform_wilcoxon_test(stats_summary, proposed_algorithm):
    """Performs the Wilcoxon test for the proposed algorithm."""
    wilcoxon_data = {}
    for entry in stats_summary:
        wilcoxon_data.setdefault(entry['Algorithm'], {}).setdefault(entry['Function'], []).extend(entry['Costs'])
    proposed_costs = wilcoxon_data.pop(proposed_algorithm, {})

    wilcoxon_results = {}
    for algorithm, data in wilcoxon_data.items():
        for benchmark, costs in proposed_costs.items():
            if benchmark in data:
                if np.array_equal(costs, data[benchmark]):
                    statistic, p_value, better_algorithm = None, None, "="
                else:
                    statistic, p_value = wilcoxon(costs, data[benchmark])
                    better_algorithm = (
                        "=" if p_value > 0.05 else
                        "+" if np.mean(data[benchmark]) >= np.mean(costs) else "-"
                    )
                wilcoxon_results.setdefault(benchmark, {})[algorithm] = {
                    'Statistic': statistic,
                    'p-value': p_value,
                    'Better Algorithm': better_algorithm
                }
    return wilcoxon_results

def plot_convergence_curves(convergence_data, benchmarks, algorithms, output_folder):
    """Plots and saves convergence curves for each benchmark."""
    os.makedirs(output_folder, exist_ok=True)
    for benchmark in benchmarks:
        plt.figure(figsize=(12, 8))
        for algorithm in algorithms:
            avg_trial = convergence_data[benchmark].get(algorithm)
            if avg_trial is not None and not avg_trial.empty:
                plt.plot(avg_trial, label=algorithm)
        plt.title(f'Convergence Curves for {benchmark}')
        plt.xlabel('Iterations')
        plt.ylabel('Average Value')
        plt.yscale("log")
        plt.legend()
        plt.grid()
        output_filename = os.path.join(output_folder, f'convergence_curve_{benchmark}.png')
        plt.savefig(output_filename, format='png', bbox_inches='tight')
        plt.close()

def save_to_excel(performance_df, friedman_results, wilcoxon_results, filename):
    """Saves performance, Friedman, and Wilcoxon results to an Excel file."""
    with pd.ExcelWriter(filename) as writer:
        if not performance_df.empty:
            performance_df.to_excel(writer, sheet_name='Performance')
        stat, p_value, algorithm_ranks = friedman_results
        pd.DataFrame({
            'Statistic': [stat],
            'p-value': [p_value],
            'Algorithm Ranks': [algorithm_ranks]
        }).to_excel(writer, sheet_name='Friedman Test', index=False)

        wilcoxon_rows = []
        for benchmark, results in wilcoxon_results.items():
            for metric, key in [('Statistic', 'Statistic'), ('p-value', 'p-value'), ('Winner', 'Better Algorithm')]:
                row = {'F': benchmark, 'Unnamed: 1': metric}
                row.update({f"{PROPOSED_ALGORITHM} vs {algo}": res[key] for algo, res in results.items()})
                wilcoxon_rows.append(row)
        if wilcoxon_rows:
            pd.DataFrame(wilcoxon_rows).set_index(['F', 'Unnamed: 1']).to_excel(writer, sheet_name='Wilcoxon')
