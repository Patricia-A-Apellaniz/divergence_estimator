# Author: Ana Jiménez & Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 15/04/2024

# Packages to import
import os

import numpy as np

from data import load_data
from colorama import Fore, Style
from joblib import Parallel, delayed

from utils import save_results, plot_box_plots, plot_n_influence, plot_paper_disc_losses, print_table
from estimator.evaluator import DivergenceEvaluator


# This function is used to calculate divergences between x_r and x_s using different estimators
def run_experiment(results_path, n, m, l, seed, experiment, verbose, data_path=None, model=None):
    if experiment in ['use_case_3', 'use_case_4']:
        if verbose:
            print('\n' + Fore.BLUE + 'Running experiment with parameters: N = ' + str(n) + ', M = ' + str(
                m) + ', L = ' + str(l) + ', seed = ' + str(seed) + Style.RESET_ALL)
    else:
        n = None
        if verbose:
            print('\n' + Fore.BLUE + 'Running experiment with parameters: M = ' + str(m) + ', L = ' + str(
                l) + ', seed = ' + str(seed) + Style.RESET_ALL)

    # Load data
    x_r, x_s, dist_r, dist_s = load_data(experiment, n, m, l, seed, data_path)

    # Run divergence estimations
    # Define evaluator
    evaluator = DivergenceEvaluator(x_r, x_s, n, m, l, dist_r, dist_s, seed, experiment, verbose)

    # Analytical KL
    evaluator.analytical_kl_calculation()

    # Ground Truth (lots of samples) Monte Carlo estimation
    evaluator.monte_carlo_gt_estimation()

    # Estimation configuration
    train_m = m / (m + 2 * l)  # Training set used to train the discriminator.
    evaluator.split_estimation_data(train_m)

    # Monte Carlo estimation
    evaluator.monte_carlo_estimation()

    # Probabilistic classifier estimation
    if experiment == 'use_case_3':
        plot_path = results_path + str(n) + '_' + str(m) + '_' + str(l) + os.sep
    elif experiment == 'use_case_4':
        plot_path = results_path + model + os.sep + str(n) + '_' + str(m) + '_' + str(l) + os.sep
    else:
        plot_path = results_path + str(m) + '_' + str(l) + os.sep
    evaluator.probabilistic_classifier_estimation(plot_path)

    if verbose:
        print('Analytical KL:', evaluator.analytical_kl)
        print('Ground Truth MC KL:', evaluator.mc_gt_kl)
        print('Ground Truth MC JS:', evaluator.mc_gt_js)
        print('MC KL:', evaluator.mc_kl)
        print('MC JS:', evaluator.mc_js)
        print('Disc KL:', evaluator.disc_kl)
        print('Disc JS:', evaluator.disc_js)

    return evaluator


def main():
    # Configuration parameters
    # Theoretical scenarios:
    #   'use_case_1': multivariate gaussian distributions comparison
    #   'use_case_2': gaussian mixed distributions comparison
    #   'use_case_3': gaussian mixture distribution and synthetic data. Synthetic data is generated by GMM.
    # Practical scenarios:
    #   'use_case_4': real and synthetic data comparison. Synthetic data is generated by CTGAN, Generator models.
    evaluate = False
    experiment = 'use_case_1'
    n_seeds = 5
    n_threads = 30
    data_path = None  # Path to data for use_case_4
    model = 'vae'  # Model used to generate synthetic data in use_case_4
    if experiment == 'use_case_1' or experiment == 'use_case_2':
        N = ['-']
        M = [20, 200, 2000]  # Number of training samples in divergence estimator
        L = [20, 200, 2000]  # Number of test samples in divergence estimator
    elif experiment == 'use_case_3':
        N = [int(i) for i in (np.linspace(10, 150,
                                          15)).tolist()]  # Number of samples in generative process. Not used in use_case_1 and use_case_2
        M = [2000]  # Number of training samples in divergence estimator
        L = [2000]  # Number of test samples in divergence estimator
    elif experiment == 'use_case_4':
        N = [10000]  # Number of samples in generative process. Not used in use_case_1 and use_case_2
        M = [7500]
        L = [1000]
    else:
        raise ValueError('Experiment not found')

    # Results path
    abs_path = os.path.dirname(os.path.abspath(__file__)) + os.sep
    results_path = abs_path + 'results' + os.sep + experiment + os.sep
    if experiment == 'use_case_3' or experiment == 'use_case_4':
        for n in N:
            for m in M:
                for l in L:
                    if experiment == 'use_case_3':
                        comb_results_path = results_path + str(n) + '_' + str(m) + '_' + str(l) + os.sep
                    else:
                        comb_results_path = results_path + model + os.sep + str(n) + '_' + str(m) + '_' + str(l) + os.sep
                    os.makedirs(comb_results_path, exist_ok=True)
        if experiment == 'use_case_4':
            data_path = abs_path + 'data' + os.sep + 'adult' + os.sep + model + os.sep
    else:
        N = [0]
        for m in M:
            for l in L:
                comb_results_path = results_path + str(m) + '_' + str(l) + os.sep
                os.makedirs(comb_results_path, exist_ok=True)

    # Evaluation process
    print('\n' + Fore.MAGENTA + experiment.upper() + Style.RESET_ALL)

    # Run experiments
    if evaluate:
        print('Training...')
        verbose = True if n_threads == 1 else False
        evaluators = Parallel(n_jobs=n_threads, verbose=10)(delayed(run_experiment)(results_path, n, m, l, seed,
                                                                                    experiment, verbose, data_path, model)
                                                            for n in N
                                                            for m in M
                                                            for l in L
                                                            for seed in range(n_seeds))
        # Save results
        print('\nSaving results...')
        save_results(evaluators, results_path, experiment, N, M, L, n_seeds, model)

    # Represent table results
    print('\nTable results...')
    print_table(results_path, experiment, model)

    # Plot error results
    print('\nPlotting results...')
    if experiment != 'use_case_3' and experiment != 'use_case_4':
        plot_box_plots(results_path, experiment, N, M, L)

    # If use case is generative process, plot N variations
    if experiment == 'use_case_3':
        plot_n_influence(results_path, experiment, N, M, L)

    # Plot M=20 and L=20, M=200 and L=200, M=2000 and L=2000 losses for paper
    if experiment != 'use_case_3' and experiment != 'use_case_4':
        plot_paper_disc_losses(results_path)


if __name__ == '__main__':
    main()
