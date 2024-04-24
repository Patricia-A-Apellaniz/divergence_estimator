# Author: Ana Jiménez & Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 17/04/2024

# Packages to import
import os

import numpy as np
import torch

import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from torch import distributions as D
from estimator.evaluator import DivergenceEvaluator


def estimate_divs(mean, n_samples, n, m, l, plot_path):
    m1 = D.MultivariateNormal(torch.tensor([0.] * 4), torch.eye(4))
    m2 = D.MultivariateNormal(torch.tensor([mean] * 4), torch.eye(4))

    m1_sample = m1.sample(torch.Size((n_samples,)))
    m2_sample = m2.sample(torch.Size((n_samples,)))

    evaluator = DivergenceEvaluator(m1_sample, m2_sample, n, m, l, m1, m2, 13, 'use_case_1', False)

    # Analytical KL
    evaluator.analytical_kl_calculation()

    # Ground Truth (lots of samples) Monte Carlo estimation
    evaluator.monte_carlo_gt_estimation()

    # Estimation configuration
    train_m = m / (m + (2 * l))  # Training set used to train the discriminator.
    evaluator.split_estimation_data(train_m)

    # Monte Carlo estimation
    evaluator.monte_carlo_estimation()

    # Probabilistic classifier estimation
    evaluator.probabilistic_classifier_estimation(plot_path, 5000)

    return evaluator.analytical_kl, evaluator.mc_js, evaluator.disc_kl, evaluator.disc_js, mean


def main():
    train = False
    n_threads = 30
    n_samples = 2400
    n = 2000
    m = 2000
    l = 200
    abs_path = os.path.dirname(os.path.abspath(__file__)) + os.sep
    results_path = abs_path + 'results_2' + os.sep + 'divergence_comparison' + os.sep
    os.makedirs(results_path, exist_ok=True)
    # lista de valores empezando en 0 y acabando en 5 con 20 valores
    # means = np.linspace(0., 10.5, 20).tolist()
    # means = np.linspace(0., 5, 20).tolist()
    means = np.linspace(0., 4, 20).tolist()

    # Parallelize estimation
    if train:
        results = Parallel(n_jobs=n_threads, verbose=10)(
            delayed(estimate_divs)(mean, n_samples, n, m, l, results_path) for mean in means)

        # Extract results
        kl_analytical = [0 for _ in range(len(means))]
        mc_js = [0 for _ in range(len(means))]
        disc_kl = [0 for _ in range(len(means))]
        disc_js = [0 for _ in range(len(means))]
        for res in results:
            kl_analytical[means.index(res[4])] = res[0].item()
            mc_js[means.index(res[4])] = res[1].item()
            disc_kl[means.index(res[4])] = res[2].item()
            disc_js[means.index(res[4])] = res[3].item()

        # Save results
        np.save(results_path + 'kl_analytical.npy', kl_analytical)
        np.save(results_path + 'disc_kl.npy', disc_kl)
        np.save(results_path + 'mc_js.npy', mc_js)
        np.save(results_path + 'disc_js.npy', disc_js)

    # Load results
    kl_analytical = np.load(results_path + 'kl_analytical.npy')
    disc_kl = np.load(results_path + 'disc_kl.npy')
    mc_js = np.load(results_path + 'mc_js.npy')
    disc_js = np.load(results_path + 'disc_js.npy')

    # Plot results
    fig = plt.figure(figsize=(13, 4))
    ax = plt.subplot(1, 2, 1)
    ax.plot(kl_analytical, disc_kl, 'o-')
    plt.xlabel('Analytical KL')
    plt.ylabel('Discriminator Estimated KL')
    ax.set_title('KL Divergence Comparison')
    plt.xlim(0, 35)
    plt.ylim(0, 35)
    plt.grid(True, alpha=0.5)
    ax = plt.subplot(1, 2, 2)
    ax.plot(mc_js, disc_js, 'o-')
    plt.xlabel('Montecarlo JS')
    plt.ylabel('Discriminator Estimated JS')
    ax.set_title('JS Divergence Comparison')
    plt.grid(True, alpha=0.5)

    fig.suptitle('Divergence Estimation Comparison')
    fig.savefig(results_path + 'kl_js_deviations.png')
    plt.show()
    plt.close(fig)
    # plt.plot(kl_analytical, disc_kl, 'o-')
    # plt.xlabel('Analytical KL')
    # plt.ylabel('Discriminator Estimated KL')
    # plt.title('KL Divergence Comparison')
    # plt.grid(True, alpha=0.5)
    # plt.savefig(abs_path + 'divs_comparison' + os.sep + 'kl.png')
    # plt.show()
    # plt.close()
    # plt.plot(mc_js, disc_js, 'o-')
    # plt.xlabel('Montecarlo JS')
    # plt.ylabel('Disc Estimated JS')
    # plt.title('JS Divergence Comparison')
    # plt.grid(True, alpha=0.5)
    # plt.savefig(abs_path + 'divs_comparison' + os.sep + 'js.png')
    # plt.show()
    # plt.close()


if __name__ == '__main__':
    main()
