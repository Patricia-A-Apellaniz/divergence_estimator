# Author: Ana Jiménez & Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 15/04/2024

# Packages to import
import torch

from estimator.divergence import KL, JS


class DivergenceEvaluator:
    def __init__(self, x_p, x_q, n, m, l, dist_p, dist_q, seed, experiment, verbose):
        self.x_p = x_p
        self.x_q = x_q
        self.n = n
        self.m = m
        self.l = l
        self.dist_p = dist_p
        self.dist_q = dist_q
        self.experiment = experiment
        self.seed = seed
        self.verbose = verbose

        self.kl = None
        self.js = None
        self.analytical_kl = None
        self.mc_gt_kl = None
        self.mc_gt_js = None
        self.mc_kl = None
        self.mc_js = None
        self.disc_kl = None
        self.disc_js = None

    def analytical_kl_calculation(self):
        if self.experiment == 'use_case_1':
            if self.analytical_kl is None:  # If different seeds, analytical kl is not calculated again
                self.analytical_kl = torch.distributions.kl_divergence(self.dist_p, self.dist_q)
        else:
            if self.verbose:
                print('Analytical KL not available for this experiment')

    def monte_carlo_gt_estimation(self, iterations=100, gt_samples=5000):
        # Define KL, JS object
        self.kl = KL(self.dist_p, self.dist_q, self.x_p, self.x_q, div='KL')
        self.js = JS(self.dist_p, self.dist_q, self.x_p, self.x_q, div='JS')

        if self.experiment == 'use_case_4':
            if self.verbose:
                print('MC ground truth divergence not available for this experiment')
        else:
            mc_gt_kl = torch.tensor(0)
            mc_gt_js = torch.tensor(0)
            for i in range(1, iterations + 1):
                # Sample from p and q
                p_test = self.dist_p.sample((gt_samples,))
                if self.experiment == 'use_case_3':
                    q_test, _ = self.dist_q.sample(gt_samples)
                    q_test = torch.tensor(q_test, dtype=torch.float32)
                else:
                    q_test = self.dist_q.sample((gt_samples,))

                # Compute KL and JS divergence using Monte Carlo
                kl1 = self.kl.mc(p_test)
                js1 = self.js.mc(p_test, q_test)

                # Update mean
                mc_gt_kl = (mc_gt_kl * (i - 1) + kl1) / i
                mc_gt_js = (mc_gt_js * (i - 1) + js1) / i

            self.mc_gt_kl = mc_gt_kl
            self.mc_gt_js = mc_gt_js

    def split_estimation_data(self, train_m):
        self.kl.split_data(train_m=train_m, n=self.n, m=self.m, l=self.l)
        self.js.split_data(train_m=train_m, n=self.n, m=self.m, l=self.l)

    # Compute KL and JS divergence using Monte Carlo
    def monte_carlo_estimation(self):
        if self.experiment == 'use_case_4':
            if self.verbose:
                print('MC ground truth divergence not available for this experiment')
        else:
            self.mc_kl = self.kl.mc()
            self.mc_js = self.js.mc()

    # Compute KL and JS divergence using our approach
    def probabilistic_classifier_estimation(self, path, epochs=5000):
        kl_training_results, kl_estimates = self.kl.forward(epochs, path)
        js_training_results, js_estimates = self.js.forward(epochs, path)
        self.disc_kl = kl_estimates
        self.disc_js = js_estimates[2]

        # TODO: do something with training results?

    def get_info(self):
        estimates = (self.analytical_kl.item() if self.analytical_kl is not None else -1,
                     self.mc_gt_kl.item() if self.mc_gt_kl is not None else -1,
                     self.mc_gt_js.item() if self.mc_gt_js is not None else -1,
                     self.mc_kl.item() if self.mc_kl is not None else -1,
                     self.mc_js.item() if self.mc_js is not None else -1,
                     self.disc_kl.item(), self.disc_js.item())
        return self.n, self.m, self.l, self.seed, estimates
