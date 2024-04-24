# Author: Ana Jiménez & Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 15/04/2024

# Packages to import
import torch

import pandas as pd
import numpy as np

from utils import set_seed
from data_utils import load_mvn, create_independent_gm, GMM, create_corr_bimodal_gm, preprocess_data


# Create multivariate gaussian distributions and sample from them
def load_multivariate_gaussian_distributions(m, l, seed, n_dims=10):
    set_seed(seed)  # same distributions for each seed
    dist_r = load_mvn(n_dims=n_dims, dist=1)
    dist_s = load_mvn(n_dims=n_dims, dist=0)
    set_seed(int(seed * 2 + 1))  # different samples for each seed
    x_r = dist_r.sample(torch.Size((m + 2 * l,)))
    x_s = dist_s.sample(torch.Size((m + 2 * l,)))
    return x_r, x_s, dist_r, dist_s


# Create gaussian mixture distributions and sample from them
def load_gaussian_mixture_distributions(m, l, seed):
    set_seed(seed)  # same distributions for each seed
    dist_r = create_independent_gm()
    dist_s = create_independent_gm(component_probs=torch.tensor([0.5, 0.5]), loc=torch.tensor([[0., 0.], [-1., -1.]]),
                                   scale=torch.tensor([.5, .5]))
    set_seed(int(seed * 2 + 1))  # different samples for each seed
    x_r_tr = dist_r.sample(torch.Size((m,)))
    x_r_ev = dist_r.sample(torch.Size((l * 2,)))
    x_r = torch.cat((x_r_tr, x_r_ev), dim=0)

    x_s_tr = dist_s.sample(torch.Size((m,)))
    x_s_ev = dist_s.sample(torch.Size((l * 2,)))
    x_s = torch.cat((x_s_tr, x_s_ev), dim=0)
    return x_r, x_s, dist_r, dist_s


# Create gaussian mixture distributions and sample from them
def load_gaussian_mixtures_distributions_generative_process(n, m, l, seed):
    set_seed(seed)  # same distributions for each seed
    dist_r = create_corr_bimodal_gm()

    x_r = dist_r.sample(torch.Size((n,)))
    dist_s = GMM(n_components=2, random_state=23)
    dist_s.fit(x_r)

    set_seed(int(seed * 2 + 1))  # different samples for each seed
    x_r_tr = dist_r.sample(torch.Size((m,)))
    x_r_ev = dist_r.sample(torch.Size((l * 2,)))
    x_r = torch.cat((x_r_tr, x_r_ev), dim=0)

    x_s_tr = torch.tensor(dist_s.sample(m)[0][0:m, :], dtype=torch.float32)
    x_s_ev = torch.tensor(dist_s.sample(l * 2)[0], dtype=torch.float32)
    x_s = torch.cat((x_s_tr, x_s_ev), dim=0)
    return x_r, x_s, dist_r, dist_s


def load_real_data_distributions_generative_process(m, l, path):
    # Load real data
    real_df = pd.read_csv(path + 'real_data.csv')
    gen_df = pd.read_csv(path + 'gen_data.csv')

    # Select same number of samples
    n_real, n_syn = real_df.shape[0], gen_df.shape[0]
    n_samples_available = min((n_syn, n_real))
    n_samples = m + (2 * l)
    if n_samples_available < n_samples:
        print(f'Warning: only {n_samples_available} samples available.')
        n_samples = n_samples_available
    real_df = real_df.sample(frac=1, random_state=0).reset_index(drop=True)
    gen_df = gen_df.sample(frac=1, random_state=0).reset_index(drop=True)
    real_df = real_df[0: n_samples]
    gen_df = gen_df[0: n_samples]

    # Normalize data
    x = pd.concat([real_df, gen_df], axis=0)
    y = np.concatenate([np.ones((real_df.shape[0],)), np.zeros((gen_df.shape[0],))])
    x = preprocess_data(x)
    real_df = x[y == 1]
    gen_df = x[y == 0]

    # Dataframe to Tensor
    real_df = real_df.values
    synthetic_df = gen_df.values
    df_r = torch.tensor(real_df, dtype=torch.float32)
    df_s = torch.tensor(synthetic_df, dtype=torch.float32)

    return df_r, df_s, None, None


def load_data(experiment, n, m, l, seed, data_path=None):
    if experiment == 'use_case_1':
        return load_multivariate_gaussian_distributions(m, l, seed)
    elif experiment == 'use_case_2':
        return load_gaussian_mixture_distributions(m, l, seed)
    elif experiment == 'use_case_3':
        return load_gaussian_mixtures_distributions_generative_process(n, m, l, seed)
    elif experiment == 'use_case_4':
        return load_real_data_distributions_generative_process(m, l, data_path)
    else:
        raise RuntimeError('Experiment not recognized')
