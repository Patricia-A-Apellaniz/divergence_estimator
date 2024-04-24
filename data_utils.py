# Author: Borja Arroyo Galende
# Email: patricia.alonsod@upm.es
# Date: 15/04/2024

# Packages to import
import math
import torch

import numpy as np
import pandas as pd

from scipy import stats
from sklearn.svm import SVR
from torch import distributions as D
from sklearn.mixture import GaussianMixture
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def load_mvn(n_dims=5, dist=1):
    """
    Generate a multivariate normal distribution with n_dims dimensions and a distance of dist between the means of
    each dimension.
    Args:
        n_dims: number of dimensions
        dist:  distance between the means of each dimension

    Returns:

    """
    cov = np.identity(n_dims).astype("float32")
    choices = [0.3, 0.5, 0.7]
    for row in range(n_dims):
        for col in range(n_dims):
            if row == col:
                cov[row, col] = 10.
            else:
                cov[row, col] = np.random.choice(choices)
    cov /= 10.
    cov = cov @ cov.T
    locs = torch.rand((n_dims,)) * dist  # To generate random means.
    mvn = torch.distributions.MultivariateNormal(loc=locs, covariance_matrix=torch.tensor(cov))
    return mvn


def create_independent_gm(component_probs=torch.tensor([0.7, 0.3]), loc=torch.tensor([[1., 1.], [-1., -1.]]),
                          scale=torch.tensor([.5, .5])):
    """Creates an independent (isotropic) mixture of Gaussians and returns it

    Args:
        component_probs (torch.Tensor, optional): The categorical probabilities. Defaults to torch.tensor([0.7, 0.3]).
        loc (torch.Tensor, optional): The location of each Gaussian component. Defaults to torch.tensor([[1., 1.], [-1., -1.]]).
        scale (torch.Tensor, optional): The scale of each Gaussian component. Defaults to torch.tensor([.5, .5]).

    Returns:
        torch.distributions.Distribution: The Gaussian mixture model
    """
    categorical = D.Categorical(probs=component_probs)
    components = D.Independent(
        D.Normal(loc=loc, scale=scale),
        1
    )
    gm = D.MixtureSameFamily(categorical, components)
    return gm


def create_corr_bimodal_gm():
    """Creates a correlated bimodal mixture of Gaussians and returns it
    Args:
        component_probs (torch.Tensor, optional): The categorical probabilities. Defaults to torch.tensor([0.7, 0.3]).
        loc (torch.Tensor, optional): The location of each Gaussian component. Defaults to torch.tensor([[1., 1.], [-1., -1.]]).
        scale (torch.Tensor, optional): The scale of each Gaussian component. Defaults to torch.tensor([.5, .5]).
    Returns:
        torch.distributions.Distribution: The Gaussian mixture model
    """
    categorical = D.Categorical(probs=torch.tensor([0.7, 0.3]))
    components = D.MultivariateNormal(
        loc=torch.Tensor([[1., 1.], [-1., -1.]]),
        covariance_matrix=torch.Tensor([
            [[1., .2],
             [.2, 1.]],
            [[1., .2],
             [.2, 1.]]
        ])
    )
    gm = D.MixtureSameFamily(categorical, components)
    return gm


class GMM:

    def __init__(self, n_components=2, covariance_type='full', random_state=0):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.gmm = GaussianMixture(n_components=self.n_components, covariance_type=self.covariance_type,
                                   random_state=self.random_state)

    def fit(self, x):
        self.gmm.fit(x)
        return self.gmm

    def sample(self, n_samples):
        return self.gmm.sample(n_samples)

    def log_prob(self, x):
        device = x.device
        return torch.tensor(self.gmm.score_samples(x), device=device)


def preprocess_data(data):
    # Strings to numbers
    data, cols = str2num(data)

    # Normalise data
    feat_distributions = get_feat_distributions(data, cols)
    norm_df = normalize_data(data, feat_distributions)

    # Impute missing data
    imp_norm_df, mask, _ = impute_data(norm_df, gen_mask=False, feat_distributions=feat_distributions)
    return imp_norm_df


def str2num(df):
    df_copy = df.copy()
    cols = []
    for col in df_copy.columns:
        if df_copy[col].dtype == 'object':
            cols.append(col)
    # Initialize the LabelEncoder
    label_encoder = OrdinalEncoder()
    label_encoder.fit(df_copy[cols])

    # Fit and transform the data
    df_copy[cols] = label_encoder.transform(df_copy[cols])

    return df_copy, cols


def get_feat_distributions(df, cols=[]):
    n_feat = df.shape[1]
    feat_dist = []
    for i in range(n_feat):
        col = df.columns[i]
        values = df.iloc[:, i].unique()
        if len(values) == 1 and math.isnan(values[0]):
            values = np.zeros((1,))
        no_nan_values = values[~pd.isnull(values)]
        if col in cols:
            feat_dist.append(('categorical', np.unique(no_nan_values).size))
            # print('Categorical feature: ', col)
        if col == 'native-country':
            feat_dist.append(('categorical', np.unique(no_nan_values).size))
            # print('Categorical feature: ', col)

        elif 'soil_type' in col and all(np.sort(no_nan_values) == np.array(
                range(int(no_nan_values.min()), int(no_nan_values.min()) + len(no_nan_values)))):
            feat_dist.append(('categorical', np.unique(no_nan_values).size))
            # print('Categorical feature: ', col)
        else:
            if no_nan_values.size <= 2 and all(np.sort(no_nan_values) == np.array(
                    range(int(no_nan_values.min()), int(no_nan_values.min()) + len(no_nan_values)))):
                feat_dist.append(('bernoulli', 1))
                # print('Bernoulli feature: ', col)
            elif np.amin(np.equal(np.mod(no_nan_values, 1), 0)):
                # Check if values are floats but don't have decimals and transform to int. They are floats because of NaNs
                if no_nan_values.dtype == 'float64':
                    no_nan_values = no_nan_values.astype(int)
                # If number of categories is less than 20, categories start in 0 and categories are consecutive numbers, then it is categorical
                if np.unique(no_nan_values).size < 20 and np.amin(no_nan_values) == 0 and all(
                        np.sort(no_nan_values) == np.array(
                            range(int(no_nan_values.min()), int(no_nan_values.min()) + len(no_nan_values)))):
                    # feat_dist.append(('categorical', np.unique(no_nan_values).size))
                    feat_dist.append(('categorical', np.max(no_nan_values) + 1))
                    # print('Categorical feature: ', col)
                else:
                    feat_dist.append(('gaussian', 2))
                    # print('Gaussian feature: ', col)
            else:
                feat_dist.append(('gaussian', 2))
                # print('Gaussian feature: ', col)

    return feat_dist


def normalize_data(raw_df, feat_distributions):
    num_patient, num_feature = raw_df.shape
    norm_df = raw_df.copy()
    for i in range(num_feature):
        col = (raw_df.columns[i])
        values = raw_df.iloc[:, i]
        if len(values) == 1 and math.isnan(values[0]):
            values = np.zeros((1,))
        no_nan_values = values[~np.isnan(values)].values
        if feat_distributions[i][0] == 'gaussian':
            loc = np.mean(no_nan_values)
            scale = np.std(no_nan_values)
        elif feat_distributions[i][0] == 'bernoulli':
            if len(np.unique(no_nan_values)) == 1:
                continue
            loc = np.amin(no_nan_values)
            scale = np.amax(no_nan_values) - np.amin(no_nan_values)
        elif feat_distributions[i][0] == 'categorical':
            loc = np.amin(no_nan_values)
            scale = 1  # Do not scale
        elif feat_distributions[i][0] == 'weibull':
            loc = -1 if 0 in no_nan_values else 0
            scale = 1
        else:
            print('Distribution ', feat_distributions[i][0], ' not normalized')
            param = np.array([0, 1])  # loc = 0, scale = 1, means that data is not modified!!
            loc = param[-2]
            scale = param[-1]
        norm_df.iloc[:, i] = (raw_df.iloc[:, i] - loc) / scale if scale != 0 else raw_df.iloc[:, i] - loc

    return norm_df.reset_index(drop=True)


def zero_imputation(data):
    imp_data = data.copy()
    imp_data = imp_data.fillna(0)
    return imp_data


def mice_imputation(data, model='bayesian'):
    imp_data = data.copy()
    if model == 'bayesian':
        clf = BayesianRidge()
    elif model == 'svr':
        clf = SVR()
    else:
        raise RuntimeError('MICE imputation base_model not recognized')
    imp = IterativeImputer(estimator=clf, verbose=2, max_iter=30, tol=1e-10, imputation_order='roman')
    imp_data.iloc[:, :] = imp.fit_transform(imp_data)
    return imp_data


def statistics_imputation(data):
    imp_data = data.copy()
    n_samp, n_feat = imp_data.shape
    for i in range(n_feat):
        values = data.iloc[:, i].values
        if any(pd.isnull(values)):
            if len(values) == 1:
                values = np.zeros((1,))
            no_nan_values = values[~pd.isnull(values)]
            if no_nan_values.size <= 2 or no_nan_values.dtype in [object, str] or np.amin(
                    np.equal(np.mod(no_nan_values, 1), 0)):
                stats_value = stats.mode(no_nan_values, keepdims=True)[0][0]
            else:
                mean_value = no_nan_values.mean()
                stats_value = mean_value
            imp_data.iloc[:, i] = [stats_value if pd.isnull(x) else x for x in imp_data.iloc[:, i]]

    return imp_data


def impute_data(df, imp_mask=True, gen_mask=False, feat_distributions=None, mode='stats'):
    # If missing data exists, impute it
    if df.isna().any().any():
        # Data imputation
        if mode == 'zero':
            imp_df = zero_imputation(df)
        elif mode == 'stats':
            imp_df = statistics_imputation(df)
        else:
            imp_df = mice_imputation(df)

        # Generate missing data mask. Our model uses it to ignore missing data, although it has been imputed
        if imp_mask:
            nans = df.isna()
            mask = nans.replace([True, False], [0, 1])
        else:
            mask = np.ones((df.shape[0], df.shape[1]))
            mask = pd.DataFrame(mask, columns=imp_df.columns)

        # Concatenate mask to data to generate synthetic missing positions too
        if gen_mask:
            mask_names = ['imp_mask_' + col for col in df.columns]
            mask.columns = mask_names
            imp_df = pd.concat([imp_df, mask], axis=1)
            tr_mask_df = mask.copy()
            tr_mask_df.columns = ['tr_mask_' + col for col in df.columns]
            mask = pd.concat([mask, tr_mask_df.replace(0, 1)], axis=1)  # It is necessary to concatenate mask to mask
            # with all ones for training purposes
            # Get new data distributions. Mask features should be bernoulli!
            feat_distributions.extend([('bernoulli', 1) for _ in range(mask.shape[1] - len(feat_distributions))])
    else:
        imp_df = df.copy()
        mask = np.ones((df.shape[0], df.shape[1]))
        mask = pd.DataFrame(mask, columns=imp_df.columns)
        mask = mask.astype(int)
    return imp_df, mask, feat_distributions
