# Author: Ana Jiménez & Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 15/04/2024

# Packages to import
import os
import torch
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import norm


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)


def plot_paper_disc_losses(results_path):
    for div in ['KL', 'JS']:
        fig = plt.figure(figsize=(20, 5))
        # fig.tight_layout(pad=5.0)
        for i, size in enumerate([20, 200, 2000]):
            ax = plt.subplot(1, 3, i + 1)
            dict_path = results_path + str(size) + '_' + str(size) + os.sep + div + '_training_results.pkl'
            with open(dict_path, 'rb') as handle:
                results = pickle.load(handle)
            ax.plot(results['training_results']['tr_loss'], label='Training Loss')
            ax.plot(results['training_results']['eval_loss'], label='Validation Loss')
            ax.set_title('M = ' + str(size) + ', L = ' + str(size))

            if div == 'JS':
                ax.axhline(results['estimates'][1], color='g', label='Training Bound', linestyle='dashed')
                ax.axhline(results['estimates'][3], color='r', label='Validation Bound', linestyle='dashed')

            # if i == 1:
            #     plt.legend(loc='upper right')
            # else:
            #     plt.legend(loc='center right')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.5)

        fig.legend(['Training Loss', 'Validation Loss', 'Training Bound', 'Validation Bound'], loc='center right')
        fig.suptitle('$D_{' + str(div) + '}$ ' + 'Discriminator Training and Validation Losses')
        fig_name = f'{div}_discriminator_losses.png'

        fig.savefig(results_path + fig_name)
        plt.show()
        plt.close(fig)


def plot_n_influence(results_path, experiment, N, M, L):
    # Load csv files for KL and JS
    kl_df = pd.read_csv(results_path + 'kl.csv')
    js_df = pd.read_csv(results_path + 'js.csv')

    # Plot N influence
    fig = plt.figure(figsize=(15, 5))
    for i, div in enumerate(['kl', 'js']):
        ax = plt.subplot(1, 2, i + 1)
        if div == 'kl':
            div_df = kl_df
        else:
            div_df = js_df
        ax.set_title('$D_{' + div + '}$ Estimation')
        plt.xlabel('N')
        plt.ylabel('Divergence Estimation')
        plt.grid(True, alpha=0.5)
        # Set highest values for M and L
        m = M[-1]
        l = L[-1]
        m_filtered_df = div_df[div_df['M'] == m]
        l_filtered_df = m_filtered_df[m_filtered_df['L'] == l]
        gt_div_values = []
        disc_div_values = []
        for n in N:
            n_filtered_df = l_filtered_df[l_filtered_df['N'] == n]
            disc_div_values.append(n_filtered_df['Disc ' + div.upper()].values[0])
            if experiment == 'use_case_3':
                gt_div_values.append(n_filtered_df['Ground Truth MC ' + div.upper()].values[0])

        # Plot values
        disc_div_values = [round(value, 3) for value in disc_div_values]
        ax.plot(N, disc_div_values, label='Discriminator ' + div.upper())
        if experiment == 'use_case_3':
            gt_div_values = [round(value, 3) for value in gt_div_values]
            ax.plot(N, gt_div_values, label='Ground Truth MC ' + div.upper())
            plt.legend()

    fig.suptitle('N Influence in Divergences Estimation')
    fig.savefig(results_path + 'divs_n_influence.png')
    plt.show()
    plt.close(fig)


def plot_box_plots(results_path, experiment, N, M, L):
    # Plot box plots
    if experiment == 'use_case_1' or experiment == 'use_case_2':
        N = ['-']
    for n in N:
        fig_kl, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(16, 6), sharey=True)
        ax_kl = [ax1, ax2, ax3]
        fig_kl.tight_layout(pad=5.0)
        fig_js, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(16, 6), sharey=True)
        ax_js = [ax1, ax2, ax3]
        fig_js.tight_layout(pad=5.0)

        bplot_kl = []
        bplot_js = []
        colors = ['lightblue', 'lightpink', 'lightyellow']
        for i, m in enumerate(M):
            if experiment == 'use_case_3':
                ax_kl[i].set_title('$D_{kl}$ Estimation for M = ' + str(m) + ' and N = ' + str(n))
                ax_js[i].set_title('$D_{js}$ Estimation for M = ' + str(m) + ' and N = ' + str(n))
            else:
                ax_kl[i].set_title('$D_{kl}$ Estimation for M = ' + str(m))
                ax_js[i].set_title('$D_{js}$ Estimation for M = ' + str(m))
            l_errors_kl = []
            l_errors_js = []
            for l in L:
                if experiment == 'use_case_3':
                    kl_df = pd.read_csv(results_path + str(n) + '_' + str(m) + '_' + str(l) + os.sep + 'seeds_kl.csv')
                    js_df = pd.read_csv(results_path + str(n) + '_' + str(m) + '_' + str(l) + os.sep + 'seeds_js.csv')
                else:
                    kl_df = pd.read_csv(results_path + str(m) + '_' + str(l) + os.sep + 'seeds_kl.csv')
                    js_df = pd.read_csv(results_path + str(m) + '_' + str(l) + os.sep + 'seeds_js.csv')
                if experiment == 'use_case_1':
                    l_errors_kl.append(np.abs(kl_df['Analytical KL'] - kl_df['Disc KL']))
                elif experiment == 'use_case_2' or experiment == 'use_case_3':
                    l_errors_kl.append(np.abs(kl_df['Ground Truth MC KL'] - kl_df['Disc KL']))
                l_errors_js.append(np.abs(js_df['Ground Truth MC JS'] - js_df['Disc JS']))
            bplot_kl.append(ax_kl[i].boxplot(l_errors_kl, vert=True, labels=L, patch_artist=True,
                                             medianprops=dict(color='red', linewidth=1.5)))
            bplot_js.append(ax_js[i].boxplot(l_errors_js, vert=True, labels=L, patch_artist=True,
                                             medianprops=dict(color='red', linewidth=1.5)))

        # fill with colors
        for plot in bplot_kl:
            for patch, color in zip(plot['boxes'], colors):
                patch.set_facecolor(color)
        for plot in bplot_js:
            for patch, color in zip(plot['boxes'], colors):
                patch.set_facecolor(color)

        # adding horizontal grid lines
        for ax in ax_kl:
            ax.yaxis.grid(True)
            ax.set_xlabel('L values')
            ax.set_ylabel('Estimation Errors')
        # adding horizontal grid lines
        for ax in ax_js:
            ax.yaxis.grid(True)
            ax.set_xlabel('L values')
            ax.set_ylabel('Estimation Errors')

        if experiment == 'use_case_3':
            fig_kl.savefig(results_path + str(n) + '_kl_error_boxplots.png')
            fig_js.savefig(results_path + str(n) + '_js_error_boxplots.png')
        else:
            fig_kl.savefig(results_path + 'kl_error_boxplots.png')
            fig_js.savefig(results_path + 'js_error_boxplots.png')
        plt.show()
        plt.close(fig_kl)
        plt.close(fig_js)

def print_table(results_path, experiment, model):
    from tabulate import tabulate
    if experiment == 'use_case_4':
        kl_df = pd.read_csv(results_path + model + os.sep + 'kl.csv')
        js_df = pd.read_csv(results_path + model + os.sep + 'js.csv')
    else:
        kl_df = pd.read_csv(results_path + 'kl.csv')
        js_df = pd.read_csv(results_path + 'js.csv')

    tab = []
    # Concatenar nombres de las columnas de kl_df y de js_df
    # columns = kl_df.columns.tolist() + js_df.columns.tolist()[3:]
    columns = kl_df.columns.tolist()[:4] + kl_df.columns.tolist()[6:] + js_df.columns.tolist()[5:]
    for i in range(len(kl_df)):
        tab.append([kl_df['N'][i],
                    kl_df['M'][i],
                    kl_df['L'][i],
                    kl_df['Analytical KL'][i],
                    # kl_df['Ground Truth MC KL'][i],
                    # kl_df['Ground Truth MC KL (CI)'][i],
                    kl_df['MC KL'][i],
                    kl_df['MC KL (CI)'][i],
                    kl_df['Disc KL'][i],
                    kl_df['Disc KL (CI)'][i],
                    # js_df['Ground Truth MC JS'][i],
                    # js_df['Ground Truth MC JS (CI)'][i],
                    js_df['MC JS'][i],
                    js_df['MC JS (CI)'][i],
                    js_df['Disc JS'][i],
                    js_df['Disc JS (CI)'][i]])

    print(tabulate(tab, headers=columns, tablefmt='orgtbl'))




def save_results(evaluators, results_path, experiment, N, M, L, n_seeds, model):
    # Save results in dictionary
    # Initialize results dictionary
    results = {}
    if experiment != 'use_case_3' and experiment != 'use_case_4':
        N = ['-']
    for n in N:
        results[n] = {}
        for m in M:
            results[n][m] = {}
            for l in L:
                results[n][m][l] = []
    # Go through parallelization output and save results in dictionary
    for ev in evaluators:
        n, m, l, seed, estimates = ev.get_info()
        if n is None:
            n = '-'
        results[n][m][l].append(estimates)
    # Save dictionary
    if experiment == 'use_case_4':
        results_path = results_path + model + os.sep
    with open(results_path + 'results.pkl', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Save results in .csv file
    kl_columns = ['N', 'M', 'L', 'Analytical KL', 'Ground Truth MC KL', 'Ground Truth MC KL (CI)', 'MC KL',
                  'MC KL (CI)',
                  'Disc KL', 'Disc KL (CI)']
    js_columns = ['N', 'M', 'L', 'Ground Truth MC JS', 'Ground Truth MC JS (CI)', 'MC JS', 'MC JS (CI)', 'Disc JS',
                  'Disc JS (CI)']
    kl_df = pd.DataFrame(columns=kl_columns)
    js_df = pd.DataFrame(columns=js_columns)

    seeds_kl_columns = ['Analytical KL', 'Ground Truth MC KL', 'MC KL', 'Disc KL', ]
    seeds_js_columns = ['Ground Truth MC JS', 'MC JS', 'Disc JS', ]
    for n in N:
        for m in M:
            for l in L:
                divs = {'analytical_kl': np.zeros((n_seeds, 1)),
                        'mc_gt_kl': np.zeros((n_seeds, 1)),
                        'mc_gt_js': np.zeros((n_seeds, 1)),
                        'mc_kl': np.zeros((n_seeds, 1)),
                        'mc_js': np.zeros((n_seeds, 1)),
                        'disc_kl': np.zeros((n_seeds, 1)),
                        'disc_js': np.zeros((n_seeds, 1))}
                # Create lists with seeds values for each estimator
                for seed in range(n_seeds):
                    res = results[n][m][l][seed]
                    divs['analytical_kl'][seed] = res[0]
                    divs['mc_gt_kl'][seed] = res[1]
                    divs['mc_gt_js'][seed] = res[2]
                    divs['mc_kl'][seed] = res[3]
                    divs['mc_js'][seed] = res[4]
                    divs['disc_kl'][seed] = res[5]
                    divs['disc_js'][seed] = res[6]

                # Save results for each M, L
                kl_data = np.concatenate((divs['analytical_kl'], divs['mc_gt_kl'], divs['mc_kl'], divs['disc_kl']),
                                         axis=1)
                js_data = np.concatenate((divs['mc_gt_js'], divs['mc_js'], divs['disc_js']), axis=1)
                seeds_kl_df = pd.DataFrame(kl_data, columns=seeds_kl_columns)
                seeds_kl_df.replace(-1, '-', inplace=True)  # Replace -1 values with '-'
                seeds_js_df = pd.DataFrame(js_data, columns=seeds_js_columns)
                seeds_js_df.replace(-1, '-', inplace=True)  # Replace -1 values with '-'

                # Saving path
                if experiment == 'use_case_3' or experiment == 'use_case_4':
                    saving_path = results_path + str(n) + '_' + str(m) + '_' + str(l) + os.sep
                else:
                    saving_path = results_path + str(m) + '_' + str(l) + os.sep
                seeds_kl_df.to_csv(saving_path + 'seeds_kl.csv', index=False)
                seeds_js_df.to_csv(saving_path + 'seeds_js.csv', index=False)

                # Save mean and confidence interval results
                z_score = norm.ppf(1 - 0.05 / 2)
                kl_df = pd.concat([kl_df, pd.DataFrame([[n, m, l,
                                                         divs[
                                                             'analytical_kl'].mean() if experiment == 'use_case_1' else '-',
                                                         divs['mc_gt_kl'].mean() if '4' not in experiment else '-',
                                                         divs['mc_gt_kl'].std() * z_score / np.sqrt(
                                                             n_seeds) if '4' not in experiment else '-',
                                                         divs['mc_kl'].mean() if '4' not in experiment else '-',
                                                         divs['mc_kl'].std() * z_score / np.sqrt(
                                                             n_seeds) if '4' not in experiment else '-',
                                                         divs['disc_kl'].mean(),
                                                         divs['disc_kl'].std() * z_score / np.sqrt(n_seeds)]],
                                                       columns=kl_columns)])
                js_df = pd.concat([js_df, pd.DataFrame([[n, m, l,
                                                         divs['mc_gt_js'].mean() if '4' not in experiment else '-',
                                                         divs['mc_gt_js'].std() * z_score / np.sqrt(
                                                             n_seeds) if '4' not in experiment else '-',
                                                         divs['mc_js'].mean() if '4' not in experiment else '-',
                                                         divs['mc_js'].std() * z_score / np.sqrt(
                                                             n_seeds) if '4' not in experiment else '-',
                                                         divs['disc_js'].mean(),
                                                         divs['disc_js'].std() * z_score / np.sqrt(n_seeds)]],
                                                       columns=js_columns)])

    kl_df.to_csv(results_path + 'kl.csv', index=False)
    js_df.to_csv(results_path + 'js.csv', index=False)
