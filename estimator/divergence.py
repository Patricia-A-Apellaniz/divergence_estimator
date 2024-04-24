# Author: Ana Jiménez & Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 15/04/2024

# Packages to import
import torch
import pickle

import pandas as pd

from sklearn.utils import shuffle
from estimator.discriminator import Discriminator
from sklearn.metrics import accuracy_score, f1_score
from estimator.est_utils import to_dataloader, plot_loss, tensor_to_dataloader


# In use_case_3 and use_case_4 we validate the generative processes.
# Compare distributions using classical ML methods for classification.
def detection_validation(p_train, q_train, p_eval, q_eval):
    x_train = torch.cat([p_train, q_train])
    y_train = torch.tensor([1.0] * len(p_train) + [0.0] * len(q_train))
    x_eval = torch.cat([p_eval, q_eval])
    y_eval = torch.tensor([1.0] * len(p_eval) + [0.0] * len(q_eval))
    x_train, y_train = shuffle(x_train, y_train, random_state=0)
    x_train = x_train.detach().cpu().numpy()
    y_train = y_train.detach().cpu().numpy()
    x_eval = x_eval.detach().cpu()
    y_eval = y_eval.detach().cpu()

    results = {}
    for model in ['RF', 'Log_Reg']:
        results[model] = {}
        if model == 'RF':
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=3, max_depth=2)
        elif model == 'Log_Reg':
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(random_state=0)
        else:
            raise NotImplementedError(f'Model {model} not available')
        clf.fit(x_train, y_train)
        y_pred_train = clf.predict(x_train)
        acc_train = accuracy_score(y_train, y_pred_train)
        f1_train = f1_score(y_train, y_pred_train)
        results[model]['train'] = {'accuracy': acc_train, 'f1': f1_train}

        y_pred_eval = clf.predict(x_eval)
        acc_eval = accuracy_score(y_eval, y_pred_eval)
        f1_eval = f1_score(y_eval, y_pred_eval)
        validation_predictions = clf.predict_proba(x_eval)
        validation_predictions = torch.tensor(validation_predictions[:, 1], dtype=torch.float32)
        loss_i = torch.nn.BCELoss()
        loss_j = loss_i(validation_predictions.clone().detach().requires_grad_(True), y_eval)
        results[model]['eval'] = {'accuracy': acc_eval, 'f1': f1_eval, 'loss': loss_j.item()}

    return results


class Divergence:
    def __init__(self, p, q, x_p, x_q, div, train_m=1.0, n=None, m=None, l=None):
        self.p = p
        self.q = q
        self.x_p = x_p
        self.x_q = x_q
        assert len(x_p) == len(x_q), 'Distribution samples must have the same length!'
        self.train_m = train_m
        self.n = n
        self.m = m
        self.l = l
        self.div = div

        # Use all data for training, evaluation and testing
        if train_m == 1.0:
            self.p_train, self.p_eval = self.x_p, self.x_p
            self.q_train, self.q_eval = self.x_q, self.x_q
            self.p_val = self.p_eval
            self.q_val = self.q_eval

    # Split data into train, validation (for early stopping) and test (to estimate)
    def split_data(self, train_m, n, m, l):
        assert 0.0 < train_m <= 1.0, 'Fraction must be in (0, 1]'
        self.n = n
        self.m = m
        self.l = l
        split_idx = int(len(self.x_p) * train_m)
        self.p_train, self.p_eval = self.x_p[:split_idx], self.x_p[split_idx:]
        self.p_test, self.p_val = self.p_eval[:len(self.p_eval) // 2], self.p_eval[len(self.p_eval) // 2:]
        self.q_train, self.q_eval = self.x_q[:split_idx], self.x_q[split_idx:]
        self.q_test, self.q_val = self.q_eval[:len(self.q_eval) // 2], self.q_eval[len(self.q_eval) // 2:]
        real_m = len(self.p_train)
        real_l = len(self.p_val)
        if real_m != self.m or real_l != self.l:
            raise ValueError(f'Expected m={self.m} and l={self.l} but got m={real_m} and l={real_l}')

    def estimate(self, log_ratio):
        pass

    def fit(self, disc_model, epochs):
        # Data loaders
        train_dl = to_dataloader(self.p_train, self.q_train)
        val_dl = to_dataloader(self.p_val, self.q_val)
        df = pd.concat([pd.DataFrame(self.p_train.cpu().numpy()), pd.DataFrame(self.q_train.cpu().numpy())])
        df['label'] = [1.0] * len(self.p_train) + [0.0] * len(self.q_train)

        # Train model
        tr_loss, eval_loss = disc_model.train_loop(train_dl, val_dl, epochs)

        # Evaluate model
        test_dl = to_dataloader(self.p_test, self.q_test, shuffle=False)
        y_pred = disc_model.predict(test_dl, sigmoid=True).clone().detach().requires_grad_(True)
        # Compute the predicted class label for each example using where and if
        y_pred_label = torch.where(y_pred > 0.5, torch.tensor(1.0), torch.tensor(0.0))
        y_eval = test_dl.dataset.y
        acc = accuracy_score(y_eval.cpu(), y_pred_label.cpu())
        f1 = f1_score(y_eval.cpu(), y_pred_label.cpu())
        loss = torch.nn.BCELoss()(y_pred, y_eval)

        # Detection validation (to compare divergence with other metrics)
        det_res = detection_validation(self.p_train, self.q_train, self.p_val, self.q_val)

        # Save results
        results = {'tr_loss': tr_loss, 'eval_loss': eval_loss, 'accuracy': acc, 'f1': f1, 'loss': loss.item(),
                   'detection_validation': det_res}

        return results

    def forward(self, epochs, path, layers=(256, 64, 32)):
        # Init model
        disc_model = Discriminator(layers)

        # Train and estimate ratio for divergences
        training_results = self.fit(disc_model, epochs)
        estimates = self.estimate(disc_model)

        # Save training_results
        results = {'training_results': training_results, 'estimates': estimates}
        training_results_path = path + self.div + '_training_results.pkl'
        with open(training_results_path, 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Plot loss
        plot_loss(training_results['tr_loss'], training_results['eval_loss'], estimates, path, self.div, self.n, self.m,
                  self.l)
        return training_results, estimates


class KL(Divergence):

    def mc(self, p_test=None):
        if p_test is None:
            p_test = self.p_test
        log_p = self.p.log_prob(p_test)
        log_q = self.q.log_prob(p_test)

        return (log_p - log_q).mean()

    def estimate(self, disc_model):
        test_dl = tensor_to_dataloader(self.p_test)
        log_ratio = disc_model.predict(test_dl, sigmoid=False)
        estimate = torch.mean(log_ratio)
        return estimate


class JS(Divergence):

    def mc(self, p_test=None, q_test=None):
        if p_test is None or q_test is None:
            p_test = self.p_test
            q_test = self.q_test

        p_p = torch.exp(self.p.log_prob(p_test))
        q_p = torch.exp(self.q.log_prob(p_test))

        p_q = torch.exp(self.p.log_prob(q_test))
        q_q = torch.exp(self.q.log_prob(q_test))

        t1 = torch.log2(2 * p_p) - torch.log2(q_p + p_p)
        t2 = (torch.log2(2 * q_q) - torch.log2(p_q + q_q))

        return (t1.mean() + t2.mean()) / 2

    def bound_divergence(self, disc_model, p, q):
        p_dl = tensor_to_dataloader(p)
        q_dl = tensor_to_dataloader(q)
        prob_p = disc_model.predict(p_dl, sigmoid=True)
        prob_q = 1 - disc_model.predict(q_dl, sigmoid=True)
        prob_p = torch.clamp(prob_p, min=1e-7, max=1)
        prob_q = torch.clamp(prob_q, min=1e-7, max=1)
        estimate = 0.5 * (1 + torch.log2(prob_p).mean() + 1 + torch.log2(prob_q).mean())
        estimate_ln = 0.5 * ((torch.log(prob_p)).mean()) + 0.5 * ((torch.log(prob_q)).mean()) + torch.log(
            torch.tensor(2))
        bound = (-2 * estimate_ln) + torch.log(torch.tensor(4))

        return estimate, bound

    def estimate(self, disc_model):
        # Train: Estimate the js divergence and the bound to the loss function
        tr_estimate, tr_bound = self.bound_divergence(disc_model, self.p_train, self.q_train)

        # Val: Estimate the js divergence and the bound to the loss function
        val_estimate, val_bound = self.bound_divergence(disc_model, self.p_test, self.q_test)

        return tr_estimate, tr_bound, val_estimate, val_bound
