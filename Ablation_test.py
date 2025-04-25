"""Ablation experiment code"""
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_X_y
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
import os

os.environ["LOKY_MAX_CPU_COUNT"] = "8"
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class OverSample(BaseEstimator, TransformerMixin):
    def __init__(self, method='cluster', random_state=42):
        # Initialize method
        self.method = method
        self.random_state = random_state
        self.valid_clusters_ = []

    def _determine_clusters(self, X):
        # The optimal number of clusters is automatically determined using BIC
        min_clusters = 2
        max_clusters = min(10, len(X) - 1)
        best_bic = np.inf

        for n in range(min_clusters, max_clusters + 1):
            gmm = GaussianMixture(n_components=n, random_state=self.random_state)
            gmm.fit(X)
            bic = gmm.bic(X)
            if bic < best_bic:
                best_bic = bic
                best_n = n
        return best_n

    def _generate_samples(self, X_min, clusters, n_samples):
        # New minority samples are synthesized according to the clustering results
        synthetic = []
        rng = check_random_state(self.random_state)

        for cluster_id in np.unique(clusters):
            cluster_data = X_min[clusters == cluster_id]
            if len(cluster_data) < 2:
                continue

            # Calculate (proportionally) the number of samples that should be synthesized for this cluster
            cluster_ratio = len(cluster_data) / len(X_min)
            n_cluster_samples = int(n_samples * cluster_ratio)

            # Randomly select two samples from the cluster for linear interpolation
            for _ in range(n_cluster_samples):
                i1, i2 = rng.choice(len(cluster_data), 2, replace=False)
                point1 = cluster_data[i1]
                point2 = cluster_data[i2]
                alpha = rng.uniform(0, 1)
                new_point = point1 + alpha * (point2 - point1)
                synthetic.append(new_point)

        return np.array(synthetic)[:n_samples]

    def _generate_samples_smote(self, X_min, n_samples):
        # Generate samples using SMOTE
        if n_samples <= 0 or len(X_min) == 0:
            return np.zeros((0, X_min.shape[1]))

        synthetic = []
        rng = check_random_state(self.random_state)
        for _ in range(n_samples):
            i, nn = rng.choice(len(X_min), 2, replace=False)
            alpha = rng.uniform(0, 1)
            new_point = X_min[i] + alpha * (X_min[nn] - X_min[i])
            synthetic.append(new_point)
        return np.array(synthetic)

    def fit_resample(self, X, y):
        # Fit the resampled data
        X, y = check_X_y(X, y)
        classes = np.unique(y)

        if len(classes) != 2:
            return X, y

        majority_class = max(classes, key=lambda c: np.sum(y == c))
        minority_class = min(classes, key=lambda c: np.sum(y == c))

        X_min = X[y == minority_class]
        X_maj = X[y == majority_class]
        n_to_generate = len(X_maj) - len(X_min)

        if n_to_generate <= 0:
            return X, y

        elif self.method == 'cluster':
            # Generate samples using clustering
            n_clusters = self._determine_clusters(X_min)
            gmm = GaussianMixture(n_components=n_clusters, random_state=self.random_state)
            clusters = gmm.fit_predict(X_min)
            synthetic_X = self._generate_samples(X_min, clusters, n_to_generate)
        elif self.method == 'smote':
            # Generate samples using SMOTE
            synthetic_X = self._generate_samples_smote(X_min, n_to_generate)

        synthetic_y = np.full(len(synthetic_X), minority_class)
        return np.vstack([X, synthetic_X]), np.concatenate([y, synthetic_y])


def stump_classify(data_matrix, dim, thresh, inequal):
    # Decision tree classifier
    m = data_matrix.shape[0]
    ret = np.ones(m)
    if inequal == 'lt':
        ret[data_matrix[:, dim] <= thresh] = -1.0  # Values less than or equal to the threshold are set to -1
    else:
        ret[data_matrix[:, dim] > thresh] = -1.0  # Values greater than the threshold are set to -1
    return ret


def build_stump(X, y, D):
    # Construct an optimal one-level decision tree
    m, n = X.shape
    best_stump = {}
    best_est = np.zeros(m)
    min_err = np.inf

    for dim in range(n):
        vals = np.unique(X[:, dim])
        for thresh in vals:
            for inequal in ('lt', 'gt'):
                pred = stump_classify(X, dim, thresh, inequal)
                err = (pred != y).astype(float)
                weighted_err = np.dot(D, err)
                if weighted_err < min_err:
                    min_err = weighted_err
                    best_est = pred.copy()
                    best_stump = {'dim': dim, 'thresh': thresh, 'ineq': inequal}
    return best_stump, min_err, best_est


def compute_group_weights(H, iteration, num_iter):
    # Sample confidence calculation
    boundary_threshold = 0.6
    bins = [0, 0.4, boundary_threshold, 1.0]  # Set the confidence margin threshold
    groups = np.digitize(H, bins) - 1
    groups = np.clip(groups, 0, 2)

    C_bar = np.zeros(3)
    for j in range(3):
        mask = (groups == j)
        if np.any(mask):
            C_bar[j] = H[mask].mean()

    i, n = iteration, num_iter
    alpha = np.tan((i - 1) * np.pi / (2 * n))

    # Calculate beta with the sigmoid function
    gamma = 0.1
    beta = 1 / (1 + np.exp(-gamma * ((2 * i / n) - 1)))

    # The dynamic offset delta of each group is adjusted according to the average confidence C bar
    delta = np.zeros(3)
    for j in range(3):
        if (C_bar[j] < 0.4) or (C_bar[j] > boundary_threshold):
            delta[j] = alpha
        else:
            delta[j] = beta * alpha

    # The inverse weight of the group is calculated according to C bar + delta
    w = 1.0 / (C_bar + delta + 1e-8)
    w_norm = w / w.sum()
    return groups, w_norm


def ada_boost_train_dynamic(X, y, num_iter=30, random_state=42, undersample_method='dynamic', oversample_method='dynamic'):
    # AdaBoost training function
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError('NOT BINARY CLASSIFICATION')

    pos, neg = classes[0], classes[1]
    y_enc = np.where(y == pos, 1, -1)
    label_map = {1: pos, -1: neg}
    m = X.shape[0]
    D = np.ones(m) / m
    agg_est = np.zeros(m)
    classifiers, betas = [], []
    rng = check_random_state(random_state)

    classes, counts = np.unique(y, return_counts=True)
    minority_class = classes[np.argmin(counts)]

    # Calculate the sample confidence
    for i in range(1, num_iter + 1):
        p = 1 / (1 + np.exp(-agg_est))
        P_correct = np.where(y_enc == 1, p, 1 - p)
        P_wrong = 1 - P_correct
        H = 1 - (P_correct - P_wrong)
        H = H / 2.0

        if undersample_method == 'none':
            sampled_maj = np.where(y_enc == -1)[0]
        elif undersample_method == 'random':
            # Random undersampling
            maj_idx = np.where(y_enc == -1)[0]
            N_minority = sum(y == minority_class)
            n_min = int(N_minority * 1.2)
            sampled_maj = maj_idx if len(maj_idx) <= n_min else rng.choice(maj_idx, n_min, replace=False)
        elif undersample_method == 'dynamic':
            # Dynamic Undersampling with Sample Confidence
            groups, w_norm = compute_group_weights(H, i, num_iter)
            N_minority = sum(y == minority_class)

            Ntarget = int(N_minority * 1.2)

            maj_idx = np.where(y_enc == -1)[0]
            maj_groups = groups[maj_idx]
            sampled_maj = []

            for j in range(3):
                group_idx = maj_idx[maj_groups == j]
                Nj = int(Ntarget * w_norm[j])
                if Nj > 0 and len(group_idx) > 0:
                    sampled = rng.choice(group_idx, min(Nj, len(group_idx)), replace=False)
                    sampled_maj.extend(sampled)

            # # If not enough, fill in the sample
            if len(sampled_maj) < Ntarget:
                remaining = np.setdiff1d(maj_idx, sampled_maj)
                sampled_maj.extend(rng.choice(remaining, Ntarget - len(sampled_maj), replace=False))
        else:
            raise ValueError("ERROR: Invalid undersample method")

        X_base = np.vstack([X[sampled_maj], X[y_enc == 1]])
        y_base = np.concatenate([y_enc[sampled_maj], y_enc[y_enc == 1]])

        if oversample_method != 'none':
            # SMOTE oversampling and GMM-based Dynamic oversampling
            overSample = OverSample(method='smote' if oversample_method == 'smote' else 'cluster', random_state=random_state)
            try:
                X_res, y_res = overSample.fit_resample(X_base, y_base)
            except Exception as e:
                print(f"ERROR: {str(e)}")
                X_res, y_res = X_base, y_base
        else:
            X_res, y_res = X_base, y_base

        # Building a stump classifier
        D_sub = np.ones(len(y_res)) / len(y_res)
        stump, error, _ = build_stump(X_res, y_res, D_sub)

        error = max(min(error, 1 - 1e-16), 1e-16)
        beta = 0.5 * np.log((1 - error) / error)

        class_est_full = stump_classify(X, stump['dim'], stump['thresh'], stump['ineq'])
        agg_est += beta * class_est_full

        D *= np.exp(-beta * y_enc * class_est_full)
        D /= D.sum()

        classifiers.append({**stump, 'beta': beta})
        betas.append(beta)

    return classifiers, betas, label_map


def ada_classify(X, classifiers, betas, label_map):
    # AdaBoost classification function
    agg = np.zeros(X.shape[0])
    for stump, beta in zip(classifiers, betas):
        agg += beta * stump_classify(X, stump['dim'], stump['thresh'], stump['ineq'])
    pred_enc = np.sign(agg)
    return np.where(pred_enc == 1, label_map[1], label_map[-1])


def calculate_gmean(y_true, y_pred):
    # Calculate G-Mean
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0
    spec = tn / (tn + fp) if (tn + fp) else 0
    return np.sqrt(sens * spec)


def run_ablation_experiments(X_train, y_train, X_test, y_test, random_state):
    experiment_conditions = [
        {'name': 'Only Random Undersampled', 'under': 'random', 'over': 'none'},
        {'name': 'Only Dynamic Undersampled', 'under': 'dynamic', 'over': 'none'},
        {'name': 'Only SMOTE', 'under': 'none', 'over': 'smote'},
        {'name': 'Only Dynamic Oversampled', 'under': 'none', 'over': 'cluster'},
        {'name': 'Random Undersampled + SMOTE', 'under': 'random', 'over': 'smote'},
    ]

    all_results = {condition['name']: {'AUC': [], 'G-Mean': []}
                   for condition in experiment_conditions}

    for condition in experiment_conditions:
        print(f"\nRunning：{condition['name']}")

        classifiers, betas, label_map = ada_boost_train_dynamic(
            X_train, y_train,
            undersample_method=condition['under'],
            oversample_method=condition['over'],
            random_state=random_state
        )
        preds = ada_classify(X_test, classifiers, betas, label_map)

        auc = roc_auc_score(y_test, preds)
        gmean = calculate_gmean(y_test, preds)

        all_results[condition['name']]['G-Mean'].append(gmean)
        all_results[condition['name']]['AUC'].append(auc)

    return all_results


if __name__ == '__main__':

    data = pd.read_csv('Experiment1/7Ydata.csv')
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    final_results = {
        'Only Random Undersampled': {'AUC': [], 'G-Mean': []},
        'Only Dynamic Undersampled': {'AUC': [], 'G-Mean': []},
        'Only SMOTE': {'AUC': [], 'G-Mean': []},
        'Only Dynamic Oversampled': {'AUC': [], 'G-Mean': []},
        'Random Undersampled + SMOTE': {'AUC': [], 'G-Mean': []}
    }

    for run in range(5):
        print(f"\n====  {run + 1} Iteration ====")

        random_seed = run
        np.random.seed(random_seed)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_seed)

        run_results = run_ablation_experiments(X_train, y_train, X_test, y_test, random_seed)

        for condition in run_results:
            for metric in run_results[condition]:
                final_results[condition][metric].append(run_results[condition][metric])

    results_data = []
    raw_data = []

    for condition_name, metrics in final_results.items():
        condition_stats = {
            'Condition': condition_name,
            'G-Mean_Mean': np.mean(metrics['G-Mean']),
            'G-Mean_Std': np.std(metrics['G-Mean']),
            'AUC_Mean': np.mean(metrics['AUC']),
            'AUC_Std': np.std(metrics['AUC'])
        }
        results_data.append(condition_stats)

        for run_idx in range(len(metrics)):
            raw_data.append({
                'Condition': condition_name,
                'Run': run_idx + 1,
                'G-Mean': metrics['G-Mean'][run_idx],
                'AUC': metrics['AUC'][run_idx]
            })

    stats_df = pd.DataFrame(results_data)
    raw_df = pd.DataFrame(raw_data)

    print("\n==== Results (mean ± std) ====")
    print(stats_df)

    output_file = '7Ydata_Ablation.xlsx'
    with pd.ExcelWriter(output_file) as writer:
        stats_df.to_excel(writer, sheet_name='Statistics', index=False)
        raw_df.to_excel(writer, sheet_name='Raw_Data', index=False)
