import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split


def load_data(file_path):
    # Read dat file
    data = []
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            data.append([float(x) for x in parts[:-1]])
            labels.append(parts[-1].strip())
    return np.array(data), np.array(labels)


def stump_classify(data_matrix, dim, thresh_val, thresh_ineq):
    # Decision tree classifier
    ret_array = np.ones((data_matrix.shape[0], 1))
    if thresh_ineq == "lt":
        ret_array[data_matrix[:, dim] <= thresh_val] = -1.0
    else:
        ret_array[data_matrix[:, dim] > thresh_val] = -1.0
    return ret_array


def build_stump(data_arr, encoded_labels, D):
    # Construct an optimal one-level decision tree
    data_matrix = np.mat(data_arr)
    label_mat = np.mat(encoded_labels).T
    m, n = data_matrix.shape
    best_stump = {}
    best_class_est = np.mat(np.zeros((m, 1)))
    min_error = np.inf

    for i in range(n):
        feature_values = np.unique(data_matrix[:, i].A1)
        for thresh_val in feature_values:
            for inequal in ["lt", "gt"]:
                predicted_vals = stump_classify(data_matrix, i, thresh_val, inequal)
                err_arr = np.mat(np.ones((m, 1)))
                err_arr[predicted_vals == label_mat] = 0
                weighted_error = D.T * err_arr
                if weighted_error < min_error:
                    min_error = weighted_error
                    best_class_est = predicted_vals.copy()
                    best_stump["dim"] = i
                    best_stump["thresh"] = thresh_val
                    best_stump["ineq"] = inequal
    return best_stump, min_error, best_class_est


def ada_boost_train_ds(data_arr, class_labels, num_it):
    # AdaBoost training function
    unique_labels = np.unique(class_labels)
    if len(unique_labels) != 2:
        raise ValueError("NOT BINARY CLASSIFICATION")
    label_map = {1: unique_labels[0], -1: unique_labels[1]}
    encoded_labels = np.where(class_labels == unique_labels[0], 1, -1)

    weak_class_arr = []
    alpha_list = []
    m = data_arr.shape[0]
    D = np.mat(np.ones((m, 1)) / m)
    agg_class_est = np.mat(np.zeros((m, 1)))

    for i in range(num_it):
        best_stump, error, class_est = build_stump(data_arr, encoded_labels, D)
        error = max(float(error.item()), 1e-16)
        alpha = 0.5 * np.log((1.0 - error) / error)
        alpha_list.append(alpha)
        best_stump["alpha"] = alpha
        weak_class_arr.append(best_stump)

        expon = np.multiply(-alpha * np.mat(encoded_labels).T, class_est)
        D = np.multiply(D, np.exp(expon))
        D /= D.sum()

    return weak_class_arr, alpha_list, label_map


def ada_classify(data_to_class, classifier_arr, alpha_list, label_map):
    # AdaBoost classification function
    data_matrix = np.mat(data_to_class)
    m = data_matrix.shape[0]
    agg_class_est = np.mat(np.zeros((m, 1)))

    for i in range(len(classifier_arr)):
        class_est = stump_classify(
            data_matrix,
            classifier_arr[i]["dim"],
            classifier_arr[i]["thresh"],
            classifier_arr[i]["ineq"]
        )
        agg_class_est += alpha_list[i] * class_est

    predictions_encoded = np.sign(agg_class_est).A1
    return np.where(predictions_encoded == 1, label_map[1], label_map[-1])


def calculate_gmean(y_true, y_pred):
    # Calculate G-Mean
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0
    spec = tn / (tn + fp) if (tn + fp) else 0
    return np.sqrt(sens * spec)


if __name__ == '__main__':
    data = pd.read_csv('Experiment1/7Ydata.csv').iloc[:, 1:]
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    results = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'gmean': [],
        'auc': []
    }

    for run in range(10):
        print(f'====  {run + 1} Iteration ====')

        random_seed = run
        np.random.seed(random_seed)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_seed)

        classifiers, betas, label_map = ada_boost_train_ds(X_train, y_train, num_it=30)
        preds = ada_classify(X_test, classifiers, betas, label_map)

        acc = accuracy_score(y_test, preds)
        macro_precision = precision_score(y_test, preds, average='macro', zero_division=0)
        macro_recall = recall_score(y_test, preds, average='macro', zero_division=0)
        macro_f1 = f1_score(y_test, preds, average='macro', zero_division=0)
        gmean = calculate_gmean(y_test, preds)
        auc = roc_auc_score(y_test, preds)

        results['accuracy'].append(acc)
        results['precision'].append(macro_precision)
        results['recall'].append(macro_recall)
        results['f1'].append(macro_f1)
        results['gmean'].append(gmean)
        results['auc'].append(auc)

        print(f'Accuracy: {acc:.4f}')
        print(f'Macro Precision: {macro_precision:.4f}')
        print(f'Macro Recall: {macro_recall:.4f}')
        print(f'Macro F1-score: {macro_f1:.4f}')
        print(f'G-Mean: {gmean:.4f}')
        print(f'AUC: {auc:.4f}')
        print()

    stats_data = []
    for metric, scores in results.items():
        stats_data.append({
            'Metric': metric.capitalize(),
            'Mean': np.mean(scores),
            'Std': np.std(scores)
        })

    stats_df = pd.DataFrame(stats_data)

    raw_data = []
    for run in range(5):
        for metric in results.keys():
            raw_data.append({
                'Run': run + 1,
                'Metric': metric.capitalize(),
                'Value': results[metric][run]
            })

    raw_df = pd.DataFrame(raw_data)

    print('==== Results ====')
    print(stats_df)
