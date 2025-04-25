import numpy as np
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize


def load_data(file_path):
    data, labels = [], []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('@'):
                continue
            parts = line.split(',')
            try:
                data.append([float(x) for x in parts[:-1]])
                labels.append(int(float(parts[-1])))
            except ValueError:
                continue
    if not labels:
        raise ValueError(f"加载失败，文件 {file_path} 中没有有效样本！")
    print(f"Loaded {len(labels)} samples from {file_path}")
    return np.array(data), np.array(labels)


def stump_predict(data_matrix, dim, thresh_val, thresh_ineq, cls_a, cls_b):
    pred = np.ones((data_matrix.shape[0],)) * cls_b
    if thresh_ineq == 'lt':
        pred[data_matrix[:, dim] <= thresh_val] = cls_a
    else:
        pred[data_matrix[:, dim] > thresh_val] = cls_a
    return pred


def build_stump(data, labels, D, classes):
    m, n = data.shape
    min_error = np.inf
    best_stump = {}
    best_pred = None

    for i in range(n):  # 特征
        feature_values = np.unique(data[:, i])
        for thresh_val in feature_values:
            for inequal in ['lt', 'gt']:
                for cls_a in classes:
                    for cls_b in classes:
                        if cls_a == cls_b:
                            continue
                        pred = stump_predict(data, i, thresh_val, inequal, cls_a, cls_b)
                        err = (pred != labels).astype(float)
                        weighted_error = np.sum(D * err)
                        if weighted_error < min_error:
                            min_error = weighted_error
                            best_pred = pred.copy()
                            best_stump = {
                                "dim": i,
                                "thresh": thresh_val,
                                "ineq": inequal,
                                "cls_a": cls_a,
                                "cls_b": cls_b
                            }
    return best_stump, min_error, best_pred


def grdsad_train_multi(data, labels, num_iter=30, random_state=42):
    np.random.seed(random_state)
    m = data.shape[0]
    classes = np.unique(labels)
    D = np.ones(m) / m
    classifiers = []
    alphas = []

    for i in range(num_iter):
        stump, error, pred = build_stump(data, labels, D, classes)
        error = np.clip(error, 1e-10, 1 - 1e-10)
        class_term = np.log(len(classes) - 1) if len(classes) > 1 else 0
        alpha = 0.5 * np.log((1 - error) / error) + class_term

        stump['alpha'] = alpha
        classifiers.append(stump)
        alphas.append(alpha)

        incorrect = (pred != labels).astype(float)
        D *= np.exp(alpha * incorrect)
        D /= D.sum()

    return classifiers, alphas, classes


def grdsad_predict_multi(data, classifiers, alphas, classes):
    m = data.shape[0]
    class_scores = np.zeros((m, len(classes)))

    for stump, alpha in zip(classifiers, alphas):
        pred = stump_predict(
            data,
            stump['dim'],
            stump['thresh'],
            stump['ineq'],
            stump['cls_a'],
            stump['cls_b']
        )
        for i, p in enumerate(pred):
            idx = np.where(classes == p)[0][0]
            class_scores[i, idx] += alpha

    final_pred = classes[np.argmax(class_scores, axis=1)]
    return final_pred


def calculate_gmean(y_true, y_pred):
    classes = np.unique(y_true)
    sensitivities = []
    for cls in classes:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))
        sens = tp / (tp + fn + 1e-10)
        sensitivities.append(sens)
    gmean = np.prod(sensitivities) ** (1 / len(sensitivities))
    return gmean


def evaluate_metrics(y_true, y_pred):
    print("Classification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    acc = accuracy_score(y_true, y_pred)
    gmean = calculate_gmean(y_true, y_pred)
    y_bin = label_binarize(y_true, classes=np.unique(y_true))
    pred_bin = label_binarize(y_pred, classes=np.unique(y_true))
    auc = roc_auc_score(y_bin, pred_bin, average='macro')
    print(f"Accuracy: {acc:.4f}, GMean: {gmean:.4f}, AUC: {auc:.4f}")
    return acc, gmean, auc


def run_repeated_holdout_have_train_test(X_train, y_train, X_test, y_test, num_iter=30):
    clf, betas, classes = grdsad_train_multi(X_train, y_train, num_iter)
    preds = grdsad_predict_multi(X_test, clf, betas, classes)
    return evaluate_metrics(y_test, preds)


if __name__ == "__main__":
    X_train, y_train = load_data('Experiment3/wine-5-fold/wine-5-4tra.dat')
    X_test, y_test = load_data('Experiment3/wine-5-fold/wine-5-4tst.dat')
    acc, gmean, auc = run_repeated_holdout_have_train_test(X_train, y_train, X_test, y_test, num_iter=30)
