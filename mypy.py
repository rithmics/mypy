import os
import tarfile
import urllib
from itertools import combinations

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from matplotlib.colors import ListedColormap
from nltk.tokenize import sent_tokenize
from sklearn.base import TransformerMixin, clone
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, auc, classification_report,
                             confusion_matrix, f1_score, mean_squared_error,
                             precision_recall_curve, precision_score, r2_score,
                             recall_score, roc_auc_score, roc_curve)
# Needs work!!!!
from sklearn.model_selection import (KFold, ShuffleSplit, StratifiedKFold,
                                     learning_curve, train_test_split)


def plot_decision_regions(X, y, classifier, test_idx=None, res=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    cmap = cm.get_cmap('rainbow')
    colors = []
    for i in np.arange(0.1, 1.0, 0.1):
        colors.append(cmap(i))
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, res),
                           np.arange(x2_min, x2_max, res))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.1, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(X[y == cl, 0],
                    X[y == cl, 1],
                    marker=markers[idx],
                    c=np.array(colors[idx]).reshape(1, -1),
                    alpha=0.6,
                    label=cl,
                    edgecolor='black')
    plt.legend(loc='best')
    plt.show()


def binary_classif_metrics(conf_matrix,
                           class_labels=[0, 1],
                           plot_cm=True,
                           print_summary=False,
                           annot_fmt='.2g'):
    TN, TP = conf_matrix[0][0], conf_matrix[1][1]
    FP, FN = conf_matrix[0][1], conf_matrix[1][0]
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    if plot_cm:
        ax = plt.subplot()
        sns.heatmap(conf_matrix, annot=True, fmt=annot_fmt, ax=ax, square=True)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        ax.xaxis.set_ticklabels([class_labels[0], class_labels[1]])
        ax.yaxis.set_ticklabels([class_labels[0], class_labels[1]])
        plt.show()
    if print_summary:
        print(' ' * 6 + '{:^16}'.format("Metrics"))
        print(' ' * 6 + '-' * 16)
        print(' ' * 6 + '{metric: <11} {score:.1f}'.format(
            metric="Accuracy", score=accuracy * 100))
        print(' ' * 6 + '{metric: <11} {score:.1f}'.format(
            metric="Precision", score=precision * 100))
        print(' ' * 6 + '{metric: <11} {score:.1f}'.format(metric="Recall",
                                                           score=recall * 100))
        print(' ' * 6 +
              '{metric: <11} {score:.1f}'.format(metric="F1", score=f1 * 100))
    return (accuracy, precision, recall, f1)


def plot_pca_variance(X, n_components=-1):
    if n_components <= 0:
        n_components = X.shape[1]

    pca = PCA(n_components=n_components)
    pca.fit(X)
    var_exp = pca.explained_variance_ratio_
    cum_var_exp = np.cumsum(pca.explained_variance_ratio_)
    feat_range = range(1, n_components + 1)
    fig, ax = plt.subplots()
    plt.bar(feat_range,
            var_exp,
            align='center',
            alpha=0.3,
            label='individual explained variance')
    plt.step(feat_range,
             cum_var_exp,
             where='mid',
             alpha=0.7,
             label='cumulative explained variance')
    plt.xticks(feat_range)
    plt.xlabel('Principal components')
    plt.ylabel('Explained variance ratio')
    plt.title('Principal Component Analysis Explained Variance')
    plt.legend(loc='best')
    plt.text(0.9, 0.94, round(cum_var_exp[-1], 3), transform=ax.transAxes)
    plt.show()


# From sklearn docs
def plot_learning_curve(estimator,
                        X,
                        y,
                        title='Learning Curves',
                        axes=None,
                        ylim=None,
                        cv=None,
                        n_jobs=None,
                        train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes,
                         train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std,
                         alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes,
                         test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std,
                         alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes,
                 train_scores_mean,
                 'o-',
                 color="r",
                 label="Training score")
    axes[0].plot(train_sizes,
                 test_scores_mean,
                 'o-',
                 color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes,
                         fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std,
                         alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("Fit times")
    axes[1].set_title("Model Scalability")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean,
                         test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std,
                         alpha=0.1)
    axes[2].set_xlabel("Fit times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Model Performance")

    return plt




def ttd_split(X, y, test_size=0.1, val_size=0.15, random_state=None):
    if 0 < (test_size + val_size) < 1.0:
        train_size = 1 - (test_size + val_size)
    else:
        raise ValueError("test_size and val_size sum must be > 0. and < 1.")

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_size, random_state=random_state)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_size/(test_size + val_size), random_state=random_state)
    return (x_train, x_val, x_test, y_train, y_val, y_test)



def plot_kmeans_sse_elbow(X, k=2, random_state=None):
    k_list = range(1, k + 1)
    sse_list = [0] * len(k_list)

    for k_i, k in enumerate(k_list):
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        kmeans.fit(X)
        clusters = kmeans.labels_
        centroids = kmeans.cluster_centers_
        sse = 0
        for i in range(k):
            cluster_i = np.where(clusters == i)
            sse += np.linalg.norm(X[cluster_i] - centroids[i])
        sse_list[k_i] = sse

    plt.plot(k_list, sse_list)
    plt.xlabel('number of clusters')
    plt.ylabel('SSE')
    plt.show()


def fetch_load_tgz_data(url, file_name, csv_file_name, folder_name=None):
    if folder_name is not None:
        path = os.path.join('data', folder_name)
        os.makedirs(path, exist_ok=True)
    else:
        path = os.getcwd()
    save_path = os.path.join(path, file_name)
    urllib.request.urlretrieve(url, save_path)
    file = tarfile.open(save_path)
    file.extractall(path=path)
    file.close()
    csv_path = os.path.join(path, csv_file_name)
    return pd.read_csv(csv_path)


# Example:
# url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz"
# fetch_load_tgz_data(u, 'housing.tgz', 'housing.csv')


def plot_feature_importance(model, X):
    """ Plot feature importance for Random Forest and SVM classifiers and
    regressors. """
    imp_vals = model.feature_importances_
    indices = np.argsort(imp_vals)[::-1]
    std = np.std([est.feature_importances_ for est in model.estimators_],
                 axis=0)
    plt.bar(range(X.shape[1]),
            imp_vals[indices],
            yerr=std[indices],
            align='center')
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.ylim([0, (imp_vals[indices][0] + std[indices][0]) + 0.1])
    plt.xlabel('feature indices')
    plt.ylabel('importance')
    plt.title('{} Feature Importances'.format(model.__class__.__name__))
    plt.show()


def plot_rf_estimators(X,
                       y,
                       estimator='clf',
                       n_trees=[25, 50, 75, 100, 150, 200, 300, 500],
                       cv=3,
                       random_state=None):
    train_mean_cv = []
    test_mean_cv = []
    for trees in n_trees:
        if estimator == 'clf':
            clf = RandomForestClassifier(n_estimators=trees)
            kfold = StratifiedKFold(n_splits=cv)
        elif estimator == 'reg':
            clf = RandomForestRegressor(n_estimators=trees)
            kfold = KFold(n_splits=cv)
        else:
            raise ValueError(
                "Expected 'clf' or 'reg'; got {}".format(estimator))
        train_scores = []
        test_scores = []
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        print(X.index)
        print(y.index)
        for train, test in kfold.split(X, y):
            clf.fit(X[train], y[train])
            train_scores.append(clf.score(X[train], y[train]))
            test_scores.append(clf.score(X[test], y[test]))
        train_mean_cv.append(np.mean(train_scores))
        test_mean_cv.append(np.mean(test_scores))
    plt.plot(n_trees, train_mean_cv, label='training scores')
    plt.plot(n_trees, test_mean_cv, label='test scores')
    plt.xticks(n_trees)
    plt.xlabel('number of trees')
    plt.ylabel('mean {}-fold CV score'.format(cv))
    plt.legend()
    plt.show







def get_perf_measures(y_actual, y_hat, return_rates=False):
    TP, FP, TN, FN = 0, 0, 0, 0

    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == 1:
            TP += 1
        if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
            FP += 1
        if y_actual[i] == y_hat[i] == 0:
            TN += 1
        if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
            FN += 1

    if return_rates:
        FPR = FP / (FP + TN)
        TPR = TP / (TP + FN)
        return (FPR, TPR)
    else:
        return (TP, FP, TN, FN)




def plot_precision_recall_curve(fitted_model,
                                X,
                                model_name='Model',
                                ravel_X=False):
    if ravel_X:
        y_pred = fitted_model.predict(X.ravel())
        y_proba = fitted_model.predict_proba(X.ravel())
    else:
        y_pred = fitted_model.predict(X)
        y_proba = fitted_model.predict_proba(X)

    # keep probabilities for the positive outcome only
    y_proba = y_proba[:, 1]
    # predict class values
    precision, recall, _ = precision_recall_curve(y_val, y_proba)
    clf_f1, clf_auc = f1_score(y_val, y_pred), auc(recall, precision)
    r, p = recall_score(y_val, y_pred), precision_score(y_val, y_pred)
    # plot the precision-recall curves
    no_skill = len(y_val[y_val == 1]) / len(y_val)
    plt.plot([0, 1.05], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.plot(recall, precision, label=model_name)
    # plot indicator
    plt.vlines(x=r, ymin=no_skill, ymax=p, color='r', linestyles='dotted')
    plt.hlines(y=p, xmin=0, xmax=r, color='r', linestyles='dotted')
    plt.plot(r, p, marker='o', markersize=6, color="r")
    # axis and labels
    plt.title('{} Precision/Recall Curve: f1={:.3f}, auc={:.3f}'.format(
        model_name, clf_f1, clf_auc))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim(0, 1.05)
    plt.ylim(no_skill - 0.05, 1.05)
    # show the legend
    plt.legend(loc='best')
    # show the plot
    plt.show()


def plot_roc_curve(fitted_model, X, y, model_name='Model', ravel_X=False):
    if ravel_X:
        y_pred = fitted_model.predict(X.ravel())
        y_proba = fitted_model.predict_proba(X.ravel())
    else:
        y_pred = fitted_model.predict(X)
        y_proba = fitted_model.predict_proba(X)

    # keep probabilities for the positive outcome only
    y_proba = y_proba[:, 1]
    # generate a no skill prediction (majority class)
    ns_probs = [0 for _ in range(len(y))]
    # calculate scores
    ns_auc = roc_auc_score(y, ns_probs)
    clf_auc = roc_auc_score(y, y_proba)
    model_fpr, model_tpr = get_perf_measures(y, y_pred, return_rates=True)
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y, ns_probs)
    clf_fpr, clf_tpr, _ = roc_curve(y, y_proba)
    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(clf_fpr, clf_tpr, label=model_name)
    # plot indicator
    plt.vlines(x=model_fpr,
               ymin=0,
               ymax=model_tpr,
               color='r',
               linestyles='dotted')
    plt.hlines(y=model_tpr,
               xmin=model_fpr,
               xmax=1,
               color='r',
               linestyles='dotted')
    plt.plot(model_fpr, model_tpr, marker='o', markersize=6, color="r")
    # axis labels
    plt.title('{} ROC Curve: No Skill AUC={:.3f}, Model AUC={:.3f}'.format(
        model_name, ns_auc, clf_auc))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.ylim(0, 1.05)
    plt.xlim(-0.05, 1)
    # show the legend
    plt.legend(loc='lower right')
    # show the plot
    plt.show()


def plot_residuals(model,
                   X_train,
                   X_test,
                   y_train,
                   y_test,
                   train_data=True,
                   test_data=True):
    if not train_data and not test_data:
        raise ValueError('train_data and/or test_data must be True')

    model.fit(X_train, y_train)
    if train_data:
        y_train_pred = model.predict(X_train)
        y_train_mse = mean_squared_error(y_train, y_train_pred)
        y_train_r2 = r2_score(y_train, y_train_pred)
        plt.scatter(y_train_pred,
                    y_train_pred - y_train,
                    alpha=0.7,
                    label='Training data, MSE={:.2f} R2={:.2f}'.format(
                        y_train_mse, y_train_r2))
    if test_data:
        y_test_pred = model.predict(X_test)
        y_test_mse = mean_squared_error(y_test, y_test_pred)
        y_test_r2 = r2_score(y_test, y_test_pred)
        plt.scatter(y_test_pred,
                    y_test_pred - y_test,
                    alpha=0.7,
                    label='Test data, MSE={:.2f} R2={:.2f}'.format(
                        y_test_mse, y_test_r2))

    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='best')
    xmin = min(y_train_pred.min(), y_test_pred.min())
    xmax = max(y_train_pred.max(), y_test_pred.max())
    plt.hlines(y=0, xmin=xmin, xmax=xmax, lw=2, color='black')
    plt.xlim([xmin, xmax])
    plt.tight_layout()
    plt.show()


def scale(X, how='std'):
    if type(X) == pd.DataFrame: X = X.values

    output = np.zeros(X.shape)
    for i in range(X.shape[1]):
        mean = X[:, i].mean()
        xmin = X[:, i].min()
        xmax = X[:, i].max()
        xstd = X[:, i].std()
        if how == 'minmax':
            col_std = (X[:, i] - xmin) / (xmax - xmin)
            output[:, i] = col_std
        elif how == 'std':
            col_std = (X[:, i] - mean) / xstd
            output[:, i] = col_std
        elif how == 'meanmin':
            col_std = (X[:, i] - mean) / (xmax - xmin)
            output[:, i] = col_std
        else:
            raise ValueError(
                "Expected 'std', 'minmax', or 'meanmin'; got '{}'".format(use))
    return output


def summarize(text, summary_length=0.5):
    txt = text.split()
    txt = ' '.join(txt)

    sent_count = len(sent_tokenize(txt))
    key = '654790eb2629a66c2061cb0bf7e049b1'
    sentences = round(sent_count * summary_length, 0)

    url = f'http://api.meaningcloud.com/summarization-1.0?key={key}&txt={txt}&sentences={sentences}'
    r = requests.post(url)
    data = r.json()
    summary = data['summary']
    summary = summary.replace('[...]', '').replace('  ', ' ')
    return summary


class LinearRegressionGD(object):
    def __init__(self, eta=0.001, n_iter=20, tol=0.001):
        self.eta = eta
        self.n_iter = n_iter
        self.tol = tol

    def fit(self, X, y):
        # Initialize weights at 0
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = y - output
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            if len(self.cost_) > 1:
                if (self.cost_[-1] - cost) <= self.tol:
                    break
            self.cost_.append(cost)
            self.intercept_ = self.w_[0]
            self.n_iter_ = i + 1
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return self.net_input(X)

    def score(self, y_true, y_pred, metric='r2'):
        if metric == 'mse':
            return mean_squared_error(y_true, y_pred, squared=True)
        elif metric == 'rmse':
            return mean_squared_error(y_true, y_pred, squared=False)
        elif metric == 'r2':
            return r2_score(y_true, y_pred)
        else:
            raise ValueError(
                "Expected 'mse', 'rmse', or 'r2'; got {}".format(metric))


class LinRegGD(object):
    def __init__(self, alpha=0.01, n_iter=30, tol=0.01):
        self.alpha = alpha
        self.n_iter = n_iter
        self.tol = tol

    def fit(self, X, y):
        self.theta_ = [np.zeros(X.shape[1] + 1)]
        self.cost_ = []
        m = X.shape[0]

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = y - output
            last_theta = self.theta_[-1]
            this_theta = np.empty(X.shape[1] + 1)
            this_theta[0] = last_theta[0] + self.alpha / m * errors.sum()
            this_theta[1:] = last_theta[1:] + self.alpha / m * X.T.dot(errors)
            cost = (errors**2).sum() / 2.0
            if len(self.cost_) > 1:
                if (self.cost_[-1] - cost) <= self.tol:
                    break
            self.cost_.append(cost)
            self.theta_.append(this_theta)
            self.iter_ = i + 1
        return self

    def net_input(self, X):
        return np.dot(X, self.theta_[-1][1:]) + self.theta_[-1][0]

    def predict(self, X):
        return self.net_input(X)

    def score(self, y_true, y_pred, metric='r2'):
        if metric == 'mse':
            return mean_squared_error(y_true, y_pred, squared=True)
        elif metric == 'rmse':
            return mean_squared_error(y_true, y_pred, squared=False)
        elif metric == 'r2':
            return r2_score(y_true, y_pred)
        else:
            raise ValueError(
                "Expected 'mse', 'rmse', or 'r2'; got {}".format(metric))


class LogisticRegressionGD(object):
    def __init__(self, alpha=0.1, n_iter=5000, tol=0.001, random_state=1):
        self.alpha = alpha
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.cost_ = []
        m = X.shape[0]

        for i in range(self.n_iter):
            pred = self.predict_proba(X)
            errors = y - pred
            cost = sum(-y.T * np.log(pred) - (1 - y).T * np.log(1 - pred))
            if len(self.cost_) > 1:
                if (self.cost_[-1] - cost) <= self.tol:
                    print(
                        'Model met tol value of {} in {} iterations with a cost of {:.3f}.'
                        .format(self.tol, self.n_iter_, self.cost_[-1]))
                    break
            self.cost_.append(cost)
            self.w_ += self.alpha / m * X.T.dot(errors)
            self.n_iter_ = i + 1

        return self

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _net_input(self, X):
        return np.dot(X, self.w_)

    def predict_proba(self, X):
        return self._sigmoid(self._net_input(X))

    def predict(self, X, thresh=0.5):
        proba = self.predict_proba(X)
        predictions = np.where(proba >= thresh, 1, 0)
        return predictions

    def score(self, y_true, y_pred, metric='accuracy', print_report=False):
        if print_report:
            print(classification_report(y_true, y_pred))

        if metric == 'accuracy':
            return accuracy_score(y_true, y_pred)
        elif metric == 'f1':
            return f1_score(y_true, y_pred)
        else:
            raise ValueError("Expected 'f1'; got {}".format(metric))

    def get_best_thresh(self, X, y, thresholds=None):
        if thresholds is None:
            thresholds = np.arange(0.1, 1, 0.05)

        scores = []
        for thresh in thresholds:
            y_pred = self.predict(X, thresh=thresh)
            scores.append(clf.score(y, y_pred))

        print(thresholds[np.argmax(scores)])
        return (thresholds, scores)


class MultiLogisticRegressionGD(object):
    def __init__(self, alpha=0.1, n_iter=5000, tol=0.001):
        self.alpha = alpha
        self.n_iter = n_iter
        self.tol = tol

    def fit(self, X, y):
        m = X.shape[0]
        class_labels = np.unique(y)

        if len(class_labels) == 2:
            self.w_ = [np.random.randn(X.shape[1]) * 0.001]
            self.cost_ = [[]]
            self.n_iter_ = [[]]
            for i in range(self.n_iter):
                pred = self.predict_proba(X)
                errors = y - pred
                cost = self._compute_cost(
                    y, pred
                )  #sum(-y.T * np.log(pred) - (1 - y).T * np.log(1 - pred))
                if len(self.cost_[0]) > 1:
                    if (self.cost_[0][-1] - cost) <= self.tol:
                        break
                self.cost_[0].append(cost)
                self.w_[0] += self.alpha / m * X.T.dot(errors)
                self.n_iter_[0] = i + 1
        elif len(class_labels) > 2:
            # Use One-vs-Rest method for multiclass classification
            self.w_ = [
                np.random.randn(X.shape[1]) * 0.001 for label in class_labels
            ]
            self.cost_ = [[] for label in class_labels]
            self.n_iter_ = [[] for label in class_labels]
            for class_label in class_labels:
                y_temp = np.where(y == class_label, 1, 0)
                # Minimize cost with batch gradient descent
                for i in range(self.n_iter):
                    pred = self.predict_class_proba(X, class_label)
                    errors = y_temp - pred
                    cost = self._compute_cost(y_temp, pred)
                    # Break loop if model converged per tol value
                    if len(self.cost_[class_label]) > 1:
                        if (self.cost_[class_label][-1] - cost) <= self.tol:
                            break
                    self.cost_[class_label].append(cost)
                    self.w_[class_label] += self.alpha / m * X.T.dot(errors)
                    self.n_iter_[class_label] = i + 1

        return self

    def _compute_cost(self, y_true, y_pred):
        return sum(-y_true.T * np.log(y_pred) -
                   (1 - y_true).T * np.log(1 - y_pred))

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _net_input(self, X, weight_idx=0):
        return np.dot(X, self.w_[weight_idx])

    def predict_class_proba(self, X, weight_idx=0):
        return self._sigmoid(self._net_input(X, weight_idx))

    def predict_proba(self, X):
        output = np.zeros((X.shape[0], len(self.w_)))
        for w in range(len(self.w_)):
            output[:, w] = self.predict_class_proba(X, w)
        if len(self.w_) == 1:
            return output.flatten()
        else:
            return output

    def predict(self, X, thresh=0.5):
        # thresh controls the prediction threshold
        # ignored for multiclass classification
        proba = self.predict_proba(X)
        if len(self.w_) == 1:
            return np.where(proba >= thresh, 1, 0)
        else:
            return np.argmax(proba, axis=1)

    def score(self, y_true, y_pred, metric='accuracy', print_report=False):
        if print_report:
            print(classification_report(y_true, y_pred))

        if metric == 'accuracy':
            return accuracy_score(y_true, y_pred)
        elif metric == 'f1':
            return f1_score(y_true, y_pred)
        else:
            raise ValueError("Expected 'f1'; got {}".format(metric))

    def get_best_thresh(self, X, thresholds=None):
        if thresholds is None:
            thresholds = np.arange(0.1, 1, 0.05)

        scores = []
        for thresh in thresholds:
            y_pred = self.predict(X, thresh=thresh)
            scores.append(clf.score(y, y_pred))

        return thresholds[np.argmax(scores)]


class NormalEquation(object):
    def fit(self, X, y):
        try:  # Try to create a two-dimensional array using X's dimensions
            X_1 = np.ones((X.shape[0], X.shape[1] + 1))
        except:  # If X is one-dimensional
            X = X.reshape(-1, 1)
            X_1 = np.ones((X.shape[0], X.shape[1] + 1))
        X_1[:, 1:] = X
        X_1_t = X_1.T
        self.theta_ = np.linalg.inv(X_1_t.dot(X_1)).dot(X_1_t).dot(y)
        return object

    def predict(self, X):
        try:
            X_1 = np.ones((X.shape[0], X.shape[1] + 1))
            X_1[:, 1:] = X
        except:
            X_1 = np.ones(X.shape[0] + 1)
            X_1[1:] = X
        prediction = self.theta_.dot(X_1)
        return prediction


# CT with pipelines
def get_transformer_feature_names(columnTransformer):
    """ Get a list of output feature names for a ColumnTransformer """
    output_features = []

    for name, pipe, features in columnTransformer.transformers_:
        if name != 'remainder':
            for i in pipe:
                trans_features = []
                if hasattr(i, 'categories_'):
                    trans_features.extend(i.get_feature_names(features))
                else:
                    trans_features = features
            output_features.extend(trans_features)

    return output_features


# Sequential backward feature selection
class SBS(TransformerMixin):
    def __init__(self,
                 estimator,
                 k_features,
                 scoring=accuracy_score,
                 test_size=0.25,
                 random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.DataFrame):
            y = y.values

        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=self.test_size,
                             random_state=self.random_state)

        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train, X_test, y_test,
                                 self.indices_)
        self.scores_ = [score]

        while dim > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X_train, y_train, X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])

        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score




def plot_ideal_k(k=3, X=None, y=None, cv=3):
    mean_acc, std_acc = [], []

    for n in range(1, k):
        kf = KFold(n_splits=cv)
        kf.get_n_splits(X)
        cv_acc, cv_std = [], []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            knn = KNeighborsClassifier(n_neighbors=n+1).fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            cv_acc.append(metrics.accuracy_score(y_test, y_pred))
            cv_std.append(np.std(y_pred==y_test) / np.sqrt(y_pred.shape[0]))
        mean_acc.append(np.mean(cv_acc))
        std_acc.append(np.mean(cv_std))

    mean_acc = np.array(mean_acc)
    std_acc = np.array(std_acc)
    print(mean_acc)

    plt.plot(range(1, k), mean_acc, 'g')
    plt.fill_between(range(1, k), mean_acc - 1 * std_acc, mean_acc + 1 * std_acc, alpha=0.10)
    plt.fill_between(range(1, k), mean_acc - 3 * std_acc, mean_acc + 3 * std_acc, alpha=0.10, color="green")
    plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
    plt.ylabel('Accuracy ')
    plt.xlabel('Number of Neighbors (k)')
    plt.tight_layout()


def plot_kmeans_clusters():
    # Initialize the plot with the specified dimensions.
    fig = plt.figure(figsize=(6, 4))

    # Colors uses a color map, which will produce an array of colors based on
    # the number of labels there are. We use set(k_means_labels) to get the
    # unique labels.
    colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

    # Create a plot
    ax = fig.add_subplot(1, 1, 1)

    # For loop that plots the data points and centroids.
    # k will range from 0-3, which will match the possible clusters that each
    # data point is in.
    for k, col in zip(range(len([[4,4], [-2, -1], [2, -3], [1, 1]])), colors):

        # Create a list of all data points, where the data poitns that are
        # in the cluster (ex. cluster 0) are labeled as true, else they are
        # labeled as false.
        my_members = (k_means_labels == k)

        # Define the centroid, or cluster center.
        cluster_center = k_means_cluster_centers[k]

        # Plots the datapoints with color col.
        ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')

        # Plots the centroids with specified color, but with a darker outline
        ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)

    # Title of the plot
    ax.set_title('KMeans')
    # Remove axis ticks
    ax.set_xticks(())
    ax.set_yticks(())
    # Show the plot
    plt.show()

    ##
    3D:
    from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
# plt.ylabel('Age', fontsize=18)
# plt.xlabel('Income', fontsize=16)
# plt.zlabel('Education', fontsize=16)
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')

ax.scatter(X[:, 1], X[:, 0], X[:, 3], c= labels.astype(np.float))


def train_val_test_split(X, y, test_size=0.1, val_size=0.2):
    if test_size + val_size < 1.0:
        train_size = 1.0 - (test_size + val_size)
    else:
        raise ValueError('sum of proportions must be < 1.0')

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=1-train_size)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_size/(test_size + val_size))

    return x_train, x_val, x_test, y_train, y_val, y_test
