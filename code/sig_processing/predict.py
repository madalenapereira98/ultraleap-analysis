# Import public packages and functions
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
import pandas as pd

from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import label_binarize
from sklearn.calibration import CalibratedClassifierCV

# Import packages for feature selection
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import MinMaxScaler


def get_repo_path_in_notebook():
    """
    Finds path of repo from Notebook.
    Start running this once to correctly find
    other modules/functions
    """
    path = os.getcwd()
    repo_name = 'ultraleap_analysis'

    while path[-len(repo_name):] != 'ultraleap_analysis':

        path = os.path.dirname(path)

    return path

repo_path = get_repo_path_in_notebook()


# Function to specify model
def select_model(model_label, input_features=None):

    if model_label == 'linearsvm':
        clf = LinearSVC()
    elif model_label == 'nonlinearsvm':
        # clf = svm.SVC(kernel='rbf', C=10.0, gamma=0.1, max_iter=10000)
        clf = svm.SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
    elif model_label == 'logit':
        clf = LogisticRegression(multi_class='multinomial', solver='lbfgs')
        # clf = LogisticRegression(random_state=0)
    elif model_label == 'randomforest':
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_label == 'kneighbors':
        clf = KNeighborsClassifier(n_neighbors=5)
    elif model_label == 'gaussiannaivebayes':
        clf = GaussianNB()
    elif model_label == 'decisiontree':
        clf = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)        
    elif model_label == 'neuralnet':
        if input_features is None:
            raise ValueError("For a neural network, the number of input features must be specified.")
        
        clf = keras.Sequential([
            layers.InputLayer(input_shape=(input_features,)),  # Input layer specifying input feature shape
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dense(5, activation='softmax')  # 5 units for 5 classes
        ])
        clf.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return clf

# Function to specify cv technique
def cv(X, y, cv_label, group_int = None):

    if cv_label == 'skf':
        cv = StratifiedKFold(n_splits=4, random_state=42, shuffle=True)
        cv_split = cv.split(X, y)
    elif cv_label == 'loocv':
        cv = LeaveOneOut()
        cv_split = cv.split(X, y)
    elif cv_label == 'logocv':
        if group_int is None:
            raise ValueError('group_int cannot be None for LeaveOneGroupOut cross-validation')
        cv = LeaveOneGroupOut()
        cv_split = cv.split(X, y, groups=group_int)

    return cv_split


def cmatrix(cm, n_classes, task, feat_selector, model, cl, cv_label, feat_file):

    # Compute evaluation metrics using the aggregated confusion matrix
    acc = np.sum(np.diag(cm)) / np.sum(cm)
    prec = np.diag(cm) / np.sum(cm, axis=0)
    rec = np.diag(cm) / np.sum(cm, axis=1)
    f1 = 2 * (prec * rec) / (prec + rec)
    
    # # Calculate the sample-weighted F1-score
    # weighted_f1 = np.nansum(f1 * np.sum(cm, axis=1)) / np.sum(cm)

    # compute the macro-averaged F1-score
    macro_avg_f1 = np.nanmean(f1)  # nanmean to avoid NaN due to division by zero
   
    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    sns.heatmap(cm, annot=True, cmap='Greens', fmt='g', cbar=True,ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix for {model} - {task}')

    # if cl == 'multiclass':
    #     ls_metrics = ['Precision', 'Recall', 'F1', 'Macro-Avg F1']
    # elif cl == 'binary':
    ls_metrics = ['Precision', 'Recall', 'F1']
    string = f'Accuracy: {round(acc,2)}\n'

    # Create dictionary to save all metrics for further export to ecxel table
    dict_metrics = {}
    dict_metrics['Accuracy'] = acc
    for met in ls_metrics:
        for i in range(n_classes):
            dict_metrics[f'{met}_{i}']=[]
    
   
    for metric in ls_metrics:
        for i in range(n_classes):
            if metric=='Precision': met = prec
            elif metric=='Recall': met = rec
            elif metric=='F1': met = f1
        
            string+=f'{metric}_{i}: {round(met[i],2)}\n'
            dict_metrics[f'{metric}_{i}'] = met[i]

    if cl == 'multiclass':
        # # Add the Sample-Weighted F1 to the metrics string and dict_metrics
        # string += f'Sample-Weighted F1: {round(weighted_f1, 2)}\n'
        # dict_metrics['Sample-Weighted F1'] = weighted_f1
        # Now append the Macro-Avg F1 to the metrics string and dict_metrics at the end
        string += f'Macro-Avg F1: {round(macro_avg_f1, 2)}\n'
        dict_metrics['Macro-Avg F1'] = macro_avg_f1

    textstr = string
    props = dict(boxstyle='round', facecolor='white')#, alpha=0.5)
    ax.annotate(textstr, xy=(1.2, 0.5), xycoords='axes fraction', fontsize=10, va='center', bbox=props)

    pred_path = os.path.join(repo_path, 'modelling', feat_file, task, feat_selector, cl, cv_label, model)
    if not os.path.exists(pred_path):
        os.makedirs(pred_path)

    fig.savefig(os.path.join(pred_path,
                            f'confusion_matrix_{task}_{cl}_{cv_label}_{model}'),
                            dpi = 300, facecolor = 'w',
                            )

    plt.close()

    df_metrics = pd.DataFrame.from_dict(dict_metrics,orient='index', columns=['value'])
    # Save the DataFrame to the Excel file
    path_metrics = os.path.join(repo_path, 'modelling', feat_file, task, feat_selector, cl, cv_label, model)
    if not os.path.exists(path_metrics):
        os.makedirs(path_metrics)
    df_metrics.to_excel(os.path.join(path_metrics, f'{task}_{feat_selector}_{cl}_{cv_label}_{model}_metrics.xlsx'))

    return df_metrics



def classifier(task, feat_selector, feat_file, cv_label, model, X, y, binary=False, group_int = None):

    if binary:
        y_binary = y > 1
        y = y_binary
        cl = 'binary'
    else:
        # y = pd.Categorical(y)
        cl = 'multiclass'    

    if model == 'neuralnet':
        clf = select_model(model,X.shape[1])
    else:
        clf = select_model(model)
    cv_split = cv(X, y, cv_label, group_int=group_int)

    n_classes = len(np.unique(y))
    cm_agg = np.zeros((n_classes, n_classes))

    # Initialize lists to store true labels and predicted probabilities for all folds
    true_labels = []
    pred_probs = []

    # loop over all folds
    for i, (train_index, test_index) in enumerate(cv_split):

        # get training and testing split for current fold
        train_X, test_X = X[train_index], X[test_index]
        train_y, test_y = y[train_index], y[test_index]

        # train classifier with train X and y
        if model == 'neuralnet':
            y_train_onehot = tf.keras.utils.to_categorical(train_y, num_classes=5)
            y_test_onehot = tf.keras.utils.to_categorical(test_y, num_classes=5)
            clf.fit(train_X, y_train_onehot, epochs=10, batch_size=32, verbose=0)
            y_pred = np.argmax(clf.predict(test_X), axis=-1)
        else:
            clf.fit(train_X, train_y)
            y_pred = clf.predict(test_X)

        # clf.fit(train_X, train_y)
        # y_pred = clf.predict(test_X)

        # y_pred = clf.predict(train_X)

        if cl == 'binary':# and (not isinstance(clf, LinearSVC) and not isinstance(clf, svm.SVC)):
            # Check if classifier has a predict_proba method
            if hasattr(clf, "predict_proba"):
                y_pred_prob = clf.predict_proba(test_X)[:, 1]
            elif model == 'neuralnet':
                y_pred_prob = clf.predict(test_X)[:, 1]
            elif isinstance(clf, LinearSVC):
                # Platt scaling for LinearSVC
                calibrated = CalibratedClassifierCV(clf, method='sigmoid', cv='prefit')
                calibrated.fit(train_X, train_y)
                y_pred_prob = calibrated.predict_proba(test_X)[:, 1]
            else:
                raise ValueError("Model does not have predict_proba method or is not recognized.")
            
            
            # # # # Predict the probabilities for the test data
            # # # y_pred_prob = clf.predict_proba(test_X)[:, 1]
            
            # Store true labels and predicted probabilities for this fold
            true_labels.extend(test_y)
            pred_probs.extend(y_pred_prob)
          
        # compute the confusion matrix for the current split
        cm_fold = confusion_matrix(test_y, y_pred, labels=np.arange(n_classes))
        # cm_fold = confusion_matrix(train_y, y_pred, labels=np.arange(n_classes))

        # aggregate the confusion matrix for the current split
        cm_agg += cm_fold

    # compute evaluation metrics using the aggregated confusion matrix
    cmatrix(cm_agg, n_classes, task, feat_selector, model, cl, cv_label, feat_file)

    if cl == 'binary':# and (not isinstance(clf, LinearSVC) and not isinstance(clf, svm.SVC)):
        # Compute the false positive rate (FPR), true positive rate (TPR), and thresholds
        fpr, tpr, thresholds = roc_curve(true_labels, pred_probs)

        # Compute the AUC score
        roc_auc = auc(fpr, tpr)

        fig = plt.figure(figsize=(10,8))

        # Plot the ROC curve
        plt.plot(fpr, tpr, label='AUC = %0.2f' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')  # Random guess line
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title(f'ROC Curve - {model}')
        plt.legend(loc='lower right')
        plt.close()

        pred_path = os.path.join(repo_path, 'modelling', feat_file, task, feat_selector, cl, cv_label, model)
        if not os.path.exists(pred_path): 
            os.makedirs(pred_path)

        fig.savefig(os.path.join(pred_path, 
                                f'roc_{task}_{cl}_{cv_label}_{model}'),
                                dpi = 300, facecolor = 'w',
                                )
        plt.close()
        
    return


def rfe_cv_feat_selector(X_df, y, model, num_folds):

    X_norm = MinMaxScaler().fit_transform(X_df.iloc[:,1:]) # y = (x – min) / (max – min)
    feat_cols = X_df.iloc[:,1:].columns

    min_features_to_select = 1  # Min number of features to consider
    clf = select_model(model)
    # clf = LogisticRegression()
    cv = StratifiedKFold(num_folds)

    rfecv = RFECV(
        estimator=clf,
        step=1,
        cv=cv,
        scoring="accuracy",
        min_features_to_select=min_features_to_select,
        n_jobs=2,
    )
    rfecv.fit(X_norm, y)

    print(f"Optimal number of features: {rfecv.n_features_}")

    selected_features = [feature for feature, selected in zip(feat_cols, rfecv.support_) if selected]
    print("Selected features:", selected_features)


    n_scores = len(rfecv.cv_results_["mean_test_score"])
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Mean test accuracy")
    plt.errorbar(
        range(min_features_to_select, n_scores + min_features_to_select),
        rfecv.cv_results_["mean_test_score"],
        yerr=rfecv.cv_results_["std_test_score"],
    )
    plt.title("Recursive Feature Elimination \nwith correlated features")
    plt.show()

    return selected_features,X_norm