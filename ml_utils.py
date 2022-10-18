# Importing Dependencies
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report, balanced_accuracy_score
from sklearn.feature_selection import RFECV, SelectFpr, SelectFromModel, SelectPercentile, SequentialFeatureSelector
from sklearn.preprocessing import RobustScaler, MinMaxScaler, MaxAbsScaler, StandardScaler, OrdinalEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from boruta import BorutaPy

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import skopt

# Defining a custom transformer to remove multicollinearity
class VifDropper(BaseEstimator, TransformerMixin):
    # Initializing the default threshold for variance inflation factor (VIF)
    def __init__(self, threshold = 2.5):
        # Default VIF threshold
        self.threshold = threshold

    # Defining a function for fitting data to custom transformer
    def fit(self, X, y = None):
        # Creating a copy of a Numpy array as Pandas dataframe
        data = pd.DataFrame(data = X).copy()
        
        # Creating a Pandas dataframe
        vif_df = pd.DataFrame()
        
        # Assigning the names of columns to a feature variable
        vif_df['feature'] = data.columns
        
        # Calculating VIF values
        vif_df['VIF'] = [VIF(exog = data.values, exog_idx = i) for i in range(len(data.columns))]
        
        # Creating an empty list
        features_with_max_vif = []
        
        # Calculating VIF values of variables based on default threshold
        while vif_df.VIF.max() > self.threshold:
            feature_with_max_vif = vif_df.loc[vif_df.VIF == vif_df.VIF.max()].feature.values[0]
            data.drop(columns = feature_with_max_vif, inplace = True)
            features_with_max_vif.append(feature_with_max_vif)
 
            vif_df = pd.DataFrame()
            vif_df['feature'] = data.columns
            vif_df['VIF'] = [VIF(exog = data.values, exog_idx = i) for i in range(len(data.columns))]
        
        # Defining the list of variables with maximum VIF values
        self.features_with_max_vif = features_with_max_vif
        
        # Returning the fitted and transformed data
        return self 
    
    # Defining a function for transforming data with custom transformer
    def transform(self, X, y = None):
        # Returning the transformed data
        return pd.DataFrame(data = X).drop(columns = self.features_with_max_vif).values
    
# Defining a function to build a classifier pipeline
def build_pipeline(classifier = None, 
                   metric = 'balanced_accuracy', 
                   data_frame = None, 
                   train_features = None, 
                   train_labels = None, 
                   apply_bayesian_optimization = True, 
                   hyperparameters = None, 
                   n_iterations = 50, 
                   scale = True, 
                   scaler_type = None,
                   drop_high_vif_features = True, 
                   apply_feature_selection = True, 
                   feature_selection_method = None, 
                   verbosity = 0):
    """
    This function is used to build a classifier pipeline.
    
    Args:
        classifier: A classifier instance.
        metric: A classification metric based on which to optimize a model.
        data_frame: A pandas data frame
        train_features: Train features.
        train_labels: Train labels.
        apply_bayesian_optimization: Whether or not to apply Bayesian Optimization to find the best hyper parameters.
        hyperparameters: A dictionary of hyperparameters.
        n_iterations: The number of repetitions for a hyperparameter tuning technique.
        scale: Whether or not to apply feature scaling.
        scaler_type: A type of a feature scaler instance as a string.
        drop_high_vif_features: Whether or not to drop features with high variance inflation factor (VIF) value.
        apply_feature_selection: Whether or not to apply feature selection.
        feature_selection_method: A type of a feature selection technique.
        verbosity: A level of verbosity to display an output of Bayesian Optimization.
        
    Returns:
        Builds a classifier pipeline.
    """
    # Creating a list of positions as string values in a descending order
    positions = []
    
    # Creating a list of ordinal features
    ordinal_features = []
    
    # Creating a list of date features
    date_features = []
    
    # Creating a list of binary features
    binary_features = [feature for feature in FEATURES if data_frame[feature].nunique() == 2]
    
    # Creating a list of numeric features
    numeric_features = [feature for feature in FEATURES if feature not in ordinal_features + date_features + binary_features]
    
    # Asserting the number of features to be equal to 192
    assert len(ordinal_features) + len(date_features) + len(binary_features) + len(numeric_features) == len(FEATURES)
    
    # Instantiating a cross validation technique
    skf = StratifiedKFold()
    
    # Pipeline for binary features
    binary_pipeline = Pipeline(steps = [('mode_imputer', SimpleImputer(strategy = 'most_frequent'))])
    
    # Pipeline for ordinal features
    ordinal_pipeline = Pipeline(steps = [('mode_imputer', SimpleImputer(strategy = 'most_frequent')), 
                                         ('ore', OrdinalEncoder(categories = [positions, positions], handle_unknown = 'use_encoded_value', unknown_value = -1))])
    
    # A condition to apply feature scaling
    if scale:
        # Creating a dictionary of feature scaler and transformer instances
        scalers_dict = {'robust':RobustScaler(), 'minmax':MinMaxScaler(), 'maxabs':MaxAbsScaler(), 'standard':StandardScaler()}
        
        # A condition to drop features with high variance inflation factor (VIF) valdues
        if drop_high_vif_features:
            # Pipeline for numeric features with variance inflation factor (VIF) and feature scaling included 
            numeric_pipeline = Pipeline(steps = [('median_imputer', SimpleImputer(strategy = 'median')), 
                                                 ('vif_dropper', VifDropper()), 
                                                 ('feature_scaler', scalers_dict.get(scaler_type))])
        else:
            # Pipeline for numeric features with only feature scaling included 
            numeric_pipeline = Pipeline(steps = [('median_imputer', SimpleImputer(strategy = 'median')), 
                                                 ('feature_scaler', scalers_dict.get(scaler_type))])
    else:
        # Pipeline for numeric features without feature scaling 
        numeric_pipeline = Pipeline(steps = [('median_imputer', SimpleImputer(strategy = 'median'))])
        
    # Feature transformer with combined pipelines
    feature_transformer = ColumnTransformer(transformers = [('binary_pipeline', binary_pipeline, binary_features),
                                                            ('ordinal_pipeline', ordinal_pipeline, ordinal_features),
                                                            ('numeric_pipeline', numeric_pipeline, numeric_features)], remainder = 'passthrough', n_jobs = -1)
    
    # Creating a condition to apply feature selection
    if apply_feature_selection:
        if feature_selection_method == 'wrapper':
            # Instantiating a wrapper feature selection instance
            feature_selector = SequentialFeatureSelector(estimator = classifier, scoring = metric, cv = skf, n_jobs = -1)
        elif feature_selection_method == 'boruta':
            # Instantiating a tree based feature selection instance
            feature_selector = BorutaPy(estimator = classifier, random_state = 42, verbose = 0)
        elif feature_selection_method == 'meta':
            # Instantiating a meta transformer feature selection instance
            feature_selector = SelectFromModel(estimator = classifier)
        elif feature_selection_method == 'statistical':
            # Instantiating a meta transformer feature selection instance
            feature_selector = SelectFpr()
        elif feature_selection_method == 'mutual_info':
            # Instantiating a meta transformer feature selection instance
            feature_selector = SelectPercentile()
        elif feature_selection_method == 'hybrid':
            # Instantiating a meta transformer feature selection instance
            feature_selector = RFECV(estimator = classifier, cv = skf, scoring = metric, n_jobs = -1)
        
        # Final classifier pipeline with feature selection
        pipe = Pipeline(steps = [('feature_reallocator', FeatureReallocator()), 
                                 ('feature_transformer', feature_transformer), 
                                 ('feature_selector', feature_selector), 
                                 ('classifier', classifier)])
    else:
        # Final classifier pipeline without feature selection
        pipe = Pipeline(steps = [('feature_reallocator', FeatureReallocator()), 
                                 ('feature_transformer', feature_transformer), 
                                 ('classifier', classifier)])
    
    # A condition to apply hyperparameter tuning with Bayesian Optimization
    if apply_bayesian_optimization:
        # Creating an operating level seed
        np.random.seed(seed = 42)
        
        # Applying Bayesian Optimization to identify the best hyperparameters
        bayes_search = skopt.BayesSearchCV(estimator = pipe, 
                                           search_spaces = hyperparameters, 
                                           n_iter = n_iterations, 
                                           scoring = metric, 
                                           n_jobs = -1, 
                                           cv = skf, 
                                           verbose = verbosity, 
                                           random_state = 42)
        
        # Fitting the training features and labels
        bayes_search.fit(X = train_features, y = train_labels)
        
        # Extracting the pipeline with the best hyperparameters
        best_pipe = bayes_search.best_estimator_
        
        # Returning the classifier pipeline with the best hyperparameters
        return best_pipe
    else:
        # Fitting train features and labels to the pipeline
        pipe.fit(X = train_features, y = train_labels)
        
        # Returning the classifier pipeline with default hyperparameters
        return pipe
    
# Defining a function to find the best probability threshold
def find_optimal_threshold(model = None, 
                           metric = None, 
                           train_features = None, 
                           train_labels = None, 
                           test_features = None, 
                           test_labels = None, 
                           beta = None):
    """
    This function is used to find out the best probability thresholds for train & test set.
    
    Args:
        model: A classifier instance.
        metric: A classification metric based on which to optimize a model.
        train_features: Train features.
        train_labels: Train labels.
        test_features: Test features.
        test_labels: Train labels.
        beta: Beta to calculate the F Beta score.
        
    Returns:
        Plots thresholding plots and identifies best probability thresholds for train & test sets.
    """
    # Creating an array of probabilities
    probabilities = np.arange(0.1, 0.91, 0.01)
    
    # Creating a dictionary of labels for evaluation metrics of a classification problem
    metrics_dict = {'accuracy':'Accuracy',
                    'positive_recall':'Positive Recall', 
                    'negative_recall':'Negative Recall',
                    'balanced_accuracy':'Balanced Accuracy', 
                    'positive_precision':'Positive Precision', 
                    'negative_precision':'Negative Precision',
                    'positive_fbeta':f'Positive F{beta}',
                    'negative_fbeta':f'Negative F{beta}'}
    
    # Creating a condition to apply thresholding to a chosen evaluation metric
    if metric == 'accuracy':
        # Calculating the accuracy score based on given probability thresholds for train and test set
        train_metrics_per_proba = [accuracy_score(y_true = train_labels, y_pred = np.where(model.predict_proba(X = train_features)[:, 1] >= proba, 1, 0)) for proba in probabilities]
        test_metrics_per_proba = [accuracy_score(y_true = test_labels, y_pred = np.where(model.predict_proba(X = test_features)[:, 1] >= proba, 1, 0)) for proba in probabilities]
        
        # Calculating the accuracy score at default threshold for train and test set
        score_at_default_threshold_train = accuracy_score(y_true = train_labels, y_pred = model.predict(X = train_features))
        score_at_default_threshold_test = accuracy_score(y_true = test_labels, y_pred = model.predict(X = test_features))
    elif metric == 'balanced_accuracy':
        # Calculating the balanced accuracy score based on given probability thresholds for train and test set
        train_metrics_per_proba = [balanced_accuracy_score(y_true = train_labels, y_pred = np.where(model.predict_proba(X = train_features)[:, 1] >= proba, 1, 0)) for proba in probabilities]
        test_metrics_per_proba = [balanced_accuracy_score(y_true = test_labels, y_pred = np.where(model.predict_proba(X = test_features)[:, 1] >= proba, 1, 0)) for proba in probabilities]
        
        # Calculating the balanced accuracy score at default threshold for train and test set
        score_at_default_threshold_train = balanced_accuracy_score(y_true = train_labels, y_pred = model.predict(X = train_features))
        score_at_default_threshold_test = balanced_accuracy_score(y_true = test_labels, y_pred = model.predict(X = test_features))
    elif metric == 'positive_precision':
        # Calculating the positive precision score based on given probability thresholds for train and test set
        train_metrics_per_proba = [precision_score(y_true = train_labels, y_pred = np.where(model.predict_proba(X = train_features)[:, 1] >= proba, 1, 0)) for proba in probabilities]
        test_metrics_per_proba = [precision_score(y_true = test_labels, y_pred = np.where(model.predict_proba(X = test_features)[:, 1] >= proba, 1, 0)) for proba in probabilities]
        
        # Calculating the positive precision score at default threshold for train and test set
        score_at_default_threshold_train = precision_score(y_true = train_labels, y_pred = model.predict(X = train_features))
        score_at_default_threshold_test = precision_score(y_true = test_labels, y_pred = model.predict(X = test_features))
    elif metric == 'negative_precision':
        # Calculating the negative precision score based on given probability thresholds for train and test set
        train_metrics_per_proba = [precision_score(y_true = train_labels, y_pred = np.where(model.predict_proba(X = train_features)[:, 1] >= proba, 1, 0), pos_label = 0) for proba in probabilities]
        test_metrics_per_proba = [precision_score(y_true = test_labels, y_pred = np.where(model.predict_proba(X = test_features)[:, 1] >= proba, 1, 0), pos_label = 0) for proba in probabilities]
    
        # Calculating the negative precision score at default threshold for train and test set
        score_at_default_threshold_train = precision_score(y_true = train_labels, y_pred = model.predict(X = train_features), pos_label = 0)
        score_at_default_threshold_test = precision_score(y_true = test_labels, y_pred = model.predict(X = test_features), pos_label = 0)
    elif metric == 'positive_recall':
        # Calculating the positive recall score based on given probability thresholds for train and test set
        train_metrics_per_proba = [recall_score(y_true = train_labels, y_pred = np.where(model.predict_proba(X = train_features)[:, 1] >= proba, 1, 0)) for proba in probabilities]
        test_metrics_per_proba = [recall_score(y_true = test_labels, y_pred = np.where(model.predict_proba(X = test_features)[:, 1] >= proba, 1, 0)) for proba in probabilities]
        
        # Calculating the positive recall score at default threshold for train and test set
        score_at_default_threshold_train = recall_score(y_true = train_labels, y_pred = model.predict(X = train_features))
        score_at_default_threshold_test = recall_score(y_true = test_labels, y_pred = model.predict(X = test_features))
    elif metric == 'negative_recall':
        # Calculating the negative recall score based on given probability thresholds for train and test set
        train_metrics_per_proba = [recall_score(y_true = train_labels, y_pred = np.where(model.predict_proba(X = train_features)[:, 1] >= proba, 1, 0), pos_label = 0) for proba in probabilities]
        test_metrics_per_proba = [recall_score(y_true = test_labels, y_pred = np.where(model.predict_proba(X = test_features)[:, 1] >= proba, 1, 0), pos_label = 0) for proba in probabilities]
        
        # Calculating the negative recall score at default threshold for train and test set
        score_at_default_threshold_train = recall_score(y_true = train_labels, y_pred = model.predict(X = train_features), pos_label = 0)
        score_at_default_threshold_test = recall_score(y_true = test_labels, y_pred = model.predict(X = test_features), pos_label = 0)
    elif metric == 'positive_fbeta':
        # Calculating the positive fbeta score based on given probability thresholds for train and test set
        train_metrics_per_proba = [fbeta_score(y_true = train_labels, y_pred = np.where(model.predict_proba(X = train_features)[:, 1] >= proba, 1, 0), beta = beta) for proba in probabilities]
        test_metrics_per_proba = [fbeta_score(y_true = test_labels, y_pred = np.where(model.predict_proba(X = test_features)[:, 1] >= proba, 1, 0), beta = beta) for proba in probabilities]
        
        # Calculating the positive fbeta score at default threshold for train and test set
        score_at_default_threshold_train = fbeta_score(y_true = train_labels, y_pred = model.predict(X = train_features), beta = beta)
        score_at_default_threshold_test = fbeta_score(y_true = test_labels, y_pred = model.predict(X = test_features), beta = beta)
    elif metric == 'negative_fbeta':
        # Calculating the negative fbeta score based on given probability thresholds for train and test set
        train_metrics_per_proba = [fbeta_score(y_true = train_labels, y_pred = np.where(model.predict_proba(X = train_features)[:, 1] >= proba, 1, 0), beta = beta, pos_label = 0) for proba in probabilities]
        test_metrics_per_proba = [fbeta_score(y_true = test_labels, y_pred = np.where(model.predict_proba(X = test_features)[:, 1] >= proba, 1, 0), beta = beta, pos_label = 0) for proba in probabilities]
        
        # Calculating the negative fbeta score at default threshold for train and test set
        score_at_default_threshold_train = fbeta_score(y_true = train_labels, y_pred = model.predict(X = train_features), beta = beta, pos_label = 0)
        score_at_default_threshold_test = fbeta_score(y_true = test_labels, y_pred = model.predict(X = test_features), beta = beta, pos_label = 0)
    
    # Identifying the best probability threshold for train & test set
    best_threshold_train = probabilities[np.array(object = train_metrics_per_proba).argmax()]
    best_threshold_test = probabilities[np.array(object = test_metrics_per_proba).argmax()]
    
    # Filtering the best score based on chosen probability thresholds for train and test set
    score_at_best_threshold_train = train_metrics_per_proba[np.array(object = train_metrics_per_proba).argmax()]
    score_at_best_threshold_test = test_metrics_per_proba[np.array(object = test_metrics_per_proba).argmax()]
    
    # Plotting probability thresholding plot for train set
    plt.subplot(1, 2, 1)
    plt.plot(probabilities, train_metrics_per_proba, label = f'{metrics_dict.get(metric)} Score')
    plt.title(label = f'Train Set {metrics_dict.get(metric)} Thresholding', fontsize = 20)
    
    # Creating a condition based on thresholding for train set
    if score_at_best_threshold_train == score_at_default_threshold_train:
        # Drawing a vertical line at default (50%) probability threshold for train set
        plt.axvline(x = 0.5, color = 'teal', label = 'Best Threshold is Default Threshold (50%)')
        
        # Readdjusting the probability threshold for train set back to the default
        best_threshold_train = 0.5
    else:
        # Drawing vertical lines at both default and best probability thresholds for train set
        plt.axvline(x = 0.5, color = 'teal', label = 'Default Threshold (50%)')
        plt.axvline(x = best_threshold_train, color = 'red', label = f'Best Train Threshold ({best_threshold_train:.0%})')
    
    plt.ylabel(ylabel = f'{metrics_dict.get(metric)} Score', fontsize = 20)
    plt.xlabel(xlabel = 'Probability', fontsize = 20)
    plt.legend(loc = 'best')
    
    # Plotting probability thresholding plot for test set
    plt.subplot(1, 2, 2)
    plt.plot(probabilities, test_metrics_per_proba, label = f'{metrics_dict.get(metric)} Score')
    plt.title(label = f'Test Set {metrics_dict.get(metric)} Thresholding', fontsize = 20)
    
    # Creating a condition based on thresholding for test set
    if score_at_best_threshold_test == score_at_default_threshold_test:
        # Drawing a vertical line at default (50%) probability threshold for test set
        plt.axvline(x = 0.5, color = 'teal', label = 'Default Threshold (50%)')
        
        # Readdjusting the probability threshold for test set back to the default
        best_threshold_test = 0.5
    else:
        # Drawing vertical lines at both default and best probability thresholds for train set
        plt.axvline(x = 0.5, color = 'teal', label = 'Default Threshold (50%)')
        plt.axvline(x = best_threshold_test, color = 'red', label = f'Best Test Threshold ({best_threshold_test:.0%})')
    
    plt.ylabel(ylabel = f'{metrics_dict.get(metric)} Score', fontsize = 20)
    plt.xlabel(xlabel = 'Probability', fontsize = 20)
    plt.legend(loc = 'best')
    plt.show()
    
    # Returning best probabiltiy threshold for train and test set
    return best_threshold_train, best_threshold_test

# Defining a function to plot Receiver Operating Characteristics (ROC) curve
def plot_roc_curve(model = None, 
                   train_features = None, 
                   train_labels = None, 
                   test_features = None, 
                   test_labels = None, 
                   algorithm_name = None):
    """
    This function is used to plot Receiver Operating Characteristics (ROC) curve for train & test set.
    
    Args:
        model: A classifier instance.
        train_features: Train features.
        train_labels: Train labels.
        test_features: Test features.
        test_labels: Train labels.
        algorithm_name: A name of an algoritm used to build the model.
        
    Returns:
        Plots Receiver Operating Characteristics (ROC) curve for train & test set.
    """
    # Calculating Area Under the Curve (AUC) score for train & test set
    auc_score_train = roc_auc_score(y_true = train_labels, y_score = model.predict_proba(X = train_features)[:, 1])
    auc_score_test = roc_auc_score(y_true = test_labels, y_score = model.predict_proba(X = test_features)[:, 1])
    
    # Calculating False Positive Rate (FPR) and True Positive Rate (TPR) for train set
    fpr, tpr, _ = roc_curve(y_true = train_labels, y_score = model.predict_proba(X = train_features)[:, 1])
    
    # Plotting Receiver Operating Characteristics (ROC) curve for train set
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label = f'{algorithm_name} Model AUC Score: {auc_score_train:.2f}', color = 'red')
    plt.plot([0, 1], [0, 1], label = 'Random Model', linestyle = '--', color = 'teal')
    plt.title(label = 'Train Set ROC Curve', fontsize = 20)
    plt.xlabel(xlabel = 'False Positive Rate', fontsize = 20)
    plt.ylabel(ylabel = 'True Positive Rate', fontsize = 20)
    plt.legend(loc = 'lower right', fontsize = 20)
    
    # Calculating False Positive Rate (FPR) and True Positive Rate (TPR) for test set
    fpr, tpr, _ = roc_curve(y_true = test_labels, y_score = model.predict_proba(X = test_features)[:, 1])
    
    # Plotting Receiver Operating Characteristics (ROC) curve for test set
    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, label = f'{algorithm_name} Model AUC Score: {auc_score_test:.2f}', color = 'red')
    plt.plot([0, 1], [0, 1], label = 'Random Model', linestyle = '--', color = 'teal')
    plt.title(label = 'Test Set ROC Curve', fontsize = 20)
    plt.xlabel(xlabel = 'False Positive Rate', fontsize = 20)
    plt.ylabel(ylabel = 'True Positive Rate', fontsize = 20)
    plt.legend(loc = 'lower right', fontsize = 20)
    plt.show()
    
# Defining a function to plot confusion matrices, recall & precision ratio
def plot_confusion_matrix(model = None, 
                          train_features = None, 
                          test_features = None, 
                          train_labels = None, 
                          test_labels = None):
    """
    This function is used to plot confusion matrices, recall & precision ratio for train & test set.
    
    Args:
        model: A classifier instance.
        train_features: Train features.
        train_labels: Train labels.
        test_features: Test features.
        test_labels: Train labels.
    
    Returns:
        Plots confusion matrices, recall and precision ratio for train & test set.
    """
    # Creating a list of class labels
    labels = ['Negative Class', 'Positive Class']
    
    # Making predictions on train and test set based on chosen probability thresholds
    train_predictions = np.where(model.predict_proba(X = train_features)[:, 1] >= train_threshold, 1, 0)
    test_predictions = np.where(model.predict_proba(X = test_features)[:, 1] >= test_threshold, 1, 0)
    
    # Creating confusion matrices for train and test set
    cm_train = confusion_matrix(y_true = train_labels, y_pred = train_predictions)
    cm_test = confusion_matrix(y_true = test_labels, y_pred = test_predictions)
    
    # Plotting confusion matrix for train set
    plt.figure(figsize = (30, 18))
    plt.subplot(2, 3, 1)
    sns.heatmap(data = cm_train, cmap = plt.cm.Blues, annot = True, fmt = '.4g', cbar = False, xticklabels = labels, yticklabels = labels)
    plt.title(label = 'Train Set Confusion Matrix', fontsize = 16)
    plt.ylabel(ylabel = 'Ground Truth', fontsize = 16)
    plt.xlabel(xlabel = 'Predictions', fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.xticks(fontsize = 16)
    
    # Calculating recall ratio for train set
    upper_array = np.divide(cm_train[0], cm_train.sum(axis = 1)[0])
    lower_array = np.divide(cm_train[1], cm_train.sum(axis = 1)[1])
    final_array = np.vstack(tup = (upper_array, lower_array))
    
    # Plotting recall ratio for train set
    plt.subplot(2, 3, 2)
    sns.heatmap(data = final_array, cmap = plt.cm.Blues, annot = True, fmt = '.0%', cbar = False, xticklabels = labels, yticklabels = labels)
    plt.title(label = 'Train Set Recall Ratio(%)', fontsize = 16)
    plt.ylabel(ylabel = 'Ground Truth', fontsize = 16)
    plt.xlabel(xlabel = 'Predictions', fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.xticks(fontsize = 16)
    
    # Calculating precision ratio for train set
    negative_predictions = np.expand_dims(a = np.divide(cm_train[:, 0], cm_train.sum(axis = 0)[0]), axis = 1)
    positive_predictions = np.expand_dims(a = np.divide(cm_train[:, 1], cm_train.sum(axis = 0)[1]), axis = 1)
    final_array = np.hstack(tup = (negative_predictions, positive_predictions))
    
    # Plotting precision ratio for train set
    plt.subplot(2, 3, 3)
    sns.heatmap(data = final_array, cmap = plt.cm.Blues, annot = True, fmt = '.0%', cbar = False, xticklabels = labels, yticklabels = labels)
    plt.title(label = 'Train Set Precision Ratio(%)', fontsize = 16)
    plt.ylabel(ylabel = 'Ground Truth', fontsize = 16)
    plt.xlabel(xlabel = 'Predictions', fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.xticks(fontsize = 16)
    
    # Plotting confusion matrix for test set
    plt.subplot(2, 3, 4)
    sns.heatmap(data = cm_test, cmap = plt.cm.Blues, annot = True, fmt = '.4g', cbar = False, xticklabels = labels, yticklabels = labels)
    plt.title(label = 'Test Set Confusion Matrix', fontsize = 16)
    plt.ylabel(ylabel = 'Ground Truth', fontsize = 16)
    plt.xlabel(xlabel = 'Predictions', fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.xticks(fontsize = 16)
    
    # Calculating recall ratio for test set
    upper_array = np.divide(cm_test[0], cm_test.sum(axis = 1)[0])
    lower_array = np.divide(cm_test[1], cm_test.sum(axis = 1)[1])
    final_array = np.vstack(tup = (upper_array, lower_array))
    
    # Plotting recall ratio for test set
    plt.subplot(2, 3, 5)
    sns.heatmap(data = final_array, cmap = plt.cm.Blues, annot = True, fmt = '.0%', cbar = False, xticklabels = labels, yticklabels = labels)
    plt.title(label = 'Test Set Recall Ratio(%)', fontsize = 16)
    plt.ylabel(ylabel = 'Ground Truth', fontsize = 16)
    plt.xlabel(xlabel = 'Predictions', fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.xticks(fontsize = 16)
    
    # Calculating precision ratio for test set
    negative_predictions = np.expand_dims(a = np.divide(cm_test[:, 0], cm_test.sum(axis = 0)[0]), axis = 1)
    positive_predictions = np.expand_dims(a = np.divide(cm_test[:, 1], cm_test.sum(axis = 0)[1]), axis = 1)
    final_array = np.hstack(tup = (negative_predictions, positive_predictions))
    
    # Plotting precision ratio for test set
    plt.subplot(2, 3, 6)
    sns.heatmap(data = final_array, cmap = plt.cm.Blues, annot = True, fmt = '.0%', cbar = False, xticklabels = labels, yticklabels = labels)
    plt.title(label = 'Test Set Precision Ratio(%)', fontsize = 16)
    plt.ylabel(ylabel = 'Ground Truth', fontsize = 16)
    plt.xlabel(xlabel = 'Predictions', fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.xticks(fontsize = 16)
    plt.show()