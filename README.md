# Script Explanation

For the sake of explainability, I have created docstrings for each function and added comments for each line so that you can grasp how each line affects the workflow.

* **VifDropper** - It is a custom transformer which can be included in a pipeline to drop highly correlated features based on variance inflation factor (VIF) threshold that is set to **2.5** by default.
- **build_pipeline** -  It is a function to build a machine learning pipeline for a classification problem.
* **find_optimal_threshold** - It is a function that applies thresholding and automatically detects the best threshold for train & test sets. Supported evaluation metrics are as follow.

    1. Precision Score (Positive & Negative) 
    2. Recall Score (Positive & Negative)
    3. F Beta Score (Positive & Negative)
    4. Balanced Accuracy Score
    5. Accuracy Score

- **plot_roc_curve** - It is a function that automatically plots **Receiver Operating Characteristics (ROC)** curve for train and test sets.
* **plot_confusion_matrix** - It is a function that automatically plots confusion matrix, recall and precision ratio for train and test sets with a given threshold.