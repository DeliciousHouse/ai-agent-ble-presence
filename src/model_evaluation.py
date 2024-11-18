from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X_test, y_test, target_names=None):
    """
    Evaluates the model's performance on the test set and logs the results.
    
    Parameters:
        model: Trained machine learning model
        X_test: Features of the test set
        y_test: True labels of the test set
        target_names: List of target class names (optional)
    """
    y_pred = model.predict(X_test)

    # Generate classification report
    if target_names is not None:
        report = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)
    else:
        report = classification_report(y_test, y_pred, zero_division=0)
    print("Classification Report:\n", report)

    # Plot and save the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("/app/data/confusion_matrix.png")
    print("Confusion matrix saved to /app/data/confusion_matrix.png.")
