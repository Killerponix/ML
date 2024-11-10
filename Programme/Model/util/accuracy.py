import numpy as np
def accuracy(y_true, y_pred):
    """
    Berechnet die Accuracy (Genauigkeit) basierend auf den tatsächlichen und den vorhergesagten Werten.

    Parameters:
    y_true (array-like): Die tatsächlichen Werte.
    y_pred (array-like): Die vorhergesagten Werte.

    Returns:
    float: Die Accuracy in Prozent.
    """
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = len(y_true)
    accuracy = correct_predictions / total_predictions
    print(f"Accuracy: {accuracy*100:.2f}%")
    return accuracy * 100  # Rückgabe in Prozent
