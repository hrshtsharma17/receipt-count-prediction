import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np

def model_eval(X_test, y_test, model, output_folder):        

    train_loss = model.history.history['loss']
    val_loss = model.history.history['val_loss']

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate regression scores
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Plot training history
    plt.figure(figsize=(10, 4))

    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Training Loss', c='blue')
    plt.plot(val_loss, label='Validation Loss', c='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    # Plot test data and predictions
    x = np.array([i for i in range(len(y_test))])
    plt.subplot(1, 2, 2)
    plt.scatter(x, y_test, label='Test Data', c='blue', marker='o')
    plt.scatter(x, y_pred, label='Predictions', c='red', marker='x')
    plt.xlabel('X_test')
    plt.ylabel('Y')
    plt.title('Test Data vs. Predictions')
    plt.legend()

    # Save the training and testing plots
    plt.savefig(os.path.join(output_folder, 'model_train_test_plots.png'))
    plt.close()

    # Save regression scores to a text file
    with open(os.path.join(output_folder, 'regression_scores.txt'), 'w') as file:
        file.write(f'Mean Squared Error (MSE): {mse}\n')
        file.write(f'Mean Absolute Error (MAE): {mae}\n')

    print("Model and results saved to:", output_folder)