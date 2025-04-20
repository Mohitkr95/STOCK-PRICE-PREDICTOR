"""
Stock Price Predictor - Main Script
Uses LSTM neural networks to predict future stock prices based on historical data.
"""
import os
import logging
import warnings
from utils.model_utils import (
    preprocess_data, create_sequences, build_lstm_model,
    prepare_test_data, evaluate_model, plot_results,
    plot_training_loss, save_prediction_results
)
from keras.callbacks import ModelCheckpoint, EarlyStopping

# Configure standard logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/stock_prediction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress TensorFlow and other warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Ensure directories exist
os.makedirs("output", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

def main():
    """
    Main execution function for stock price prediction
    """
    # Configuration parameters
    DATA_PATH = 'data/nse-tata-global.csv'
    SEQUENCE_LENGTH = 60
    SPLIT_POINT = 987
    
    logger.info("Starting stock price prediction process")
    
    # Data preprocessing
    logger.info("Loading and preprocessing data")
    data = preprocess_data(DATA_PATH)
    
    # Creating sequences for LSTM
    logger.info("Creating sequence data for LSTM model")
    x_train, y_train, train, test, scaler, dataset = create_sequences(
        data, SEQUENCE_LENGTH, SPLIT_POINT
    )
    logger.info(f"Training set size: {train.shape[0]} records")
    logger.info(f"Test set size: {test.shape[0]} records")
    logger.info(f"Training sequences shape: {x_train.shape}")
    
    # Building and training the model
    logger.info("Building LSTM model architecture")
    model = build_lstm_model(input_shape=(x_train.shape[1], 1))
    model.summary(print_fn=logger.info)
    
    # Set up callbacks for better training
    checkpoint = ModelCheckpoint(
        filepath='models/best_model.keras',
        monitor='loss',
        save_best_only=True,
        verbose=1
    )
    early_stopping = EarlyStopping(
        monitor='loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    logger.info("Training LSTM model")
    history = model.fit(
        x_train, y_train,
        epochs=50,
        batch_size=32,
        verbose=1,
        callbacks=[checkpoint, early_stopping]
    )
    
    # Plot training loss
    logger.info("Generating training loss visualization")
    plot_training_loss(history, 'output/training_loss.png')
    
    # Generate predictions
    logger.info("Preparing test data for predictions")
    X_test = prepare_test_data(data, len(test), SEQUENCE_LENGTH, scaler)
    logger.info(f"Test sequences shape: {X_test.shape}")
    
    logger.info("Making predictions")
    predictions = model.predict(X_test, verbose=1)
    predictions = scaler.inverse_transform(predictions)
    
    # Evaluate model performance
    logger.info("Evaluating model performance")
    metrics = evaluate_model(test, predictions)
    logger.info("Model Performance Metrics:")
    logger.info(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}")
    logger.info(f"R² Score: {metrics['r2']:.4f}")
    logger.info(f"Prediction Accuracy: {metrics['accuracy']:.2f}%")
    
    # Visualize predictions
    logger.info("Generating stock price prediction visualization")
    train_data = data[:SPLIT_POINT]
    test_data = data[SPLIT_POINT:].copy()
    test_data['Predictions'] = predictions
    plot_results(train_data, test_data, predictions, 'output/stock_price_prediction.png')
    
    # Save prediction results
    logger.info("Saving prediction results")
    save_prediction_results(test_data, predictions, 'output/prediction_results.csv')
    
    logger.info("Stock price prediction process completed successfully")

if __name__ == "__main__":
    main()