# Arda Mavi
import os
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from get_dataset import get_dataset
from get_model import get_model, save_model

epochs = 100
batch_size = 5

def train_model(model, X, X_test, Y, Y_test):
    """Train the model with checkpoints and TensorBoard logging."""
    # Create checkpoints directory if it doesn't exist
    checkpoint_dir = 'Data/Checkpoints/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Define callbacks
    callbacks = [
        ModelCheckpoint(
            os.path.join(checkpoint_dir, 'best_weights.h5'),
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode='auto'
        ),
        TensorBoard(
            log_dir=os.path.join(checkpoint_dir, 'logs'),
            histogram_freq=1,
            write_graph=True,
            write_images=False
        )
    ]
    
    # Train the model
    model.fit(
        X, Y,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_test, Y_test),
        shuffle=True,
        callbacks=callbacks
    )

    return model

def main():
    """Main function to get data, train the model, and save it."""
    # Get dataset
    X, X_test, Y, Y_test = get_dataset()
    
    # Build and train model
    model = get_model()
    model = train_model(model, X, X_test, Y, Y_test)
    
    # Save model and weights
    save_model(model)

if __name__ == '__main__':
    main()

