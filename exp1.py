import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# Step 2: Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Step 3: Preprocess the data
x_train = x_train.reshape((x_train.shape[0], 28*28)).astype("float32") / 255
x_test = x_test.reshape((x_test.shape[0], 28*28)).astype("float32") / 255

# Step 4: Create DNN model (Deeper Network)
model = models.Sequential([
    layers.Dense(512, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Step 5: Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Step 6: Train model
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Step 7: Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test)
print("\nTest Accuracy:", test_acc)

# Step 8: Display sample predictions
predictions = model.predict(x_test[:5])
predicted_labels = np.argmax(predictions, axis=1)

for i in range(5):
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {predicted_labels[i]}, Actual: {y_test[i]}")
    plt.axis('off')
    plt.show()
