import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image

# Step 1: Prepare the Dataset
data_dir = 'small_img_set/cropped_traffic_lights'
labels = []
images = []

for entry in os.listdir(data_dir):
    if entry.endswith('.json'):
        continue  # Skip JSON files

    image_path = os.path.join(data_dir, entry)
    image = Image.open(image_path)
    image = image.resize((224, 224))  # Resize the image
    images.append(np.array(image))

    # Parse the label from the file name or annotations
    if 'red' in entry:
        labels.append('red')
    elif 'green' in entry:
        labels.append('green')
    else:
        labels.append('other')

# Step 2: Split the Dataset
train_data, test_data, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Step 3: Build the Neural Network
base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
predictions = tf.keras.layers.Dense(3, activation='softmax')(x)
model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

# Step 4: Train the Neural Network
train_labels = LabelEncoder().fit_transform(train_labels)
train_dataset = tf.data.Dataset.from_tensor_slices((np.array(train_data), train_labels)).batch(32)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        logits = model(images, training=True)
        loss_value = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss_value


num_epochs = 10
for epoch in range(num_epochs):
    for batch_images, batch_labels in train_dataset:
        loss = train_step(batch_images, batch_labels)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}')

# Step 5: Evaluate the Model
test_labels = LabelEncoder().fit_transform(test_labels)
test_dataset = tf.data.Dataset.from_tensor_slices((np.array(test_data), test_labels)).batch(32)
test_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()

for batch_images, batch_labels in test_dataset:
    logits = model(batch_images, training=False)
    test_accuracy_metric.update_state(batch_labels, logits)

test_accuracy = test_accuracy_metric.result().numpy()
print(f"Test accuracy: {test_accuracy}")

# Step 6: Save the Model
model.save('traffic_light_classifier.h5')
