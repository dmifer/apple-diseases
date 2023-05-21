import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dropout

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# Load and preprocess data
print("Loading and preprocessing data...")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_generator = train_datagen.flow_from_directory('apple_dataset\\train', target_size=(224, 224), batch_size=32, class_mode='categorical', subset='training')
validation_generator = train_datagen.flow_from_directory('apple_dataset\\validation', target_size=(224, 224), batch_size=32, class_mode='categorical', subset='validation')

# Load pre-trained MobileNetV2 and customize the classifier
print("Loading pre-trained MobileNetV2 and customizing the classifier...")
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(4, activation='softmax')(x)  # Assuming 3 disease types + 1 healthy class
dropout_layer = Dropout(0.5)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

callbacks = [early_stopping, model_checkpoint]

# Compile and train the model
print("Compiling and training the model...")
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, epochs=25, validation_data=validation_generator)

# Unfreeze some layers of the base model for fine-tuning
print("Unfreezing some layers of the base model for fine-tuning...")
for layer in model.layers[:100]:
    layer.trainable = False
for layer in model.layers[100:]:
    layer.trainable = True

# Compile the model with a lower learning rate for fine-tuning
print("Compiling the model with a lower learning rate for fine-tuning...")
model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with fine-tuning
print("Training the model with fine-tuning...")
model.fit(train_generator, epochs=10, validation_data=validation_generator)

# Save the trained model
print("Saving the trained model...")
model.save('apple_disease_classifier.h5')

# Evaluate the model on test data (assuming you have a separate test set)
print("Evaluating the model on test data...")
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory('apple_dataset\\test', target_size=(224, 224), batch_size=32, class_mode='categorical')
test_loss, test_acc = model.evaluate(test_generator)
print('Test accuracy:', test_acc)