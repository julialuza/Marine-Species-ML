import os
import tensorflow as tf
import matplotlib.pyplot as plt

#komponenty modelu
EfficientNetB1 = tf.keras.applications.efficientnet.EfficientNetB1
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
Model = tf.keras.Model
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
GlobalAveragePooling2D = tf.keras.layers.GlobalAveragePooling2D
Adam = tf.keras.optimizers.Adam
EarlyStopping = tf.keras.callbacks.EarlyStopping

#ściezki
TRAIN_DIR = 'dataset/train'
VAL_DIR = 'dataset/val'
MODEL_SAVE_PATH = 'model/saved_model/model_efficientnetB1_13.keras'

#lista klas
CLASSES = sorted(os.listdir(TRAIN_DIR))
IMG_SIZE = (240, 240)
BATCH_SIZE = 32

#budowa modelu
base_model = EfficientNetB1(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
#uzycie EfficientNetB1 bez ostatnich warstw (include_top=False), z wagami z ImageNet
x = base_model.output # pobieramy ostatnie wyjście z EfficientNet
x = GlobalAveragePooling2D()(x) # spłaszczamy cechy przestrzenne do wektora
x = Dropout(0.3)(x) # zastosowanie Dropoutu - wyłączenie 30% neuronów losowo
x = Dense(384, activation='relu')(x) # gęsta warstwa ukryta z aktywacją ReLU
predictions = Dense(len(CLASSES), activation='softmax')(x) # warstwa wyjściowa z aktywacją softmax

model = Model(inputs=base_model.input, outputs=predictions) # budujemy kompletny model

# zamrożenie pierwszych 80 warstw (transfer learning)
for layer in base_model.layers[:80]:
    layer.trainable = False

# kompilacja modelu
loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05) # funkcja straty z wygładzeniem etykiet
model.compile(optimizer=Adam(learning_rate=1e-4), loss=loss_fn, metrics=['accuracy']) # kompilacja

# augmentacja
train_datagen = ImageDataGenerator(
    rescale=1./255, # przeskalowanie pikseli do zakresu 0–1
    rotation_range=20, # obrót obrazu o losowy kąt
    zoom_range=0.2, # losowe przybliżenia
    width_shift_range=0.1, # przesunięcie w poziomie
    height_shift_range=0.1, # przesunięcie w pionie
    horizontal_flip=True # odbicie w poziomie
)
val_datagen = ImageDataGenerator(rescale=1./255) # walidacja bez augmentacji

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
)
val_gen = val_datagen.flow_from_directory(
    VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
)

# EarlyStopping - wcześniejsze zatrzymanie treningu jeśli nie ma poprawy wyników
early_stop = EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True)

#faza 1: trening z zamroxonymi warstwami
history1 = model.fit(train_gen, epochs=10, validation_data=val_gen)

#faza 2: fine-tuning (odmrażanie kolejnych warstw)
for layer in base_model.layers[80:]:
    layer.trainable = True

#kompilacja z mniejszym learning rate do delikatnego dostrajania
model.compile(optimizer=Adam(learning_rate=1e-5), loss=loss_fn, metrics=['accuracy'])
#kontynuacja treningu z EarlyStopping
history2 = model.fit(train_gen, epochs=30, validation_data=val_gen, callbacks=[early_stop])

#zapis modelu
model.save(MODEL_SAVE_PATH)
print("\u2705 Model zapisany jako:", MODEL_SAVE_PATH)

#=====wykresy======
acc = history1.history['accuracy'] + history2.history['accuracy']
val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
loss = history1.history['loss'] + history2.history['loss']
val_loss = history1.history['val_loss'] + history2.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, acc, label='Train Accuracy')
plt.plot(epochs, val_acc, label='Validation Accuracy')
plt.xlabel('Epoka')
plt.ylabel('Dokładność')
plt.title('Dokładność modelu (EfficientNetB1)')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epoka')
plt.ylabel('Strata (Loss)')
plt.title('Strata modelu (EfficientNetB1)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('accuracy_plot_b1_13.png')
plt.show()
