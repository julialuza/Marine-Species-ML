import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn.metrics

#ten kod - evaluate_model.py - służył do oceny naszego gotowego modelu i sprawdzenie go na zbiorze testowym
#ścieżki
MODEL_PATH = 'model/saved_model/model_efficientnetB1_13.keras'
TEST_DIR = 'dataset/test'
IMG_SIZE = (240, 240)
BATCH_SIZE = 32 #liczba obrazów, które model przetwarza jednocześnie

#zaladuj model
model = tf.keras.models.load_model(MODEL_PATH)

#generator danych testowych
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

#klasy
class_indices = test_gen.class_indices
class_labels = list(class_indices.keys())

#predykcje
predictions = model.predict(test_gen)
y_pred = np.argmax(predictions, axis=1)
y_true = test_gen.classes

#raport klasyfikacji
report = sklearn.metrics.classification_report(
    y_true, y_pred, target_names=class_labels, output_dict=True
)

#=====wykresy======
#confusion matrix
conf_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
plt.imshow(conf_matrix, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(ticks=np.arange(len(class_labels)), labels=class_labels, rotation=45, ha='right')
plt.yticks(ticks=np.arange(len(class_labels)), labels=class_labels)
plt.colorbar()
plt.tight_layout()
plt.savefig('confusion_matrix_B1_13.png')
plt.show()

#precyzja dla danej klasy
precisions = [report[label]['precision'] for label in class_labels]
plt.figure(figsize=(14, 6))
bars = plt.bar(class_labels, precisions, color='skyblue')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1)
plt.ylabel('Precision')
plt.title('Precision per Class')
for bar, val in zip(bars, precisions):
    plt.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f'{val:.2f}', ha='center', fontsize=8)
plt.tight_layout()
plt.savefig('precision_per_class_B1_13.png')
plt.show()

#średnie wyniki
accuracy = report['accuracy']
macro_avg = report['macro avg']
weighted_avg = report['weighted avg']

print(f"Dokładność ogólna (accuracy): {accuracy:.2f}")