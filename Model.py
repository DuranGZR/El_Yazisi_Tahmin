import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 1 MNIST Veri Kümesini Yükleme
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2 Veriyi Normalleştirme (0-255 → 0-1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# 3 Şekli Düzgün Hale Getirme (CNN İçin)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# 4 OpenCV Kullanarak Veri Artırma (Data Augmentation)
def augment_image(image):
    image = image.reshape(28, 28)  # OpenCV işlemleri için 2D yapıya dönüştür
    rows, cols = image.shape[:2]

    # Rastgele Döndürme (-15 ile 15 derece arasında)
    angle = np.random.randint(-15, 15)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    image = cv2.warpAffine(image, M, (cols, rows))

    # Yatay ve Dikey Kaydırma (-5 ile 5 piksel arası)
    shift_x = np.random.randint(-5, 5)
    shift_y = np.random.randint(-5, 5)
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    image = cv2.warpAffine(image, M, (cols, rows))

    # Yakınlaştırma (0.9x ile 1.1x arasında)
    zoom_factor = np.random.uniform(0.9, 1.1)
    zoomed = cv2.resize(image, None, fx=zoom_factor, fy=zoom_factor)

    # Geriye dönüştürme
    zoomed = cv2.resize(zoomed, (28, 28))

    return zoomed.reshape(28, 28, 1)  # Tekrar CNN için uygun forma getir

#  Veri artırma işlemini uygulayarak yeni bir eğitim seti oluştur
augmented_x_train = np.array([augment_image(img) for img in x_train])
augmented_y_train = np.copy(y_train)  # Etiketler aynı kalıyor

# 5 Modeli Tanımlama (Gelişmiş CNN)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),  # Giriş katmanı
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),  # Overfitting’i önlemek için
    tf.keras.layers.Dense(10, activation='softmax')  # Çıktı katmanı (10 rakam için)
])

# 6️ Modeli Derleme (AdamW + Loss + Accuracy)
model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# 7️ Eğitim Callback (Grafik İçin)
class TrainingHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracy = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracy.append(logs.get('accuracy'))


history = TrainingHistory()

# 8️ Modeli Eğitme (OpenCV ile artırılmış verilerle)
model.fit(augmented_x_train, augmented_y_train, batch_size=128, epochs=20, validation_split=0.1, callbacks=[history])

# 9️ Modeli Değerlendirme
test_loss, test_acc = model.evaluate(x_test, y_test)
print("\n Test Doğruluğu: {:.2f}%".format(test_acc * 100))

#  Eğitim Sürecini Grafikle Gösterme
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.losses, label="Loss")
plt.title("Eğitim Kaybı")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.accuracy, label="Accuracy")
plt.title("Eğitim Doğruluğu")
plt.legend()
plt.show()

#  Modeli Kaydetme
model.save("mnist_digit_classifier_opencv.h5")
print("\n Model başarıyla kaydedildi!")

#  Modeli Yükleme ve Kullanma
loaded_model = tf.keras.models.load_model("mnist_digit_classifier_opencv.h5")

#  Test İçin Tahmin Yapma
i = 5  # İlk test verisi
prediction = np.argmax(loaded_model.predict(np.array([x_test[i]])))
print("\n Tahmin: {}, Gerçek Etiket: {}".format(prediction, y_test[i]))


#  Görseli Gösterme
def display(index):
    img = x_test[index].reshape(28, 28)
    plt.title(f"Tahmin: {prediction}, Gerçek: {y_test[index]}")
    plt.imshow(img, cmap='gray')
    plt.show()


# Eğer tahmin doğruysa göster
if prediction == y_test[i]:
    display(i)
