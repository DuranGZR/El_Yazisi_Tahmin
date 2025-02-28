# 🖊️ El Yazısı Rakam Tanıma (MNIST + OpenCV + CNN)

Bu proje, **MNIST el yazısı rakam veri setini** kullanarak bir **derin öğrenme modeli (CNN)** eğitir ve ardından **OpenCV** ile etkileşimli bir çizim arayüzü sunar. Kullanıcı, ekrana çizdiği rakamları modele tanıtıp gerçek zamanlı tahmin alabilir.

---


## 📊 Veri Kümesi ve Ön İşleme

- Kullanılan veri seti: [MNIST](http://yann.lecun.com/exdb/mnist/).
- Görseller 0-255 aralığından **0-1** aralığına normalize edildi.
- Görseller CNN uyumluluğu için boyutlandırıldı: **(28, 28, 1)**

### 🛠️ Veri Artırma (Augmentation)

OpenCV kullanarak eğitim verilerine aşağıdaki artırmalar (augmentations) uygulandı:
- **Döndürme:** Rastgele -15 ile +15 derece arası.
- **Kaydırma:** X ve Y eksenlerinde rastgele -5 ile +5 piksel arası.
- **Yakınlaştırma:** Rastgele %90 ile %110 arası ölçekleme.

---

## 🧠 Model Mimarisi

Model, **Convolutional Neural Network (CNN)** yapısındadır:

| Katman | Detaylar |
| --- | --- |
| Conv2D + BatchNorm + MaxPooling | 32 filtre, 3x3, ReLU |
| Conv2D + BatchNorm + MaxPooling | 64 filtre, 3x3, ReLU |
| Conv2D + BatchNorm + MaxPooling | 128 filtre, 3x3, ReLU |
| Flatten | |
| Dense + Dropout | 128 nöron, ReLU, %50 dropout |
| Dense (Output) | 10 sınıf, Softmax |

### 🔧 Derleme Ayarları
- **Optimizer:** AdamW
- **Loss:** Sparse Categorical Crossentropy
- **Metric:** Accuracy

---

## 📈 Eğitim Süreci

- Eğitim verisi: MNIST + OpenCV ile artırılmış veri
- **Batch size:** 128
- **Epoch sayısı:** 20
- **Validation Split:** %10
- Eğitim süreci boyunca kayıp (loss) ve doğruluk (accuracy) grafikleri kaydedildi ve gösterildi.

---

## 🖥️ OpenCV Tabanlı Çizim ve Tahmin Arayüzü

Eğitilen model, **main.py** ile OpenCV kullanarak etkileşimli bir arayüzde test edilebilir.

### 🖌️ Arayüz Özellikleri

| Tuş | İşlev |
| --- | --- |
| **Fare** | Çizim yapar |
| **ENTER** | Tahmin yapar |
| **C** | Ekranı tamamen temizler |
| **T** | Ekranı hafif temizler (soft clear) |
| **+ / -** | Fırça boyutunu değiştirir |
| **ESC** | Çıkış yapar |

### 🎨 Arayüz Görseli

Çizim yapılacak alan, tahmin sonucu ve kullanım talimatları ekranda gösterilir.

---

## 🔗 Kullanılan Kütüphaneler

- TensorFlow
- OpenCV
- NumPy
- Matplotlib

---

## 📦 Projeyi Çalıştırma

### 1️⃣ Modeli Eğitmek
```bash
python model.py
