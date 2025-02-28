import numpy as np
import tensorflow as tf
import cv2

# Modeli yükle
model = tf.keras.models.load_model("mnist_digit_classifier_opencv.h5")

# Pencereyi büyütelim ve düzgün bir arka plan oluşturalım
canvas = np.ones((500, 500), dtype=np.uint8) * 255  # Daha büyük beyaz alan
brush_size = 10  # Fırça boyutu
drawing = False  # Çizim yapılıyor mu?

# Tahmin sonucu değişkeni
prediction_text = "Tahmin: ?"

# Fare olaylarını takip eden fonksiyon
def draw(event, x, y, flags, param):
    global drawing, brush_size

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(canvas, (x, y), brush_size, 0, -1, cv2.LINE_AA)  # Anti-aliasing ile yumuşatma
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

# OpenCV penceresi oluştur
cv2.namedWindow("El Yazisi Tanima", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("El Yazisi Tanima", draw)

while True:
    display_canvas = canvas.copy()

    # Kenarlık ekleyerek daha modern bir görünüm sağlayalım
    cv2.rectangle(display_canvas, (0, 0), (500, 60), (230, 230, 230), -1)  # Üst bilgi alanı (gri arka plan)
    cv2.rectangle(display_canvas, (0, 450), (500, 500), (200, 200, 200), -1)  # Alt tahmin alanı (gri arka plan)

    # Kullanıcı için talimatları ekleyelim (Daha okunaklı font ile)
    cv2.putText(display_canvas, "ENTER: Tahmin | C: Temizle | +/-: Firca | T: Soft Temizle",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2, cv2.LINE_AA)

    # Tahmin metni kutunun içinde ve şık bir font ile ortalanmış şekilde gösterelim
    text_size = cv2.getTextSize(prediction_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
    text_x = (500 - text_size[0]) // 2  # Ortalamak için x pozisyonu
    cv2.putText(display_canvas, prediction_text, (text_x, 485),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)

    # Fırça boyutunu göstermek için küçük bir ikon ekleyelim
    cv2.circle(display_canvas, (460, 30), brush_size // 2, (50, 50, 50), -1, cv2.LINE_AA)

    cv2.imshow("El Yazısı Tanıma", display_canvas)  # Pencereyi göster
    key = cv2.waitKey(1)

    if key == 13:  # ENTER tuşuna basınca tahmin yap
        # Çizimi modele uygun hale getir
        img_resized = cv2.resize(canvas, (28, 28))  # 28x28'e küçült
        img_resized = img_resized / 255.0  # Normalizasyon
        img_resized = 1 - img_resized  # Siyah-beyaz ters çevirme (MNIST formatı için)
        img_resized = img_resized.reshape(1, 28, 28, 1)  # Model için uygun şekle getir

        # Model ile tahmin yap
        prediction = np.argmax(model.predict(img_resized))

        # Sonucu değişken olarak tut
        prediction_text = f"Tahmin: {prediction}"

        print(f"\n📌 Model Tahmini: {prediction}")

    elif key == ord("c"):  # 'C' tuşuna basınca ekranı tamamen temizle
        canvas = np.ones((500, 500), dtype=np.uint8) * 255
        prediction_text = "Tahmin: ?"

    elif key == ord("t"):  # 'T' tuşuna basınca hafif şeffaf temizleme (Soft Temizleme)
        canvas = cv2.addWeighted(canvas, 0.5, np.ones_like(canvas) * 255, 0.5, 0)

    elif key == ord("+"):  # '+' tuşuna basınca fırça boyutunu artır
        brush_size = min(brush_size + 5, 50)

    elif key == ord("-"):  # '-' tuşuna basınca fırça boyutunu azalt
        brush_size = max(brush_size - 5, 5)

    elif key == 27:  # ESC tuşuna basınca çık
        break

cv2.destroyAllWindows()
