# Mars Rover Crater Detection 🚀🔴

Bu proje, otonom keşif araçlarının (Rover) ve endüstriyel AGV sistemlerinin seyir güvenliğini sağlamak amacıyla geliştirilmiş bir derin öğrenme (CNN) modülüdür. Model, araç kamerasından veya sensörlerinden alınan yüzey görüntülerini analiz ederek rotada bir krater veya tehlikeli çukur olup olmadığını sınıflandırır.

## 📌 Proje Özeti
Otonom sistemlerde seyir güvenliği fiziksel donanımın bütünlüğü için kritik bir öneme sahiptir. Bu projede, PyTorch kullanılarak sıfırdan bir Evrişimli Sinir Ağı (CNN) mimarisi inşa edilmiş ve Mars/Ay yüzeyi fotoğrafları üzerinde eğitilmiştir. 

Modelin temel navigasyon felsefesi **"False Positive" (Yanlış Pozitif) toleransına** dayanır. Sistem, emin olamadığı yüksek kontrastlı uçurumları veya karmaşık pürüzlü zeminleri potansiyel "Krater" olarak işaretleyerek otonom aracın güvenli ve tedbirli bir rota çizmesini sağlar. Tehlikeyi atlayıp aracın hasar alması (False Negative) riskine karşı, güvenli hata yaklaşımı benimsenmiştir.

## 🛠️ Kullanılan Teknolojiler
* **Programlama Dili:** Python
* **Derin Öğrenme Kütüphanesi:** PyTorch, Torchvision
* **Veri İşleme & Görselleştirme:** NumPy, Matplotlib, PIL

## 📊 Veri Seti ve Ön İşleme
Otonom sürüş testlerinde veri sızıntısını (data leakage) önlemek amacıyla veri seti etiketlerine göre sıfırdan karıştırılmış ve dengeli bir şekilde bölünmüştür:
* **Train (Eğitim):** %70 
* **Validation (Doğrulama):** %15
* **Test:** %15

Modelin zorlu kamera açılarına ve sarsıntılara karşı genelleme yeteneğini artırmak için **Eğitim** kümesine rastgele yatay çevirme (`RandomHorizontalFlip`) ve 15 derecelik döndürme (`RandomRotation`) uygulanmıştır. Tüm görüntüler ağa verilmeden önce `224x224` boyutuna getirilmiş ve ImageNet standartlarında normalize edilmiştir.

## 🧠 Model Mimarisi ve Eğitim
* 3 adet `Conv2D` ve `MaxPool2d` katmanından oluşan, sıfırdan tasarlanmış hafif sıklet bir CNN mimarisi.
* **Optimizer:** Adam (Öğrenme Oranı: 0.001)
* **Kayıp Fonksiyonu (Loss):** CrossEntropyLoss
* **Epoch Sayısı:** 10

## 📈 Performans ve Metrikler
10 epoch'luk eğitim sonucunda model, eğitim ve doğrulama kayıplarını (loss) eşzamanlı olarak düşürerek aşırı öğrenmeye (overfitting) girmeden istikrarlı bir grafik sergilemiştir:
* **Doğrulama Başarısı (Validation Accuracy):** %95.00
* **Test Başarısı (Test Accuracy):** %87.50

## 🚀 Kurulum ve Kullanım

1. Projeyi bilgisayarınıza klonlayın:
`git clone https://github.com/esra-ylmz8572tech/mars-rover-crater-detection.git`

2. Gerekli kütüphaneleri yükleyin:
`pip install torch torchvision pillow`

3. Modeli lokal bir fotoğrafla test etmek için örnek scripti çalıştırın:
`python tahmin_et.py`

## 🔭 Gelecek Geliştirmeler (Hedefler)
* Modelin OpenCV kullanılarak ZED 2 veya standart web kameraları ile gerçek zamanlı (real-time) video akışına entegre edilmesi.
* ROS 2 ortamında otonom duruş ve engel aşma (obstacle avoidance) algoritmalarıyla entegre çalışacak bir node haline getirilmesi.
