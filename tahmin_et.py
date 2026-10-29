import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# 1. ADIM: İskelet Modeli Tanımlama (Eğitimdeki kodun aynısı)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 2. ADIM: Eğittiğimiz Beyni (.pth) İskelete Yükleme
model = SimpleCNN()
# Bilgisayarında GPU olmayabileceğini varsayarak 'cpu' modunda açıyoruz
model.load_state_dict(torch.load('rover_crater_model.pth', map_location=torch.device('cpu')))
model.eval() # Tahmin moduna al

# 3. ADIM: Görüntü Ön İşleme (Eğitimde kullandığımız valid/test kurallarının BİREBİR aynısı)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 4. ADIM: Fotoğrafı Alıp Tahmin Yapan Fonksiyon
def fotografa_bak(foto_yolu):
    try:
        # Fotoğrafı yükle ve RGB renk formatına çevir
        img = Image.open(foto_yolu).convert('RGB')
        
        # Fotoğrafı 224x224 yap ve tensöre (matematiksel diziye) çevir
        img_tensor = transform(img)
        
        # PyTorch fotoğrafları yığın (batch) halinde bekler. 
        # Tek bir fotoğraf verdiğimiz için boyutuna hayali bir yığın (1) ekliyoruz
        img_tensor = img_tensor.unsqueeze(0) 
        
        # Modele fotoğrafı göster ve tahmini al
        with torch.no_grad():
            cikis = model(img_tensor)
            _, tahmin_edilen = torch.max(cikis, 1)
            
        # Sınıflar klasör isimlerine göre alfabetik sıradadır: 0->crater, 1->no_crater
        siniflar = ['Krater Var (Tehlike)', 'Krater Yok (Düz Zemin)']
        sonuc = siniflar[tahmin_edilen.item()]
        
        print(f"📷 İncelenen Fotoğraf: {foto_yolu}")
        print(f"🤖 Yapay Zeka Diyor Ki: {sonuc}\n")
        
    except FileNotFoundError:
        print(f"HATA: '{foto_yolu}' adında bir fotoğraf bulunamadı!")

# --- BURADAN KONTROL EDEBİLİRSİN ---
# Soru işaretli olan yere, test etmek istediğin fotoğrafın adını yaz:
fotografa_bak('deneme_fotografi.png')