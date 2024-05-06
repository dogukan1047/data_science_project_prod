import csv
import os
from PIL import Image
import numpy as np

def image_to_vector(image_path):
    image = Image.open(image_path).convert("L")  # Görüntüyü gri ölçekli olarak aç
    image = image.resize((28, 28))  # Görüntüyü 28x28 piksele yeniden boyutlandır
    image_array = np.array(image)  # Görüntüyü bir NumPy dizisine dönüştür
    image_vector = image_array.flatten()  # Görüntüyü bir vektöre düzleştir
    return image_vector

def append_csv_from_images(image_dir, output_csv, auto_labels):
    # Dosya yoksa yeni bir dosya oluştur, varsa 'a' modunda aç
    mode = 'w' if not os.path.exists(output_csv) else 'a'
    
    with open(output_csv, mode, newline='') as csv_file:
        fieldnames = ['image_vector', 'label']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        # Dosya yoksa başlık satırını yaz
        if mode == 'w':
            writer.writeheader()

        for filename in os.listdir(image_dir):
            if filename.endswith(".jpg"):
                image_path = os.path.join(image_dir, filename)
                image_vector = image_to_vector(image_path)  # Görüntüyü vektöre dönüştür
                index = int(filename.split("_")[1].split(".")[0])
                label_index = (index - 1) // 10
                label = auto_labels[label_index]  # Otomatik etiketleri döngüye al
                writer.writerow({'image_vector': ','.join(map(str, image_vector)), 'label': label})

# Örnek kullanım
image_dir = "./output/output2"  # Parçaların bulunduğu dizin
output_csv = "./labels1.csv"  # Oluşturulacak veya güncellenecek CSV dosyasının adı
auto_labels = [0,1,2,3,4,5,6,7,8,9]  # Otomatik etiketler
append_csv_from_images(image_dir, output_csv, auto_labels)
