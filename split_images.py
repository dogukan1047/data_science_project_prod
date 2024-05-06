from PIL import Image
import os

def split_image(image_path, rows, cols, should_square=False, should_cleanup=False, output_dir="./output/"):
    # Görüntüyü yükle
    image = Image.open(image_path)
    
    # Görüntünün boyutlarını al
    width, height = image.size
    
    # Her parçanın boyutunu hesapla
    piece_width = width // cols
    piece_height = height // rows
    
    # Parça sayısını hesapla
    num_pieces = rows * cols
    
    # Görüntüyü parçalara böl
    for i in range(rows):
        for j in range(cols):
            # Parça boyutlarına göre kesme koordinatlarını hesapla
            left = j * piece_width
            top = i * piece_height
            right = (j + 1) * piece_width
            bottom = (i + 1) * piece_height
            
            # Kare olacak şekilde ayarla
            if should_square:
                size = max(piece_width, piece_height)
                diff_width = size - (right - left)
                diff_height = size - (bottom - top)
                left -= diff_width // 2
                top -= diff_height // 2
                right += diff_width - (diff_width // 2)
                bottom += diff_height - (diff_height // 2)
                
            # Parçayı kes ve kaydet
            piece = image.crop((left, top, right, bottom))
            output_path = os.path.join(output_dir, f"piece_{i * cols + j + 1}.jpg")
            piece.save(output_path)
    
    # Temizlik işlemi
    if should_cleanup:
        image.close()

# Örnek kullanım
split_image("./dataset_tp\\New.jpg", 10, 10, should_square=False, should_cleanup=False, output_dir="./output/output2/")