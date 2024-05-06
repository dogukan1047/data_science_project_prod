import pickle

# El ile girilen veriyi kullanıcıdan al
image_vector_input = input("Enter the image vector (comma-separated values): ")
image_vector = list(map(int, image_vector_input.split(',')))

# Eğitilmiş modeli yükle
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Tahmin yap
prediction = model.predict([image_vector])

# Tahmin sonucunu yazdır
print("Prediction:", prediction[0])

