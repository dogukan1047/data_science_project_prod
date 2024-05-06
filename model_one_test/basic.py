import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle

# CSV dosyasını oku
data = pd.read_csv("labels1.csv")

# Girdi (X) ve hedef (y) değişkenlerini ayır
X = data['image_vector'].apply(lambda x: list(map(int, x.split(',')))).tolist()
y = data['label']

# Verileri normalize et
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Modeli tanımla ve eğit
model = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Modelin performansını test et
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model accuracy:", accuracy)

# Eğitilmiş modeli bir pickle dosyasına kaydet
with open('trained_model.pkl', 'wb') as f:
    pickle.dump(model, f)
