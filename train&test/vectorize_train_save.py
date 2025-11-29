import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import joblib  # Modelleri kaydetmek için
import os
import re
import nltk
from nltk.corpus import stopwords

# --- AYARLAR ---
HUMAN_DATA_PATH = "datasets/temiz_insan_verisi.csv"
AI_DATA_PATH = "datasets/temiz_ai_verisi.csv"
SAVE_DIR = "saved_models" # Dosyaların kaydedileceği klasör

# Klasör yoksa oluştur
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# --- 1. VERİ VE TEMİZLİK ---
nltk.download('stopwords')

def strict_clean(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text) 
    words = text.split()
    stop_words = set(stopwords.words('english'))
    custom_stops = {'abstract', 'summary', 'title', 'introduction', 'conclusion', 'paper', 'keywords'} 
    all_stops = stop_words.union(custom_stops)
    clean_words = [w for w in words if w not in all_stops and len(w) > 2]
    return " ".join(clean_words)

def prepare_data():
    print("1. Veriler Yükleniyor ve Hazırlanıyor...")
    df_human = pd.read_csv(HUMAN_DATA_PATH).dropna(subset=['cleaned_text'])
    df_ai = pd.read_csv(AI_DATA_PATH).dropna(subset=['cleaned_text'])
    df = pd.concat([df_human, df_ai], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Temizlik
    print("   -> Metin temizliği yapılıyor...")
    df['final_text'] = df['cleaned_text'].apply(strict_clean)
    return df

# --- 2. EĞİTİM VE KAYIT ---
def train_and_save():
    df = prepare_data()
    
    # A. Vektörleştirici (Vectorizer)
    print("\n2. TF-IDF Vektörleştirici Eğitiliyor...")
    vectorizer = TfidfVectorizer(max_features=5000, min_df=5)
    X = vectorizer.fit_transform(df['final_text']).toarray()
    y = df['label'].values
    
    # KAYIT 1: Vectorizer (Çok Önemli!)
    # Bunu kaybetmek, sözlüğü kaybetmek gibidir.
    joblib.dump(vectorizer, os.path.join(SAVE_DIR, 'tfidf_vectorizer.joblib'))
    print(f"✅ Vectorizer kaydedildi: {SAVE_DIR}/tfidf_vectorizer.joblib")

    # B. Model 1: Logistic Regression
    print("\n3. Logistic Regression Eğitiliyor ve Kaydediliyor...")
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X, y)
    joblib.dump(lr_model, os.path.join(SAVE_DIR, 'logistic_regression_model.joblib'))
    print("✅ Logistic Regression kaydedildi.")

    # C. Model 2: Random Forest
    print("\n4. Random Forest Eğitiliyor ve Kaydediliyor...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X, y)
    joblib.dump(rf_model, os.path.join(SAVE_DIR, 'random_forest_model.joblib'))
    print("✅ Random Forest kaydedildi.")

    # D. Model 3: Neural Network
    print("\n5. Neural Network Eğitiliyor ve Kaydediliyor...")
    nn_model = Sequential([
        Dense(64, input_shape=(X.shape[1],), activation='relu'), # input_dim yerine input_shape modern yöntem
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    nn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    nn_model.fit(X, y, epochs=8, batch_size=32, verbose=1)
    
    
    nn_model.save(os.path.join(SAVE_DIR, 'neural_network_model.keras'))
    print(f" Neural Network kaydedildi: {SAVE_DIR}/neural_network_model.keras")

    print("\n TÜM MODELLER BAŞARIYLA PAKETLENDİ!")

if __name__ == "__main__":
    train_and_save()