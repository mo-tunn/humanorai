import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
import re
import nltk
from nltk.corpus import stopwords
import os
import warnings

# --- AYARLAR ---
# Gereksiz uyarıları gizle
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

HUMAN_DATA_PATH = "datasets/temiz_insan_verisi.csv"
AI_DATA_PATH = "datasets/temiz_ai_verisi.csv"

# NLTK Kontrolü
nltk.download('stopwords', quiet=True)

# 1. TEMİZLİK FONKSİYONU
def strict_clean(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text) 
    words = text.split()
    stop_words = set(stopwords.words('english'))
    custom_stops = {'abstract', 'summary', 'title', 'introduction', 'conclusion', 'paper', 'keywords'} 
    all_stops = stop_words.union(custom_stops)
    clean_words = [w for w in words if w not in all_stops and len(w) > 2]
    return " ".join(clean_words)

# 2. MODELLERİ EĞİTME VE HAZIRLAMA
def train_all_models():
    print("\n[SISTEM] Veriler yukleniyor ve modeller egitiliyor... Lutfen bekleyin.")
    
    # Veri Yükleme
    if not os.path.exists(HUMAN_DATA_PATH) or not os.path.exists(AI_DATA_PATH):
        raise FileNotFoundError("Veri setleri bulunamadi!")

    df_human = pd.read_csv(HUMAN_DATA_PATH).dropna(subset=['cleaned_text'])
    df_ai = pd.read_csv(AI_DATA_PATH).dropna(subset=['cleaned_text'])
    df = pd.concat([df_human, df_ai], ignore_index=True)
    
    # Temizlik
    df['final_text'] = df['cleaned_text'].apply(strict_clean)
    
    # Vektörleştirme
    print("   -> TF-IDF Vektorlestirici hazirlaniyor...")
    vectorizer = TfidfVectorizer(max_features=5000, min_df=5)
    X = vectorizer.fit_transform(df['final_text']).toarray()
    y = df['label'].values
    
    models = {}

    # Model 1: Logistic Regression
    print("   -> Logistic Regression egitiliyor...")
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X, y)
    models['Logistic Regression'] = lr_model

    # Model 2: Random Forest
    print("   -> Random Forest egitiliyor...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X, y)
    models['Random Forest'] = rf_model

    # Model 3: Neural Network
    print("   -> Neural Network egitiliyor...")
    nn_model = Sequential([
        Input(shape=(X.shape[1],)),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    nn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    nn_model.fit(X, y, epochs=5, batch_size=32, verbose=0)
    models['Neural Network'] = nn_model

    print("[SISTEM] Tum modeller kullanima hazir!\n")
    return models, vectorizer

# 3. ÇOKLU TAHMİN FONKSİYONU
def predict_with_all_models(text, models, vectorizer):
    # Temizlik ve Vektörleştirme
    clean_text = strict_clean(text)
    vectorized_text = vectorizer.transform([clean_text]).toarray()
    
    results = {}
    
    # Her model için tahmin al
    for name, model in models.items():
        if name == 'Neural Network':
            prob = model.predict(vectorized_text, verbose=0)[0][0]
        else:
            prob = model.predict_proba(vectorized_text)[0][1]
        results[name] = prob
        
    return results

def get_interpretation(score):
    if score > 0.85: return "🤖 KESIN AI"
    if score > 0.60: return "🤔 MUHTEMEL AI"
    if score > 0.40: return "⚖️ BELIRSIZ"
    if score > 0.15: return "👤 MUHTEMEL INSAN"
    return "🧠 KESIN INSAN"

# --- ANA UYGULAMA ---
if __name__ == "__main__":
    # Eğitim
    models, vectorizer = train_all_models()
    
    print("="*60)
    print("🚀 MULTI-MODEL AI DETECTOR (FRONTEND DEMO)")
    print("="*60)
    print("Cikmak icin 'q' yazin.\n")
    
    while True:
        user_input = input("📝 Metni yapistirin: ")
        
        if user_input.lower() == 'q':
            break
            
        if len(user_input) < 50:
            print("⚠️ Lutfen daha uzun bir metin girin (Min 50 karakter).")
            continue
            
        print("\n" + "-"*60)
        print("📊 ANALIZ SONUCLARI")
        print("-"*60)
        
        predictions = predict_with_all_models(user_input, models, vectorizer)
        
        # Sonuçları tablo gibi yazdır
        print(f"{'MODEL':<25} | {'OLASILIK':<10} | {'KARAR'}")
        print("-" * 60)
        
        avg_prob = 0
        for name, prob in predictions.items():
            avg_prob += prob
            percent = prob * 100
            decision = get_interpretation(prob)
            print(f"{name:<25} | %{percent:.2f}     | {decision}")
            
        print("-" * 60)
        
        # Ortalama Sonuç (Ensemble Mantığı)
        avg_prob /= len(predictions)
        print(f"{'GENEL ORTALAMA':<25} | %{avg_prob*100:.2f}     | {get_interpretation(avg_prob)}")
        print("\n")