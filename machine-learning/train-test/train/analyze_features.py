import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# --- AYARLAR ---
CURRENT_DIR = os.getcwd()
VECTOR_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "vectorization")
MODEL_DIR = CURRENT_DIR

# Dosya Yolları
MODEL_PATH = os.path.join(MODEL_DIR, "model_Logistic_Regression.pkl")
VECTORIZER_PATH = os.path.join(VECTOR_DIR, "tfidf_vectorizer.pkl")

def main():
    print("🕵️ Overfitting Analizi: Model Neyi Öğrendi?")
    
    # Dosyaları Yükle
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        print("❌ Model veya Vektörleştirici dosyası bulunamadı.")
        return

    print("📂 Modeller yükleniyor...")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
        
    # Feature İsimlerini Al (Kelimeler)
    feature_names = vectorizer.get_feature_names_out()
    
    # Katsayıları Al (Logistic Regression Coefficients)
    # Binary classification olduğu için model.coef_[0] bize tek bir dizi verir.
    # Pozitif değerler -> Class 1 (AI)
    # Negatif değerler -> Class 0 (Human)
    coefs = model.coef_[0]
    
    # Sıralama
    # En negatiften en pozitife doğru sırala
    sorted_indices = np.argsort(coefs)
    
    print("\n--- 🧑 HUMAN SINIFI İÇİN EN GÜÇLÜ İPUÇLARI (Top 20) ---")
    # En negatif 20 değer (Human göstergeleri)
    top_human_indices = sorted_indices[:20]
    for idx in top_human_indices:
        print(f"   {feature_names[idx]:<20} (Score: {coefs[idx]:.4f})")
        
    print("\n--- 🤖 AI SINIFI İÇİN EN GÜÇLÜ İPUÇLARI (Top 20) ---")
    # En pozitif 20 değer (AI göstergeleri), sondan geriye
    top_ai_indices = sorted_indices[-20:][::-1]
    for idx in top_ai_indices:
        print(f"   {feature_names[idx]:<20} (Score: {coefs[idx]:.4f})")

    # Grafik Çiz
    plt.figure(figsize=(12, 8))
    
    # Human Features
    human_features = [feature_names[i] for i in top_human_indices]
    human_scores = [coefs[i] for i in top_human_indices]
    
    # AI Features
    ai_features = [feature_names[i] for i in top_ai_indices]
    ai_scores = [coefs[i] for i in top_ai_indices]
    
    # Birleştir
    all_features = human_features + ai_features[::-1]
    all_scores = human_scores + ai_scores[::-1]
    colors = ['blue'] * 20 + ['red'] * 20
    
    plt.barh(all_features, all_scores, color=colors)
    plt.title("Model Feature Importance (Blue=Human, Red=AI)")
    plt.xlabel("Coefficient Value")
    plt.tight_layout()
    plt.savefig(os.path.join(CURRENT_DIR, "feature_importance_analysis.png"))
    print("\n📊 Grafik kaydedildi: feature_importance_analysis.png")
    
    print("\n🧐 YORUM:")
    print("Eğer yukarıdaki kelimeler çok jenerik (the, and, of) değil de çok spesifik (wikipedia, news, dataset, title) ise model ezberliyor demektir.")

if __name__ == "__main__":
    main()
