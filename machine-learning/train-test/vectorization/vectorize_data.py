import pandas as pd
import numpy as np
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# --- AYARLAR ---
# Scriptin bulunduğu klasör (vectorization)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Bir üst klasör (raw_datasets)
RAW_DATASETS_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = SCRIPT_DIR
INPUT_FILE = os.path.join(RAW_DATASETS_DIR, "final_dataset_cleaned.csv")

def get_top_n_words(corpus, n=20):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

def main():
    print(f"🚀 Vektörizasyon İşlemi Başlatılıyor...")
    print(f"📂 Veri Kaynağı: {INPUT_FILE}")

    if not os.path.exists(INPUT_FILE):
        print("❌ HATA: final_dataset_cleaned.csv bulunamadı! Lütfen bir üst dizini kontrol et.")
        return

    # Veriyi Oku
    df = pd.read_csv(INPUT_FILE)
    print(f"✅ Veri Okundu: {len(df)} satır")
    
    # Null ve boş string check
    df = df.dropna(subset=['text'])
    df = df[df['text'].str.strip().astype(bool)]
    
    print(f"✅ Temiz Veri (Boşlar atıldı): {len(df)} satır")

    # 1. EN SIK TEKRAR EDEN KELİMELER (Frequency Analysis)
    print("\n📊 En Sık Geçen Kelimeler Analiz Ediliyor (Top 200)...")
    
    def safe_get_top_words(text_series, n=200):
        try:
            if text_series.empty:
                return []
            return get_top_n_words(text_series, n)
        except ValueError:
            return []

    # Human (0) Top Words
    human_text = df[df['label'] == 0]['text']
    top_human = safe_get_top_words(human_text, 200)
    
    # AI (1) Top Words
    ai_text = df[df['label'] == 1]['text']
    top_ai = safe_get_top_words(ai_text, 200)
    
    print("\n--- 🧑 HUMAN (En Sık 20 Kelime - Özet) ---")
    for word, freq in top_human[:20]: # Konsola sadece 20 tane bas
        print(f"{word}: {freq}")

    print("\n--- 🤖 AI (En Sık 20 Kelime - Özet) ---")
    for word, freq in top_ai[:20]: # Konsola sadece 20 tane bas
        print(f"{word}: {freq}")

    # Sonuçları TXT'ye yaz
    with open(os.path.join(OUTPUT_DIR, "top_words_analysis.txt"), "w", encoding="utf-8") as f:
        f.write("--- HUMAN TOP 200 WORDS ---\n")
        for word, freq in top_human:
            f.write(f"{word}: {freq}\n")
        f.write("\n--- AI TOP 200 WORDS ---\n")
        for word, freq in top_ai:
            f.write(f"{word}: {freq}\n")
    print(f"💾 Kelime analizi (Top 200) kaydedildi: {os.path.join(OUTPUT_DIR, 'top_words_analysis.txt')}")

    # 2. VEKTÖRİZASYON (TF-IDF)
    print("\n🧮 TF-IDF Vektörizasyonu Başlıyor...")
    print("   (Bu işlem veri boyutuna göre biraz zaman alabilir)")
    
    # max_features=5000 -> En önemli 5000 kelimeyi al (Boyutu yönetilebilir tutmak için)
    tfidf = TfidfVectorizer(max_features=5000)
    
    X = tfidf.fit_transform(df['text'])
    y = df['label'].values
    
    print(f"✅ Vektörizasyon Tamamlandı. Matris Boyutu: {X.shape}")

    # Kaydetme (Pickle ile)
    print("💾 Vektörler ve Model Kaydediliyor...")
    
    with open(os.path.join(OUTPUT_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(tfidf, f)
        
    with open(os.path.join(OUTPUT_DIR, "X_tfidf_matrix.pkl"), "wb") as f:
        pickle.dump(X, f)
        
    with open(os.path.join(OUTPUT_DIR, "y_labels.pkl"), "wb") as f:
        pickle.dump(y, f)

    print(f"✅ Tüm dosyalar '{OUTPUT_DIR}' klasörüne kaydedildi.")

if __name__ == "__main__":
    main()
