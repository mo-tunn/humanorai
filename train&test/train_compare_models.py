import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, 
                             mean_absolute_error, r2_score, roc_auc_score)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
import os
import re
import nltk
from nltk.corpus import stopwords
import warnings

# --- AYARLAR VE KISITLAMALAR ---
# Gereksiz uyarıları kapat (Temiz terminal için)
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Pandas görünüm ayarları (Tabloların kaymasını engeller)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

# Kütüphane indirmeleri
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Dosya Yolları
HUMAN_DATA_PATH = "datasets/temiz_insan_verisi.csv"
AI_DATA_PATH = "datasets/temiz_ai_verisi.csv"
PLOT_FILENAME = "model_performans_grafigi.png"

# ==========================================
# 1. YARDIMCI FONKSİYONLAR
# ==========================================

def load_data():
    print("[BILGI] Veriler yukleniyor...")
    if not os.path.exists(HUMAN_DATA_PATH) or not os.path.exists(AI_DATA_PATH):
        raise FileNotFoundError(f"HATA: Dosyalar bulunamadi! Lutfen '{HUMAN_DATA_PATH}' ve '{AI_DATA_PATH}' yollarini kontrol et.")

    df_human = pd.read_csv(HUMAN_DATA_PATH)
    df_ai = pd.read_csv(AI_DATA_PATH)
    
    # NaN temizliği
    df_human = df_human.dropna(subset=['cleaned_text'])
    df_ai = df_ai.dropna(subset=['cleaned_text'])

    # Birleştirme ve Karıştırma
    df_final = pd.concat([df_human, df_ai], ignore_index=True)
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"   - Insan Verisi Sayisi : {len(df_human)}")
    print(f"   - AI Verisi Sayisi    : {len(df_ai)}")
    print(f"   - Toplam Veri Seti    : {len(df_final)}")
    return df_final

def strict_clean(text):
    """
    Agresif temizlik: Stop words, noktalama ve 'abstract' gibi sizinti kelimelerini atar.
    """
    # 1. Küçük harf ve sadece harfler
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text) 
    
    # 2. Kelimelere böl
    words = text.split()
    
    # 3. YASAKLI KELİME LİSTESİ
    stop_words = set(stopwords.words('english'))
    custom_stops = {'abstract', 'summary', 'title', 'introduction', 'conclusion', 'paper', 'keywords'} 
    all_stops = stop_words.union(custom_stops)
    
    # 4. Temizle (2 harften kısa kelimeleri de at)
    clean_words = [w for w in words if w not in all_stops and len(w) > 2]
    
    return " ".join(clean_words)

def get_metrics(y_true, y_pred, y_prob=None):
    metrics = {}
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    metrics['F1 Score'] = f1_score(y_true, y_pred)
    metrics['ROC-AUC'] = roc_auc_score(y_true, y_prob) if y_prob is not None else 0.0
    metrics['MAE'] = mean_absolute_error(y_true, y_pred) 
    metrics['R2 Score'] = r2_score(y_true, y_pred)
    return metrics

def analyze_model_features(model, vectorizer, top_n=20):
    """
    Modelin en çok hangi kelimelere ağırlık verdiğini gösterir.
    """
    print("\n" + "="*60)
    print("DEDEKTIF MODU: MODEL KARAR KRITERLERI ANALIZI")
    print("="*60)
    
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]
    df_feats = pd.DataFrame({'Kelime': feature_names, 'Katsayi': coefs})
    
    top_ai = df_feats.sort_values(by='Katsayi', ascending=False).head(top_n)
    top_human = df_feats.sort_values(by='Katsayi', ascending=True).head(top_n)
    
    # Tablo formatında yazdırma
    print(f"\n[ AI (LABEL 1) ICIN EN GUCLU {top_n} IPUCU ]")
    print("-" * 40)
    print(top_ai.to_string(index=False))
    
    print(f"\n[ INSAN (LABEL 0) ICIN EN GUCLU {top_n} IPUCU ]")
    print("-" * 40)
    print(top_human.to_string(index=False))
    print("\n" + "="*60)

def plot_and_save_results(results):
    df_res = pd.DataFrame(results).T
    
    print("\n" + "="*60)
    print("FINAL PERFORMANS TABLOSU")
    print("="*60)
    print(df_res)
    print("="*60)
    
    # Grafik Ayarları
    plt.style.use('ggplot') # Daha profesyonel görünüm
    plt.figure(figsize=(12, 6))
    
    metrics_to_plot = ['Accuracy', 'F1 Score', 'ROC-AUC']
    ax = df_res[metrics_to_plot].plot(kind='bar', figsize=(12, 6), edgecolor='black', rot=0)
    
    plt.title('AI vs Human Model Performans Karsilastirmasi', fontsize=14)
    plt.ylabel('Skor (0-1)', fontsize=12)
    plt.xlabel('Modeller', fontsize=12)
    plt.ylim(0.80, 1.01) # Detayları görmek için alt sınırı yüksek tuttum
    plt.legend(loc='lower right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Çubukların üzerine değerleri yaz
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.4f}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 9), 
                    textcoords='offset points',
                    fontsize=9)

    plt.tight_layout()
    
    # KAYDETME İŞLEMİ
    plt.savefig(PLOT_FILENAME, dpi=300)
    print(f"\n[BILGI] Grafik basariyla '{PLOT_FILENAME}' olarak kaydedildi.")
    
    # Ekranda göster (Opsiyonel, terminal ortamında hata verirse try-except ile geçilir)
    try:
        plt.show()
    except:
        pass

# ==========================================
# 2. ANA BORU HATTI (PIPELINE)
# ==========================================

def main():
    # A. Veri Yükleme
    df = load_data()
    
    # B. Temizlik (Strict Mode)
    print("\n[ISLEM] Agresif Temizlik Yapiliyor (Stopwords + Leakage)...")
    df['final_text'] = df['cleaned_text'].apply(strict_clean)
    
    # C. Vektörleştirme
    print("[ISLEM] TF-IDF Vektorlestirme (Max Features: 5000)...")
    tfidf = TfidfVectorizer(max_features=5000, min_df=5)
    X = tfidf.fit_transform(df['final_text']).toarray()
    y = df['label'].values
    
    # D. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = {}

    # --- MODEL 1: LOGISTIC REGRESSION ---
    print("\n--- Model 1: Logistic Regression Egitiliyor ---")
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    
    # Tahminler
    y_pred_lr = lr_model.predict(X_test)
    y_prob_lr = lr_model.predict_proba(X_test)[:, 1]
    results['Logistic Regression'] = get_metrics(y_test, y_pred_lr, y_prob_lr)
    
    # *** ANALİZ ***
    analyze_model_features(lr_model, tfidf)

    # --- MODEL 2: RANDOM FOREST ---
    print("\n--- Model 2: Random Forest Egitiliyor ---")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    results['Random Forest'] = get_metrics(y_test, rf_model.predict(X_test), rf_model.predict_proba(X_test)[:, 1])

    # --- MODEL 3: NEURAL NETWORK ---
    print("\n--- Model 3: Neural Network (Deep Learning) Egitiliyor ---")
    # Keras Input shape uyarısını düzeltmek için Input layer kullanıyoruz
    nn_model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    nn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Eğitimi başlat (Sessiz mod: verbose=0)
    nn_model.fit(X_train, y_train, epochs=8, batch_size=32, validation_split=0.1, verbose=0)
    
    y_prob_nn = nn_model.predict(X_test).ravel()
    y_pred_nn = (y_prob_nn > 0.5).astype(int)
    results['Neural Network'] = get_metrics(y_test, y_pred_nn, y_prob_nn)

    # --- SONUÇLAR VE KAYIT ---
    plot_and_save_results(results)
    print("\n[BASARILI] Tum islemler tamamlandi.")

if __name__ == "__main__":
    main()