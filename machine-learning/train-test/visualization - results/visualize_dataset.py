import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
from collections import Counter

# --- AYARLAR ---
CURRENT_DIR = os.getcwd()
INPUT_FILE = os.path.join(CURRENT_DIR, "humanorai-2 - Kopya/train-test/visualization/final_dataset_cleaned.csv")

def main():
    print(f"📊 Veri Seti Analiz Ediliyor: {INPUT_FILE}")
    
    if not os.path.exists(INPUT_FILE):
        print("❌ Dosya bulunamadı! Lütfen önce temizlik işlemini yapın.")
        return

    try:
        df = pd.read_csv(INPUT_FILE)
    except Exception as e:
        print(f"❌ Dosya okunamadı: {e}")
        return

    print(f"✅ Toplam Satır: {len(df)}")
    print("--- Önizleme ---")
    print(df.head())

    # Label açıklamaları (0: Human, 1: AI varsayımıyla, kullanıcının veri setine göre kontrol edilmeli)
    # Genelde clean scriptinde Human=0, AI=1 düzeni korunuyor
    
    # 1. SINIF DAĞILIMI
    plt.figure(figsize=(8, 6))
    sns.countplot(x='label', data=df, palette='viridis')
    plt.title("Sınıf Dağılımı (0: Human, 1: AI)")
    plt.xlabel("Label")
    plt.ylabel("Sayı")
    plt.savefig(os.path.join(CURRENT_DIR, "viz_class_distribution.png"))
    print("💾 Kaydedildi: viz_class_distribution.png")
    plt.close()

    # 2. KELİME SAYISI ANALİZİ
    print("📈 Kelime sayıları hesaplanıyor...")
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
    
    print(df.groupby('label')['word_count'].describe())

    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='word_count', hue='label', kde=True, bins=50, palette={0: 'blue', 1: 'red'}, alpha=0.5)
    plt.title("Kelime Sayısı Dağılımı (Human vs AI)")
    plt.xlabel("Kelime Sayısı")
    plt.ylabel("Frekans")
    plt.xlim(0, df['word_count'].quantile(0.99)) # Outlierları görselden kes
    plt.savefig(os.path.join(CURRENT_DIR, "viz_word_count_hist.png"))
    print("💾 Kaydedildi: viz_word_count_hist.png")
    plt.close()

    # 3. KUTU GRAFİĞİ (Boxplot) - Outlierları görmek için
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='label', y='word_count', data=df, palette='Set2')
    plt.title("Kelime Sayısı Kutu Grafiği")
    plt.ylim(0, df['word_count'].quantile(0.99))
    plt.savefig(os.path.join(CURRENT_DIR, "viz_word_count_boxplot.png"))
    print("💾 Kaydedildi: viz_word_count_boxplot.png")
    plt.close()

    # 4. EN SIK GÖRÜLEN KELİMELER & WORDCLOUD
    print("☁️ Wordcloud hazırlanıyor...")
    
    STOPWORDS_CHECK = set(['the', 'and', 'is', 'in', 'to', 'of', 'a', 'it', 'that', 'for']) 

    def generate_wordcloud(text_series, title, filename):
        text_combined = " ".join(str(t) for t in text_series)
        wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100, stopwords=STOPWORDS_CHECK).generate(text_combined)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title)
        plt.savefig(os.path.join(CURRENT_DIR, filename))
        print(f"💾 Kaydedildi: {filename}")
        plt.close()

    # Human Wordcloud
    generate_wordcloud(df[df['label'] == 0]['text'], "Wordcloud: Human Generated Text", "viz_wordcloud_human.png")
    
    # AI Wordcloud
    generate_wordcloud(df[df['label'] == 1]['text'], "Wordcloud: AI Generated Text", "viz_wordcloud_ai.png")

    # 5. EN ŞÜPHELİ (ÇOK KISA) VERİLER
    print("\n--- Çok Kısa Metin Kontrolü (<10 Kelime) ---")
    short_texts = df[df['word_count'] < 10]
    if not short_texts.empty:
        print(f"⚠️ 10 kelimeden kısa {len(short_texts)} satır var:")
        print(short_texts[['text', 'label']].head(10))
    else:
        print("✅ Çok kısa metin bulunmadı.")

    print("\n✅ Analiz tamamlandı. Grafikleri inceleyebilirsiniz.")
    plt.show() # Tüm pencereleri açık tut

if __name__ == "__main__":
    main()
