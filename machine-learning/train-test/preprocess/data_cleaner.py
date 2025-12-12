import pandas as pd
import re
import glob
import os
import string
import matplotlib.pyplot as plt
import seaborn as sns

# --- AYARLAR ---
CURRENT_DIR = os.getcwd()
OUTPUT_FILE = os.path.join(CURRENT_DIR, "final_dataset_cleaned.csv")

# English Stopwords List (Hardcoded to avoid checking NLTK download)
STOPWORDS = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", 
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", 
    "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", 
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", 
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", 
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", 
    "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", 
    "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", 
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", 
    "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", 
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", 
    "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
])

# Overfitting'i engellemek için "Hard Mode" kelimeleri (Modelin çok kolay öğrendiği kelimeler)
HARD_MODE_STOPWORDS = set([
    # Human ipuçları
    "show", "propose", "use", "two", "using", "used", "well", "said", "important", 
    "also", "new", "percent", "study", "results", "however", 
    # AI ipuçları
    "demonstrate", "significant", "crucial", "robust", "often", "leading", "offers", 
    "typically", "various", "conclusion", "summary", "key", "addition", "furthermore",
    "increasingly", "landscape", "realm", "foster", "delve"
])

# Hepsini birleştir
STOPWORDS.update(HARD_MODE_STOPWORDS)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # 0. Ön Temizlik (BOM, whitespace)
    text = text.strip()

    # 1. Başlangıçtaki "Title:" satırını kontrol et ve sil
    lines = text.splitlines()
    if lines:
        first_line_clean = lines[0].strip().lower()
        if first_line_clean.startswith("title"):
            lines = lines[1:]
            text = "\n".join(lines)

    # Regex ile kalan Title kalıntıları (eğer satır başı değilse veya yapışık ise)
    # Satır sonu (\n) OLABİLİR VEYA OLMAYABİLİR ($)
    text = re.sub(r'^\s*title\s*[:\-]?\s*.*?(\n|$)', '', text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Diğer AI kalıpları
    ai_starters = [
        "Here is a formal encyclopedia entry", 
        "Here is a news article",
        "Certainly, here is",
        "As an AI language model",
        "Sure! Here is",
        "In this article",
    ]
    for phrase in ai_starters:
        if phrase.lower() in text.lower():
            text = re.sub(f"(?i){phrase}.*?[\n\.:]", "", text, count=1)

    # 2. Markdown ve gürültü temizliği
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'__(.*?)__', r'\1', text)
    text = re.sub(r'^#+\s', '', text)
    text = re.sub(r'^\s*-\s', '', text)
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\[edit\]', '', text)
    text = re.sub(r'http\S+', '', text)

    # 3. Noktalama İşaretleri
    # Unicode punctuation dahil etmek için
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 4. Genel Düzenleme ve Stopwords
    text = text.lower()
    text = text.replace('\n', ' ')
    words = text.split()
    
    # Stopwords temizliği
    filtered_words = [w for w in words if w not in STOPWORDS]
    
    return " ".join(filtered_words)

def main():
    print("🚀 Veri Temizleme ve Birleştirme Başlatılıyor...")
    
    # Hedef Dosyalar
    target_files = [
        "mistral_3k.csv",
        "llama3_3k.csv",
        "human_cnn_news_3k.csv",
        "human_wikipedia_3k.csv",
        "arxiv_gemini-2.5_combinated_20k.csv"
    ]
    
    df_list = []
    
    for filename in target_files:
        path = os.path.join(CURRENT_DIR, filename)
        if not os.path.exists(path):
            print(f"⚠️ Dosya Bulunamadı: {filename}")
            continue
            
        try:
            # Okuma denemesi
            try:
                # on_bad_lines='skip' ile bozuk satırları atlarız
                df = pd.read_csv(path, encoding='utf-8', on_bad_lines='skip')
            except UnicodeDecodeError:
                df = pd.read_csv(path, encoding='latin-1', on_bad_lines='skip')
            
            print(f"📖 Okundu: {filename} - Sütunlar: {list(df.columns)}")
            
            # Sütun isimlerini düzeltme (label;;; gibi hatalar için)
            df.columns = [c.strip().replace(';;;', '') for c in df.columns]
            
            # Gerekli sütunları bulma
            text_col = None
            label_col = None
            
            for col in df.columns:
                if 'text' in col.lower():
                    text_col = col
                if 'label' in col.lower():
                    label_col = col
            
            if text_col and label_col:
                # Sadece text ve label al
                clean_df = df[[text_col, label_col]].copy()
                clean_df.columns = ['text', 'label'] # Standartlaştır
                df_list.append(clean_df)
                print(f"✅ Eklendi: {filename} ({len(clean_df)} satır)")
            else:
                print(f"❌ Sütun eksik (text/label bulunamadı): {filename}")

        except Exception as e:
            print(f"❌ Hata ({filename}): {e}")

    if not df_list:
        print("❌ Hiçbir dosya işlenemedi.")
        return

    # Birleştirme
    full_df = pd.concat(df_list, ignore_index=True)
    print(f"\n📊 Toplam Satır (Birleşmiş): {len(full_df)}")

    # Temizlik
    print("🧼 Temizleme işlemi uygulanıyor...")
    full_df = full_df.dropna(subset=['text'])
    full_df['text'] = full_df['text'].apply(clean_text)
    
    # Boş kalanları at
    full_df = full_df[full_df['text'].str.len() > 0]
    
    # Kelime sayısı filtresi
    full_df = full_df[full_df['text'].apply(lambda x: len(str(x).split()) > 5)] # Çok kısaları at
    
    print(f"✅ Temizlik Tamamlandı. Kalan Satır: {len(full_df)}")
    
    # Görselleştirme
    print("\n--- İstatistikler ---")
    print(full_df['label'].value_counts())
    
    # SON KONTROL: Eğer hala "title" ile başlayan varsa uçur (Brute Force)
    # "title" kelimesiyle başlayan geçerli cümleleri kaybetme riski var ama user özellikle "Title:" artifactlerinden bahsetti.
    # Genelde clean_text sonrası "title " (boşluklu) kalıyorsa artifact olma ihtimali yüksek.
    bad_rows = full_df[full_df['text'].str.startswith('title ')]
    if not bad_rows.empty:
        print(f"⚠️ Son kontrolde {len(bad_rows)} 'title' ile başlayan satır siliniyor.")
        full_df = full_df[~full_df['text'].str.startswith('title ')]

    # DUPLIKE KONTROLÜ VE TEMİZLİĞİ (User isteği üzerine eklendi)
    duplicate_count = full_df.duplicated(subset=['text']).sum()
    if duplicate_count > 0:
        print(f"✂️ {duplicate_count} adet kopya (duplicate) satır tespit edildi ve siliniyor...")
        full_df.drop_duplicates(subset=['text'], inplace=True)

    # Kaydetme
    full_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n💾 Dosya Kaydedildi: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()