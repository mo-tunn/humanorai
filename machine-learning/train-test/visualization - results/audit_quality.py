import pandas as pd
import re
import os

# --- AYARLAR ---
CURRENT_DIR = os.getcwd()
INPUT_FILE = os.path.join(CURRENT_DIR, "final_dataset_cleaned.csv")

def main():
    print(f"🕵️ Detaylı Kalite Kontrolü: {INPUT_FILE}")
    
    if not os.path.exists(INPUT_FILE):
        print("❌ Dosya yok.")
        return

    df = pd.read_csv(INPUT_FILE)
    
    # 1. HTML Tag Kontrolü
    # <br>, <div>, &nbsp; gibi ifadeler kalmış mı?
    html_pattern = r'<[a-z][\s\S]*?>|&[a-z]+;'
    html_rows = df[df['text'].str.contains(html_pattern, regex=True, na=False)]
    print(f"\n1️⃣ HTML/XML Artığı İçeren Satırlar: {len(html_rows)}")
    if not html_rows.empty:
        print("   Örnek:", html_rows['text'].iloc[0][:100])

    # 2. Çok Uzun Kelime (Garbage String) Kontrolü
    # Normal bir İngilizce kelime genelde 20-30 karakteri geçmez. 
    # 40+ karakterli "aaaaaaaa..." veya url kalıntıları var mı?
    def has_long_word(text, threshold=45):
        words = str(text).split()
        for w in words:
            if len(w) > threshold:
                return True
        return False

    long_word_rows = df[df['text'].apply(lambda x: has_long_word(x))]
    print(f"\n2️⃣ Anormal Uzun Kelime İçeren Satırlar (>45 karakter): {len(long_word_rows)}")
    if not long_word_rows.empty:
        print("   Örnek:", long_word_rows['text'].iloc[0][:100] + "...")

    # 3. Karakter Temizliği (Non-Alphanumeric Oranı)
    # Temizlik sonrası sadece harf, sayı ve temel boşluk kalmalıydı.
    # Eğer temizlemeyi çok sıkı yaptıysak (noktalama sildik), sadece boşluk ve harf olmalı.
    # Özel karakter yoğunluğuna bakalım.
    def special_char_ratio(text):
        if not text: return 0
        special = len(re.findall(r'[^a-zA-Z0-9\s]', str(text)))
        return special / len(str(text))

    # %10'dan fazla özel karakter barındıran satırlar (kirli olabilir)
    dirty_rows = df[df['text'].apply(lambda x: special_char_ratio(x) > 0.1)]
    print(f"\n3️⃣ Yüksek Özel Karakter Oranı (>%10) Olan Satırlar: {len(dirty_rows)}")
    if not dirty_rows.empty:
        print("   Örnek:", dirty_rows['text'].iloc[0][:100])

    # 4. Tekrarlayan Satırlar (Exact Duplicates)
    # Temizleyici scriptte dedupe yapmıştık ama bir daha bakalım
    dupes = df.duplicated(subset=['text']).sum()
    print(f"\n4️⃣ Tekrarlayan (Duplicate) Satırlar: {dupes}")

    # Grafik oluştur
    try:
        import matplotlib.pyplot as plt
        
        metrics = ['HTML Tags', 'Long Words', 'Special Chars', 'Duplicates']
        counts = [len(html_rows), len(long_word_rows), len(dirty_rows), dupes]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics, counts, color=['red', 'orange', 'blue', 'purple'])
        
        # Değerleri yazdır
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + (yval*0.05), int(yval), ha='center', va='bottom', fontsize=12, fontweight='bold')
            
        plt.title("Data Quality Audit Results")
        plt.ylabel("Count of Issues")
        plt.savefig(os.path.join(CURRENT_DIR, "audit_summary.png"))
        print("\n💾 Kalite özeti kaydedildi: audit_summary.png")
        plt.close()
    except Exception as e:
        print(f"\n⚠️ Grafik çizilemedi: {e}")

    print("\n--- SONUÇ YORUMU ---")
    if len(html_rows) < 50 and len(long_word_rows) < 50 and len(dirty_rows) < 100:
        print("✅ Veri oldukça temiz görünüyor. Ufak tefek kaçaklar model başarısını etkilemez.")
    else:
        print("⚠️ Bazı temizlik problemleri görünüyor, yukarıdaki sayıları user ile paylaş.")

if __name__ == "__main__":
    main()
