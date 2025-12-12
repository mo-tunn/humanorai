import wikipediaapi
import csv
import time

csv_path = "wikipedia_ai_articles.csv"
TARGET_ROW_COUNT = 3000  # Hedeflenen satır sayısı

# Başlangıç konuları (Seed topics)
topics_queue = [
    "Large language model",
    "Artificial intelligence",
    "Machine learning",
    "Deep learning",
    "Transformer (machine learning)",
    "Natural language processing",
    "Neural network",
    "Generative artificial intelligence",
    "AI safety",
    "Artificial general intelligence",
    "Reinforcement learning",
    "Computer vision",
    "Supervised learning",
    "Unsupervised learning",
    "Data science",
    "Turing test",
    "Expert system",
    "Perceptron"
]

wiki = wikipediaapi.Wikipedia(
    language="en",
    user_agent="HumanOrAIProject/1.0 (contact@example.com)" 
)

visited_topics = set() # Aynı sayfayı tekrar çekmemek için
collected_rows = 0

def get_text_chunks(text, min_words=100, max_words=150):
    """
    Uzun bir metni alır, kelimelere böler ve belirtilen aralıkta parçalar (chunk) oluşturur.
    """
    words = text.replace('\n', ' ').split() # Satır sonlarını boşlukla değiştir ve kelimelere ayır
    chunks = []
    
    # 100 kelimelik adımlarla ilerle ama 150 kelime al (Overlapping/Kaydırma yapılabilir ama burada düz mantık gidiyoruz)
    # Veri çeşitliliği için tam bloklar halinde alalım.
    current_idx = 0
    while current_idx < len(words):
        # Rastgelelik veya doğal bitiş için 100 ile 150 arası bir kesit alalım
        # Burada basitçe 120 kelimelik bloklar alıyoruz, bu aralığa uyar.
        end_idx = current_idx + 120 
        
        chunk_words = words[current_idx:end_idx]
        
        # Eğer parça 100 kelimeden azsa (makale sonu vb.) almayalım
        if len(chunk_words) >= min_words:
            # 150'den fazlaysa kırpalım (gerçi yukarıda 120 ayarladık ama garanti olsun)
            if len(chunk_words) > max_words:
                chunk_words = chunk_words[:max_words]
            
            chunks.append(" ".join(chunk_words))
        
        current_idx = end_idx # Bir sonraki bloğa geç
        
    return chunks

print("Veri toplama işlemi başladı...")

with open(csv_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["title", "text", "label"])

    while collected_rows < TARGET_ROW_COUNT and len(topics_queue) > 0:
        current_topic = topics_queue.pop(0) # Listeden bir konu al
        
        if current_topic in visited_topics:
            continue
            
        visited_topics.add(current_topic)
        
        try:
            page = wiki.page(current_topic)
            
            if page.exists():
                # Makaleyi parçalara ayır
                chunks = get_text_chunks(page.text)
                
                rows_added_from_this_page = 0
                for chunk in chunks:
                    if collected_rows >= TARGET_ROW_COUNT:
                        break
                    
                    writer.writerow([page.title, chunk, 0])
                    collected_rows += 1
                    rows_added_from_this_page += 1
                
                print(f"[✓] {page.title} -> {rows_added_from_this_page} parça eklendi. (Toplam: {collected_rows}/{TARGET_ROW_COUNT})")

                # Eğer hala hedef sayıya ulaşamadıysak ve kuyruk azaldıysa
                # Mevcut sayfanın linklerini kuyruğa ekle (Konu genişletme)
                if collected_rows < TARGET_ROW_COUNT:
                    links = page.links
                    for title in links.keys():
                        if title not in visited_topics and title not in topics_queue:
                            # Sadece konuyla alakalı olabilecekleri almak zor (AI ismini filtreleyebiliriz ama
                            # şimdilik Wikipedia'nın context yapısına güveniyoruz)
                            topics_queue.append(title)
            
            else:
                print(f"[!] Sayfa bulunamadı: {current_topic}")
                
        except Exception as e:
            print(f"[Error] {current_topic} işlenirken hata: {e}")
            # API hatası durumunda bekleme
            time.sleep(1)

print(f"\nİşlem tamamlandı! Toplam {collected_rows} satır veri '{csv_path}' dosyasına kaydedildi.")