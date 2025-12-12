from huggingface_hub import InferenceClient
import pandas as pd
import csv
import time


HF_TOKEN = "" # 


repo_id = "meta-llama/Meta-Llama-3-14B-Instruct"

client = InferenceClient(token=HF_TOKEN)

source_csv = "human_cc_news_3k.csv" 
output_csv = "llama3_ai_articles.csv"
TARGET_COUNT = 3000

# Dosya okuma işlemleri
try:
    df = pd.read_csv(source_csv)
    titles = df['title'].tolist()
    
    # Eğer haber başlıkları 3000'den azsa, eldeki kadarını alalım veya 
    # wikipedia csv'sini de okuyup birleştirebilirsin.
    if len(titles) > TARGET_COUNT:
        titles = titles[:TARGET_COUNT]
    
    print(f"{len(titles)} başlık için Llama 3 (Chat Modu) üretimi başlıyor...")

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["title", "text", "label"])

        for i, title in enumerate(titles):
            
            # Chat formatına uygun mesaj yapısı
            messages = [
                {"role": "system", "content": "You are a journalist. Write a short news snippet based on the headline provided."},
                {"role": "user", "content": f"Write a news article snippet about '{title}'. It must be between 100 and 150 words."}
            ]
            
            try:
                # text_generation YERİNE chat_completion kullanıyoruz
                response = client.chat_completion(
                    model=repo_id,
                    messages=messages,
                    max_tokens=250,
                    temperature=0.7
                )
                
                # Yanıtı al (OpenAI yapısına benzer döner)
                ai_text = response.choices[0].message.content
                
                # Temizlik
                ai_text = ai_text.strip().replace('\n', ' ')
                
                writer.writerow([title, ai_text, 1])
                
                if i % 20 == 0:
                    print(f"[Llama] {i+1}/{len(titles)} tamamlandı.")
                
                # API limitine takılmamak için bekleme
                time.sleep(2) 
                    
            except Exception as e:
                print(f"Hata ({title}): {e}")
                time.sleep(5)

    print("Llama verisi tamamlandı.")

except FileNotFoundError:
    print(f"Hata: '{source_csv}' dosyası bulunamadı. Lütfen önce haber verilerini çekin.")