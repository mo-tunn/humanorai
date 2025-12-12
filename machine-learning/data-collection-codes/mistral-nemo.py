from huggingface_hub import InferenceClient
import pandas as pd
import csv
import time


HF_TOKEN = "" 


repo_id = "mistralai/Mistral-Nemo-Instruct-2407"

client = InferenceClient(token=HF_TOKEN)

source_csv = "human_wikipedia_3k.csv" # Kaynak olarak Wikipedia başlıklarını kullanıyoruz
output_csv = "mistral_ai_articles.csv"
TARGET_COUNT = 3000

try:
    df = pd.read_csv(source_csv)
    titles = df['title'].tolist()
    
    # Başlık sayısı kontrolü
    if len(titles) > TARGET_COUNT:
        titles = titles[:TARGET_COUNT]
    
    print(f"{len(titles)} başlık için Mistral üretimi başlıyor...")

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["title", "text", "label"]) # label 1 = AI

        for i, title in enumerate(titles):
            
            # Mistral Prompt Yapısı
            messages = [
                {"role": "system", "content": "You are a technical encyclopedia writer. Write a formal, informative snippet about the topic."},
                {"role": "user", "content": f"Write a short article about '{title}'. Keep it between 100-150 words."}
            ]
            
            try:
                response = client.chat_completion(
                    model=repo_id,
                    messages=messages,
                    max_tokens=250,
                    temperature=0.7
                )
                
                ai_text = response.choices[0].message.content
                ai_text = ai_text.strip().replace('\n', ' ')
                
                writer.writerow([title, ai_text, 1])
                
                if i % 20 == 0:
                    print(f"[Mistral] {i+1}/{len(titles)} tamamlandı.")
                
                # Rate limit için bekleme
                time.sleep(2) 
                    
            except Exception as e:
                print(f"Hata ({title}): {e}")
                time.sleep(5)

    print("Mistral verisi tamamlandı.")

except FileNotFoundError:
    print(f"Hata: '{source_csv}' dosyası bulunamadı. Lütfen önce Wikipedia verilerini çekin.")