import ollama
import pandas as pd
from tqdm import tqdm
import time
import os
import json
import re

# --- YAPILANDIRMA ---
MODEL_NAME = "qwen2.5:1.5b"
OUTPUT_FILE = "ai_generated_dataset_json.csv" # Dosya adı
TOTAL_TARGET = 3080
CATEGORIES_COUNT = 11
TARGET_PER_CATEGORY = int(TOTAL_TARGET / CATEGORIES_COUNT) 
BATCH_SIZE = 5 

categories_map = {
    "Bilgisayar ve Toplum": "Ethical implications of AI, algorithmic bias, fairness",
    "Sosyal ve Bilgi Ağları": "Social media behavior analysis, mental health in social networks",
    "Fizik ve Toplum": "Sustainability, carbon footprint, energy efficiency in tech",
    "Tarihsel Veri Analizi": "Historical demographics, industrial revolution data analysis",
    "Bilgisayar Bilimlerinde Mantık": "Mathematical logic in CS, game theory, reasoning",
    "Kantitatif Biyoloji": "CRISPR, gene editing, genetic engineering models",
    "Fiziksel İnsan Hareketliliği": "Human mobility patterns, urban transportation analysis",
    "Metin Analizi": "NLP for literary texts, narrative analysis, storytelling",
    "Kantitatif Finans": "Cryptocurrency dynamics, bitcoin, blockchain markets",
    "Bilgisayar Grafikleri": "Generative art, NFT aesthetics, digital art algorithms",
    "Kriptografi ve Güvenlik": "IoT security, intrusion detection, phishing defense"
}

def clean_json_response(content):
    """
    Modelin yanıtını JSON objesine çevirir.
    """
    try:
        # Markdown code block (```json ... ```) varsa temizle
        content = re.sub(r'```json', '', content)
        content = re.sub(r'```', '', content)
        content = content.strip()
        
        # JSON'u parse et
        data = json.loads(content)
        
        # Anahtar kontrolü (abstracts listesi dönmeli)
        if "abstracts" in data and isinstance(data["abstracts"], list):
            return data["abstracts"]
        else:
            return []
            
    except json.JSONDecodeError:
        # Nadiren JSON bozuk gelirse boş dön, bir sonraki turda tekrar dener
        return []

def generate_batch_abstracts(topic, count=5):
    # Prompt'u JSON isteyecek şekilde değiştirdik
    prompt = f"""
    You are a data generator. Output a valid JSON object containing exactly {count} academic abstracts about "{topic}".
    
    JSON FORMAT:
    {{
        "abstracts": [
            "Abstract text 1...",
            "Abstract text 2...",
            "Abstract text 3..."
        ]
    }}

    RULES:
    1. Do NOT include titles.
    2. Do NOT include intro/outro text.
    3. Output ONLY valid JSON.
    4. Each abstract must be 100-150 words.
    """
    
    try:
        response = ollama.chat(model=MODEL_NAME, messages=[
            {'role': 'user', 'content': prompt}
        ], options={'temperature': 0.7, 'format': 'json'}) # format='json' ile modeli zorluyoruz

        content = response['message']['content']
        return clean_json_response(content)
        
    except Exception as e:
        print(f"Hata: {e}")
        return []

def main():
    if not os.path.exists(OUTPUT_FILE):
        pd.DataFrame(columns=['category', 'abstract', 'label']).to_csv(OUTPUT_FILE, index=False)
    
    print(f"🚀 JSON Modu (Kusursuz Ayrıştırma) Başlatılıyor: {MODEL_NAME}")
    
    for category, topic in categories_map.items():
        try:
            df_current = pd.read_csv(OUTPUT_FILE)
            existing_count = len(df_current[df_current['category'] == category])
        except:
            existing_count = 0
        
        if existing_count >= TARGET_PER_CATEGORY:
            print(f"✅ {category} tamamlandı. Geçiliyor.")
            continue

        to_generate = TARGET_PER_CATEGORY - existing_count
        pbar = tqdm(total=to_generate, desc=category[:15])
        
        current_generated = 0
        while current_generated < to_generate:
            needed = to_generate - current_generated
            batch_size = 5 if needed >= 5 else needed
            
            # JSON formatında veri iste
            abstracts = generate_batch_abstracts(topic, batch_size)
            
            valid_rows = []
            for abs_text in abstracts:
                # Basit bir temizlik daha yapalım (Satır başı boşlukları vs.)
                clean_text = abs_text.strip()
                
                # İçinde "Title:" geçenleri temizle (Nadiren sızabilir)
                clean_text = re.sub(r'^\*\*.*?\*\*\s*', '', clean_text) 
                
                if len(clean_text) > 50 and current_generated < to_generate:
                    valid_rows.append({
                        'category': category,
                        'abstract': clean_text,
                        'label': 1
                    })
                    current_generated += 1
                    pbar.update(1)
            
            if valid_rows:
                pd.DataFrame(valid_rows).to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
            
        pbar.close()

    print("\n🎉 Veri Seti (JSON Formatıyla) Hazır!")

if __name__ == "__main__":
    main()