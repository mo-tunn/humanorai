import pandas as pd
import numpy as np
import os
import pickle
import re
import string
import warnings

# --- AYARLAR ---
warnings.filterwarnings('ignore')
CURRENT_DIR = os.getcwd()
VECTOR_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "vectorization")
MODEL_DIR = CURRENT_DIR

# English Stopwords (Hardcoded for consistency with training)
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

HARD_MODE_STOPWORDS = set([
    "show", "propose", "use", "two", "using", "used", "well", "said", "important", 
    "also", "new", "percent", "study", "results", "however", 
    "demonstrate", "significant", "crucial", "robust", "often", "leading", "offers", 
    "typically", "various", "conclusion", "summary", "key", "addition", "furthermore",
    "increasingly", "landscape", "realm", "foster", "delve"
])
STOPWORDS.update(HARD_MODE_STOPWORDS)

def clean_text_for_prediction(text):
    if not isinstance(text, str): return ""
    
    # 0. Strip
    text = text.strip()
    
    # 1. Başlangıçtaki "Title:" ve benzeri yapılar (Training temizliğiyle uyum)
    lines = text.splitlines()
    if lines:
        first_line_clean = lines[0].strip().lower()
        if first_line_clean.startswith("title"):
             lines = lines[1:]
    text = "\n".join(lines)
    text = re.sub(r'^\s*title\s*[:\-]?\s*.*?(\n|$)', '', text, flags=re.IGNORECASE | re.MULTILINE)

    # 2. Markdown ve gürültü
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'__(.*?)__', r'\1', text)
    text = re.sub(r'^#+\s', '', text)
    text = re.sub(r'http\S+', '', text)

    # 3. Noktalama
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 4. Stopwords
    text = text.lower()
    text = text.replace('\n', ' ')
    words = text.split()
    filtered_words = [w for w in words if w not in STOPWORDS]
    
    return " ".join(filtered_words)

def load_resources():
    print("⏳ Kaynaklar yükleniyor...")
    
    # Vectorizer
    vec_path = os.path.join(VECTOR_DIR, "tfidf_vectorizer.pkl")
    if not os.path.exists(vec_path):
        print("❌ Vektörleştirici bulunamadı.")
        return None, None

    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)

    # Modeller
    models = {}
    model_files = [f for f in os.listdir(MODEL_DIR) if f.startswith("model_") and f.endswith(".pkl")]
    
    for mf in model_files:
        model_name = mf.replace("model_", "").replace(".pkl", "")
        try:
            with open(os.path.join(MODEL_DIR, mf), "rb") as f:
                models[model_name] = pickle.load(f)
        except Exception as e:
            print(f"⚠️ {mf} yüklenemedi: {e}")

    print(f"✅ {len(models)} model yüklendi: {', '.join(models.keys())}")
    return vectorizer, models

def get_interpretation(prob):
    # Model 1 (AI) olma olasılığı veriyor diyelim.
    # Genelde predict_proba outputu [p_0, p_1] döner. p_1 AI olasılığıdır.
    score = prob
    if score > 0.85: return "🤖 KESIN AI"
    if score > 0.60: return "🤔 MUHTEMEL AI"
    if score > 0.40: return "⚖️ BELIRSIZ"
    if score > 0.15: return "👤 MUHTEMEL INSAN"
    return "🧠 KESIN INSAN"

def main():
    vectorizer, models = load_resources()
    if not vectorizer or not models:
        return

    print("\n" + "="*60)
    print("🚀 HUMAN OR AI? - MANUAL TESTER")
    print("="*60)
    print("Çıkmak için 'q' yazın.\n")

    while True:
        user_input = input("📝 Metni yapıştırın: ")
        
        if user_input.lower() == 'q':
            pass
            break
        
        if len(user_input) < 10:
            print("⚠️ Çok kısa metin.")
            continue
            
        print("\n🔄 Analiz ediliyor...")
        
        # Temizlik ve Vektör
        cleaned = clean_text_for_prediction(user_input)
        vec_input = vectorizer.transform([cleaned]) # Sparse matrix döner
        
        # Dense convert (Neural Network vb. için gerekebilir ama sklearn MLP sparse kabul eder genelde)
        # Ama kodu garantiye almak için toarray yapmıyorum çünkü TF-IDF sparse'dır. 
        # Sadece Naive Bayes dense isteyebilir mi? Sklearn NB sparse destekler.
        
        print("\n" + "-"*75)
        print(f"{'MODEL':<25} | {'OLASILIK (AI)':<15} | {'KARAR'}")
        print("-" * 75)
        
        total_prob = 0
        valid_models = 0

        for name, model in models.items():
            try:
                # predict_proba var mı?
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(vec_input)[0][1] # Class 1 (AI) olasılığı
                else:
                    # LinearSVC gibi modellerde varsayılan olarak proba yok, decision_function var
                    # Veya sadece predict var.
                    pred = model.predict(vec_input)[0]
                    prob = float(pred) # 0.0 veya 1.0 (Kesin karar)
                
                total_prob += prob
                valid_models += 1
                
                decision = get_interpretation(prob)
                print(f"{name:<25} | %{prob*100:6.2f}          | {decision}")
            except Exception as e:
                print(f"{name:<25} | HATA              | {e}")

        if valid_models > 0:
            avg_prob = total_prob / valid_models
            print("-" * 75)
            print(f"{'GENEL ORTALAMA':<25} | %{avg_prob*100:6.2f}          | {get_interpretation(avg_prob)}")
        print("\n")

if __name__ == "__main__":
    main()
