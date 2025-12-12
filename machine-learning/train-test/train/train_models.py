import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- AYARLAR ---
CURRENT_DIR = os.getcwd()
# vectorization klasörü bir üst dizinde 'vectorization' adıyla
VECTOR_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "vectorization")
OUTPUT_DIR = CURRENT_DIR

# Dosya Yolları
X_PATH = os.path.join(VECTOR_DIR, "X_tfidf_matrix.pkl")
Y_PATH = os.path.join(VECTOR_DIR, "y_labels.pkl")

def save_plot_confusion_matrix(y_test, y_pred, title, filename):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Human', 'AI'], yticklabels=['Human', 'AI'])
    plt.title(title)
    plt.ylabel('Gerçek Etiket')
    plt.xlabel('Tahmin Edilen')
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Grafik kaydedildi: {filename}")

def main():
    print("🚀 Model Eğitimi Başlatılıyor...")
    
    # 1. VERİ YÜKLEME
    print(f"📂 Veriler Yükleniyor: {VECTOR_DIR}")
    if not os.path.exists(X_PATH) or not os.path.exists(Y_PATH):
        print(f"❌ HATA: Veri dosyaları bulunamadı! Lütfen önce vectorization işlemini yapın.\nAranan: {X_PATH}")
        return

    with open(X_PATH, "rb") as f:
        X = pickle.load(f)
    with open(Y_PATH, "rb") as f:
        y = pickle.load(f)

    print(f"✅ Veri Yüklendi. X: {X.shape}, y: {y.shape}")

    # Tip dönüşümü ve Temizlik (Corrupted data fix)
    # Örnek bozuk veri: '1";' -> '1'
    # Sadece ilk integer karakteri al
    try:
        y_str = y.astype(str)
        y_clean = []
        for val in y_str:
            # Sadece rakamları çek
            import re
            match = re.search(r'(\d+)', val)
            if match:
                y_clean.append(int(match.group(1)))
            else:
                y_clean.append(0) # Fallback
        y = np.array(y_clean)
    except Exception as e:
        print(f"⚠️ Label conversion warning: {e}")
        y = y.astype(int)

    # 2. TRAIN-TEST SPLIT
    print("✂️ Eğitim ve Test Seti Ayrılıyor (%80 Train, %20 Test)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"   Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # Modeller
    models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000),
        "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Neural_Network": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    }

    results = {}

    # 3. MODEL EĞİTİMİ VE DEĞERLENDİRME
    for name, model in models.items():
        print(f"\n⚙️  Eğitiliyor: {name}...")
        model.fit(X_train, y_train)
        
        # Tahmin
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        
        print(f"   ✅ {name} Accuracy: {acc:.4f}")
        
        # Modeli Kaydet
        model_filename = f"model_{name}.pkl"
        with open(os.path.join(OUTPUT_DIR, model_filename), "wb") as f:
            pickle.dump(model, f)
        print(f"   💾 Model Kaydedildi: {model_filename}")

        # Confusion Matrix Çiz ve Kaydet
        save_plot_confusion_matrix(y_test, y_pred, f"{name} Confusion Matrix", f"cm_{name}.png")
        
        # Detaylı Raporu TXT olarak kaydet
        report = classification_report(y_test, y_pred, target_names=['Human', 'AI'])
        with open(os.path.join(OUTPUT_DIR, f"report_{name}.txt"), "w") as f:
            f.write(f"Model: {name}\nAssuming Human=0, AI=1\n\n")
            f.write(report)

    # 4. KARŞILAŞTIRMA GRAFİĞİ
    print("\n📈 Karşılaştırma Grafiği Çiziliyor...")
    plt.figure(figsize=(10, 6))
    bars = plt.bar(results.keys(), results.values(), color=['blue', 'green', 'orange'])
    plt.ylim(0, 1.1)
    plt.title("Model Doğruluk (Accuracy) Karşılaştırması")
    plt.ylabel("Accuracy Score")
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.4f}", ha='center', va='bottom', fontsize=12, fontweight='bold')
        
    plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison_accuracy.png"))
    print("📊 Grafik kaydedildi: model_comparison_accuracy.png")
    plt.close()

    print("\n✅ Tüm işlemler başarıyla tamamlandı.")

if __name__ == "__main__":
    main()
