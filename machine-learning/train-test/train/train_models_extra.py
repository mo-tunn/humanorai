import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- AYARLAR ---
CURRENT_DIR = os.getcwd()
VECTOR_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "vectorization")
OUTPUT_DIR = CURRENT_DIR

# Dosya Yolları
X_PATH = os.path.join(VECTOR_DIR, "X_tfidf_matrix.pkl")
Y_PATH = os.path.join(VECTOR_DIR, "y_labels.pkl")

def save_plot_confusion_matrix(y_test, y_pred, title, filename):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=['Human', 'AI'], yticklabels=['Human', 'AI'])
    plt.title(title)
    plt.ylabel('Gerçek Etiket')
    plt.xlabel('Tahmin Edilen')
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Grafik kaydedildi: {filename}")

def main():
    print("🚀 Ek Modellerin Eğitimi Başlatılıyor...")
    
    # 1. VERİ YÜKLEME
    if not os.path.exists(X_PATH) or not os.path.exists(Y_PATH):
        print("❌ HATA: Veri dosyaları bulunamadı!")
        return

    with open(X_PATH, "rb") as f:
        X = pickle.load(f)
    with open(Y_PATH, "rb") as f:
        y = pickle.load(f)

    # Tip dönüşümü ve Temizlik (Corrupted data fix - Same as before)
    try:
        y_str = y.astype(str)
        y_clean = []
        for val in y_str:
            import re
            match = re.search(r'(\d+)', val)
            if match:
                y_clean.append(int(match.group(1)))
            else:
                y_clean.append(0)
        y = np.array(y_clean)
    except Exception as e:
        y = y.astype(int)

    # 2. TRAIN-TEST SPLIT
    print("✂️ Eğitim ve Test Seti Ayrılıyor (%80 Train, %20 Test)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 5 YENİ MODEL
    models = {
        "Linear_SVM": LinearSVC(dual=False, random_state=42),
        "Naive_Bayes": MultinomialNB(),
        "Gradient_Boosting": GradientBoostingClassifier(n_estimators=50, random_state=42), # Hız için 50 n_estimators
        "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42),
        "Decision_Tree": DecisionTreeClassifier(random_state=42)
    }

    results = {}

    # 3. TRAINING LOOP
    for name, model in models.items():
        print(f"\n⚙️  Eğitiliyor (Ekstra): {name}...")
        try:
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results[name] = acc
            
            print(f"   ✅ {name} Accuracy: {acc:.4f}")
            
            # Save
            with open(os.path.join(OUTPUT_DIR, f"model_{name}.pkl"), "wb") as f:
                pickle.dump(model, f)
            
            # CM
            save_plot_confusion_matrix(y_test, y_pred, f"{name} Confusion Matrix", f"cm_{name}.png")
            
            # Report
            report = classification_report(y_test, y_pred, target_names=['Human', 'AI'])
            with open(os.path.join(OUTPUT_DIR, f"report_{name}.txt"), "w") as f:
                f.write(f"Model: {name}\n\n{report}")
        except Exception as e:
            print(f"❌ Eğitim Hatası ({name}): {e}")

    # 4. COMPA CHART
    print("\n📈 Karşılaştırma Grafiği (Ek Modeller)...")
    plt.figure(figsize=(12, 6))
    bars = plt.bar(results.keys(), results.values(), color=['purple', 'cyan', 'lime', 'magenta', 'brown'])
    plt.ylim(0, 1.1)
    plt.title("Ekstra Modellerin Doğruluk Karşılaştırması")
    plt.ylabel("Accuracy Score")
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.4f}", ha='center', va='bottom', fontsize=12, fontweight='bold')
        
    plt.savefig(os.path.join(OUTPUT_DIR, "extra_models_comparison.png"))
    print("✅ İşlem tamamlandı.")

if __name__ == "__main__":
    main()
