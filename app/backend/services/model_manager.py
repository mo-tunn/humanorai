import os
import joblib
import numpy as np
from .text_processor import strict_clean

MODELS = {}
VECTORIZER = None

def load_models_from_disk(base_dir):
    """Loads models and vectorizer from the specified directory."""
    global MODELS, VECTORIZER
    
    # Adjust path safely
    # Assuming base_dir is app/backend directory passed from main
    model_dir = os.path.join(base_dir, '..', '..', 'machine-learning', 'train-test', 'saved_models')
    model_dir = os.path.abspath(model_dir)

    print(f"[ModelManager] Loading models from: {model_dir}")

    try:
        VECTORIZER = joblib.load(os.path.join(model_dir, 'tfidf_vectorizer.pkl'))
        
        model_files = {
            'AdaBoost': 'model_AdaBoost.pkl',
            'Decision Tree': 'model_Decision_Tree.pkl',
            'Gradient Boosting': 'model_Gradient_Boosting.pkl',
            'Linear SVM': 'model_Linear_SVM.pkl',
            'Logistic Regression': 'model_Logistic_Regression.pkl',
            'Naive Bayes': 'model_Naive_Bayes.pkl',
            'Neural Network': 'model_Neural_Network.pkl',
            'Random Forest': 'model_Random_Forest.pkl'
        }

        for name, filename in model_files.items():
            path = os.path.join(model_dir, filename)
            if os.path.exists(path):
                MODELS[name] = joblib.load(path)
                print(f"[ModelManager] {name} loaded.")
            else:
                print(f"[WARNING] {name} not found: {filename}")
                
        print("[ModelManager] All models loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to load models: {str(e)}")

def predict_with_all_models(text):
    if not VECTORIZER:
        raise RuntimeError("Models not loaded. Call load_models_from_disk first.")

    # Cleaning and Vectorization
    clean_text = strict_clean(text)
    
    try:
        vectorized_text = VECTORIZER.transform([clean_text]).toarray()
    except ValueError:
        return {'Error': 0.5}

    results = {}
    
    for name, model in MODELS.items():
        try:
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(vectorized_text)[0][1]
            else:
                pred = model.predict(vectorized_text)
                if isinstance(pred, (list, np.ndarray)) and len(pred) > 0:
                     val = pred[0]
                     if isinstance(val, (list, np.ndarray)):
                         prob = float(val[0])
                     else:
                         prob = float(val)
                else:
                    prob = float(pred)
            results[name] = prob
        except Exception as e:
            print(f"[ERROR] Error predicting with {name}: {e}")
            results[name] = 0.5

    return results

def get_interpretation(score):
    if score > 0.85: return "DEFINITELY AI"
    if score > 0.60: return "LIKELY AI"
    if score > 0.40: return "UNCERTAIN"
    if score > 0.15: return "LIKELY HUMAN"
    return "DEFINITELY HUMAN"
