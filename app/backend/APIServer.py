from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import uvicorn
import warnings

# Custom Modules
from schemas import TextRequest
from services import model_manager
from utils.file_parser import extract_text_from_file

# --- CONFIG ---
warnings.filterwarnings('ignore')

app = FastAPI(title="Human or AI? API", version="1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static Files
base_dir = os.path.dirname(os.path.abspath(__file__))
frontend_dir = os.path.join(base_dir, '..', 'frontend')
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

# --- EVENTS ---

@app.on_event("startup")
async def startup_event():
    try:
        model_manager.load_models_from_disk(base_dir)
    except Exception as e:
        print(f"[FATAL] Could not load models: {e}")
        # In a real app we might want to shut down, but printing is okay for now
        # raise e 

# --- ENDPOINTS ---

@app.post("/predict")
def predict_text(request: TextRequest):
    """Analyzes text and returns predictions from all models."""
    user_text = request.text
    
    if not user_text or len(user_text) < 50:
        raise HTTPException(status_code=400, detail="Minimum 50 characters required for analysis.")

    # Get predictions
    predictions = model_manager.predict_with_all_models(user_text)
    
    if 'Error' in predictions:
        raise HTTPException(status_code=500, detail="Error during text processing.")

    # Format results
    results = []
    total_prob = 0
    
    for name, prob in predictions.items():
        total_prob += prob
        results.append({
            'model': name,
            'probability_percent': round(float(prob) * 100, 2),
            'decision': model_manager.get_interpretation(prob)
        })

    avg_prob = total_prob / len(predictions) if predictions else 0
    
    return {
        "individual_results": results,
        "ensemble_average_percent": round(float(avg_prob) * 100, 2),
        "ensemble_decision": model_manager.get_interpretation(avg_prob),
    }

@app.post("/predict_file")
def predict_file_endpoint(file: UploadFile = File(...)):
    """File upload endpoint (Supports: .pdf, .docx, .txt)"""
    
    text = extract_text_from_file(file)
    
    if not text or len(text.strip()) < 50:
         raise HTTPException(status_code=400, detail="Minimum 50 characters required for analysis.")
         
    predictions = model_manager.predict_with_all_models(text)
    
    if 'Error' in predictions:
        raise HTTPException(status_code=500, detail="Error during text processing.")

    results = []
    total_prob = 0
    
    for name, prob in predictions.items():
        total_prob += prob
        results.append({
            'model': name,
            'probability_percent': round(float(prob) * 100, 2),
            'decision': model_manager.get_interpretation(prob)
        })

    avg_prob = total_prob / len(predictions) if predictions else 0
    
    return {
        "individual_results": results,
        "ensemble_average_percent": round(float(avg_prob) * 100, 2),
        "ensemble_decision": model_manager.get_interpretation(avg_prob),
    }

# Root mount must be last
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
