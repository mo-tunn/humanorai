import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import sys
import os

# Add backend to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from APIServer import app

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def mock_model_manager():
    """
    Mocks the model_manager services to avoid loading actual ML models.
    """
    with patch("services.model_manager.load_models_from_disk") as mock_load, \
         patch("services.model_manager.predict_with_all_models") as mock_predict, \
         patch("services.model_manager.get_interpretation") as mock_interpret:
        
        # Default behavior
        mock_load.return_value = None
        mock_predict.return_value = {
            "AdaBoost": 0.95,
            "Logistic Regression": 0.10
        }
        
        # Re-implement simple logic for interpretation mock if needed, or just return fixed
        def side_effect_interpret(score):
            if score > 0.6: return "LIKELY AI"
            return "LIKELY HUMAN"
            
        mock_interpret.side_effect = side_effect_interpret
        
        yield {
            "load": mock_load,
            "predict": mock_predict,
            "interpret": mock_interpret
        }
