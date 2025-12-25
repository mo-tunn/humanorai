import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Add backend to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import model_manager

class TestModelManager:
    def setup_method(self):
        # Reset globals before each test
        model_manager.MODELS = {}
        model_manager.VECTORIZER = None

    def teardown_method(self):
        model_manager.MODELS = {}
        model_manager.VECTORIZER = None

    def test_get_interpretation(self):
        assert model_manager.get_interpretation(0.90) == "DEFINITELY AI"
        assert model_manager.get_interpretation(0.65) == "LIKELY AI"
        assert model_manager.get_interpretation(0.50) == "UNCERTAIN"
        assert model_manager.get_interpretation(0.20) == "LIKELY HUMAN"
        assert model_manager.get_interpretation(0.10) == "DEFINITELY HUMAN"

    @patch("services.model_manager.joblib.load")
    @patch("services.model_manager.os.path.exists")
    def test_load_models_success(self, mock_exists, mock_load):
        # Mock file system to return True for existence
        mock_exists.return_value = True
        
        # Mock joblib to return a dummy object
        mock_obj = MagicMock()
        mock_load.return_value = mock_obj
        
        base_dir = "/dummy"
        model_manager.load_models_from_disk(base_dir)
        
        assert model_manager.VECTORIZER is not None
        # Validating that models were populated. 8 models in the list.
        assert len(model_manager.MODELS) == 8 

    @patch("services.model_manager.joblib.load")
    @patch("services.model_manager.os.path.exists")
    def test_load_models_partial(self, mock_exists, mock_load):
        # Simulate only some files exist
        # We need a side_effect for exists to return False for some
        # The code checks tfidf_vectorizer.pkl first (must exist implicitly or it crashes inside try block before loop?)
        # Code: VECTORIZER = joblib.load(...) -> if this fails, it raises.
        # Then loops over models.
        
        def exists_side_effect(path):
            if "tfidf" in path: return True
            if "AdaBoost" in path: return True
            return False
            
        mock_exists.side_effect = exists_side_effect
        mock_load.return_value = MagicMock()
        
        model_manager.load_models_from_disk("/dummy")
        
        assert model_manager.VECTORIZER is not None
        assert "AdaBoost" in model_manager.MODELS
        assert "Decision Tree" not in model_manager.MODELS

    def test_predict_no_models_loaded(self):
        with pytest.raises(RuntimeError):
            model_manager.predict_with_all_models("Should fail")

    def test_predict_vectorizer_error(self):
        model_manager.VECTORIZER = MagicMock()
        model_manager.VECTORIZER.transform.side_effect = ValueError("Vocab error")
        
        result = model_manager.predict_with_all_models("text")
        assert "Error" in result

    def test_predict_model_types(self):
        # Test both predict_proba and predict models
        model_manager.VECTORIZER = MagicMock()
        mock_transform_result = MagicMock()
        mock_transform_result.toarray.return_value = [[0, 1]] # Dummy vector
        model_manager.VECTORIZER.transform.return_value = mock_transform_result
        
        # Model with predict_proba
        model_proba = MagicMock()
        model_proba.predict_proba.return_value = [[0.2, 0.8]]
        
        # Model with predict returning scalar
        model_pred_scalar = MagicMock()
        del model_pred_scalar.predict_proba # Ensure it doesn't have it
        model_pred_scalar.predict.return_value = [0.7]

        # Model with predict returning array
        model_pred_array = MagicMock()
        del model_pred_array.predict_proba
        model_pred_array.predict.return_value = [[0.6]] # 2D array case?
        
        model_manager.MODELS = {
            "ProbaModel": model_proba,
            "ScalarModel": model_pred_scalar,
            "ArrayModel": model_pred_array
        }
        
        results = model_manager.predict_with_all_models("some text")
        
        assert results["ProbaModel"] == 0.8
        assert results["ScalarModel"] == 0.7
        # The code logic for predict is: 
        # pred = model.predict(...)
        # if list/ndarray and len>0: val = pred[0] check if list/ndarray...
        # So [[0.6]] -> pred[0] is [0.6], val[0] is 0.6. Correct.
        assert results["ArrayModel"] == 0.6

    def test_predict_model_exception(self):
        model_manager.VECTORIZER = MagicMock()
        # Fix mock for vectorizer here too implicitly or explicitly if needed, 
        # but in strict_clean(text) call it might fail if we don't mock? 
        # Actually strict_clean is imported. 
        # VECTORIZER.transform(...).toarray() is called.
        mock_res = MagicMock()
        mock_res.toarray.return_value = [[0, 1]]
        model_manager.VECTORIZER.transform.return_value = mock_res
        
        bad_model = MagicMock()
        bad_model.predict_proba.side_effect = Exception("Inference failed")
        
        model_manager.MODELS = {"BadModel": bad_model}
        
        results = model_manager.predict_with_all_models("text")
        assert results["BadModel"] == 0.5 # Default fallback

    @patch("services.model_manager.joblib.load")
    def test_load_models_fatal_error(self, mock_load):
        # Test the outer try/except block in load_models_from_disk
        mock_load.side_effect = Exception("Disk corrupted")
        
        with pytest.raises(RuntimeError) as excinfo:
            model_manager.load_models_from_disk("/dummy")
        
        assert "Failed to load models" in str(excinfo.value)
