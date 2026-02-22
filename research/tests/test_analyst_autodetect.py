
import os
import sys
import unittest
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from unittest.mock import patch, MagicMock
from research.scientist.llm.backend import create_backend
from research.scientist.persona import Aria

class TestAnalystAutodetect(unittest.TestCase):
    
    @patch('research.scientist.llm.ollama.requests.get')
    def test_autodetect_gemma(self, mock_get):
        # Setup mock Ollama response with a small Gemma model
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "models": [
                {"name": "gemma2:2b", "size": 1.6 * 1024**3},
                {"name": "qwen2.5-coder:3b", "size": 1.9 * 1024**3},
                {"name": "phi4:latest", "size": 15 * 1024**3}
            ]
        }
        mock_get.return_value = mock_resp
        
        # Ensure no env vars interfering
        with patch.dict(os.environ, {}, clear=True):
            backend = create_backend(is_analyst=True)
            
            self.assertIsNotNone(backend)
            self.assertEqual(backend.name, "ollama")
            self.assertEqual(backend.model, "gemma2:2b")
            self.assertEqual(backend.keep_alive, 0) # Immediate unload

    @patch('research.scientist.llm.ollama.requests.get')
    def test_fallback_to_primary(self, mock_get):
        # Ollama not available
        mock_get.side_effect = Exception("Connection refused")
        
        with patch.dict(os.environ, {"ARIA_LLM_BACKEND": "anthropic"}, clear=True):
            # create_backend(is_analyst=True) should return primary if autodetect fails
            backend = create_backend(is_analyst=True)
            self.assertIsNotNone(backend)
            self.assertEqual(backend.name, "anthropic")

    @patch('research.scientist.llm.ollama.requests.get')
    @patch('research.scientist.llm.anthropic.AnthropicBackend.generate')
    def test_aria_uses_analyst(self, mock_anthropic_gen, mock_ollama_get):
        # Setup mock Ollama
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"models": [{"name": "gemma2:2b", "size": 1.6 * 1024**3}]}
        mock_ollama_get.return_value = mock_resp
        
        # Setup mock Anthropic for primary
        mock_anthropic_gen.return_value = MagicMock(text="Strategic plan", tokens_used=10)
        
        with patch.dict(os.environ, {"ARIA_LLM_BACKEND": "anthropic"}, clear=True):
            aria = Aria()
            aria._continuous_mode = False
            
            # This should use primary (Anthropic)
            aria.plan_strategy("context")
            self.assertTrue(mock_anthropic_gen.called)
            
            # This should use analyst (Ollama/Gemma)
            # We must ensure ARIA_LLM_BACKEND is NOT seen by the analyst create_backend call
            # if we want auto-detection to trigger.
            with patch.dict(os.environ, {}, clear=True):
                with patch('research.scientist.llm.ollama.requests.post') as mock_post:
                    mock_post.return_value.status_code = 200
                    mock_post.return_value.json.return_value = {"response": "Summary result", "eval_count": 5}
                    
                    # Force analyst initialization
                    analyst = aria._get_analyst_llm()
                    self.assertEqual(analyst.name, "ollama")
                    self.assertEqual(analyst.model, "gemma2:2b")
                    
                    aria.experiment_summary({"total": 10}, context="context")
                    
                    # Verify Ollama was called
                    self.assertTrue(mock_post.called, "Ollama post was not called for analyst task")
                    args, kwargs = mock_post.call_args
                    self.assertEqual(kwargs['json']['model'], "gemma2:2b")
                    self.assertEqual(kwargs['json']['keep_alive'], 0)

if __name__ == '__main__':
    unittest.main()
