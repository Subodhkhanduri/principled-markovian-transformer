# tests/test_models.py
import pytest
import torch
from src.models import PrincipledMarkovianTransformer

class TestPrincipledMarkovianTransformer:
    def test_forward_pass(self):
        """Test basic forward pass."""
        model = PrincipledMarkovianTransformer()
        x = torch.randn(2, 100, 768)
        output = model(x)
        assert output.shape == (2, 100, 768)
    
    def test_deterministic_output(self):
        """Test deterministic behavior."""
        model = PrincipledMarkovianTransformer()
        x = torch.randn(2, 100, 768)
        
        torch.manual_seed(42)
        output1 = model(x)
        
        torch.manual_seed(42)
        output2 = model(x)
        
        assert torch.allclose(output1, output2, atol=1e-6)
    
    def test_expert_utilization(self):
        """Test expert utilization."""
        model = PrincipledMarkovianTransformer()
        x = torch.randn(2, 100, 768)
        
        with torch.no_grad():
            output, expert_stats = model(x, return_expert_stats=True)
        
        # Check that all experts are used
        assert len(expert_stats['utilization']) == 4
        assert all(util > 0 for util in expert_stats['utilization'])
