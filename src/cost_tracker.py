"""Cost tracking for LiteLLM API calls."""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json


@dataclass
class APICall:
    """Represents a single API call with cost information."""
    component: str
    call_type: str  # 'completion' or 'embedding'
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    timestamp: str


class CostTracker:
    """Tracks costs for all LiteLLM API calls."""
    
    def __init__(self):
        self.calls: List[APICall] = []
        self._component_costs: Dict[str, float] = {}
        self._model_costs: Dict[str, float] = {}
        
    def record_call(
        self,
        component: str,
        call_type: str,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost: float = 0.0,
        timestamp: Optional[str] = None
    ):
        """Record an API call with its cost."""
        if timestamp is None:
            from datetime import datetime
            timestamp = datetime.now().isoformat()
            
        call = APICall(
            component=component,
            call_type=call_type,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            timestamp=timestamp
        )
        
        self.calls.append(call)
        
        # Update running totals
        self._component_costs[component] = self._component_costs.get(component, 0) + cost
        self._model_costs[model] = self._model_costs.get(model, 0) + cost
        
    def get_total_cost(self) -> float:
        """Get total cost across all API calls."""
        return sum(call.cost for call in self.calls)
    
    def get_component_costs(self) -> Dict[str, float]:
        """Get costs broken down by component."""
        return self._component_costs.copy()
    
    def get_model_costs(self) -> Dict[str, float]:
        """Get costs broken down by model."""
        return self._model_costs.copy()
    
    def get_call_type_costs(self) -> Dict[str, float]:
        """Get costs broken down by call type (completion vs embedding)."""
        type_costs = {}
        for call in self.calls:
            type_costs[call.call_type] = type_costs.get(call.call_type, 0) + call.cost
        return type_costs
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        total_calls = len(self.calls)
        total_cost = self.get_total_cost()
        
        # Token statistics
        total_input_tokens = sum(call.input_tokens for call in self.calls)
        total_output_tokens = sum(call.output_tokens for call in self.calls)
        
        # Call type breakdown
        completion_calls = sum(1 for call in self.calls if call.call_type == 'completion')
        embedding_calls = sum(1 for call in self.calls if call.call_type == 'embedding')
        
        return {
            'total_cost': round(total_cost, 4),
            'total_calls': total_calls,
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
            'completion_calls': completion_calls,
            'embedding_calls': embedding_calls,
            'component_costs': {k: round(v, 4) for k, v in self.get_component_costs().items()},
            'model_costs': {k: round(v, 4) for k, v in self.get_model_costs().items()},
            'call_type_costs': {k: round(v, 4) for k, v in self.get_call_type_costs().items()}
        }
    
    def print_summary(self):
        """Print a formatted cost summary."""
        stats = self.get_stats()
        
        print(f"\n💰 COST SUMMARY")
        print("=" * 50)
        print(f"Total Cost: ${stats['total_cost']:.4f}")
        print(f"Total API Calls: {stats['total_calls']}")
        print(f"Total Input Tokens: {stats['total_input_tokens']:,}")
        print(f"Total Output Tokens: {stats['total_output_tokens']:,}")
        
        print(f"\n📊 BREAKDOWN BY CALL TYPE:")
        for call_type, cost in stats['call_type_costs'].items():
            call_count = stats.get(f'{call_type}_calls', 0)
            print(f"  {call_type.title()}: ${cost:.4f} ({call_count} calls)")
        
        print(f"\n🧩 BREAKDOWN BY COMPONENT:")
        for component, cost in sorted(stats['component_costs'].items()):
            percentage = (cost / stats['total_cost'] * 100) if stats['total_cost'] > 0 else 0
            print(f"  {component}: ${cost:.4f} ({percentage:.1f}%)")
        
        print(f"\n🤖 BREAKDOWN BY MODEL:")
        for model, cost in sorted(stats['model_costs'].items()):
            percentage = (cost / stats['total_cost'] * 100) if stats['total_cost'] > 0 else 0
            print(f"  {model}: ${cost:.4f} ({percentage:.1f}%)")
        
        print("=" * 50)
    
    def save_detailed_report(self, filepath: str):
        """Save detailed cost report to JSON file."""
        report = {
            'summary': self.get_stats(),
            'detailed_calls': [
                {
                    'component': call.component,
                    'call_type': call.call_type,
                    'model': call.model,
                    'input_tokens': call.input_tokens,
                    'output_tokens': call.output_tokens,
                    'cost': call.cost,
                    'timestamp': call.timestamp
                }
                for call in self.calls
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"💾 Detailed cost report saved to: {filepath}")


# Global cost tracker instance
global_cost_tracker = CostTracker()


def get_cost_tracker() -> CostTracker:
    """Get the global cost tracker instance."""
    return global_cost_tracker