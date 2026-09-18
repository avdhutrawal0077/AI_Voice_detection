from models.schemas import Verdict
from typing import List, Dict, Tuple

def aggregate_scores(window_results: List[Dict]) -> Tuple[Verdict, float]:
    """
    Applies temporal aggregation over a series of window results to determine the final session verdict.
    """
    if not window_results:
        return Verdict.UNCERTAIN, 0.0
        
    scores = [w["risk_score"] for w in window_results if "risk_score" in w]
    
    if not scores:
        return Verdict.UNCERTAIN, 0.0
        
    # Simple average for the prototype; can be upgraded to weighted accumulation
    avg_score = sum(scores) / len(scores)
    
    if avg_score > 0.6:
        final_verdict = Verdict.AI
    elif avg_score < 0.4:
        final_verdict = Verdict.HUMAN
    else:
        final_verdict = Verdict.UNCERTAIN
        
    return final_verdict, avg_score