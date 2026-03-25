from typing import Literal

def evaluate_curriculum(
    current_diff: int,
    avg_score: float,
    med_score: float,
    consecutive_drops: int,
) -> tuple[int, int, Literal["promote", "demote", "none"]]:
    """
    Evaluates the training metrics to determine if the curriculum difficulty 
    should be increased or decreased.
    
    Returns:
        (new_diff, new_consecutive_drops, action)
    """
    if current_diff > 1 and avg_score < 50.0:
        consecutive_drops += 1
        if consecutive_drops >= 3:
            return current_diff - 1, 0, "demote"
    else:
        consecutive_drops = 0
        
    if med_score >= 300.0 and current_diff < 6:
        return current_diff + 1, 0, "promote"
        
    return current_diff, consecutive_drops, "none"
