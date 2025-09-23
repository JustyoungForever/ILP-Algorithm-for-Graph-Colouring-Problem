from typing import List, Dict, Set
import math

def choose_colors_after_fixing(
    allowed_colors: List[int],
    rc_y: Dict[int, float],
    z_LP: float,
    UB: int,
    reserved_colors: Set[int],
    eps: float = 1e-6,
) -> List[int]:
    """
    Safe prefix-shrink rule driven by the LP lower bound:
      K_new = min(UB, max(LB_reserved, ceil(z_LP)))
    - LB_reserved = max(reserved_colors)+1 (typically the clique size)
    - Always return a contiguous prefix 0..K_new-1 to keep invariants.
    - We never expand K here; we only shrink or keep the same.
    """
    if not allowed_colors:
        return []

    LB_reserved = (max(reserved_colors) + 1) if reserved_colors else 0
    k_from_lp = int(math.ceil(z_LP - 1e-9))  # ceil(z_LP) with tiny slack for numerical noise
    K_new = min(UB, max(LB_reserved, k_from_lp))

    K_prev = len(allowed_colors)
    if K_new >= K_prev:
        return allowed_colors  # do not expand; keep as-is
    return list(range(K_new))
