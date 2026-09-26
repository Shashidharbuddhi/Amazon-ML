"""
split_data.py: Leakage-free split utilities grouping strictly by Source 1 entity ID.
"""
import hashlib
from typing import Set, Tuple, List

def get_entity_split_group(entity_id: str, val_ratio: float = 0.2, seed: int = 42) -> str:
    """
    Deterministically hash an entity_id to 'train' or 'val'.
    Prevents leakage by ensuring all candidate pairs for an entity_id belong to either train or val.
    """
    h = hashlib.sha256(f"{entity_id}_{seed}".encode("utf-8")).hexdigest()
    int_val = int(h[:8], 16)
    threshold = int(val_ratio * 0xFFFFFFFF)
    return "val" if int_val < threshold else "train"

def split_entity_ids(entity_ids: List[str], val_ratio: float = 0.2, seed: int = 42) -> Tuple[Set[str], Set[str]]:
    """
    Split a list of unique Source 1 entity IDs into train and validation sets.
    """
    train_ids = set()
    val_ids = set()
    for eid in entity_ids:
        group = get_entity_split_group(eid, val_ratio=val_ratio, seed=seed)
        if group == "val":
            val_ids.add(eid)
        else:
            train_ids.add(eid)
    return train_ids, val_ids
