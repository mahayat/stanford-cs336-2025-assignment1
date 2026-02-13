from typing import List, Dict, Tuple
import json

# Helper functions for saving/loading vocab and merges

def save_vocab(vocab: Dict[int, bytes], filepath: str):
    """Save vocabulary to a JSON file."""
    # Convert bytes to list of ints for JSON serialization
    vocab_serializable = {k: list(v) for k, v in vocab.items()}
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(vocab_serializable, f)


def save_merges(merges: List[Tuple[bytes, bytes]], filepath: str):
    """Save merges to a JSON file."""
    # Convert bytes tuples to list of lists of ints
    merges_serializable = [[list(pair[0]), list(pair[1])] for pair in merges]
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(merges_serializable, f)


def load_vocab(filepath: str) -> Dict[int, bytes]:
    """Load vocabulary from a JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    return {int(k): bytes(v) for k, v in vocab_data.items()}


def load_merges(filepath: str) -> List[Tuple[bytes, bytes]]:
    """Load merges from a JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        merges_data = json.load(f)
    return [(bytes(pair[0]), bytes(pair[1])) for pair in merges_data]