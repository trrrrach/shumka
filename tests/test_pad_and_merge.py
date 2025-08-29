import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Provide a minimal stub for pydub to satisfy analyzer imports during tests
import types
pydub_stub = types.ModuleType("pydub")

class AudioSegment:
    pass

pydub_stub.AudioSegment = AudioSegment
sys.modules.setdefault("pydub", pydub_stub)

from analyzer import pad_and_merge

def test_pad_and_merge_overlapping_after_padding():
    # intervals overlap after applying padding
    chunks = [(100, 200), (250, 300)]
    result = pad_and_merge(chunks, pad_ms=50)
    assert result == [[50, 350]]

def test_pad_and_merge_respects_merge_gap():
    # intervals are merged when gap is within merge_gap_ms
    chunks = [(0, 100), (150, 200)]
    result = pad_and_merge(chunks, pad_ms=0, merge_gap_ms=50)
    assert result == [[0, 200]]

    # intervals remain separate when gap exceeds merge_gap_ms
    chunks_far = [(0, 100), (250, 300)]
    result_far = pad_and_merge(chunks_far, pad_ms=0, merge_gap_ms=50)
    assert result_far == [[0, 100], [250, 300]]
