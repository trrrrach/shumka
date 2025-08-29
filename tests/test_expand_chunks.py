import os
import sys
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Provide a minimal stub for pydub to satisfy analyzer imports during tests
pydub_stub = types.ModuleType("pydub")
class AudioSegment:
    pass
pydub_stub.AudioSegment = AudioSegment
sys.modules.setdefault("pydub", pydub_stub)

from analyzer import expand_chunks


def test_expand_chunks_basic_and_bounds():
    chunks = [(100, 200)]
    result = expand_chunks(chunks, expand_ms=50, total_ms=500)
    assert result == [[50, 250]]

    # respects boundaries at start and end without merging distant chunks
    edge_chunks = [(0, 50), (480, 500)]
    edge_result = expand_chunks(edge_chunks, expand_ms=50, total_ms=500)
    assert edge_result == [[0, 100], [430, 500]]


def test_expand_chunks_merges_overlaps():
    # two chunks that are separated but become overlapping after expansion
    chunks = [(100, 160), (200, 260)]

    # expanding by 80ms causes the ranges to overlap and merge into one interval
    result = expand_chunks(chunks, expand_ms=80, total_ms=500)
    assert result == [[20, 340]]
    assert len(result) == 1


def test_expand_chunks_no_overlaps_in_result():
    chunks = [(10, 20), (40, 50), (400, 410)]
    result = expand_chunks(chunks, expand_ms=10, total_ms=500)

    # first two chunks merge, but final intervals should be non-overlapping
    assert result == [[0, 60], [390, 420]]
    for (s1, e1), (s2, e2) in zip(result, result[1:]):
        assert e1 < s2
