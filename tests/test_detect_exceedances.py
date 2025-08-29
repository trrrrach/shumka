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

from analyzer import detect_exceedances


def test_detect_exceedances_handles_hysteresis_and_open_end():
    rms_dbfs = [0, 1, 4, 3.5, 2.5, 1, 0, 4, 5, 4]
    hop_ms = 100
    baseline_db = 0
    delta_db = 3
    hyst_db = 1
    min_event_ms = 200

    expected = [[200, 500], [700, 1000]]
    result = detect_exceedances(
        rms_dbfs, hop_ms, baseline_db, delta_db, hyst_db, min_event_ms
    )
    assert result == expected
