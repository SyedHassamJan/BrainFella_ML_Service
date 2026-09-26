"""Unit tests that need no Whisper weights."""

import base64

import pytest

from app import voice_analysis as va


class TestDecodeAudio:
    def test_decodes_plain_and_data_uri(self):
        b64 = base64.b64encode(b"RIFFdata").decode()
        assert va.decode_audio(b64) == b"RIFFdata"
        assert va.decode_audio("data:audio/wav;base64," + b64) == b"RIFFdata"

    def test_rejects_bad_base64_and_empty(self):
        for bad in ("not base64 !!!", ""):
            with pytest.raises(va.VoiceError) as e:
                va.decode_audio(bad)
            assert e.value.code == "invalid_audio"

    def test_rejects_oversized(self):
        with pytest.raises(va.VoiceError) as e:
            va.decode_audio("A" * (va.MAX_B64_CHARS + 1))
        assert e.value.code == "audio_too_large"


class TestSpeechMetrics:
    def test_no_words(self):
        assert va.compute_speech_metrics([], 5.0) == {"speakingRate": 0.0, "pauseCount": 0}

    def test_speaking_rate_uses_speech_span_not_clip_length(self):
        # 10 words evenly across 0..5 s, clip padded to 30 s -> 120 wpm, not 20 wpm
        words = [(i * 0.5, i * 0.5 + 0.4) for i in range(10)]
        m = va.compute_speech_metrics(words, 30.0)
        assert m["speakingRate"] == pytest.approx(10 / 4.9 * 60, abs=0.2)
        assert m["pauseCount"] == 0

    def test_counts_only_gaps_at_or_above_threshold(self):
        words = [(0.0, 0.5), (0.6, 1.0), (1.69, 2.0), (2.7, 3.0), (5.0, 5.5)]
        # gaps: 0.1, 0.69 (<0.7, no), 0.7 (yes), 2.0 (yes)
        assert va.compute_speech_metrics(words, 6.0)["pauseCount"] == 2

    def test_very_short_span_falls_back_to_clip_duration(self):
        m = va.compute_speech_metrics([(0.0, 0.3), (0.35, 0.6)], 4.0)
        assert m["speakingRate"] == pytest.approx(2 / 4.0 * 60, abs=0.1)
