from app.fusion.rules import late_fusion_rule


def test_late_fusion_returns_high_for_stressed_inputs():
    result = late_fusion_rule(
        {"sad": 0.8, "happy": 0.2},
        {"stressed": 0.9, "calm": 0.1},
    )
    assert result.stress_level == "high"


def test_late_fusion_weights_can_shift_result():
    face_probabilities = {"happy": 0.9, "sad": 0.1}
    voice_probabilities = {"stressed": 0.9, "calm": 0.1}

    face_heavy = late_fusion_rule(
        face_probabilities,
        voice_probabilities,
        weights={"face": 0.9, "voice": 0.1},
    )
    voice_heavy = late_fusion_rule(
        face_probabilities,
        voice_probabilities,
        weights={"face": 0.1, "voice": 0.9},
    )

    assert face_heavy.stress_score < voice_heavy.stress_score
