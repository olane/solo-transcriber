from app.pipeline.separate import order_stems


def test_order_stems_uses_the_canonical_order_then_extras():
    assert order_stems({"other", "bass", "vocals", "kazoo", "drums"}) == [
        "vocals",
        "drums",
        "bass",
        "other",
        "kazoo",
    ]
