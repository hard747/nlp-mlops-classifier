from train.train_intent_classifier import build_label_mappings


def test_build_label_mappings_is_deterministic_and_sorted():
    labels = ["cancel_order", "greet", "cancel_order", "refund"]
    id2label, label2id = build_label_mappings(labels)

    assert id2label == {0: "cancel_order", 1: "greet", 2: "refund"}
    assert label2id == {"cancel_order": 0, "greet": 1, "refund": 2}


def test_build_label_mappings_round_trips():
    labels = ["b", "a", "c"]
    id2label, label2id = build_label_mappings(labels)

    for label, idx in label2id.items():
        assert id2label[idx] == label


def test_build_label_mappings_num_labels_matches_unique_count():
    labels = ["x"] * 10 + ["y"] * 5 + ["z"]
    id2label, _ = build_label_mappings(labels)
    assert len(id2label) == 3
