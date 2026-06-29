import pandas as pd

from src.benchmark.ontology import filter_semantic_droplets, normalize_semantic_class_ids


def test_semantic_class_mapping_unifies_prediction_and_reference_droplets() -> None:
    df = pd.DataFrame({
        "class_id": [0, 1, 3, 4],
        "global_id": [10, 11, 12, 13],
    })

    normalized = normalize_semantic_class_ids(df)
    assert normalized["semantic_class_id"].tolist() == [0, 3, 3, 4]

    droplets = filter_semantic_droplets(df)
    assert droplets["global_id"].tolist() == [11, 12]
