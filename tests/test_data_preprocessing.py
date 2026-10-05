import pandas as pd
import pytest

from src.data_audit import audit_dataset, identify_target
from src.data_preprocessing import build_preprocessor, prepare_data


def test_audit_identifies_binary_status_target():
    data = pd.DataFrame({"feature": [1, 2, 3], "loan_status": [0, 1, 0]})
    report = audit_dataset(data)
    assert report["target_column"] == "loan_status"
    assert report["shape"] == {"rows": 3, "columns": 2}


def test_target_identification_rejects_ambiguous_data():
    data = pd.DataFrame({"a_status": [0, 1], "b_status": [1, 0]})
    with pytest.raises(ValueError, match="ambiguous"):
        identify_target(data)


def test_prepare_data_removes_duplicates_and_preserves_stratification(tmp_path):
    data = pd.DataFrame(
        {
            "income": [10, 20, 30, 40, 50, 60, 70, 80],
            "category": ["a", "a", "b", "b", "a", "b", "a", "b"],
            "loan_status": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    path = tmp_path / "data.csv"
    pd.concat([data, data.iloc[[0]]]).to_csv(path, index=False)
    result = prepare_data(path, test_size=0.25)
    assert result["removed_duplicate_rows"] == 1
    assert len(result["X_train"]) + len(result["X_test"]) == len(data)
    assert result["y_train"].value_counts().to_dict() == {0: 3, 1: 3}
    assert result["preprocessor"].transformers[0][0] == "numerical"


def test_preprocessor_handles_missing_and_unknown_categories():
    preprocessor = build_preprocessor(["income"], ["home"])
    train = pd.DataFrame({"income": [10, None], "home": ["RENT", "OWN"]})
    test = pd.DataFrame({"income": [20], "home": ["NEW_CATEGORY"]})
    preprocessor.fit(train)
    transformed = preprocessor.transform(test)
    assert transformed.shape[0] == 1
    assert transformed.shape[1] == 3
