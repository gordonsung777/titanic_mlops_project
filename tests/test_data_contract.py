from titanic_mlops.data import FEATURE_COLUMNS, TARGET_COLUMN, load_titanic, split_features_target


def test_titanic_file_has_required_contract():
    df = load_titanic("data/titanic.csv")
    assert TARGET_COLUMN in df.columns
    for column in FEATURE_COLUMNS:
        assert column in df.columns
    assert df[TARGET_COLUMN].isin([0, 1]).all()


def test_split_features_target_shapes_match():
    df = load_titanic("data/titanic.csv")
    X, y = split_features_target(df)
    assert len(X) == len(y)
    assert list(X.columns) == FEATURE_COLUMNS
