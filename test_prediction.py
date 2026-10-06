from src.predict import predict_rentals


def test_prediction():

    sample_data = [
        2,
        1,
        7,
        0,
        1,
        1,
        0.70,
        0.65,
        0.60,
        0.15,
        2
    ]

    result = predict_rentals(
        sample_data
    )

    assert isinstance(
        result,
        float
    )

    assert result >= 0