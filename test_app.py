import pytest
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True

    with app.test_client() as test_client:
        yield test_client


def test_home(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.get_data(as_text=True) == (
        "Iris Classifier API is Running!"
    )


def test_predict_get(client):
    response = client.get(
        "/predict",
        query_string={
            "f1": 5.1,
            "f2": 3.5,
            "f3": 1.4,
            "f4": 0.2,
        },
    )

    assert response.status_code == 200
    assert response.get_json() == {"prediction": "setosa"}
    


def test_predict_post(client):
    response = client.post(
        "/predict",
        json={"features": [5.1, 3.5, 1.4, 0.2]},
    )

    assert response.status_code == 200
    assert response.get_json() == {"prediction": "setosa"}
    


def test_missing_get_parameters(client):
    response = client.get("/predict?f1=5.1")

    assert response.status_code == 400
    assert "error" in response.get_json()


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"features": [5.1, 3.5]},
        {"features": ["wrong", 3.5, 1.4, 0.2]},
    ],
)
def test_invalid_post_input(client, payload):
    response = client.post("/predict", json=payload)

    assert response.status_code == 400
    assert "error" in response.get_json()