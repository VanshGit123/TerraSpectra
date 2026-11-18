from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'


def test_farm_metadata():
    response = client.get('/api/farms')
    assert response.status_code == 200
    assert response.json()['farms'][0]['area_acres'] == 1000


def test_metrics_area_sums_to_total():
    response = client.get('/api/metrics')
    assert response.status_code == 200
    data = response.json()
    assert round(data['high_risk_acres'] + data['medium_risk_acres'] + data['low_risk_acres'], 1) == 1000.0


def test_geojson_layer():
    response = client.get('/api/layers')
    assert response.status_code == 200
    payload = response.json()
    assert payload['type'] == 'FeatureCollection'
    assert len(payload['features']) > 0
