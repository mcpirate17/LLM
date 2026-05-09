from research.scientist.api import create_app
from research.synthesis.templates import TEMPLATES


def test_template_names_api_returns_full_template_registry():
    app = create_app(notebook_path=":memory:")
    client = app.test_client()

    response = client.get("/api/template-names")

    assert response.status_code == 200
    names = response.get_json()["names"]
    assert names == sorted(TEMPLATES.keys())
