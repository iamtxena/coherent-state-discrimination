import pytest
from .conftest import create_app
from src.database import database
from src.controllers.results import ResultsController


@pytest.fixture(scope='module')
def test_client():
    app = create_app()
    # Create a test client using the Flask application configured for testing
    with app.test_client() as testing_client:
        with app.app_context():
            yield testing_client


# def test_status(test_client):
#     response = test_client.get('/status')
#     print(response)
#     assert response.status_code == 200
#     print(response.data)


def test_results_controller_constructor(test_client):
    response = test_client.get('/status')
    results_controller = ResultsController(database=database)
    print(results_controller.get_best_training_execution_results_for_all_alphas())
