import os
from flask import Flask, make_response, jsonify, request
from flask_cors import CORS
from config import postgres
from src.utils.errors import HttpError, HttpErrorSetup
from src.database import db, database
from src.utils.logger import logger

from src.controllers.results import ResultsController
from src.typings.execution_result import (ExecutionResultInput,
                                          TrainingExecutionResultInput,
                                          TrainingExecutionResult,
                                          TestingExecutionResultInput,
                                          TestingExecutionResult)


app = Flask(__name__)


@app.errorhandler(Exception)
def handle_exception(ex):
    """Return JSON instead of HTML for any Exception."""
    http_error = ex
    if not isinstance(ex, HttpError):
        http_error = HttpError(HttpErrorSetup({
            'title': type(ex).__name__,
            'status': 500,
            'detail': str(ex)
        }), 500)

    response = jsonify(http_error.error)
    response.status_code = http_error.status_code
    response.content_type = "application/json"
    return response, response.status_code


CORS(app)
app.config['JSON_SORT_KEYS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = postgres.DATABASE_CONNECTION_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.app_context().push()
db.init_app(app)
# db.drop_all

db.create_all()

# API version
version = os.environ.get("PUBLIC_API_VERSION", "v1")
EXTERNAL_PUBLIC_SERVICE_URL = os.environ.get("EXTERNAL_PUBLIC_SERVICE_URL", "localhost")
API_URL = f'{EXTERNAL_PUBLIC_SERVICE_URL}/api/{version}'
logger.info(f'CSD API SERVICE URL: {API_URL}')

results_controller = ResultsController(database=database)


@ app.route("/status", methods=["GET"])
def health_check():
    logger.debug("Health check: OK")
    return make_response("OK", 200)


""" RESULTS """


@ app.route(f'/api/{version}/results', methods=['POST'])
def create_result():
    """
    Create a new result
    """
    result_type = request.args.get('type', 'training', type=str)
    if result_type == 'testing':
        body: TestingExecutionResultInput = request.get_json()
        testing_result = results_controller.add_testing_execution_result(
            testing_execution_result_input=body)
        return make_response(testing_result.__dict__, 201)
    body: TrainingExecutionResultInput = request.get_json()
    training_result = results_controller.add_training_execution_result(
        training_execution_result_input=body)
    return make_response(training_result.__dict__, 201)


@ app.route(f'/api/{version}/results', methods=['GET'])
def list_results():
    """
    GET all results
    """
    result_type = request.args.get('type', 'training', type=str)
    if result_type == 'testing':
        testing_results = results_controller.get_best_testing_execution_results_for_all_alphas()
        return make_response(testing_results.__dict__, 201)

    training_results = results_controller.get_best_training_execution_results_for_all_alphas()
    return make_response(training_results.__dict__, 201)
