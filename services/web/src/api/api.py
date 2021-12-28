import os
from flask import Flask, make_response, jsonify
from flask_cors import CORS
from config import postgres
from src.utils.errors import HttpError, HttpErrorSetup
from src.database import db
from src.utils.logger import logger

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


@ app.route("/status", methods=["GET"])
def health_check():
    logger.debug("Health check: OK")
    return make_response("OK", 200)
