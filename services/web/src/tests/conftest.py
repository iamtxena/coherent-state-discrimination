from flask import Flask
from src.database import db

import os
from dotenv import load_dotenv
load_dotenv()

user = os.environ.get('POSTGRES_USER')
password = os.environ.get('POSTGRES_PASSWORD')
host = os.environ.get('POSTGRES_HOST', 'localhost')
database = 'csd_flask'
port = os.environ.get('POSTGRES_PORT', '5432')

DATABASE_TEST_CONNECTION_URI = f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}'


def create_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_TEST_CONNECTION_URI
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.app_context().push()
    db.init_app(app)
    db.create_all()
    return app
