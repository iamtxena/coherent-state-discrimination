import os

user = os.environ.get('POSTGRES_USER')
password = os.environ.get('POSTGRES_PASSWORD')
host = os.environ.get('POSTGRES_HOST', 'postgres_db')
database = os.environ.get('PUBLIC_POSTGRES_DB', 'csd_flask')
port = os.environ.get('POSTGRES_PORT', '5432')

DATABASE_CONNECTION_URI = f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}'
