from src.database import db


class OptimizedParameters(db.Model):

    __tablename__ = 'optimized_parameters'
    id = db.Column(db.Integer, primary_key=True)
    parameters = db.Column(db.Text, nullable=False)


def __init__(self,
             parameters: str):
    self.parameters = parameters


def __eq__(self, other) -> bool:
    if not isinstance(other, OptimizedParameters):
        return False
    return (self.id == other.id and self.parameters == other.parameters)
