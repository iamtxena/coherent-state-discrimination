from src.database import db


class AdditionalResult(db.Model):

    __tablename__ = 'additional_results'
    id = db.Column(db.Integer, primary_key=True)
    distance_to_homodyne_probability = db.Column(db.Float, nullable=False)
    time_in_seconds = db.Column(db.Float, nullable=False)
    bit_error_rate = db.Column(db.Float, nullable=False)


def __init__(self,
             distance_to_homodyne_probability: float,
             time_in_seconds: float,
             bit_error_rate: float):
    self.distance_to_homodyne_probability = distance_to_homodyne_probability
    self.time_in_seconds = time_in_seconds
    self.bit_error_rate = bit_error_rate


def __eq__(self, other) -> bool:
    if not isinstance(other, AdditionalResult):
        return False

    return (self.id == other.id and
            self.distance_to_homodyne_probability == other.distance_to_homodyne_probability and
            self.time_in_seconds == other.time_in_seconds and
            self.bit_error_rate == other.bit_error_rate)
