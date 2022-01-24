from src.database import db


class TrainingOptions(db.Model):

    __tablename__ = 'training_options'
    id = db.Column(db.Integer, primary_key=True)
    learning_rate = db.Column(db.Float, nullable=True)
    learning_steps = db.Column(db.Integer, nullable=True)
    cutoff_dimensions = db.Column(db.Integer, nullable=True)
    full_batch_used = db.Column(db.Boolean, nullable=False, default=True)


def __init__(self,
             learning_rate: float,
             learning_steps: int,
             cutoff_dimensions: int,
             full_batch_used: bool = True):
    self.learning_rate = learning_rate
    self.learning_steps = learning_steps
    self.cutoff_dimensions = cutoff_dimensions
    self.full_batch_used = full_batch_used


def __eq__(self, other) -> bool:
    if not isinstance(other, TrainingOptions):
        return False

    return (self.id == other.id and
            self.learning_rate == other.learning_rate and
            self.learning_steps == other.learning_steps and
            self.cutoff_dimensions == other.cutoff_dimensions and
            self.full_batch_used == other.full_batch_used)
