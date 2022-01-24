from src.database import db
from src.models.alphas import Alpha
from src.models.architectures import Architecture
from src.models.additional_results import AdditionalResult


class TestingResult(db.Model):

    __tablename__ = 'testing_results'
    id = db.Column(db.Integer, primary_key=True)
    success_probability = db.Column(db.Float, nullable=False)
    alpha_id = db.Column(db.Integer, db.ForeignKey(f'{Alpha.__tablename__}.id'), nullable=False)
    architecture_id = db.Column(db.Integer, db.ForeignKey(f'{Architecture.__tablename__}.id'), nullable=False)
    additional_result_id = db.Column(db.Integer, db.ForeignKey(
        f'{AdditionalResult.__tablename__}.id'), nullable=False)


def __init__(self,
             success_probability: float,
             alpha_id: int,
             architecture_id: int,
             additional_result_id: int):
    self.success_probability = success_probability
    self.alpha_id = alpha_id
    self.architecture_id = architecture_id
    self.additional_result_id = additional_result_id


def __eq__(self, other) -> bool:
    if not isinstance(other, TestingResult):
        return False

    return (self.id == other.id and
            self.success_probability == other.success_probability and
            self.alpha_id == other.alpha_id and
            self.architecture_id == other.architecture_id and
            self.additional_result_id == other.additional_result_id)
