from src.database import db
from src.models.alphas import Alpha
from src.models.architectures import Architecture
from src.models.training_options import TrainingOptions
from src.models.additional_results import AdditionalResult
from src.models.optimized_parameters import OptimizedParameters


class TrainingResult(db.Model):

    __tablename__ = 'training_results'
    id = db.Column(db.Integer, primary_key=True)
    success_probability = db.Column(db.Float, nullable=False)
    alpha_id = db.Column(db.Integer, db.ForeignKey(f'{Alpha.__tablename__}.id'), nullable=False)
    architecture_id = db.Column(db.Integer, db.ForeignKey(f'{Architecture.__tablename__}.id'), nullable=False)
    training_options_id = db.Column(db.Integer, db.ForeignKey(f'{TrainingOptions.__tablename__}.id'), nullable=False)
    additional_result_id = db.Column(db.Integer, db.ForeignKey(
        f'{AdditionalResult.__tablename__}.id'), nullable=False)
    optimized_parameters_id = db.Column(db.Integer, db.ForeignKey(
        f'{OptimizedParameters.__tablename__}.id'), nullable=False)


def __init__(self,
             success_probability: float,
             alpha_id: int,
             architecture_id: int,
             training_options_id: int,
             additional_result_id: int,
             optimized_parameters_id: int):
    self.success_probability = success_probability
    self.alpha_id = alpha_id
    self.architecture_id = architecture_id
    self.training_options_id = training_options_id
    self.additional_result_id = additional_result_id
    self.optimized_parameters_id = optimized_parameters_id


def __eq__(self, other) -> bool:
    if not isinstance(other, TrainingResult):
        return False

    return (self.id == other.id and
            self.success_probability == other.success_probability and
            self.alpha_id == other.alpha_id and
            self.architecture_id == other.architecture_id and
            self.training_options_id == other.training_options_id and
            self.additional_result_id == other.additional_result_id and
            self.optimized_parameters_id == other.optimized_parameters_id)
