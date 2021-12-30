from src.database import db


class Architecture(db.Model):

    __tablename__ = 'architectures'
    id = db.Column(db.Integer, primary_key=True)
    number_modes = db.Column(db.Integer, nullable=False, default=1)
    squeezing = db.Column(db.Boolean, nullable=False, default=False)
    number_ancillas = db.Column(db.Integer, nullable=False, default=0)
    number_layers = db.Column(db.Integer, nullable=False, default=1)


def __init__(self,
             number_modes: int,
             squeezing: bool,
             number_ancillas: int,
             number_layers: int):
    self.number_modes = number_modes
    self.squeezing = squeezing
    self.number_ancillas = number_ancillas
    self.number_layers = number_layers


def __eq__(self, other) -> bool:
    if not isinstance(other, Architecture):
        return False

    return (self.id == other.id and
            self.number_modes == other.number_modes and
            self.squeezing == other.squeezing and
            self.number_ancillas == other.number_ancillas and
            self.number_layers == other.number_layers)
