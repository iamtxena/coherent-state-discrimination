from src.database import db


class Alpha(db.Model):

    __tablename__ = 'alphas'
    id = db.Column(db.Integer, primary_key=True)
    alpha = db.Column(db.Float)


def __init__(self,
             alpha: float):
    self.alpha = alpha


def __eq__(self, other) -> bool:
    if not isinstance(other, Alpha):
        return False
    return (self.id == other.id and self.alpha == other.alpha)
