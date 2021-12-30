from src.database import db


class Alpha(db.Model):

    __tablename__ = 'alphas'
    alpha = db.Column(db.Float, primary_key=True)


def __init__(self,
             alpha: float):
    self.alpha = alpha


def __eq__(self, other) -> bool:
    if not isinstance(other, Alpha):
        return False
    return (self.alpha == other.alpha)
