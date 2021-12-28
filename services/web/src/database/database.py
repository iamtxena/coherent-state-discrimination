from typing import List, Tuple
from flask_sqlalchemy import Pagination

from . import db


def get_all(model):
    data = model.query.all()
    return data


def get_instance(model, id):
    data = model.query.filter_by(id=id).all()[0]
    return data


def add_instance(model, **kwargs):
    instance = model(**kwargs)
    db.session.add(instance)
    commit_changes()
    return instance


def delete_instance(model, id):
    model.query.filter_by(id=id).delete()
    commit_changes()


def edit_instance(model, id, **kwargs):
    instance = model.query.filter_by(id=id).all()[0]
    updated = False
    for attr, new_value in kwargs.items():
        if getattr(instance, attr) != new_value:
            setattr(instance, attr, new_value)
            updated = True
    if updated:
        commit_changes()
    return instance, updated


def commit_changes():
    db.session.commit()


def paginate(model, page, per_page) -> Tuple[Pagination, List[dict]]:
    paginated_data: Pagination = model.query.paginate(page=page, per_page=per_page)
    items = [item.to_dict() for item in paginated_data.items]
    return paginated_data, items


def paginate_instances_filtered(model, page, per_page, user_id) -> Tuple[Pagination, List[dict]]:
    paginated_data: Pagination = model.query.filter_by(user_id=user_id).paginate(page=page, per_page=per_page)
    items = [item.to_dict() for item in paginated_data.items]
    return paginated_data, items
