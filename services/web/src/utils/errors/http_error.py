# http_error.py
from typing import TypedDict


class HttpErrorSetup(TypedDict):
    title: str
    status: int
    detail: str


class HttpError(Exception):
    def __init__(self, error: HttpErrorSetup, status_code, ex: Exception = None):
        self.error = error
        self.status_code = status_code
        if ex:
            self.error['detail'] = f"{type(ex).__name__}: {self.error['detail']} {str(ex)}"
