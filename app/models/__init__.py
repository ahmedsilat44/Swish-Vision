# Import all models here so that Base.metadata is fully populated whenever
# this package is imported — no need to list individual models in main.py.
from app.models.user import User  # noqa: F401
from app.models.session import SessionModel as Session  # noqa: F401
from app.models.revoked_token import RevokedToken  # noqa: F401
from app.models.angle_frame import AngleFrame  # noqa: F401
from app.models.shot_event import ShotEvent  # noqa: F401
from app.models.report import Report  # noqa: F401
