from celery import Celery
from src.core.config import config

celery_app = Celery(
    "embed_worker",
    broker=config.CELERY_BROKER_URL,
    backend=config.CELERY_RESULT_BACKEND,
)

celery_app.conf.imports = (
    "tasks",
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,

    broker_connection_retry_on_startup=True,
    broker_connection_timeout=5,
    broker_connection_max_retries=3,

    result_backend_connection_timeout=5,
    result_backend_connection_max_retries=3,

    task_store_eager_result=False,
)
