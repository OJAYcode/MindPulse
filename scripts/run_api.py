import _bootstrap  # noqa: F401

import uvicorn

from app.api.main import app
from app.utils.config import get_settings


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
