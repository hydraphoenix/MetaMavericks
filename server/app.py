# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Metamavericks Env Environment.

This module creates an HTTP server that exposes the MetamavericksEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from metamavericks_env.models import MetamavericksAction, MetamavericksObservation
    from server.metamavericks_env_environment import MetamavericksEnvironment
except ImportError:
    # Fallback for different environment structures
    try:
        from ..models import MetamavericksAction, MetamavericksObservation
        from .metamavericks_env_environment import MetamavericksEnvironment
    except (ImportError, ValueError):
        from models import MetamavericksAction, MetamavericksObservation
        from server.metamavericks_env_environment import MetamavericksEnvironment


# Create the app with web interface and README integration
app = create_app(
    MetamavericksEnvironment,
    MetamavericksAction,
    MetamavericksObservation,
    env_name="metamavericks_env",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)

from fastapi import Request
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
app.add_middleware(ProxyHeadersMiddleware, trusted_proxies="*")

@app.get("/reset")
async def reset_get():
    return {"status": "ok", "message": "Reset endpoint is active. Use POST to reset."}

@app.get("/")
async def root():
    return {"status": "ok", "message": "AquaSAR-Env OpenEnv Server"}

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    logger.info(f"Headers: {request.headers}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

from fastapi.middleware.cors import CORSMiddleware

# Enable CORS for the submission bot and other external access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # Must be False when allow_origins is ["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.
    """
    import uvicorn
    import os
    
    # Allow port override by environment variable
    port = int(os.getenv("PORT", port))
    
    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    main()
