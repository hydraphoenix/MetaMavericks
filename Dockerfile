FROM python:3.10-slim

# Install system dependencies for image processing and git
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create user with UID 1000 (Strictly required for Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user

# Set environment variables
ENV PATH="/home/user/.local/bin:$PATH"
ENV HOME="/home/user"
ENV PYTHONPATH="$HOME/app"

WORKDIR $HOME/app

# Copy files with correct ownership
COPY --chown=user . $HOME/app

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    "openenv-core>=0.2.2" \
    numpy \
    matplotlib \
    imageio \
    gymnasium \
    stable-baselines3 \
    openai \
    torch

# Install the local environment package in editable mode so it's recognized as a package
RUN pip install -e ./metamavericks_env

# Hugging Face Spaces listens on port 7860
EXPOSE 7860

# Correct CMD for the OpenEnv server in this project.
# We run the server module from the root to ensure all relative imports work.
# We use --log-level debug to help diagnose deployment issues.
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--log-level", "debug"]
