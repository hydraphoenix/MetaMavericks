FROM python:3.10-slim

# Install system dependencies for image processing
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Create user with UID 1000 (Strictly required for Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user

# Set environment variables
ENV PATH="/home/user/.local/bin:$PATH"
ENV HOME="/home/user"

WORKDIR $HOME/app

# Copy files with correct ownership
COPY --chown=user . $HOME/app

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn pydantic "openenv-core" numpy matplotlib imageio

EXPOSE 7860

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
