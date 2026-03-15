# Use official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Non-root user setup required by Hugging Face Spaces
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy requirements and install them
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY --chown=user . .

# Train the ML model at build time (no external data/network needed)
RUN python backend/build_model.py

# Expose the standard Hugging Face port
EXPOSE 7860

# Run the FastAPI server
CMD ["uvicorn", "backend.server:app", "--host", "0.0.0.0", "--port", "7860"]
