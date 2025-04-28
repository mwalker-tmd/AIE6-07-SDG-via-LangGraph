# Use Python 3.11 as base image
FROM python:3.11-slim

# Add build argument for version tracking
ARG BUILD_VERSION=1.0.0
ENV BUILD_VERSION=${BUILD_VERSION}

# Set working directory
WORKDIR /app

# Install UV
RUN pip install uv

# Copy pyproject.toml and install dependencies
COPY pyproject.toml .
RUN uv venv && \
    . .venv/bin/activate && \
    uv pip install -e .

# Copy the application code
COPY preprocess/ preprocess/
COPY graph/ graph/
COPY app.py .

# Create necessary directories
RUN mkdir -p generated data

# Copy data after creating directory
COPY data/ data/

# Create a shell script to run the application
RUN echo '#!/bin/bash\n\
echo "Starting application version ${BUILD_VERSION}"\n\
source /app/.venv/bin/activate\n\
PORT=${PORT:-8501}\n\
exec /app/.venv/bin/streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0' > /app/run.sh && \
    chmod +x /app/run.sh

# Expose the default port Streamlit runs on
EXPOSE ${PORT:-8501}

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=development
ENV LANGCHAIN_TRACING_V2=false
ENV PATH="/app/.venv/bin:$PATH"
ENV PORT=8501

# Command to run the application
CMD ["/app/run.sh"] 