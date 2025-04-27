# Use Python 3.11 as base image
FROM python:3.11-slim

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
source /app/.venv/bin/activate\n\
exec /app/.venv/bin/streamlit run app.py --server.port=8501 --server.address=0.0.0.0' > /app/run.sh && \
    chmod +x /app/run.sh

# Expose the port Streamlit runs on
EXPOSE 8501

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=development
ENV LANGCHAIN_TRACING_V2=false
ENV PATH="/app/.venv/bin:$PATH"

# Command to run the application
CMD ["/app/run.sh"] 