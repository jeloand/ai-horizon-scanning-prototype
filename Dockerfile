FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source package and install it
COPY src/ ./src/
RUN pip install -e ./src

# Download spaCy model (required for stakeholder NER)
RUN python -m spacy download en_core_web_sm

# Copy the entire application
COPY . .

# Create necessary directories
RUN mkdir -p /app/data

# Expose Streamlit default port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Set environment variables for production
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Run streamlit
CMD ["streamlit", "run", "streamlit/app_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
