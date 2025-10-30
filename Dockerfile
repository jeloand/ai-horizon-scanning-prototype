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

# Download spaCy model (required for stakeholder NER)
RUN python -m spacy download en_core_web_sm

# Copy the entire application
COPY . .

# Create necessary directories
RUN mkdir -p /app/data

# Expose Streamlit default port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run streamlit
ENTRYPOINT ["streamlit", "run", "streamlit/app_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
