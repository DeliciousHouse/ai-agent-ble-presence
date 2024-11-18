FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a new user 'bkam' and set permissions
RUN useradd -m bkam

# Set the working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade pip

# Copy the rest of the application code
COPY . .

# Change ownership of the /app directory to 'bkam'
RUN chown -R bkam:bkam /app

# Switch to the new user
USER bkam

# Not required if not using ports, but needed for healthcheck if using HTTP
EXPOSE 8000

# Run the main.py script
CMD ["python", "main.py"]
