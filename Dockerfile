# Use an official lightweight Python image
FROM python:3.9-slim

# Set working directory in container
WORKDIR /app

# Copy application files to container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port your app runs on
EXPOSE 8080

# Run the app
CMD ["python", "app.py"]
