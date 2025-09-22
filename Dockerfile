# Use official lightweight Python image
FROM python:3.12-slim

# Set working directory in the container
WORKDIR /app

# Copy local code to the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn mlflow pandas pydantic xgboost scikit-learn==1.4.2 datetime pyspark

# Expose port 8000 for the FastAPI app
EXPOSE 8000

# Command to run the FastAPI app with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
