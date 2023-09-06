# syntax=docker/dockerfile:1.2
FROM python:3.9
# put your docker configuration here

# Set the working directory to /app
WORKDIR /app

# Copy the application files into the container
COPY challenge ./challenge
COPY requirements.txt .
COPY modelo_train.pkl .

# Install production dependencies
RUN pip install -r requirements.txt

# Expose the port on which your FastAPI application runs (adjust as needed)
EXPOSE 8080

# Start command to run the application
CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8080"]
