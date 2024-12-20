# Use an official Python runtime as a base image
FROM python:3.6

# Expose the port on which the Flask app will run
EXPOSE 5000

# Set the working directory for the app
WORKDIR /app

# Copy the requirements file and install the dependencies
COPY requirements.txt /app
RUN pip install -r requirements.txt

# Copy the Flask app and set the command to run the app
COPY app.py /app
CMD python app.py
