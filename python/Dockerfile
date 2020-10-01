# Base image for the container
FROM python:3.7

# Copy the file to the container
COPY requirements.txt .

RUN apt-get update -y
RUN apt-get install -y libspatialindex-dev

# Use pip to install the requirements from the file
RUN pip install -r requirements.txt
