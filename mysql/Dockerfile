# Use this image as base
FROM mysql


##### Setup to create the database

# Name of the database to create
ENV MYSQL_DATABASE bike_movement

# Password for the root user
ENV MYSQL_ROOT_USER root
ENV MYSQL_ROOT_PASSWORD password

# Create a user with a new password
ENV MYSQL_USER user
ENV MYSQL_PASSWORD password


# Add the content of the sql-scripts/ directory to your image
# All scripts in docker-entrypoint-initdb.d/ are automatically
# executed during container startup
COPY ./sql-scripts/ /docker-entrypoint-initdb.d/

