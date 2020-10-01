CREATE TABLE bike_locations (
id INT NOT NULL AUTO_INCREMENT,
bike_id INT,
bike_latitude  FLOAT,
bike_longitude FLOAT,
bike_time INT,
cluster_id SMALLINT DEFAULT NULL,
PRIMARY KEY (id)
);
