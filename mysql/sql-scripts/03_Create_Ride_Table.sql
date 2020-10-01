CREATE TABLE bike_rides (
id INT NOT NULL AUTO_INCREMENT,
bike_id INT,
bike_latitude FLOAT,
bike_longitude FLOAT,
bike_time INT,
cluster_id SMALLINT, 
bike_latitude_next FLOAT,
bike_longitude_next FLOAT,
cluster_id_next SMALLINT,
ride_duration INT,
bee_line FLOAT,
bee_speed FLOAT,
distance_center_start FLOAT,
distance_center_end FLOAT,
PRIMARY KEY (id)
);
