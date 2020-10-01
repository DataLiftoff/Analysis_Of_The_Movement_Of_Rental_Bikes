CREATE TABLE city_clusters (
id INT NOT NULL AUTO_INCREMENT,
name VARCHAR(50),
code VARCHAR(2),
city VARCHAR(50),
country VARCHAR(50),
lat FLOAT, 
lng FLOAT,
cluster_id SMALLINT,
timezone VARCHAR(50) DEFAULT NULL,
n_bikes INT,
lat_mean FLOAT,
lat_std FLOAT,
lng_mean FLOAT,
lng_std FLOAT,
n_rides INT DEFAULT NULL,
duration_mean FLOAT DEFAULT NULL,
duration_std FLOAT DEFAULT NULL,
bee_line_mean FLOAT DEFAULT NULL,
bee_line_std FLOAT DEFAULT NULL,
bee_speed_mean FLOAT DEFAULT NULL,
bee_speed_std FLOAT DEFAULT NULL,
radius FLOAT DEFAULT NULL,
city_analyse BOOLEAN DEFAULT 0,
last_osm_graph INT DEFAULT 0,
last_rides_shortest INT DEFAULT 0,
last_city_plots INT DEFAULT 0,
PRIMARY KEY (id)
);