CREATE TABLE management (
id INT NOT NULL AUTO_INCREMENT,
last_location_insert INT DEFAULT 0,
last_cluster_search INT DEFAULT 0,
last_kmeans_model INT DEFAULT 0,
last_predict_clusters INT DEFAULT 0,
last_cluster_table INT DEFAULT 0,
last_rides_table INT DEFAULT 0,
last_enrich_cluster_table INT DEFAULT 0,
last_osm_download INT DEFAULT 0,
PRIMARY KEY (id)
);
