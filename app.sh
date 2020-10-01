# Analyse The Data Of Hire Bikes

##### Setup #####

CLUSTER_SEARCH=false;
KMEANS_RECOMPUTE=false;


##### Execute Docker-Compose #####

# Insert The Data From Files To MySQL Database
docker-compose -f docker-compose/01_insert_to_database.yml up --abort-on-container-exit --exit-code-from python --remove-orphans;


# Perform Cluster Search For Optimal Number Of Clusters
if $CLUSTER_SEARCH;
then docker-compose -f docker-compose/02_kmeans_cluster_search.yml up --abort-on-container-exit --exit-code-from python --remove-orphans;
fi;


# Compute A Kmeans Model With Optimal Number Of Clusters
if $KMEANS_RECOMPUTE;
then docker-compose -f docker-compose/03_kmeans_create_model.yml up --abort-on-container-exit --exit-code-from python --remove-orphans;
fi;


# Predict The Cluster For Each Datapoint
docker-compose -f docker-compose/04_kmeans_predict_clusters.yml up --abort-on-container-exit --exit-code-from python --remove-orphans;


# Aggregate Data For The Clusters
docker-compose -f docker-compose/05_create_cluster_table.yml up --abort-on-container-exit --exit-code-from python --remove-orphans;


# Create A Table With Bike Trips Between Two Points
docker-compose -f docker-compose/06_create_rides_table.yml up --abort-on-container-exit --exit-code-from python --remove-orphans;


# Enrich The Clusters Data With Aggregated Trips
docker-compose -f docker-compose/07_enrich_cluster_table.yml up --abort-on-container-exit --exit-code-from python --remove-orphans;


# Download Streetmaps And Store The Graphs In ZEO Database
docker-compose -f docker-compose/08_download_osm_graph.yml up --abort-on-container-exit --exit-code-from python --remove-orphans;


# Compute The Shortest Paths For The Trips On The Streetmaps
docker-compose -f docker-compose/09_create_shortest_rides_table.yml up --detach --scale python=7 --remove-orphans;

sleep 20;
while (docker ps | grep python > /dev/null);
do sleep 10;
done;

docker-compose -f docker-compose/09_create_shortest_rides_table.yml down;


# Randomize The Shortest Paths For The Trips On The Streetmaps
docker-compose -f docker-compose/10_create_altered_rides_table.yml up --detach --scale python=7 --remove-orphans;

sleep 20;
while (docker ps | grep python > /dev/null);
do sleep 10;
done;

docker-compose -f docker-compose/10_create_altered_rides_table.yml down;


# Plot The Data As Timeseries, Barplots And Heatmaps
docker-compose -f docker-compose/11_create_city_plots.yml up --detach --scale python=6 --remove-orphans;

sleep 20;
while (docker ps | grep python > /dev/null);
do sleep 10;
done;

docker-compose -f docker-compose/11_create_city_plots.yml down;


# Plot An Overall Graph For German Cities
docker-compose -f docker-compose/12_create_germany_plot.yml up --abort-on-container-exit --exit-code-from python --remove-orphans;
