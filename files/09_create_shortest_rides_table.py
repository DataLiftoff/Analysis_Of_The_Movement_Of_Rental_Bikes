##### Import Libraries

# To do linear algebra
import numpy as np

# To load OpenStreetMap-data
import osmnx as ox

# To store data
import pandas as pd

# To work with networks
import networkx as nx

# To count objects
from collections import Counter

# To search efficiently
from sklearn.neighbors import KDTree

# To connect to databases
from helperFunctions import Graph
from helperFunctions import connectZEO
from helperFunctions import connectMySQL
from helperFunctions import connectMongo
from helperFunctions import containerWait
from helperFunctions import managementUpdate
from helperFunctions import managementCompare


##### Run Program

# Wait for the MySQL server to start
containerWait()


try:

  # Connect to ZODB
  root = connectZEO()

  # Connect to MongoDb
  cluster, db_mongo, collection = connectMongo(collection='shortest')

  # Connect to MySQL and get cursor
  db_mysql, cur = connectMySQL()


  ##### Iterate Over All Cities And Download OSM-Graph

  # Iterate over all cities
  while True:

    # SQL query
    sql_query = 'SELECT cluster_id FROM city_clusters WHERE city_analyse=1 AND last_osm_graph>last_rides_shortest ORDER BY RAND() LIMIT 1'

    # Execute the query
    cur.execute(sql_query)
    cluster_id = cur.fetchall()

    # Check if clusters are left
    if not cluster_id:
      break

    # Extract data
    cluster_id = cluster_id[0][0]

    # Update timestamp for city
    managementUpdate('last_rides_shortest', cur, db_mysql, table='city_clusters', row='cluster_id', id=cluster_id)


    # SQL query
    sql_query = 'SELECT bike_latitude, bike_longitude, bike_latitude_next, bike_longitude_next FROM bike_rides WHERE cluster_id={}'.format(cluster_id)

    # Execute the query
    cur.execute(sql_query)
    data = cur.fetchall()

    df_city = pd.DataFrame(data, columns=[['Lat', 'Lng', 'Lat_next', 'Lng_next']])


    # Load the graph from the ZODB
    graph = root[cluster_id].graph



    ##### Compute Shortest Rides On OSM

    # Get all nodes and edges of the street network
    nodes, edges = ox.graph_to_gdfs(graph)

    # Create a k-dimensional tree of the position of the nodes
    tree = KDTree(nodes[['y', 'x']], 
                  metric='euclidean')

    # Get the closest point to the start- and end-point in the graph 
    start_idx = tree.query(df_city[['Lat', 'Lng']], k=1, return_distance=False)
    end_idx = tree.query(df_city[['Lat_next', 'Lng_next']], k=1, return_distance=False)

    # Flatten start- and end-idx
    start_idx = np.array([i[0] for i in start_idx])
    end_idx = np.array([i[0] for i in end_idx])

    # Create mask for different start/end points
    mask = start_idx != end_idx


    # Get the closest nodes
    closest_node_to_start = nodes.iloc[start_idx[mask]].index.values
    closest_node_to_end = nodes.iloc[end_idx[mask]].index.values


    # Count occurences of rides
    counter_rides = Counter(zip(closest_node_to_start, closest_node_to_end))

    
    # Iterate over all start/end point pairs
    for (start, end), count in counter_rides.items():
        
        # Create key for dictionary
        key = str((start, end))


        # Get precomputed shortest paths
        precomputed_shortest = collection.find_one({'_id':key})
        
        
        # Check if ride has to be recomputed
        if precomputed_shortest:
            
            # Copy precomputed ride to rides
            collection.update_one({'_id':key}, {'$set':{'_id':key, 'Ride':list(map(int, precomputed_shortest['Ride'])), 'Count':count, 'Length':precomputed_shortest['Length'], 'Cluster_Id':cluster_id}}, upsert=True)
            
        else:
            try:
                # Compute the shortest ride
                ride = nx.shortest_path(graph, start, end, weight='length')
                
                # Compute ride length
                ride_length = sum(ox.utils_graph.get_route_edge_attributes(graph, ride, attribute='length'))

                # Copy newcomputed ride to rides
                collection.update_one({'_id':key}, {'$set':{'_id':key, 'Ride':list(map(int, ride)), 'Count':count, 'Length':ride_length, 'Cluster_Id':cluster_id}}, upsert=True)

            except:
                pass



except mysql.connector.Error as error:
  print(error)

finally:
  if (db_mysql.is_connected()):
    db_mysql.close()
    print('Database has been closed.')
