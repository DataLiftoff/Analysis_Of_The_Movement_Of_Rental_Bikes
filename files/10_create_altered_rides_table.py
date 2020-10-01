##### Import Libraries

# To load OpenStreetMap-data
import osmnx as ox

# To work with networks
import networkx as nx

# To connect to databases
from helperFunctions import Graph
from helperFunctions import alterRoute
from helperFunctions import connectZEO
from helperFunctions import connectMySQL
from helperFunctions import connectMongo
from helperFunctions import containerWait
from helperFunctions import managementUpdate


##### Run Program

# Wait for the MySQL server to start
containerWait()


try:

  # Connect to ZODB
  root = connectZEO()

  # Connect to MongoDb
  cluster, db_mongo, collection_altered = connectMongo(collection='altered')
  collection_shortest = db_mongo['shortest']

  # Connect to MySQL and get cursor
  db_mysql, cur = connectMySQL()


  ##### Iterate Over All Cities And Alter The Path 

  # Iterate over all cities
  while True:

    # SQL query
    sql_query = 'SELECT cluster_id FROM city_clusters WHERE city_analyse=1 AND last_osm_graph>last_rides_altered ORDER BY RAND() LIMIT 1'

    # Execute the query
    cur.execute(sql_query)
    cluster_id = cur.fetchall()

    # Check if clusters are left
    if not cluster_id:
      break

    # Extract data
    cluster_id = cluster_id[0][0]

    # Update timestamp for city
    managementUpdate('last_rides_altered', cur, db_mysql, table='city_clusters', row='cluster_id', id=cluster_id)


    # Load the graph from the ZODB
    graph = root[cluster_id].graph


    
    # Iterate over all shortest rides
    for item in collection_shortest.find({'Cluster_Id':cluster_id}):
        
        # Create key for dictionary
        key = item['_id']


        # Get precomputed altered paths
        precomputed_altered = collection_altered.find_one({'_id':key})

        # Stop to much computation
        if precomputed_altered and item['Count']==precomputed_altered['Count']:
          continue
        
        # Check if ride has to be recomputed
        if precomputed_altered:
            
            # Copy precomputed ride to rides
            rides = precomputed_altered['Rides']
            lengths = precomputed_altered['Lengths']
            count = precomputed_altered['Count']

        else:
            # Set default values
            rides = []
            lengths = []
            count = 0


        # Compute the number of missing rides
        n_missing_rides = item['Count'] - count


        # Create an altered ride for each count
        for _ in range(n_missing_rides):
            
          # Alter and save each ride
          ride_altered = alterRoute(item['Ride'], graph, p=0.2)
            
          # Compute ride length
          ride_length = sum(ox.utils_graph.get_route_edge_attributes(graph, ride_altered, attribute='length'))


          # Store the altered rides
          rides.append(list(map(int, ride_altered)))
          lengths.append(ride_length)
          count += 1


        # Update the altered rides in the database
        collection_altered.update_one({'_id':key}, {'$set':{'_id':key, 'Rides':rides, 'Count':count, 'Lengths':lengths, 'Cluster_Id':cluster_id}}, upsert=True)



except mysql.connector.Error as error:
  print(error)

finally:
  if (db_mysql.is_connected()):
    db_mysql.close()
    print('Database has been closed.')
