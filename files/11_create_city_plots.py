##### Import Libraries

# To work with OSM
import osmnx as ox

# To connect to databases
from helperFunctions import Graph
from helperFunctions import connectZEO
from helperFunctions import createFolder
from helperFunctions import connectMySQL
from helperFunctions import connectMongo
from helperFunctions import getAllCities
from helperFunctions import getCityRides
from helperFunctions import plotGraphOSM
from helperFunctions import containerWait
from helperFunctions import plotHeatmapOSM
from helperFunctions import managementUpdate
from helperFunctions import plotTripsPerCity
from helperFunctions import plotTripsPerWeekday
from helperFunctions import plotTripsDistribution
from helperFunctions import plotHeatmapAlteredOSM
from helperFunctions import plotHeatmapAlteredSpeedOSM

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
    sql_query = 'SELECT cluster_id FROM city_clusters WHERE city_analyse=1 AND last_rides_altered>last_city_plots AND code="DE" ORDER BY RAND() LIMIT 1'

    # Execute the query
    cur.execute(sql_query)
    cluster_id = cur.fetchall()

    # Check if clusters are left
    if not cluster_id:
      break

    # Extract data
    cluster_id = cluster_id[0][0]

    # Update timestamp for city
    managementUpdate('last_city_plots', cur, db_mysql, table='city_clusters', row='cluster_id', id=cluster_id)


    ##### Prepare The Folder
    
    folder = 'build'
    createFolder(cluster_id, folder)


    ##### Get The Data

    # Get the city data
    df_cities, df_city, city = getAllCities(cur, cluster_id)

    # Filter for german cities
    df_cities = df_cities[df_cities['code']=='DE']

    # Get the rides data
    df_rides = getCityRides(cur, cluster_id, df_city['timezone'].values[0])


    ##### Create Plots

    # Plot trips per city
    plotTripsPerCity(df_cities, folder, cluster_id, city)

    # Plot trips per day
    plotTripsDistribution(df_rides, folder, cluster_id, city)

    # Plot trips per weekday
    plotTripsPerWeekday(df_rides, folder, cluster_id, city)


    ##### Plot OSM

    # Load the graph from the ZODB
    graph = root[cluster_id].graph

    # Plot OSM graph
    plotGraphOSM(graph, df_rides, city, cluster_id, folder)

    # Plot heatmap on OSM
    plotHeatmapOSM(graph, collection_shortest, cluster_id, city, folder)

    # Plot heatmap altered on OSM
    plotHeatmapAlteredOSM(graph, collection_altered, cluster_id, city, folder)

    # Plot heatmap of speed on OSM
    plotHeatmapAlteredSpeedOSM(graph, df_rides, collection_altered, cluster_id, city, folder)
    



except mysql.connector.Error as error:
  print(error)

finally:
  if (db_mysql.is_connected()):
    db_mysql.close()
    print('Database has been closed.')
