##### Import Libraries

# To measure time
import time

# To connect to ZODB
import transaction

# To load OpenStreetMap-data
import osmnx as ox

# To connect to databases
from helperFunctions import Graph
from helperFunctions import connectZEO
from helperFunctions import connectMySQL
from helperFunctions import containerWait
from helperFunctions import managementUpdate
from helperFunctions import managementCompare


##### Run Program

# Wait for the MySQL server to start
containerWait()


try:

  # Connect to ZODB
  root = connectZEO()

  # Connect to MySQL and get cursor
  db_mysql, cur = connectMySQL()


  # Check timestamps
  if managementCompare('last_enrich_cluster_table', 'last_osm_download', cur=cur, db_mysql=db_mysql):


    ##### Iterate Over All Cities And Download OSM-Graph

    # SQL query
    sql_query = 'SELECT cluster_id FROM city_clusters WHERE city_analyse=1'

    # Execute the query
    cur.execute(sql_query)
    cluster_ids = cur.fetchall()

    # Iterate over all cities
    for cluster_id in cluster_ids:

      # Extract the data
      cluster_id = cluster_id[0]

      # SQL query
      sql_query = 'SELECT lat, lng FROM city_clusters WHERE cluster_id={}'.format(cluster_id)

      # Execute the query
      cur.execute(sql_query)
      point = cur.fetchone()


      # Download the OSM-graph for the city
      graph = ox.graph_from_point(point,
                                  dist=5000, 
                                  dist_type='bbox', 
                                  network_type='bike', 
                                  simplify=True, 
                                  retain_all=False, 
                                  truncate_by_edge=False,
                                  clean_periphery=True,
                                  custom_filter=None)

      # Store the graph into the ZODB
      root[cluster_id] = Graph(graph)
      transaction.commit()


      # Update timestamp for city
      sql_query = 'UPDATE {} SET {}={} WHERE cluster_id={}'.format('city_clusters', 'last_osm_graph', int(time.time()), cluster_id)
      cur.execute(sql_query)
      db_mysql.commit()


    # Update timestamp
    managementUpdate('last_osm_download', cur=cur, db_mysql=db_mysql)


except mysql.connector.Error as error:
  print(error)


finally:
  if (db_mysql.is_connected()):
    db_mysql.close()
    print('Database has been closed.')
