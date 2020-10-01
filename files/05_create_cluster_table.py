##### Import Libraries

# To store the model
import pickle

# To store data
import pandas as pd

# To compute the clusters
from sklearn.cluster import KMeans

# To convert lat/lng to time zones
from timezonefinder import TimezoneFinder

# To connect to databases
from helperFunctions import connectMySQL
from helperFunctions import containerWait
from helperFunctions import managementUpdate
from helperFunctions import managementCompare
from helperFunctions import mapClustersToCities


##### Setup Program

# Path for the plot
path_model = 'build/KMeans_Model.p'



##### Run Program

# Wait for the MySQL server to start
containerWait()


try:

  # Connect to MySQL and get cursor
  db_mysql, cur = connectMySQL()


  # Check timestamps
  if managementCompare('last_predict_clusters', 'last_cluster_table', cur=cur, db_mysql=db_mysql):


    # Load the kmeans model
    kmeanModel = pickle.load(open(path_model, 'rb'))

    # Get the centers for the clusters
    kmeanCenters = kmeanModel.cluster_centers_



    ##### Map Clusters To Cities

    # Clusters to city names
    dict_cities = mapClustersToCities(kmeanCenters)

    # Create a dataframe for the cities
    df_cities = pd.DataFrame.from_dict(dict_cities, orient='index')
    
    # Add time zones to cities dataframe
    tf = TimezoneFinder()
    df_cities['Timezone'] = df_cities[['Lat', 'Lng']].apply(lambda x: tf.timezone_at(lat=x[0], lng=x[1]), axis=1)



    ##### Get Aggregated Data From Database

    # SQL query
    sql_query = 'SELECT cluster_id, COUNT(DISTINCT(bike_id)), AVG(bike_latitude), STDDEV(bike_latitude), AVG(bike_longitude), STDDEV(bike_longitude) FROM bike_locations GROUP BY cluster_id'

    # Execute the query
    cur.execute(sql_query)
    data = cur.fetchall()

    # Merge both dataframes
    df_data = pd.DataFrame(data, columns=['Center', 'N_Bikes', 'Lat_Mean', 'Lat_Std', 'Lng_Mean', 'Lng_Std'])
    df_cities = pd.merge(df_cities, df_data, on='Center')



    ##### Insert Data To Table

    # SQL statement for update
    sql_query = 'INSERT IGNORE INTO city_clusters (name, code, city, country, lat, lng, cluster_id, timezone, n_bikes, lat_mean, lat_std, lng_mean, lng_std) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'

    cur.executemany(sql_query, df_cities.values.tolist())
    db_mysql.commit()


    # Update timestamp
    managementUpdate('last_cluster_table', cur=cur, db_mysql=db_mysql)



except mysql.connector.Error as error:
  print(error)


finally:
  if (db_mysql.is_connected()):
    db_mysql.close()
    print('Database has been closed.')
