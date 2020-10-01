##### Import Libraries

# To store data
import pandas as pd

# To connect to databases
from helperFunctions import connectMySQL
from helperFunctions import containerWait
from helperFunctions import managementUpdate
from helperFunctions import managementCompare
from helperFunctions import computeSphereDistance


##### Setup Program

# Variables to filter for
min_beeline=50
max_beeline=20000
min_speed=0.5
max_speed=10
min_duration=150
max_duration=18000
max_distance = 10000

# Chunk size for input to database
size = 100000



##### Run Program

# Wait for the MySQL server to start
containerWait()


try:

  # Connect to MySQL and get cursor
  db_mysql, cur = connectMySQL()


  # Check timestamps
  if managementCompare('last_cluster_table', 'last_rides_table', cur=cur, db_mysql=db_mysql):


    ##### Get The Point And City Dataframes

    # SQL query
    sql_query = 'SHOW columns FROM city_clusters'

    # Get column names
    cur.execute(sql_query)
    columns = [i[0] for i in cur.fetchall()]


    # SQL query
    sql_query = 'SELECT * FROM city_clusters'

    # Get dataframe
    cur.execute(sql_query)
    df_cities = pd.DataFrame(cur.fetchall(), columns=columns)


    # SQL query
    sql_query = 'SHOW columns FROM bike_locations'

    # Get column names
    cur.execute(sql_query)
    columns = [i[0] for i in cur.fetchall()]


    # SQL query
    sql_query = 'SELECT * FROM bike_locations'

    # Get dataframe
    cur.execute(sql_query)
    df_points = pd.DataFrame(cur.fetchall(), columns=columns)



    ##### Generate The Rides Dataframe

    # Sort the dataframe to get sequential locations
    df_rides = df_points.sort_values(['bike_id', 'bike_time']).reset_index(drop=True)

    # Combine the current and the next location into one row
    df_rides = pd.concat([df_rides, df_rides.shift(-1).rename(columns={i:i+'_next' for i in df_rides.columns})], axis=1).dropna()

    # Set correct type for column
    df_rides['cluster_id_next'] = df_rides['cluster_id_next'].astype('int')

    # Filter for different locations for the same bike
    df_rides = df_rides[((df_rides['bike_latitude']!=df_rides['bike_latitude_next']) | (df_rides['bike_longitude']!=df_rides['bike_longitude_next'])) & (df_rides['bike_id']==df_rides['bike_id_next'])].drop(['bike_id_next'], axis=1)

    # Compute the duration of the location change
    df_rides['duration'] = df_rides['bike_time_next'] - df_rides['bike_time']

    # Drop unused columns
    df_rides.drop('bike_time_next', axis=1, inplace=True)

    # Filter for ride duration
    df_rides = df_rides[df_rides['duration'].between(min_duration, max_duration)]

    # Compute the beeline between the sequential locations
    df_rides['bee_line'] = computeSphereDistance(df_rides[['bike_latitude', 'bike_longitude', 'bike_latitude_next', 'bike_longitude_next']].values)

    # Compute the speed of bikes
    df_rides['bee_speed'] = df_rides['bee_line'] / df_rides['duration']

    # Filter rides by length of beeline, speed and duration
    df_rides = df_rides[df_rides['bee_line'].between(min_beeline, max_beeline) & df_rides['bee_speed'].between(min_speed, max_speed)]

    # Join rides and center coordinates
    df_tmp = pd.merge(df_rides[['bike_latitude', 'bike_longitude', 'bike_latitude_next', 'bike_longitude_next', 'cluster_id', 'cluster_id_next']], df_cities[['cluster_id', 'lat', 'lng']], on='cluster_id').drop('cluster_id', axis=1)
    df_tmp = pd.merge(df_tmp, df_cities[['cluster_id', 'lat', 'lng']], left_on='cluster_id_next', right_on='cluster_id').drop('cluster_id', axis=1)

    # Compute start and end distance to center
    df_rides['distance_center_start'] = computeSphereDistance(df_tmp[['bike_latitude', 'bike_longitude', 'lat_x', 'lng_x']].values)
    df_rides['distance_center_end'] = computeSphereDistance(df_tmp[['bike_latitude_next', 'bike_longitude_next', 'lat_y', 'lng_y']].values)

    # Filter rides by distance of start- and end-point to center
    df_rides = df_rides[(df_rides['distance_center_start']<max_distance) | (df_rides['distance_center_end']<max_distance)]

    # Drop unused columns
    df_rides.drop(['id', 'id_next'], axis=1, inplace=True)



    ##### Insert Data To Table

    # SQL statement for insert
    sql_query = 'INSERT IGNORE INTO bike_rides (bike_id, bike_latitude, bike_longitude, bike_time, cluster_id, bike_latitude_next, bike_longitude_next, cluster_id_next, ride_duration, bee_line, bee_speed, distance_center_start, distance_center_end) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'

    # Iterate over the dataframe
    i = 0
    while True:

      data = df_rides.values[i*size:i*size+size]
      i += 1

      # Check for dataframe end
      if not data.any():
        break

      cur.executemany(sql_query, data.tolist())
      db_mysql.commit()


    # Update timestamp
    managementUpdate('last_rides_table', cur=cur, db_mysql=db_mysql)



except mysql.connector.Error as error:
  print(error)


finally:
  if (db_mysql.is_connected()):
    db_mysql.close()
    print('Database has been closed.')
