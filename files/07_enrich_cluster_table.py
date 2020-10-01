##### Import Libraries

# To store data
import pandas as pd

# To connect to databases
from helperFunctions import connectMySQL
from helperFunctions import containerWait
from helperFunctions import managementUpdate
from helperFunctions import managementCompare


##### Setup Program

# Number of rides for cities to analyse
n_rides = 5000



##### Run Program

# Wait for the MySQL server to start
containerWait()


try:

  # Connect to MySQL and get cursor
  db_mysql, cur = connectMySQL()


  # Check timestamps
  if managementCompare('last_rides_table', 'last_enrich_cluster_table', cur=cur, db_mysql=db_mysql):

    ##### Get Aggregated Data From Database

    # SQL query
    sql_query = 'SELECT cluster_id, COUNT(bike_id), AVG(ride_duration), STDDEV(ride_duration), AVG(bee_line), STDDEV(bee_line), AVG(bee_speed), STDDEV(bee_speed) FROM bike_rides GROUP BY cluster_id'

    # Execute the query
    cur.execute(sql_query)
    data = cur.fetchall()

    # Create dataframe
    df_data = pd.DataFrame(data, columns=['cluster_id', 'n_rides', 'duration_mean', 'duration_std', 'bee_line_mean', 'bee_line_std', 'bee_speed_mean', 'bee_speed_std'])


    ##### Update Data In Table

    # SQL statement for update
    sql_query = 'UPDATE city_clusters SET n_rides=%s, duration_mean=%s, duration_std=%s, bee_line_mean=%s, bee_line_std=%s, bee_speed_mean=%s, bee_speed_std=%s WHERE cluster_id=%s'

    cur.executemany(sql_query, df_data[['n_rides', 'duration_mean', 'duration_std', 'bee_line_mean', 'bee_line_std', 'bee_speed_mean', 'bee_speed_std', 'cluster_id']].values.tolist())
    db_mysql.commit()


    ##### Update Cities To Analyse

    # SQL statement for update
    sql_query = 'UPDATE city_clusters SET city_analyse=1 WHERE n_rides>{}'.format(n_rides)

    cur.execute(sql_query)
    db_mysql.commit()


    # Update timestamp
    managementUpdate('last_enrich_cluster_table', cur=cur, db_mysql=db_mysql)


except mysql.connector.Error as error:
  print(error)


finally:
  if (db_mysql.is_connected()):
    db_mysql.close()
    print('Database has been closed.')
