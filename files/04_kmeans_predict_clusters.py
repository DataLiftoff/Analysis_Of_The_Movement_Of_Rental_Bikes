##### Import Libraries

# To store the model
import pickle

# To compute the clusters
from sklearn.cluster import KMeans

# To connect to databases
from helperFunctions import connectMySQL
from helperFunctions import containerWait
from helperFunctions import managementUpdate
from helperFunctions import managementCompare



##### Setup Program

# Path for the plot
path_model = 'build/KMeans_Model.p'

# Size off the data chunk
size = 100000



##### Run Program

# Wait for the MySQL server to start
containerWait()


try:

  # Connect to MySQL and get cursor
  db_mysql, cur1 = connectMySQL(buffered=True)


  # Check timestamps
  if managementCompare('last_location_insert', 'last_predict_clusters', cur=cur1, db_mysql=db_mysql) or managementCompare('last_kmeans_model', 'last_predict_clusters', cur=cur1, db_mysql=db_mysql):


    # Get second cursor
    cur2 = db_mysql.cursor()

    # Load the kmeans model
    kmeanModel = pickle.load(open(path_model, 'rb'))



    ##### Get All Datapoints Tp Predict Clusters

    # Select statement
    sql_query = 'SELECT id, bike_latitude, bike_longitude FROM bike_locations'

    # Execute the data query
    cur1.execute(sql_query)


    # Iterate over all data chunks
    while True:
      X = cur1.fetchmany(size)

      # Check if data has been returned
      if not X:
        break


      # Split the id and the data
      id, X = zip(*[[i[0], i[1:]] for i in X])

      # Build and fit the model 
      y = kmeanModel.predict(X).tolist()

      # Combine id and prediction for database update
      data = list(zip(list(y), id))


      # SQL statement for update
      sql_query = 'UPDATE bike_locations SET cluster_id=%s WHERE id=%s'

      cur2.executemany(sql_query, data)
      db_mysql.commit()


    # Update timestamp
    managementUpdate('last_predict_clusters', cur=cur1, db_mysql=db_mysql)



except mysql.connector.Error as error:
  print(error)

finally:
  if (db_mysql.is_connected()):
    db_mysql.close()
    print('Database has been closed.')
