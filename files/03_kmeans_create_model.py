##### Import Libraries

# To store the model
import pickle

# To get random indices
from numpy.random import choice

# To compute the clusters
from sklearn.cluster import KMeans

# To connect to databases
from helperFunctions import connectMySQL
from helperFunctions import containerWait
from helperFunctions import managementUpdate


##### Setup Program

# Path for the plot
path_model = 'build/KMeans_Model.p'

# Number of datapoints for the KMeans
size = 1000000

# Number of clusters
number_of_clusters = 130



##### Run Program

# Wait for the MySQL server to start
containerWait()


try:

  # Connect to MySQL and get cursor
  db_mysql, cur = connectMySQL()



  ##### Get Highest Index

  # Select statement
  sql_query = 'SELECT MAX(id) FROM bike_locations'

  # Get highest index in database
  cur.execute(sql_query)
  max_index = cur.fetchone()[0]



  ##### Generate Random Indices For KMeans

  # Choose indices for KMeans
  random_indices = choice(max_index, size=size, replace=False) +1



  ##### Get Datapoints For KMeans

  # Select statement
  sql_query = 'SELECT bike_latitude, bike_longitude FROM bike_locations WHERE id IN {}'.format(tuple(random_indices))

  # Get index datapoints
  cur.execute(sql_query)
  X = cur.fetchall()



  ##### Compute And Store KMeans Model

  # Build and fit the model 
  kmeanModel = KMeans(n_clusters=number_of_clusters, n_jobs=4) 
  kmeanModel.fit(X)

  # Store the model
  pickle.dump(kmeanModel, open(path_model, 'wb'))

  # Update timestamp
  managementUpdate('last_kmeans_model', cur=cur, db_mysql=db_mysql)


except mysql.connector.Error as error:
  print(error)


finally:
  if (db_mysql.is_connected()):
    db_mysql.close()
    print('Database has been closed.')
