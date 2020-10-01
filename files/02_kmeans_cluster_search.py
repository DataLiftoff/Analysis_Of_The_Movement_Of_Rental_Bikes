##### Import Libraries

# To get random indices
from numpy.random import choice

# To create plots
import matplotlib.pyplot as plt

# To compute the clusters
from sklearn.cluster import KMeans

# To connect to databases
from helperFunctions import connectMySQL
from helperFunctions import containerWait
from helperFunctions import managementUpdate


##### Setup Program

# Path for the plot
path_plot = 'build/KMeans_Elbow_Cluster_Search.pdf'

# Number of datapoints for the KMeans
size = 250000

# List with number of clusters to test
search_space = list(range(40, 151, 5))



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


  ##### Perform Cluster Search

  # Create empty variable for MSE
  mse = []

  # Iterate over the search space
  for k in search_space:

    # Build and fit the model 
    kmeanModel = KMeans(n_clusters=k, n_jobs=4) 
    kmeanModel.fit(X)

    # Get and save mse
    mse.append(kmeanModel.inertia_)


  ##### Plot Cluster Search MSE Results

  plt.figure(figsize=(15, 6))
  plt.plot(search_space, mse, 'bx-')

  plt.title('KMeans Elbow Method For The Clusters With {} Datapoints'.format(len(X)))
  plt.xlabel('Number Of Clusters') 
  plt.ylabel('MSE / Inertia')

  plt.yscale('log')
        
  plt.tight_layout()
  plt.savefig(path_plot)

  # Update timestamp
  managementUpdate('last_cluster_search', cur=cur, db_mysql=db_mysql)


except mysql.connector.Error as error:
  print(error)


finally:
  if (db_mysql.is_connected()):
    db_mysql.close()
    print('Database has been closed.')
