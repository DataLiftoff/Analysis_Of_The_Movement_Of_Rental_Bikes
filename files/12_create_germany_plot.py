##### Import Libraries

# To walk in directories
import os

# To store data
import pandas as pd

# To create interactive plots
import plotly.express as px

# To connect to databases
from helperFunctions import connectMySQL
from helperFunctions import containerWait
from helperFunctions import plotRawDataTable


##### Setup

folder= 'build'


##### Run Program

# Wait for the MySQL server to start
containerWait()


try:

  # Connect to MySQL and get cursor
  db_mysql, cur = connectMySQL()


  ##### Create Plot

  # Get data
  sql_query = '''
  SELECT * 
  FROM city_clusters
  WHERE city_analyse=1
  AND code="DE"'''

  # Execute the query
  cur.execute(sql_query)
  data = cur.fetchall()

  # Get columns
  sql_query = '''SHOW COLUMNS FROM city_clusters'''
  cur.execute(sql_query)
  columns = [col[0].capitalize() for col in cur.fetchall()]

  # Create dataframe
  df_cities = pd.DataFrame(data, columns=columns)
  df_cities.rename({'N_rides':'Trips'}, inplace=True, axis=1)



  # Plot cities on interactive map
  fig = px.scatter_geo(df_cities, 
                       lat='Lat', 
                       lon='Lng',
                       size='Trips',
                       text='City',
                       center={'lat': 51.1657, 'lon': 10.4515},
                       width=1000, 
                       height=800, 
                       scope='europe',
                       title='Which Cities Can Be Analysed?')
  fig.update_layout(geo={'projection':{'scale':6}},
                    font_family = 'Courier New',
                    font_color = '#003f6e')

  # Save the plot
  path = os.path.join(folder, 'Trips_In_German_Cities.html')
  fig.write_html(path)


  # Plot raw data table
  plotRawDataTable(folder)

    



except mysql.connector.Error as error:
  print(error)

finally:
  if (db_mysql.is_connected()):
    db_mysql.close()
    print('Database has been closed.')
