##### Import Libraries

# To search directories
import os

# To connect to databases
from helperFunctions import connectMySQL
from helperFunctions import containerWait
from helperFunctions import managementUpdate



##### Setup Program

# Folder for the data files to load
folder_data = 'data'



##### Run Program

# Wait for the MySQL server to start
containerWait(sleep=20)

try:

  # Connect to MySQL and get cursor
  db_mysql, cur = connectMySQL()



  ##### Check If Update Is Needed

  # Get last insert-timestamp
  sql_query = 'SELECT last_location_insert FROM management WHERE id=1'
  cur.execute(sql_query)
  time_insert = cur.fetchone()

  # Get last file-timestamp
  time_files = max([os.path.getatime(os.path.join(folder_data, file)) for file in os.listdir(folder_data)])


  # Gather rows for insert
  data = []


  # Check for update
  if not time_insert or time_files>time_insert[0]:

    ##### Get Inserted Files

    # List all data files and their recording-times
    files = [[os.path.join(folder_data, file), int(file.split('.')[0].split('_')[-1])] for file in os.listdir(folder_data)]



    ##### Insert New Files

    # Insert statement
    sql_query = "INSERT IGNORE INTO bike_locations (bike_id, bike_latitude, bike_longitude, bike_time) VALUES (%s,%s, %s, %s)"

    # Get all unique times in database
    cur.execute('SELECT DISTINCT bike_time FROM bike_locations')
    response = [val[0] for val in cur.fetchall()]


    # Iterate over all files
    for file, bike_time in files:

      # Check if file has been added to the database
      if int(bike_time) in response:
        continue

      # Open the file
      with open(file) as f:

        # Iterate over all lines
        for line in f.readlines()[1:]:

          # Clean and split data
          bike_number, bike_latitude, bike_longitude = line.strip().split(',')

          # Filter for correct coordinates 
          if bike_latitude and bike_longitude and (-90<float(bike_latitude)) and (float(bike_latitude)<90) and (-180<float(bike_longitude)) and (float(bike_longitude)<180):

            # Store the data
            data.append([bike_number, bike_latitude, bike_longitude, bike_time])


      # Check if enough rows have been gathered
      if len(data) > 50000:

        # Insert data row into database
        cur.executemany(sql_query, data)
        db_mysql.commit()
        data = []

    # Update timestamp
    managementUpdate('last_location_insert', cur=cur, db_mysql=db_mysql)


except mysql.connector.Error as error:
  print(error)


finally:
  # Insert the remaining rows
  if len(data)>0:
    cur.executemany(sql_query, data)
    db_mysql.commit()

  if (db_mysql.is_connected()):
    db_mysql.close()
    print('Database has been closed.')
