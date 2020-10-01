# To create nice plots
#import seaborn as sns

# To compute KDEs
#import scipy.stats as st

# To create plots
#import matplotlib as mpl

# To create nice plots
#import matplotlib.cm as cm

# To create nice plots
#from matplotlib.colors import LinearSegmentedColormap





## Wait for container start up
def containerWait(sleep=10):
    '''
    Input:
    sleep - time to sleep
    
    Output:
    '''

    # To measure time
    import time

    # Wait for containers to start
    time.sleep(sleep)





## Update Timestamp In Management-Table
def managementUpdate(column, cur, db_mysql, table='management', row='id', id=1):
    '''
    Input:
    column - column to update
    
    Output:
    '''

    # To measure time
    import time

    # Update the timestamp
    sql_query = 'UPDATE {} SET {}={} WHERE {}={}'.format(table, column, int(time.time()), row, id)
    cur.execute(sql_query)
    db_mysql.commit()





## Compare Timestamps In Management-Table
def managementCompare(column1, column2, cur, db_mysql):
    '''
    Input:
    column1 - older column
    column2 - younger column
    
    Output:
    bool - column1 younger than column2
    '''

    # Compare the timestamps
    sql_query = 'SELECT id FROM management WHERE {}>{}'.format(column1, column2)
    cur.execute(sql_query)
    return cur.fetchone()





## Connect To ZEO Database
def connectZEO(host='docker-compose_zeo_1', port=8090):
    '''
    Input:
    host - host server
    port - port to connect to
    
    Output:
    returns database-root
    '''

    # To connect to ZEO
    import ZEO

    # Connect to ZODB
    connection = ZEO.connection((host, port))
    return connection.root()





## Connect To Mongo Database
def connectMongo(collection, host='docker-compose_mongo_1', port=27017):
    '''
    Input:
    host - host server
    port - port to connect to
    
    Output:
    returns database-collection
    '''

    # To connect to MongoDB
    from pymongo import MongoClient

    # Connect to ZODB
    cluster = MongoClient('{}:{}'.format(host, port))
    db_mongo = cluster['rides']
    collection = db_mongo[collection]
    return cluster, db_mongo, collection





## Connect To MySQL Database
def connectMySQL(host='docker-compose_mysql_1', user='morris', password='password', database='bike_movement', buffered=False):
    '''
    Input:
    host - host server
    user - database user
    password - database password
    database - database to use
    buffered- bool for cursor
    
    Output:
    returns database-connection and cursor
    '''
    
    # To connect to databases
    import mysql.connector
    
    # Connect to database
    db = mysql.connector.connect(host=host,
                                 user=user,
                                 password=password,
                                 database=database)
    
    # Get cursor for execution
    cur = db.cursor(buffered=buffered)
    
    return (db, cur)





## Class For Graph Storage
import persistent
class Graph(persistent.Persistent):

    def __init__(self, graph):
        self.graph = graph





## Map Clusters To City Names
def mapClustersToCities(kmeans_centers):
    '''
    Input:
    kmeans_centers - center points for all clusters
    
    Output:
    dictionary with cluster city data
    '''
    
    # To get reverse geocodes
    import reverse_geocode
    
    # Create an empthy dictionary for the cities
    dict_cities = {}
    
    
    # Geocode the cluster center to a city and country name
    geocode_data = reverse_geocode.search(kmeans_centers)

    
    # Iterate over all centers
    for i, (center, (lat, lng)) in enumerate(zip(geocode_data, kmeans_centers)):

        # Get the data
        city = center['city'].replace('/', '_')
        code = center['country_code']
        country = center['country']
        
        # Create city-country name
        name = ', '.join([city, country])

        
        # Add the data to the dictionary
        dict_cities[i] = {'Name':name, 'Code':code, 'City':city, 'Country':country, 'Lat':lat, 'Lng':lng, 'Center':i}
    
    return dict_cities





## Compute distance on sphere
def computeSphereDistance(X, radius_earth=6371009):
    '''
    Input:
    X - 2D array of form [(Lat1, Lng1, Lat2, Lng2)]
    radius_earth - radius of sphere
    
    Output:
    array with distances on sphere (fast)
    '''

    # To do linear algebra
    import numpy as np
    
    # Convert lat/lng to radians
    X_radians = np.radians(X)

    # Compute lat/lng differences
    diff_lat = X_radians[:, 2] - X_radians[:, 0]
    diff_lng = X_radians[:, 3] - X_radians[:, 1]

    # Auxiliary calculations
    a = np.sin(diff_lat/2)**2 + np.cos(X_radians[:, 0]) * np.cos(X_radians[:, 2]) * np.sin(diff_lng/2)**2
    b = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    # Compute and return distance
    return radius_earth * b





## Create Folder For A City
def createFolder(cluster_id, folder):
    '''
    Input:
    cluster_id - id of the current cluster
    folder - folder for the builds
    
    Output:
    creates a folder for the city
    '''

    # To walk in directories
    import os
    
    # Check if folder has been created
    if str(cluster_id) not in os.listdir(folder):

      # Create missing folder
      os.mkdir(os.path.join(folder, str(cluster_id)))
    
    return None





## Combine Clusters To Have Reduced Cities To Analyse
def combineClusters(df_cities, analyse_cities):
    '''
    Input:
    df_cities - dataframewith all cities
    analyse_cities - list of cities and clusters to analyse
    
    Output:
    combines clusters and returns a reduced number of cities to analyse
    '''
    
    # Empty variable to store data
    data = []
    
    
    # Iterate all cities to analyse
    for city, center in analyse_cities:
        
        # Empty variable to store data
        nearby_centers = []
        
        # Get nearby centers 
        nearby = df_cities[df_cities['Center']==center]['Nearby'].values[0]

        # Save the data
        nearby_centers.append(center)
        nearby_centers.extend(nearby)
        data.append(tuple(sorted(nearby_centers)))
    
    
    # Empty variable to store data
    combined_centers = []
    
    # Iterate all all unique center clusters
    for centers in set(data):
        
        # Filter and sort the data
        df_tmp = df_cities[df_cities['Center'].isin(centers)].sort_values('N_Rides', ascending=False)[['Name', 'Center']]
        
        # Save the data
        combined_centers.append(df_tmp.values[0].tolist())
        
    return pd.DataFrame(combined_centers).values





## Compute Nearby Centers For Combining
def computeNearbyCenters(df_cities, max_center_distance=10000):
    '''
    Input:
    df_cities - dataframe with all cities 
    max_center_distance - maximal distance between cluster centern
    
    Output:
    computes nearby clusters and returns list
    '''
    
    # Empty variable 
    data = []
    
    # Iterate over all cities
    for center, lat, lng in df_cities[['Center', 'Lat', 'Lng']].values:
        
        # Set up data for distance computation
        coordinates_center = np.repeat([[lat, lng]], len(df_cities), axis=0)
        coordinates_other = df_cities[['Lat', 'Lng']].values
        
        # Compute distances between city and clusters
        center_distances = computeSphereDistance(np.concatenate([coordinates_center, coordinates_other], axis=1))
        
        # Filter for nearby clusters and remove self
        nearby_centers = df_cities[center_distances<max_center_distance]['Center'].values
        nearby_centers = nearby_centers[nearby_centers!=center].tolist()
        
        # Store data
        data.append(nearby_centers)
    
    return data





## Get All Nearby Clusters By Iteration
def getNearbyClusters(city, df_cities):
    '''
    Input:
    city - current city
    df_cities - dataframe with all cities
    
    Output:
    return list of nearby clusters to filter for
    '''
    
    # Empty variable to store nearby clusters
    data = []
    
    
    # Create queue for iteration
    container = deque()
    
    # Fill queue with initial nearby clusters
    container.extend(df_cities[df_cities['Name']==city]['Center'].values)


    # Iterate over nearby clusters until the queue is empty
    while container:
        
        # Get a new center
        center = container.pop()
        
        # Add center if it has not been added before
        if center not in data:
            data.append(center)
        
        # Get nearby clusters of current center
        nearby_clusters = df_cities[df_cities['Center']==center]['Nearby'].values[0]
        
        
        # Iterate over all values
        for value in nearby_clusters:
            
            # Add nearby cluster to queue if it has not been added before
            if value not in data and value not in container:
                container.append(value)
    
    return data





## Combine Multiple Graphs Into One
def combineGraphs(df_cities, city, folder_build):
    '''
    Inpute:
    df_cities - dataframe with all cities
    city - current city
    folder_build - folder for the builds
    
    Output:
    returns a combined graph of all city centers
    '''
    
    # Create a mapping from centers to city names
    mapping_center_to_city = {center:name for name, center in df_cities[['Name', 'Center']].values}
    
    # Get nearby cluster
    nearby_cluster = getNearbyClusters(city, df_cities)
    
    
    # Get unique city names to load
    cities = [mapping_center_to_city[cluster] for cluster in nearby_cluster]
    
    
    # Get all paths for the graphs
    path_graphs = ['{}/{}_{}/Graph.p'.format(folder_build, city, cluster) for city, cluster in zip(cities, nearby_cluster)]
    
    # Load allstored graphs
    graphs = [pickle.load(open(path, 'rb')) for path in path_graphs]
    
    # Combine and return graphs
    return nx.compose_all(graphs)





## Get All Data For The Cities
def getAllCities(cur, cluster_id):
    '''
    Input:
    cur - cursor to mysql db
    cluster_id - cluster id of the current city
    
    Output:
    df_cities - dataframe with all cities
    df_city - dataframe with the current city
    city - cityname
    '''

    # To store data
    import pandas as pd
    
    # Execute SQL query to get the data
    sql_query = '''SELECT * 
                   FROM city_clusters
                   WHERE city_analyse=1'''
    cur.execute(sql_query)
    data = cur.fetchall()

    # Execute SQL query to get the columns
    sql_query = '''SHOW COLUMNS FROM city_clusters'''
    cur.execute(sql_query)
    columns = [col[0] for col in cur.fetchall()]

    # Create the dataframes
    df_cities = pd.DataFrame(data, columns=columns)
    df_city = df_cities[df_cities['cluster_id']==cluster_id]

    city = df_city['city'].values[0]

    return df_cities, df_city, city





## Get All Rides For The Current City
def getCityRides(cur, cluster_id, timezone):
    '''
    Input:
    cur - cursor to mysql db
    cluster_id - cluster id of the current city
    timezone - timezone of the cluster from df_city
    
    Output:
    df_rides - dataframe with all city rides
    '''
    
    # To store data
    import pandas as pd
    
    # Execute query to get all rides of the current city
    sql_query = '''SELECT * 
                   FROM bike_rides
                   WHERE cluster_id={}'''.format(cluster_id)
    cur.execute(sql_query)
    data = cur.fetchall()

    # Execute query to get the columns
    sql_query = '''SHOW COLUMNS FROM bike_rides'''
    cur.execute(sql_query)
    columns = [col[0] for col in cur.fetchall()]

    
    # Create the dataframe
    df_rides = pd.DataFrame(data, columns=columns)

    # Convert timestamp to datetime and add time zone
    df_rides['date'] = pd.to_datetime(df_rides['bike_time'], unit='s').dt.tz_localize('UTC')

    # Create a new index with the timestamp
    df_rides.set_index('date', inplace=True)

    # Convert the timestamp to the correct time zone
    df_rides.index = df_rides.index.tz_convert(timezone)

    return df_rides





## Plot Trips Per City
def plotTripsPerCity(df_cities, folder, cluster_id, city):
    '''
    Input:
    df_cities - dataframe with all cities
    folder - folder to store the builds
    cluster_id - cluster id of the current city
    city - city to highlight
    
    Output:
    creates an interactive bar chart with trips per city
    '''
    
    # To create interactive plots
    import plotly.graph_objs as go
    
    # To walk directories
    import os
    
    
    # Aggregate data
    df_tmp = df_cities.groupby('city').agg({'n_rides':'sum'}).sort_values('n_rides')

    # Create colors
    color = ['#003f6e' if id==False else '#29bdef' for id in df_tmp.index==city]

    # Define bars
    data = go.Bar(x = df_tmp['n_rides'],
                  y = df_tmp.index,
                  name = 'Trips',
                  marker_color = color,
                  orientation = 'h',
                  width = 0.8,
                  marker = dict(color = '#003f6e',#29bdef
                                line=dict(color = '#ffffff',
                                          width = 0.0)),
                  opacity = 1.0,
                  hovertemplate ='%{y}: %{x}')

    # Setup layout
    layout = dict(title = 'How many trips are there per city?',

                  font_family='Courier New',
                  font_color='#003f6e',

                  xaxis = dict(title = 'Number of trips',
                               tickangle = 0,
                               titlefont = dict(size = 16),
                               tickfont = dict(size = 14)),
                  yaxis = dict(title = 'City',
                               tickangle = 0,
                               titlefont = dict(size = 16),
                               tickfont = dict(size = 14)),

                  legend = dict(x = 0,
                              y = 1.0,
                              bgcolor = '#000000',
                              bordercolor = '#000000'),

                  barmode = 'group',
                  bargap = 0.15,
                  bargroupgap = 0.1,
                  paper_bgcolor = '#ffffff',
                  plot_bgcolor = '#E0E0E0',
                  height=800)

    # Create the plot
    fig = go.Figure(data=[data], layout=layout)
    
    # Save the plot
    path = os.path.join(os.path.join(folder, str(cluster_id)), 'Trips_Per_City.html')
    fig.write_html(path)
    
    return None





## Plot Trips Per Day
def plotTripsDistribution(df_rides, folder, cluster_id, city):
    '''
    Input:
    df_rides - dataframe with all rides
    folder - folder to store the builds
    cluster_id - cluster id of the current city
    city - current city
    
    Output:
    creates an interactive line chart with trips distribution
    '''
    
    # To create interactive plots
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    # To walk in directories
    import os

    
    # Setup subplots
    fig = make_subplots(rows=2, cols=1, subplot_titles=['Trips per Hour', 'Trips per Day'])

    
    # Resample dataframe
    df_tmp = df_rides.resample('h')['id'].count().rolling(5, min_periods=3, center=True).mean()

    # Create trace
    fig.add_trace(
        go.Scatter(x=df_tmp.index, 
                   y=df_tmp, 
                   name = 'Per Hour', 
                   line=dict(color='#003f6e', width=3),
                   hovertemplate ='Trips: %{y:.1f}'), row=1, col=1)


    # Resample dataframe
    df_tmp = df_rides.resample('d')['id'].count().rolling(3, min_periods=3, center=True).mean().round()

    # Create trace
    fig.add_trace(
        go.Scatter(x=df_tmp.index, 
                   y=df_tmp, 
                   name = 'Per Day', 
                   line=dict(color='#003f6e', width=3),
                   hovertemplate ='Trips: %{y:.0f}'), row=2, col=1)


    # Update layout
    fig.update_layout(height = 800, 
                      width = 900, 
                      title_text = 'How Many Trips Are There In {} Per Day?'.format(city),
                      font_family = 'Courier New',
                      font_color = '#003f6e', 
                      paper_bgcolor = '#ffffff',
                      plot_bgcolor = '#E0E0E0',
                      hovermode="x",
                      showlegend=False)

    fig.update_xaxes(title_text='Date', row=1, col=1)
    fig.update_yaxes(title_text='Trips', row=1, col=1)
    fig.update_xaxes(title_text='Date', row=2, col=1)
    fig.update_yaxes(title_text='Trips', row=2, col=1)

    # Save the plot
    path = os.path.join(os.path.join(folder, str(cluster_id)), 'Trips_Distribution_{}.html'.format(city))
    fig.write_html(path)
    
    return None





## Plot trips per weekday
def plotTripsPerWeekday(df_rides, folder, cluster_id, city):
    '''
    Input:
    df_rides - dataframe with all rides
    folder - folder to store the builds
    cluster_id - cluster id of the current city
    city - current city
    
    Output:
    creates an interactive line chart with trips distribution
    '''
    
    # To create interactive plots
    import plotly.graph_objects as go

    # To walk in directories
    import os
    
    
    # Create a mapping for weekdays
    weekdays= {0:'Monday', 
               1:'Tuesday', 
               2:'Wednesday', 
               3:'Thursday', 
               4:'Friday', 
               5:'Saturday', 
               6:'Sunday'}


    # Count all bike movements per hour
    df_tmp = df_rides.resample('h')['id'].count()

    # Smooth the counts with a rolling mean and drop empthy rows
    df_tmp = df_tmp.rolling(3, center=True).mean().dropna()

    # Compute mean rides per weekday and hour
    df_tmp = df_tmp.groupby([df_tmp.index.weekday, df_tmp.index.hour]).agg('mean').to_frame()

    # Rename indexes
    df_tmp.index.set_names(['Weekday', 'Hour'], inplace=True)

    # Reset index
    df_tmp.reset_index(inplace=True)

    # Pivot table to dispay graphs
    df_tmp = df_tmp.pivot_table(index='Hour', columns='Weekday', values='id', aggfunc='mean').round(1)


    data = []

    for i in range(df_tmp.shape[1]):
        data.append(go.Scatter(x=df_tmp.index, 
                               y=df_tmp[i], 
                               name = weekdays[i],
                               hovertemplate ='Trips: %{y:.1f}'))

    layout = dict(title = 'How Are The Trips Distributed Over The Weekday In {}?'.format(city),

                  font_family='Courier New',
                  font_color='#003f6e',

                  xaxis = dict(title = 'Hour of the day',
                               tickangle = 0,
                               titlefont = dict(size = 16),
                               tickfont = dict(size = 14),showgrid=True, gridcolor='#7fd8f5'),
                  yaxis = dict(title = 'Trips',
                               tickangle = 0,
                               titlefont = dict(size = 16),
                               tickfont = dict(size = 14),showgrid=True, gridcolor='#7fd8f5'),

                  legend = dict(x = 0,
                              y = 1.0,
                              bgcolor = '#ffffff',
                              bordercolor = '#000000'),

                  hovermode="x",

                  paper_bgcolor = '#ffffff',
                  plot_bgcolor = '#ffffff',
                  height=600)


    # Create the plot
    fig = go.Figure(data=data, layout=layout)

    # Save the plot
    path = os.path.join(os.path.join(folder, str(cluster_id)), 'Trips_Per_Weekday_{}.html'.format(city))
    fig.write_html(path)

    return None





## Plot Graph OSM
def plotGraphOSM(graph, df_rides, city, cluster_id, folder):
    '''
    Input:
    graph - OSM graph of the city
    df_rides - dataframe with all trips in the city
    city - name of current city
    cluster_id - cluster id of current city
    folder - folder for the builds
    
    Output:
    creates a graph plot with sampled route lines
    '''

    # To create plots
    import matplotlib.pyplot as plt
    
    # To work with OSM
    import osmnx as ox
    
    # To walk in directories
    import os
    
    
    # Create path to store plot
    path_plot = os.path.join(os.path.join(folder, str(cluster_id)), 'OSM_Graph_{}.pdf'.format(city))
    
    
    # Create OSM plot
    fig, ax = ox.plot_graph(graph, 
                            bbox=None,
                            #axis_off=False, 
                            figsize = (15, 12),
                            #fig_height=15, 
                            #fig_width=None, 
                            #margin=0.02,
                            bgcolor='w',
                            show=False, 
                            close=False, 
                            dpi=300, 
                            #annotate=False, 
                            node_color='none', 
                            node_size=15, 
                            node_alpha=1, 
                            node_edgecolor='none', 
                            node_zorder=1, 
                            edge_color='#999999', 
                            edge_linewidth=1, 
                            edge_alpha=1, 
                            #use_geom=True
                           )
    
    # Iterate over sampled rides
    for lat, lng, lat_next, lng_next in df_rides[['bike_latitude', 'bike_longitude', 'bike_latitude_next', 'bike_longitude_next']].drop_duplicates().sample(750, replace=True).values:
        plt.plot([lng, lng_next], [lat, lat_next], 'm')
        
    plt.title('OSM Network Of {}'.format(city))
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.tight_layout()
    plt.savefig(path_plot)
    #plt.show()
    
    return None





## Plot All Empthy Graphs
def plotOSMGraphs(analyse_cities, df_cities, df_rides, folder_build, path_rides):
    '''
    Input:
    analyse_cities - list of cities and clusters to analyse
    df_cities - dataframe with all cities
    df_rides - dataframe with all rides
    folder_build - path for the builds
    path_rides - path to the rides dataframe
    
    Output:
    creates a graph plot with sampled route lines
    '''
    
    # Iterate over all graph cities
    for city, cluster, lat, lng, radius in df_cities[df_cities['Center'].isin(analyse_cities[:, 1])][['Name', 'Center', 'Lat', 'Lng', 'Radius']].values:

        # Set path for pickle graph
        path_graph = '{}/{}_{}/Graph.p'.format(folder_build, city, cluster)
        
        # Set path to store the plot
        path_plot = '{}/{}_{}/Graph.pdf'.format(folder_build, city, cluster)
        
        
        # Check if plot exists
        if not olderThan(path_rides, path_plot):
        
            # Load the stored file
            graph = combineGraphs(df_cities, city, folder_build)


            # Plot the street-network
            fig, ax = ox.plot_graph(graph, 
                                    bbox=None,
                                    #axis_off=False, 
                                    figsize = (15, 12),
                                    #fig_height=15, 
                                    #fig_width=None, 
                                    #margin=0.02,
                                    bgcolor='w',
                                    show=False, 
                                    close=False, 
                                    dpi=300, 
                                    #annotate=False, 
                                    node_color='none', 
                                    node_size=15, 
                                    node_alpha=1, 
                                    node_edgecolor='none', 
                                    node_zorder=1, 
                                    edge_color='#999999', 
                                    edge_linewidth=1, 
                                    edge_alpha=1, 
                                    #use_geom=True
                                   )
            
            
            # Create a dataframe with nearby rides
            df_city = createCityDataFrame(df_rides, df_cities, city)
            
            
            # Sample and iterate rides
            for lat, lng, lat_next, lng_next in df_city[['Lat', 'Lng', 'Lat_next', 'Lng_next']].drop_duplicates().sample(750, replace=True).values:

                # Plot the line
                plt.plot([lng, lng_next], [lat, lat_next], 'm')

            plt.title('OSM Network Of {}: {:.0f}m'.format(city, radius))
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.tight_layout()
            plt.savefig(path_plot)
            plt.show()
    
    return None





## Plot Heatmap On OSM
def plotHeatmapOSM(graph, collection_shortest, cluster_id, city, folder):
    '''
    Input:
    graph - OSM graph of the city
    collection_shortest - connection to mongo
    cluster_id - cluster id of current city
    city - name of current city
    folder - folder for the builds
    
    Output:
    creates a graph plot with sampled route lines
    '''
    
    # To walk in directories
    import os
    
    # To count things
    from collections import Counter
    
    # To work with OSM
    import osmnx as ox
    
    # To do linear algebra
    import numpy as np

    # To store data
    import pandas as pd
    
    # To create plots
    import matplotlib as mpl

    # To create nice plots
    import matplotlib.cm as cm

    # To create plots
    import matplotlib.pyplot as plt

    
    # Create path for the plot
    path_plot = path_plot = os.path.join(os.path.join(folder, str(cluster_id)), 'OSM_Heatmap_{}.pdf'.format(city))


    # Create counter for nodes
    counter = Counter()

    # Iterate over all rides
    for document in collection_shortest.find({'Cluster_Id':cluster_id}):

        # Get ride count and all ride nodes
        count = document['Count']
        ride = document['Ride']

        # Update counter with new nodes and counts
        counter.update(ride*count)


    # Get the nodes and edges for filtering
    nodes, edges = ox.graph_to_gdfs(graph)


    # Compute mean occurence of edges with nodes
    edges_mean = np.mean([edges['u'].apply(lambda x: counter[x]), edges['v'].apply(lambda x: counter[x])], axis=0)


    # Filter unimportant edges
    edges_plot = edges[edges_mean>1]
    edges_mean = edges_mean[edges_mean>1]


    # Compute quantile of edges mean for clipping to high values
    quantile = np.quantile(edges_mean, q=0.95)
    edges_mean = np.clip(edges_mean, a_min=0, a_max=quantile)


    # Get all nodes for the edges
    nodes_plot = nodes.loc[pd.unique(edges_plot[['u', 'v']].values.ravel('K'))]


    # Add important attribute to prevent crashing
    if not hasattr(edges_plot, 'gdf_name'):
        edges_plot.gdf_name = 'unnamed_nodes'

    if not hasattr(nodes_plot, 'gdf_name'):
        nodes_plot.gdf_name = 'unnamed_nodes'


    # Set up color mapper
    cmap = cm.binary
    norm = mpl.colors.Normalize(vmin=0, vmax=quantile)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)


    # Convert the mean of both nodes of an edge to the colormap
    edges_color = mapper.to_rgba(edges_mean)


    # Convert filtered dataframes back to graph
    graph = ox.utils_graph.graph_from_gdfs(nodes_plot, edges_plot)

    # Plot the street-network
    fig, ax = ox.plot_graph(graph,
                            bbox=None,
                            #axis_off=False, 
                            figsize=(15,12),
                            #fig_height=15, 
                            #fig_width=None, 
                            #margin=0.01,
                            bgcolor='w',
                            show=False, 
                            close=False, 
                            dpi=300, 
                            #annotate=False, 
                            node_color='none', 
                            node_size=15, 
                            node_alpha=1, 
                            node_edgecolor='none', 
                            node_zorder=1, 
                            edge_color=edges_color, 
                            edge_linewidth=2, 
                            edge_alpha=1, 
                            #use_geom=True
                           )

    cbar = plt.colorbar(mapper)
    cbar.set_label('Relative')

    plt.title('Heatmap Of Rides In {}'.format(city))
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.tight_layout()
    plt.savefig(path_plot)
    
    return None





## Compute An Alternative Ride On The Street-Network
def alterRoute(ride, graph, p=0.05):
    '''
    Input:
    ride - list of nodes to alter
    graph - graph to query for rides
    p - probobility to turn at each node
    
    Output:
    creates randomised alternative ride
    '''

    # To do linear algebra
    import numpy as np

    # To inspect networks
    import networkx as nx

    
    # Compute the index-list for the turns
    turns = np.random.choice([0, 1], size=len(ride), p=[1-p, p])
    
    # Get the indices where the ride has to be altered
    turn_index = np.argwhere(turns==1)
    
    # Check if the ride has to be alteres at all (turn exists AND turn is not last node)
    if (turn_index.size) and (len(ride)-1!=turn_index[0][0]):
        
        # Get the target node
        end = ride[-1]
        
        # Get the first index to change the ride
        turn_index = turn_index[0][0]
        
        # Create variable for new ride (dont alter nodes before turn-node)
        new_ride = ride[:turn_index +1]
        
        # Get the node where the ride changes
        turn_node = ride[turn_index]
        
        # Get the next node from the shortest ride
        shortest_next_node = ride[turn_index +1]
        
        # Get all neighbours of the turning node
        all_neighbours = [n for n in graph.neighbors(turn_node)]

        # Remove shortest and turn node
        possible_neighbours = list(set(all_neighbours).difference([turn_node, shortest_next_node]))

        
        # Check if there are neighbours left
        if possible_neighbours:

            # Choose a random new node
            random_neighbour = np.random.choice(possible_neighbours)

        else:
            # If there are no neighbours take the shortest ride
            random_neighbour = shortest_next_node
        
        try:
            # Get a new ride from the random neighbour to the end
            turn_ride = nx.shortest_path(graph, 
                                         random_neighbour,
                                         end, 
                                         weight='length')
        except:
            # Return original ride if there is no connection
            turn_ride = ride[turn_index +1:]
        
        # Recursively alter the new ride
        altered_ride = alterRoute(turn_ride, graph, p)
        
        # Extend the new ride with the new path
        new_ride.extend(altered_ride)
        
    else:
        new_ride = ride
        
    return new_ride





## Plot Heatmap On OSM
def plotHeatmapAlteredOSM(graph, collection_altered, cluster_id, city, folder):
    '''
    Input:
    graph - OSM graph of the city
    collection_altered - connection to mongo
    cluster_id - cluster id of current city
    city - name of current city
    folder - folder for the builds
    
    Output:
    creates a graph plot with sampled route lines
    '''
    
    # To walk in directories
    import os
    
    # To count things
    from collections import Counter
    
    # To work with OSM
    import osmnx as ox
    
    # To do linear algebra
    import numpy as np

    # To store data
    import pandas as pd
    
    # To create plots
    import matplotlib as mpl

    # To create nice plots
    import matplotlib.cm as cm

    # To create plots
    import matplotlib.pyplot as plt

    
    # Create path for the plot
    path_plot = path_plot = os.path.join(os.path.join(folder, str(cluster_id)), 'OSM_Heatmap_Altered_{}.pdf'.format(city))


    # Create counter for nodes
    counter = Counter()

    # Iterate over all rides
    for document in collection_altered.find({'Cluster_Id':cluster_id}):

        # Update counter with new nodes and counts
        for ride in document['Rides']:
          counter.update(ride)


    # Get the nodes and edges for filtering
    nodes, edges = ox.graph_to_gdfs(graph)


    # Compute mean occurence of edges with nodes
    edges_mean = np.mean([edges['u'].apply(lambda x: counter[x]), edges['v'].apply(lambda x: counter[x])], axis=0)


    # Filter unimportant edges
    edges_plot = edges[edges_mean>1]
    edges_mean = edges_mean[edges_mean>1]


    # Compute quantile of edges mean for clipping to high values
    quantile = np.quantile(edges_mean, q=0.95)
    edges_mean = np.clip(edges_mean, a_min=0, a_max=quantile)


    # Get all nodes for the edges
    nodes_plot = nodes.loc[pd.unique(edges_plot[['u', 'v']].values.ravel('K'))]


    # Add important attribute to prevent crashing
    if not hasattr(edges_plot, 'gdf_name'):
        edges_plot.gdf_name = 'unnamed_nodes'

    if not hasattr(nodes_plot, 'gdf_name'):
        nodes_plot.gdf_name = 'unnamed_nodes'


    # Set up color mapper
    cmap = cm.binary
    norm = mpl.colors.Normalize(vmin=0, vmax=quantile)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)


    # Convert the mean of both nodes of an edge to the colormap
    edges_color = mapper.to_rgba(edges_mean)


    # Convert filtered dataframes back to graph
    graph = ox.utils_graph.graph_from_gdfs(nodes_plot, edges_plot)

    # Plot the street-network
    fig, ax = ox.plot_graph(graph,
                            bbox=None,
                            #axis_off=False, 
                            figsize=(15,12),
                            #fig_height=15, 
                            #fig_width=None, 
                            #margin=0.01,
                            bgcolor='w',
                            show=False, 
                            close=False, 
                            dpi=300, 
                            #annotate=False, 
                            node_color='none', 
                            node_size=15, 
                            node_alpha=1, 
                            node_edgecolor='none', 
                            node_zorder=1, 
                            edge_color=edges_color, 
                            edge_linewidth=2, 
                            edge_alpha=1, 
                            #use_geom=True
                           )

    cbar = plt.colorbar(mapper)
    cbar.set_label('Relative')

    plt.title('Heatmap Of Altered Trips In {}'.format(city))
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.tight_layout()
    plt.savefig(path_plot)
    
    return None





## Plot Heatmap On OSM
def plotHeatmapAlteredSpeedOSM(graph, df_rides, collection_altered, cluster_id, city, folder, n=10):
    '''
    Input:
    graph - OSM graph of the city
    df_rides - dataframe with all trips
    collection_altered - connection to mongo
    cluster_id - cluster id of current city
    city - name of current city
    folder - folder for the builds
    n - number of edges to compute mean speed
    
    Output:
    creates a graph plot with sampled route lines
    '''
    
    # To walk in directories
    import os
    
    # To work with OSM
    import osmnx as ox
    
    # To do linear algebra
    import numpy as np

    # To store data
    import pandas as pd
    
    # To create plots
    import matplotlib as mpl

    # To create nice plots
    import matplotlib.cm as cm

    # To create plots
    import matplotlib.pyplot as plt 

    # To search efficiently
    from sklearn.neighbors import KDTree

    # To create nice plots
    from matplotlib.colors import LinearSegmentedColormap

    
    # Create path for the plot
    path_plot = path_plot = os.path.join(os.path.join(folder, str(cluster_id)), 'OSM_Heatmap_Altered_Speedmap_{}.pdf'.format(city))


    # Get the nodes and edges for filtering
    nodes, edges = ox.graph_to_gdfs(graph)
    
    # Create a k-dimensional tree of the position of the nodes
    tree = KDTree(nodes[['y', 'x']], 
                  metric='euclidean')
    
    # Get the closest point to the start- and end-point in the graph 
    start_idx = tree.query(df_rides[['bike_latitude', 'bike_longitude']], k=1, return_distance=False)
    end_idx = tree.query(df_rides[['bike_latitude_next', 'bike_longitude_next']], k=1, return_distance=False)

    # Flatten start- and end-idx
    start_idx = np.array([i[0] for i in start_idx])
    end_idx = np.array([i[0] for i in end_idx])
    
    # Create mask for different start/end points
    mask = start_idx != end_idx
    
    # Get the closest nodes
    closest_node_to_start = nodes.iloc[start_idx[mask]].index.values
    closest_node_to_end = nodes.iloc[end_idx[mask]].index.values
    
    # Create keys for altered routes dict
    keys = [str((a, b)) for a, b in zip(closest_node_to_start, closest_node_to_end)]
    
    
    # Filter for the durations
    durations = df_rides['ride_duration'].values[mask]
    
    
    # Dictionary for all nodes to get mean speeds
    dict_speed = {}

    # Iterate over all keys
    for key, duration in zip(keys, durations):
        
        # Get all alternative routes for the key
        values = collection_altered.find_one({'_id':key})
        
        if values:
            
            routes = values['Rides']
            lengths = values['Lengths']
            

            # Choose a random index for routes
            index = np.random.randint(len(lengths))

            # Get specific route and length
            route = routes[index]
            length = lengths[index]

            # Compute mean speed of route
            speed = length / duration


            # Iterate over all nodes
            for node in route:


                # Add node to dictionary with its mean speed
                if node in dict_speed.keys():

                    dict_speed[node].append(speed)

                else:
                    dict_speed[node] = [speed]
                    
                    
    # Iterate over all edges
    edge_speeds = np.array([np.mean(dict_speed[start] + dict_speed[end]) if start in dict_speed.keys() and end in dict_speed.keys() and len(dict_speed[start])>n and len(dict_speed[end])>n else np.nan for start, end in edges[['u', 'v']].values])

    mask = np.invert(np.isnan(edge_speeds))


    # Filter unimportant edges
    edges_plot = edges[mask]
    edge_speeds_mean = edge_speeds[mask]


    # Compute quantile of edges mean for clipping to high values
    quantile_upper = np.quantile(edge_speeds_mean, q=0.95)
    quantile_lower = np.quantile(edge_speeds_mean, q=0.05)
    edge_speeds_mean = np.clip(edge_speeds_mean, a_min=0, a_max=quantile_upper)


    # Get all nodes for the edges
    nodes_plot = nodes.loc[pd.unique(edges_plot[['u', 'v']].values.ravel('K'))]


    # Add important attribute to prevent crashing
    if not hasattr(edges_plot, 'gdf_name'):
        edges_plot.gdf_name = 'unnamed_nodes'

    if not hasattr(nodes_plot, 'gdf_name'):
        nodes_plot.gdf_name = 'unnamed_nodes'


    # Set up color mapper
    colors = [(0.8, 0, 0), (1, 0.5, 0), (1, 1, 0), (0.5, 1, 0), (0, 0.8, 0)]
    cmap = LinearSegmentedColormap.from_list('Speeds', colors, N=100)
    norm = mpl.colors.Normalize(vmin=quantile_lower, vmax=quantile_upper)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)


    # Convert the mean of both nodes of an edge to the colormap
    edges_color = mapper.to_rgba(edge_speeds_mean)


    # Convert filtered dataframes back to graph
    graph = ox.utils_graph.graph_from_gdfs(nodes_plot, edges_plot)


    # Plot the street-network
    fig, ax = ox.plot_graph(graph,
                            bbox=None,
                            #axis_off=False, 
                            figsize=(15,12),
                            #fig_height=15, 
                            #fig_width=None, 
                            #margin=0.01,
                            bgcolor='w',
                            show=False, 
                            close=False, 
                            dpi=300, 
                            #annotate=False, 
                            node_color='none', 
                            node_size=15, 
                            node_alpha=1, 
                            node_edgecolor='none', 
                            node_zorder=1, 
                            edge_color=edges_color, 
                            edge_linewidth=2, 
                            edge_alpha=1, 
                            #use_geom=True
                           )

    cbar = plt.colorbar(mapper)
    cbar.set_label('Relative')

    plt.title('Heatmap Of Speeds In {}'.format(city))
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.tight_layout()
    plt.savefig(path_plot)
    
    return None





## Plot Raw Data Table
def plotRawDataTable(folder):
    '''
    Input:
    folder - folder for the builds
    
    Output:
    creates a table plot with raw data
    '''
    
    # To create interactive plots
    import plotly.graph_objects as go

    # To store data
    import pandas as pd
    
    # To do linear algebra
    import numpy as np
    
    # To walk directories
    import os
    

    # Generate a random data distribution
    unix_timestamp = np.random.randint(low=1500000000, high=1600000000, size=5)
    bike_id = np.random.randint(low=40000, high=100000, size=5)
    bike_latitude = np.round(np.random.uniform(low=47, high=55, size=5), 5)
    bike_longitude = np.round(np.random.uniform(low=6, high=15, size=5), 5)

    # Create the dataframe
    df = pd.DataFrame(zip(unix_timestamp, bike_id, bike_latitude, bike_longitude), columns=['UNIX_Timestamp', 'Bike_ID', 'Bike_Latitude', 'Bike_Longitude'])
    
    
    path_plot = os.path.join(folder, 'Raw_Data_Table.html')

    # Create the trace
    data = go.Table(header=dict(values=list(df.columns),
                                fill_color='#E0E0E0',
                                align='left'),
                    cells=dict(values=[df.UNIX_Timestamp, df.Bike_ID, df.Bike_Latitude, df.Bike_Longitude],
                               fill_color='#F0F0F0',
                               align='left'))

    # Create the layout
    layout = go.Layout(title_text = 'What Does The Raw Data Look Like?',
                       font_family = 'Courier New',
                       font_color = '#003f6e', 
                       height = 350)

    # Create the figure
    fig = go.Figure(data=[data], layout=layout)
    fig.write_html(path_plot)
    
    return None
