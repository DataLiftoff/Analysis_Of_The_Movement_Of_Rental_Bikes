# Analysis Of The Movement Of Rental Bikes

The APIs of bike sharing companies can be used to gain insights into bicycle traffic in cities. In addition to movement patterns in everyday life, the road network itself can also be analyzed.

This repository shows how a map of the well-developed areas of a city is created from the individual GPS coordinates of the parked bicycles and how insights into the behavior of the population can be gained.


# 1. Where does the raw data come from?

In order to be able to operate bike sharing, every parked bike is recorded by the respective services with its GPS coordinates. This current data is then forwarded via API so that users can see where the bikes are currently parked.
The [WoBike](https://github.com/ubahnverleih/WoBike) project offers an overview of the public APIs

The locations of the [nextbike](https://www.nextbike.de/) API were recorded for this project for a month and a half. The recorded data only includes the time, a unique ID for each bicycle and the GPS coordinates of the bicycles parked. No data is available for bikes currently rented.

![Raw_Data](/build/Raw_Data_Table.png)


# 2. How is the data processed?

With the help of the K-Means algorithm, the individual GPS coordinates of the bicycles are combined into approx. 100 global clusters. A city and a time zone can be assigned to the individual clusters using reverse geocoding.

Because two different, consecutive locations of a bicycle require a trip between the positions, the data is filtered for such trips. With the UNIX time stamps, the journey time is also known.
For further analyzes, outliers in the trip data are removed and only clusters within Germany are considered.

![Trips_Per_City](/build/35/Trips_Per_City.png)


# 3. What is the driving behavior like in Bremen?

The analysis of the data for the city of Bremen is considered as an example. Over the period of data collection, a weekly pattern can be seen in the number of trips and two peak times per day.

![Trips_Per_Day](/build/35/Trips_Distribution_Bremen.png)

If the number of trips is grouped according to the day of the week, a strong correlation to the usual working hours can be determined. Even an earlier end of work on Friday can be booked.

![Trips_Per_Weekday](/build/35/Trips_Weekday_Bremen.png)


# 4. Which urban areas are heavily used?

The road network of Open Street Map is used to link the trips of the cyclists with the actual infrastructure. In this way, the shortest route on the road network can be calculated using the start and end points of the bike trip.

Because this can still lead to a distortion, a randomization is inserted into the calculation of the path that is likely to be used.
Although no individual trip can be precisely determined with this approach, patterns for preferred roads are averaged over many randomized trips. 

![Heatmap_Location](/build/35/Heatmap_Location_Bremen.png)


# 5. How fast do the bicycles move around the city?

The average speed of the cyclist can be determined with the duration of the trip and the route calculated on the Open Street Map. This, too, only leads to a meaningful illustration in a mix with many trips.

![Heatmap_Speed](/build/35/Heatmap_Speed_Bremen.png)


# 6. Conclusion

Insights into the work and weekend culture of the population can be gained from the location data of rental bicycles alone.

It is also possible to draw conclusions about the expansion and efficiency of the city infrastructure.
In this way, not only residential and work areas of a city can be identified, but also much-used streets and areas in which bicycle traffic is efficiently regulated.

It should be noted that the randomized trips of this project can only represent an approximation of reality and that it becomes even more meaningful with more data. 


# 7. Outlook

It is possible that patterns of individuals can be traced in the data if they travel regular routes.

Based on the analysis of a single city, the comparison of cities with each other should be exciting.




