# Analysis Of The Movement Of Rental Bikes

Um Einsichten in den Fahrradverkehr von Städten zu bekommen, können die APIs von Bike Sharing Unternehmen heran gezogen werden. Neben Bewegungsmustern des alltäglichen Lebens kann auch das Straßennetz an sich analysiert werden.

Dieses Repository zeigt wie aus den einzelnen GPS-Koordinaten der abgestellten Fahrräder eine Karte der gut entickelten Gebiete einer Stadt entsteht und wie Einblicke in die Verhaltensweisen der Bevölkerung gewonnen werden können. 


# 1. Wo kommen die Rohdaten her?

Um Bike Sharing betreiben zu können, ist jedes abgestellte Fahrrad mit seinen GPS-Koordinaten von den jeweiligen Diensten erfasst. Diese aktuellen Daten werden dann per API weitergeleitet, sodass die Nutzer einsehen können wo die Fahrräder gerade geparkt sind. 
Einen Überblick über die öffentlichen APIs bietet das Projekt [WoBike](https://api.nextbike.net/reservation/geojson/flexzone_all.json).

Für eineinhalb Monate wurden für dieses Projekt die Standorte der [nextbike](https://www.nextbike.de/)-API aufgezeichnet. Die festgehaltenen Daten umfassen ausschließlich den Zeitpunkt, eine eindeutige ID für jedes Fahrrad und die GPS-Koordinaten der abgestellten Fahrräder. Für momentan gemietete Räder liegen keine Daten vor.

![Raw_Data](/build/Raw_Data_Table.png)


# 2. Wie werden die Daten aufbereitet?

Mit Hilfe des K-Means-Algorithmus werden die einzelnen GPS-Koordinaten der Fahrräder zu ca. 100 weltweiten Clustern zusammengefasst. Mittels reverse geocoding kann eine Stadt und eine Zeitzone den einzelnen Clustern zugeordnet werden. 

Weil zwei unterschiedliche, aufeinander folgende Standorte eines Fahrrades eine Fahrt zwischen den Positionen vorraussetzt, werden die Daten nach solchen Trips gefiltert. Mit den UNIX Zeitstempeln ist auch gleich die Fahrtdauer bekannt. 
Für die weiteren Analysen werden Ausreißer in den Daten der Trips entfernt und auch nur Cluster innerhalb von Deutschland betrachtet. 

![Trips_Per_City](/build/35/Trips_Per_City.png)


# 3. Wie sieht das Fahrverhalten in Bremen aus?

Exemplarisch wird die Analyse der Daten an der Stadt Bremen betrachtet. Über den Zeitraum der Datenerhebung lässt sich ein wöchentliches Muster in der Anzahl der Trips und auch zwei Stoßzeiten pro Tag erkennen. 

![Trips_Per_Day](/build/35/Trips_Distribution_Bremen.png)

Wird die Anzahl der Trips gruppiert nach Wochentag dargestellt, so lässt sich ein starker Zusammenhang zu den üblichen Arbeitszeiten feststellen. Selbst ein früherer Feierabend am Freitag kann belegt werden. 

![Trips_Per_Weekday](/build/35/Trips_Weekday_Bremen.png)


# 4. Welche Stadtgebiete werden stark befahren?

Um die Trips der Fahrradfahrer mit der tatsächlichen Infrastruktur zu verknüpfen, wird das Straßennetz von Open-Street-Map verwendet. So kann mit dem Start- und dem End-Punkt des Fahrrad-Trips der kürzeste Weg auf dem Straßennetz berechnet werden. Weil dies noch zu einer Verzerrung führen kann, wird eine Randomisierung in die Berechnung des wahrscheinlich genutzen Weges eingefügt. 
Zwar kann mit diesem Ansatz kein einzelner Trip genau bestimmt werden, doch gemittelt über viele randomisierte Trips ergeben sich Muster für bevorzugte Straßen. 

![Heatmap_Location](/build/35/Heatmap_Location_Bremen.png)


# 5. Wie schnell bewegen sich die Fahrräder im Stadtgebiet fort?

Mit der Fahrtdauer und der von Open-Street-Map berechneten Fahrtsrecke, lässt sich die durschnitliche Geschwindigkeit des Radfahrers bestimmen. Auch dies führt erst im Mix mit vielen Trips zu einer aussagekräftigen Abbildung.

![Heatmap_Speed](/build/35/Heatmap_Speed_Bremen.png)


# 6. Fazit

Allein aus den Standortdaten von Leihfahrrädern können Einblicke in die Arbeits- und Wochenends-Kultur der Bevölkerung gewonnen werden. 
Auch ist es möglich Rückschlüsse auf den Ausbau und die Effizienz der Stadtinfrastruktur zu ziehen.

Zu beachten ist dabei, dass die randomisierten Trips nur eine Annäherung an die Wirklichkeit darstellen können und mit mehr Daten noch an Aussagekraft gewinnen. 


# 7. Ausblick

Möglicherweise lassen sich Muster von einzelnen Individuen in den Daten aufspüren, wenn sie regelmäßige Routen zurücklegen.  
Aufbauend auf der Analyse einer einzelnen Stadt, sollte der Vergleich von Städten untereinander Spannendes bereit halten. 




