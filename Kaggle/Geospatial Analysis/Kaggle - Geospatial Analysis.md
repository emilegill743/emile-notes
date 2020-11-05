# Kaggle - Geospatial Analysis

- https://www.kaggle.com/learn/geospatial-analysis

---

## Contents

- [Kaggle - Geospatial Analysis](#kaggle---geospatial-analysis)
  - [Contents](#contents)
  - [Reading Data](#reading-data)
  - [Visualising Data](#visualising-data)
    - [Plotting GeoDataFrames on the same axis](#plotting-geodataframes-on-the-same-axis)
  - [Coordinate Reference Systems](#coordinate-reference-systems)
    - [Setting the CRS](#setting-the-crs)
    - [Re-projecting](#re-projecting)
  - [Attributes of geometric objects](#attributes-of-geometric-objects)
  - [Creating Interactive Maps](#creating-interactive-maps)
    - [Markers](#markers)
    - [Marker Clusters](#marker-clusters)
    - [Bubble Maps](#bubble-maps)
    - [Heatmaps](#heatmaps)
    - [Choropleth maps](#choropleth-maps)
  - [Manipulating Geospatial Data](#manipulating-geospatial-data)
    - [Geocoding](#geocoding)
    - [Table joins](#table-joins)
      - [Attribute joins](#attribute-joins)
      - [Spatial joins](#spatial-joins)

---

## Reading Data

We can use the GeoPandas library to get started reading geospatial data in Python.

```python
import geopandas as gpd
```

There are many different file formats of geospatial data: shapefile, GeoJSON, KML, GPKG. shapefile is the most common format; however, all of these file types can be quickly loading using geopandas:

```python
gdf = gpd.read_file()
```

Data is read into a **GeoDataFrame** type object, a subclass of the pandas DataFrame class. We can, therefore, apply any of the traditional pandas attributes/methods to a GeoDataFrame, but also make use of the additional features of GeoPandas.

## Visualising Data

We can quickly visualise the data in a GeoDataFrame using its `plot()` method, which returns as matplotlib subplot object.

```python
gdf.plot()
```

Every GeoDataFrame object contains a special "geometry" column which contains all the geometric objects to be displayed when we call the `plot()` method. This column may containa variety of different datatypes; however, typically this will be a **Point**, **LineString** or **Polygon**.

![Geometry data types](https://i.imgur.com/N1llefr.png)

### Plotting GeoDataFrames on the same axis

The `plot()` method takes several optional parameters that can be used to customise the appearance of the plot. Setting a value for `ax` ensures that all of the information is plotted on a single map.

```python
# Define a base map with county boundaries
ax = counties.plot(figsize=(10,10), color='none', edgecolor='gainsboro', zorder=3)

# Add wild lands, campsites, and foot trails to the base map
wild_lands.plot(color='lightgreen', ax=ax)
campsites.plot(color='maroon', markersize=2, ax=ax)
trails.plot(color='black', markersize=1, ax=ax)
```
## Coordinate Reference Systems

If we want to display a 3D object- such as the Earths surface- on to a 2D map, we will have to use a **map projection** to render it as a flat surface.

Map projections can never be 100% accurate. Each projection will distort the surface in some way, so as to preserve some other useful property. For example *equal-area* projections (such as 'Lambert Cylindrical Equal Area' or 'Africa Albers Equal Area Conic') preserve area and *equidistant* projections (such as 'Azimuthal Equidistant projection') preserve distance.

![Map Projections](https://i.imgur.com/noBRRNR.png)

To show how projected points correspond to real locations, we use a **coordinate reference system (CRS)**.

### Setting the CRS

When we create a GeoDataFrame from a shapefile, the CRS is already imported for us.

```python
In: # Load a GeoDataFrame containing regions in Ghana
    regions = gpd.read_file("/Map_of_Regions_in_Ghana.shp")
    
    print(regions.crs)

Out: {'init' : 'epsg:32630}
``` 

Coordinate reference systems are referenced by [**European Petroleum Survey Group (EPSG)**](http://www.epsg.org/) codes. For example, the above GeoDataFrame uses [EPSG 32630](https://epsg.io/32630), more commonly known as the 'Mercator' projection. This projection preserves angles, making it useful for sea navigation, but slightly distorts area.

When creating a GeoDataFrame from as CSV file, we have to manually set the CRS:

```python
# Create a DataFrame with health facilities in Ghana
facilities_df = pd.read_csv("../input/geospatial-learn-course-data/ghana/ghana/health_facilities.csv")

# Convert the DataFrame to a GeoDataFrame
facilities = gpd.GeoDataFrame(facilities_df, geometry=gpd.points_from_xy(facilities_df.Longitude, facilities_df.Latitude))

# Set the coordinate reference system (CRS) to EPSG 4326
facilities.crs = {'init': 'epsg:4326'}

# View the first five rows of the GeoDataFrame
facilities.head()
```
### Re-projecting

Re-projecting refers to the process of changing the CRS. This can be done in GeoPandas with the `to_crs()` method. This method modifies the 'geometry' column, to shift the CRS: all other columns are left as they are.

```python
# Changing CRS to known EPSG code
facilities.to_crs(epsg=32630).head()
```

If there is no EPSG code available in GeoPandas for the CRS, we can change the CRS using a **proj4 string**. E.g. converting to a latitude/longitude coordinate system:

```python
# Change the CRS to EPSG 4326
regions.to_crs('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')
```
## Attributes of geometric objects

For an arbitrary GeoDataFrame, the type in the 'geometry' column depends on what we are trying to show. For example, we might use:

- a Point for the epicentre of an earthquake,
- a LineString for a street, or
- a Polygon to show country boundaries.

All three types of geometric objects have built in attributes that we can use to quickly analyse the dataset.

Accessing x, y coordinates of a Point object:

```python
# Get x-coordinate of each point
facilities.geometry.x.head()
```

Length of a LineString object:

```python
# Get length of LineString
gdf.geometry.length
```

Area of a Polygon object:

```python
# Calculate the area (sq m) of each polygon in the GeoDataFrame 
regions.loc[:, "AREA"] = regions.geometry.area / 10**6

print("Area of Ghana: {} square kilometers".format(regions.AREA.sum()))
```

## Creating Interactive Maps

We can use the `folium` package to create interactive maps.

```python
import folium
from folium import Choropleth, Circle, Marker
from folium.plugins import HeatMap, MarkerCluster
```

Creating a simple map using `folium.Map()`:

```python
# Create a map
m_1 = folium.Map(location=[42.32,-71.0589], tiles='openstreetmap', zoom_start=10)

# Display the map
m_1
```

- `location` : sets the intial centre of the map by lat, log

- `tiles` : changes the styling of the map (options can be found [here](https://github.com/python-visualization/folium/tree/master/folium/templates/tiles))

- `zoom_start` : sets initial level of zoom of the map, higher values zoom in closer

### Markers

We can add markers to a map using `folium.Marker()`:

```python

daytime_robberies = crimes[((crimes.OFFENSE_CODE_GROUP == 'Robbery') & \
                            (crimes.HOUR.isin(range(9,18))))]

# Create a map
m_2 = folium.Map(location=[42.32,-71.0589], tiles='cartodbpositron', zoom_start=13)

# Add points to the map
for idx, row in daytime_robberies.iterrows():
    Marker([row['Lat'], row['Long']]).add_to(m_2)

# Display the map
m_2
```

### Marker Clusters

`folium.plugins.MarkerCluster` can help declutter the map if we have many markers.

```python
# Create the map
m_3 = folium.Map(location=[42.32,-71.0589],
                 tiles='cartodbpositron',
                 zoom_start=13)

# Add points to the map
mc = MarkerCluster()

for idx, row in daytime_robberies.iterrows():
    if not math.isnan(row['Long']) and not math.isnan(row['Lat']):
        mc.add_child(Marker([row['Lat'], row['Long']]))

m_3.add_child(mc)

# Display the map
m_3
```

### Bubble Maps

A **bubble map** uses circles of varying size, color to show the relationship between location and two other variables.

We can create a bubble map using `folium.Circle()` to iteratively add circles.

```python
# Create a base map
m_4 = folium.Map(location=[42.32,-71.0589], tiles='cartodbpositron', zoom_start=13)

def color_producer(val):
    if val <= 12:
        return 'forestgreen'
    else:
        return 'darkred'

# Add a bubble map to the base map
for i in range(0,len(daytime_robberies)):

    Circle(
        location=[daytime_robberies.iloc[i]['Lat'],
                  daytime_robberies.iloc[i]['Long']],
        radius=20,
        color=color_producer(
                daytime_robberies.iloc[i]['HOUR'])
    ).add_to(m_4)

# Display the map
m_4
```
- `location` : list containing the centre of the circle in lat, long

- `radius` : sets radius of circle
    - Can implement a mapping function (similar to `color_producer()`) to vary radii

- `color` : sets color of circle
    - Implementing a function (e.g. `color_producer()` above) allows us to vary the color of each circle.

### Heatmaps

We can create a **heatmap** using `folium.plugins.HeatMap()`.

```python
# Create a base map
m_5 = folium.Map(location=[42.32,-71.0589], tiles='cartodbpositron', zoom_start=12)

# Add a heatmap to the base map
HeatMap(data=crimes[['Lat', 'Long']], radius=10).add_to(m_5)

# Display the map
m_5
```

- `data` : the DataFrame containing the locations we'd like to plot

- `radius` : controls the smoothness of the heatmap. Higher values make the heatmap look smoother (fewer gaps)

### Choropleth maps

We can create a **choropleth map** using `folium.Choropleth()`.

```python
# GeoDataFrame with geographical boundaries of Boston police districts
districts_full = gpd.read_file('../input/geospatial-learn-course-data/Police_Districts/Police_Districts/Police_Districts.shp')
districts = districts_full[["DISTRICT", "geometry"]].set_index("DISTRICT")

# Number of crimes in each police district
plot_dict = crimes.DISTRICT.value_counts()
plot_dict.head()

# Create a base map
m_6 = folium.Map(location=[42.32,-71.0589], tiles='cartodbpositron', zoom_start=12)

# Add a choropleth map to the base map
Choropleth(geo_data=districts.__geo_interface__, 
           data=plot_dict, 
           key_on="feature.id", 
           fill_color='YlGnBu', 
           legend_name='Major criminal incidents (Jan-Aug 2018)'
          ).add_to(m_6)

# Display the map
m_6
```

- `geo_data` : GeoJSON FeatureCollection containing boundaries of geographical area.
    - Above we get a GeoJSON FeatureCollection using the `__geo_interface__` attribute of our GeoDataFrame.

- `data` : Pandas Series containing values used to color-code each geographical area.

- `key_on` : variable in GeoJSON file to bind data to.

- `fill_color` : sets color scale

- `legend_name` : labels the legend in the top right corner of the map

## Manipulating Geospatial Data

### Geocoding

**Geocoding** is the process of converting the name of a place or an address to a location on a map.

We can use geopandas to help us with geocoding.

```python
from geopandas.tools import geocode
```

All we need to provide is the **name or address** as `str` and the **name of the provider**.

`geocode()` will return a GeoDataFrame with two columns: **geometry** (lat, long) and **address** (full address).

```python
# Geocoding based on OpenStreetMap Nominatim geocoder
result = geocode("The Great Pyramid of Giza",
                 provider="nominatim)

point = result.geometry.iloc[0]
print("Latitude:", point.y)
print("Longitude:", point.x)
```

Example, top 100 universities in Europe:

```python
universities = pd.read_csv("../input/geospatial-learn-course-data/top_universities.csv")
universities.head()
```

| | Name                   |
|-| ---------------------- |
|0| University of Oxford   |
|1| University of Cambridge|
|2| Imperial College London|
|3| ETH Zurich             |
|4| UCL                    |

Applying geocoder to each row:

```python
def my_geocoder(row):
    try:
        point = geocode(row, provider='nominatim').geometry.iloc[0]
        return pd.Series({'Latitude': point.y,
                          'Longitude': point.x,
                          'geometry': point})
    except:
        return None

universities[['Latitude',
              'Longitude',
              'geometry']] = universities.apply(
                                lambda x: my_geocoder(x['Name']),
                                axis=1)

print("{}% of addresses were geocoded!".format(
     (1 - sum(np.isnan(universities["Latitude"]))
      / len(universities)) * 100))

# Drop universities that were not successfully geocoded
universities = universities.loc[~np.isnan(universities["Latitude"])]
universities = gpd.GeoDataFrame(universities,
                                geometry=universities.geometry)
universities.crs = {'init': 'epsg:4326'}
universities.head()
```

Visualising locations:

```python
# Create a map
m = folium.Map(location=[54, 15], tiles='openstreetmap', zoom_start=2)

# Add points to the map
for idx, row in universities.iterrows():
    Marker([row['Latitude'], row['Longitude']], popup=row['Name']).add_to(m)

# Display the map
m
```

### Table joins

#### Attribute joins

We can join GeoDataFrames in the same way we do with regular pandas DataFrames, to combine information matching values on an shared index.

```python
# Use an attribute join to merge data about countries in Europe
europe = europe_boundaries.merge(europe_stats, on="name")
europe.head()
```

#### Spatial joins

With a **spatial join** we combine GeoDataFrames based on the spatial relationship between objects in the "geometry" columns. We can do this using `gpd.sjoin()`.

```python
# Use spatial join to match universities to countries in Europe
european_universities = gpd.sjoin(universities, europe)

# Investigate the result
print("We located {} universities.".format(len(universities)))
print("Only {} of the universities were located in Europe \
     (in {} different countries).\
     ".format(len(european_universities),
              len(european_universities.name.unique())))

european_universities.head()
```

If a Point object from the `universities` GeoDataFrame intersects a Polygon object from the `europe` DataFrame, the corresponding rows are combined and added as a single row in the `european universities` DataFrame. Otherwise they are omitted from the results.

The `gpr.sjoin()` method can also be customised for different kinds of joins, with `how='left'` etc.






















