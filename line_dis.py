from shapely.geometry import LineString
a=LineString([(1,1),(3,5)])
b=LineString([(1,1),(5,2)])
print(a.distance(b))
