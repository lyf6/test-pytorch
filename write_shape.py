import shapefile
import csv

csv_file = '/home/yf/Documents/test-pytorch/2019jh_long-crop.csv'
tareget_shape = '/home/yf/Documents/test-pytorch/2019_jh-crop'
fp = open(csv_file)
lines = fp.read().splitlines()
w = shapefile.Writer(shapefile.POLYGON)
w.autoBalance = 1
w.field('BUFF_DIST', 'C', '40')
count = 0
for line in lines:
    point_list = []
    polygon_list = []
    point = []
    tmp_loc = line.split(',')
    label_name = tmp_loc[0]
    tmp_loc = tmp_loc[1:]
    #print(tmp_loc)
    count = count+1
    for id in range(0, len(tmp_loc), 2):
        #print(tmp_loc[id], tmp_loc[id+1])
        point.append(float(tmp_loc[id]))
        point.append(float(tmp_loc[id+1]))
        point_list.append(point)
        point = []
        #print(point_list)
    #print(polygon_list)
    polygon_list.append(point_list)
    #print(polygon_list)
    w.poly(parts=polygon_list)
    w.record(label_name)
w.save(tareget_shape)