import laspy
from laspy.file import File
import numpy as np
import os


dirs = '/home/yf/disk/pointcloud/Data'
listdir=os.listdir(dirs)
target = '/home/yf/disk/pointcloud/57.las'
hdr = laspy.header.Header(point_format=2)
outfile = laspy.file.File(target, mode="w", header=hdr)
all_x = []
all_y = []
all_z = []
all_r = []
all_g = []
all_b = []
outfile.header.scale= [0.01,0.01,0.01]

for dir in listdir:
    dir = os.path.join(dirs,dir)
    for file in os.listdir(dir):
        path = os.path.join(dir,file)
        inFile = File(path, mode = "r")
        all_x=np.append(all_x, inFile.x, axis=0)
        all_y=np.append(all_y, inFile.y, axis=0)
        all_z=np.append(all_z,inFile.z, axis=0)
        all_r=np.append(all_r, inFile.red, axis=0)
        all_g=np.append(all_g, inFile.green, axis=0)
        all_b=np.append(all_b, inFile.blue, axis=0)

outfile.x=np.array(all_x)
outfile.y=np.array(all_y)
outfile.z=np.array(all_z)
outfile.red=np.array(all_r)
outfile.green=np.array(all_g)
outfile.blue=np.array(all_b)
xmin = np.min(outfile.x)
ymin = np.min(outfile.y)
zmin = np.min(outfile.z)
outfile.header.offset = np.array([xmin,ymin,zmin])

outfile.close()