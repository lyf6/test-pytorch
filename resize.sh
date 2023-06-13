infn=/home/yf/disk/jinzhongqiao/Production_3_ortho_merge.tif
outfn=/home/yf/disk/jinzhongqiao/Production_3_ortho_merge_scale.tif

xres=0.1
yres=0.1
resample_alg=bilinear
gdalwarp -of GTiff -tr ${xres} ${yres} -r ${resample_alg} ${infn} ${outfn}
