source_dir=/home/yf/disk/buildings/inria/created
img_dir=${source_dir}/img
ann_dir=${source_dir}/mask
work_dir=${source_dir}
id_class_map_file=./class_list.txt
echo $source_dir
python mask2coco.py --imgdir $img_dir --anndir $ann_dir --id_class_map_file $id_class_map_file --workdir $work_dir