import os
import csv

folder_path = "./data/lidar_oa_steering"
output_path = "./data/lidar_oa_steering_clip10"

files = os.listdir(folder_path)

for file in files:
    file_path = os.path.join(folder_path, file)
    with open(file_path, 'r', newline='') as f:
        output_file_path = os.path.join(output_path, file)
        with open(output_file_path, 'w', newline='') as o:
            writer = csv.writer(o)
            for line_num, line in enumerate(f, 1):
                elements = line.strip().split(',')
                output_line = []
                for e in elements:
                    if float(e) > 10:
                        output_line.append(10.0)
                    else:
                        output_line.append(e)
                # send the output to the same output file name, but to the output_path dir
                writer.writerow(output_line)
