# vinet imu reading

import csv
import numpy as np

def readIMU_File(path):
    imu = []
    count = 0
    with open(path) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            if count == 0:
                count += 1
                continue
            parsed = [float(row[1]), float(row[2]), float(row[3]), 
                        float(row[4]), float(row[5]), float(row[6])]
            imu.append(parsed)
            
    return np.array(imu)

imu = readIMU_File('/data/euroc/Raw/MH_01_easy/mav0/imu0/data.csv')

print(imu)

