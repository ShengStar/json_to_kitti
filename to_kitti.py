
import json
import os
import math
import numpy as np
import random
import shutil

random.seed(0)
np.random.seed(0)


label_path = '/home/lixusheng/SUSTechPOINTS/data/views/label/'
calibFile = '/home/lixusheng/SUSTechPOINTS/data/views/calib/front.json'
sourceImagePath = '/home/lixusheng/SUSTechPOINTS/data/views/camera/front/'
sourceVelodynePath = '/home/lixusheng/SUSTechPOINTS/data/views/bin/'



kittiLabelPath = '/home/lixusheng/kitti/training/label_2/'
kittiCalibPath = '/home/lixusheng/kitti/training/calib/'
imageSetsPath = '/home/lixusheng/kitti/ImageSets/'
trainingPath = '/home/lixusheng/kitti/training/'
testingPath = '/home/lixusheng/kitti/testing/'

errorFlag = 0

def calibLabelFileGen(Path, fname, istrain=True):
    if istrain:
        if os.path.exists(Path + "label_2/" + fname.replace('json', 'txt')):
            os.remove(Path + "label_2/" + fname.replace('json', 'txt'))
        with open(label_path + fname)as fp:
            # json content
            jsonContent = json.load(fp)
            if (len(jsonContent) == 0):
                print("the json annotation file is empty, please check file: ", fname)
                return 1
            else:           
                for i in range(len(jsonContent)):
                    content = jsonContent[i]
                    psr = content["psr"]
                    position = psr["position"]
                    scale = psr["scale"]
                    rotation = psr["rotation"]
                    #lidar -> camera
                    pointXYZ = np.array([position["x"], position["y"], position["z"], 1]).T 
                    camPosition = np.matmul(Tr_velo_to_cam, pointXYZ) #Tr_velo_to_cam @ pointXYZ camera coordinate position
                    #print(invExtrinsic @ pointXYZ)


                    # kitti content
                    kittiDict = {}
                    kittiDict["objectType"] = content["obj_type"]
                    kittiDict["truncated"] = "1.0"
                    kittiDict["occluded"] = "0"
                    kittiDict["alpha"] = "0.0"
                    kittiDict["bbox"] = [0.00, 0.00, 50.00, 50.00]  # should be higher than 50
                    kittiDict["diamensions"] = [scale['z'], scale['y'], scale['x']] #height, width, length
                    kittiDict["location"] = [camPosition[0], camPosition[1] + float(scale["z"])/2 , camPosition[2]  ] # camera coordinate
                    kittiDict["rotation_y"] = -math.pi/2 - rotation["z"]
                    
                    # write txt files
                    
                    
                    with open(Path + "label_2/" + fname.replace('json', 'txt'), 'a+') as f:
                        for item in kittiDict.values():
                            if isinstance(item, list):
                                for temp in item:
                                    f.writelines(str(temp) + " ")
                            else:      
                                f.writelines(str(item)+ " ")
                        f.writelines("\n")

    # write calibration files
    with open(Path + "calib/" + fname.replace('json', 'txt'), 'w') as f:
        P2 =  np.array(intrinsic).reshape(3,3)
        P2 = np.insert(P2, 3, values=np.array([0,0,0]), axis=1)

        f.writelines("P0: ")
        for num in P2.flatten():
            f.writelines(str(num)+ " ")
        f.writelines("\n")

        f.writelines("P1: ")        
        for num in P2.flatten():
            f.writelines(str(num)+ " ")
        f.writelines("\n")


        f.writelines("P2: ")
        for num in P2.flatten():
            f.writelines(str(num)+ " ")
        f.writelines("\n")

        f.writelines("P3: ")
        for num in P2.flatten():
            f.writelines(str(num)+ " ")
        f.writelines("\n")


        f.writelines("R0_rect: ")
        for num in np.eye(3,3).flatten():
            f.writelines(str(num)+ " ")
        f.writelines("\n")

        f.writelines("Tr_velo_to_cam: ")
        for temp in Tr_velo_to_cam[:3].flatten():
            f.writelines(str(temp) + " ")
        f.writelines("\n")

        f.writelines("Tr_imu_to_velo: ")
        for temp in Tr_velo_to_cam[:3].flatten():
            f.writelines(str(temp) + " ")

    return 0


# SUST image coordinate, kitti camera coordinate
def getCalibMatrix():
    with open(calibFile) as fp:
        calib = json.load(fp)
    return calib["extrinsic"], calib["intrinsic"]


extrinsic, intrinsic = getCalibMatrix()

Tr_velo_to_cam = np.array(extrinsic).reshape(4,4)
print("Tr_velo_to_cam Extrinsic: ", Tr_velo_to_cam)


# read all annotation json files
files = os.listdir(label_path)
total_num = len(files)
testing_num = 4
training_num = total_num - testing_num
print("total files num:", total_num)
print("training files num:", training_num)
print("testing files num:", testing_num)





fileLists = ["test.txt", "train.txt", "trainval.txt", "val.txt"]
for fileName in fileLists:
    if fileName == "test.txt":
        with open(imageSetsPath + fileName, 'w') as f:
            for i in range(training_num,total_num):     
                errorFlag = calibLabelFileGen(testingPath ,files[i], istrain=False)
                if errorFlag:
                    pass
                else:    
                    shutil.copy(sourceImagePath  + files[i].replace("json", "bmp"), testingPath + 'image_2/' + files[i].replace("json", "bmp")) # copy Image
                    shutil.copy(sourceVelodynePath  + files[i].replace("json", "bin"), testingPath + 'velodyne/' + files[i].replace("json", "bin")) # copy bin
                    f.writelines(files[i].strip(".json") + "\n")

    elif fileName == "train.txt":
        with open(imageSetsPath + fileName, 'w') as f:
            for i in range(training_num):
                errorFlag = calibLabelFileGen(trainingPath , files[i], istrain=True)
                if errorFlag:
                    pass
                else:
                    shutil.copy(sourceImagePath  + files[i].replace("json", "bmp"), trainingPath + 'image_2/' + files[i].replace("json", "bmp")) # copy Image
                    shutil.copy(sourceVelodynePath  + files[i].replace("json", "bin"), trainingPath + 'velodyne/' + files[i].replace("json", "bin")) # copy bin 
                    f.writelines(files[i].strip(".json") + "\n")   

    else:
        shutil.copy(imageSetsPath + 'train.txt', imageSetsPath + fileName)


