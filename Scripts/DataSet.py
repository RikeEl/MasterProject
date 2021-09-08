import SimpleITK as sitk
import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
import math
import torch


class DataLoaderPNG8(Dataset):
    def __init__(self, ):
        imageDict = getImagesOfFolder("../Data/PNG8")
        patientData = []
        segmentation = []
        n_samples = len(imageDict["flair"])
        for i in range(n_samples):
            patientData.append(
                np.array([imageDict["flair"][i], imageDict["t1"][i], imageDict["t1ce"][i], imageDict["t2"][i]]))
            segmentation.append(
                torch.from_numpy(np.array(imageDict["seg"][i])[:, :, np.newaxis]).permute(2, 0, 1).numpy())
        self.patientData = (torch.from_numpy(np.array(patientData)) - 0) / (255.0-0)
        self.segmentation = torch.from_numpy(np.array(segmentation))
        self.n_samples = n_samples

    def __getitem__(self, index):
        print(index)
        return self.patientData[index], self.segmentation[index]

    def __len__(self):
        return self.n_samples


class DataLoaderPNG16(Dataset):
    def __init__(self, ):
        imageDict = getImagesOfFolder("../Data/PNG8")
        patientData = []
        segmentation = []
        n_samples = len(imageDict["flair"])
        for i in range(n_samples):
            patientData.append(
                np.array([imageDict["flair"][i], imageDict["t1"][i], imageDict["t1ce"][i], imageDict["t2"][i]]))
            segmentation.append(
                torch.from_numpy(np.array(imageDict["seg"][i])[:, :, np.newaxis]).permute(2, 0, 1).numpy())
        self.patientData = torch.from_numpy(np.array(patientData))
        self.segmentation = torch.from_numpy(np.array(segmentation))
        self.n_samples = n_samples

    def __getitem__(self, index):
        return self.patientData[index], self.segmentation[index]

    def __len__(self):
        return self.n_samples


def getImagesOfFolder(dir_path):
    dict = {"flair": [], "t1": [], "t1ce": [], "t2": [], "seg": []}
    for element in glob.glob(dir_path + "/*.png"):
        imageNames = element.split("/")[-1]
        modality = imageNames.split("_")[-1].split(".")[0]
        data = sitk.ReadImage(element)
        img = sitk.GetArrayFromImage(data)
        dict[modality].append(img)
    return dict
