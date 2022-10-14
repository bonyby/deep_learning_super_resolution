import numpy as np
import glob
from PIL import Image
from patchify import patchify

# image = Image.open("./Datasets/T91/T91_Original/t1.png")
# img = np.array(image)
# patches = patchify(img, (33,33,3), step=14)
# counter = 0
# for i in range(patches.shape[0]):
#     for j in range(patches.shape[1]):
#         counter += 1
#         patch = Image.fromarray(patches[i, j, 0])
#         patch.save("./Datasets/T91/T91_HR_Patches/t1_" + str(counter) + ".png")


def makeDatasetIntoPatches(datasetPath, newPatchesPath):
    images = []
    #Get names of all image files in dataset and save the images as np arrays
    for f in glob.iglob("./Datasets/" + datasetPath + "/*"):
        name = (f.split("\\")[1]).split(".")[0] # remove path before file name and file extension
        images.append((name, np.array(Image.open(f))))

    for image in images:
        (name, img) = image
        patches = patchify(img, (33,33,3), step=14) 
        counter = 0
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                counter += 1
                patch = Image.fromarray(patches[i, j, 0])
                patch.save("./Datasets/" + newPatchesPath + "/" + name + "_" + str(counter) + ".png")

# ----------- code ------------
#makeDatasetIntoPatches("T91/T91_Original", "T91/T91_HR_Patches"