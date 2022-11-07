import numpy as np
import glob
from PIL import Image
from patchify import patchify
import cv2

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
    # Get names of all image files in dataset and save the images as np arrays
    for f in glob.iglob("./Datasets/" + datasetPath + "/*"):
        # remove path before file name and file extension
        name = (f.split("\\")[1]).split(".")[0]
        images.append((name, np.array(Image.open(f))))

    for image in images:
        (name, img) = image
        patches = patchify(img, (33, 33, 3), step=14)
        counter = 0
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                counter += 1
                patch = Image.fromarray(patches[i, j, 0])
                patch.save("./Datasets/" + newPatchesPath + "/" +
                           name + "_" + str(counter) + ".png")


def makeBlurredImages(imagesPath, blurredPath):
    images = []

    for f in glob.iglob("./Datasets/" + imagesPath + "/*"):
        # remove path before file name and file extension
        name = (f.split("\\")[1]).split(".")[0]
        images.append((name, np.array(Image.open(f))))

    for image in images:
        (name, img) = image
        h, w, _ = img.shape
        low_res_img = cv2.resize(
            img, (int(w*1/3), int(h*1/3)), interpolation=cv2.INTER_CUBIC)

        high_res_upscale = cv2.resize(
            low_res_img, (w, h), interpolation=cv2.INTER_CUBIC)

        high_res_upscale_PIL = Image.fromarray(high_res_upscale)
        high_res_upscale_PIL.save(
            "./Datasets/" + blurredPath + "/" + name + "_blurred_" + ".png")



def makeBlurredImagesx2(imagesPath, blurredPath):
    images = []

    for f in glob.iglob("./Datasets/" + imagesPath + "/*"):
        # remove path before file name and file extension
        name = (f.split("\\")[1]).split(".")[0]
        images.append((name, np.array(Image.open(f))))

    for image in images:
        (name, img) = image
        h, w, _ = img.shape
        low_res_img = cv2.resize(
            img, (int(w*1/2), int(h*1/2)), interpolation=cv2.INTER_CUBIC)

        high_res_upscale = cv2.resize(
            low_res_img, (w, h), interpolation=cv2.INTER_CUBIC)

        high_res_upscale_PIL = Image.fromarray(high_res_upscale)
        high_res_upscale_PIL.save(
            "./Datasets/" + blurredPath + "/" + name + "_blurred_" + ".png")


def makeDatasetIntoPatchDatasetx3(datasetPath, newPatchesPath, lowResPath, highResPath):
    images = []
    # Get names of all image files in dataset and save the images as np arrays
    for f in sorted(glob.iglob("./Datasets/" + datasetPath + "/*")):
        # remove path before file name and file extension
        name = (f.split("\\")[1]).split(".")[0]
        images.append((name, np.array(Image.open(f))))

    for image in images:
        (name, img) = image
        patches = patchify(img, (33, 33, 3), step=14)
        counter = 0
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                counter += 1
                patch = patches[i, j, 0]
                patch_img = Image.fromarray(patch)
                patch_img.save("./Datasets/" + newPatchesPath +
                               "/" + name + "_" + str(counter) + ".png")

                h, w, _ = patch.shape
                low_res_img = cv2.resize(
                    patch, (int(w*1/3), int(h*1/3)), interpolation=cv2.INTER_CUBIC)

                high_res_upscale = cv2.resize(low_res_img, (w, h),
                                              interpolation=cv2.INTER_CUBIC)

                low_res_img_PIL = Image.fromarray(low_res_img)
                high_res_upscale_PIL = Image.fromarray(high_res_upscale)

                low_res_img_PIL.save(
                    "./Datasets/" + lowResPath + "/" + name + "_lowRes_" + str(counter) + ".png")
                high_res_upscale_PIL.save(
                    "./Datasets/" + highResPath + "/" + name + "_upscaled_" + str(counter) + ".png")


def makeDatasetIntoPatchDatasetx2(datasetPath, newPatchesPath, lowResPath, highResPath):
    images = []
    # Get names of all image files in dataset and save the images as np arrays
    for f in sorted(glob.iglob("./Datasets/" + datasetPath + "/*")):
        # remove path before file name and file extension
        name = (f.split("\\")[1]).split(".")[0]
        images.append((name, np.array(Image.open(f))))

    for image in images:
        (name, img) = image
        patches = patchify(img, (32, 32, 3), step=14)
        counter = 0
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                counter += 1
                patch = patches[i, j, 0]
                patch_img = Image.fromarray(patch)
                patch_img.save("./Datasets/" + newPatchesPath +
                               "/" + name + "_" + str(counter) + ".png")

                h, w, _ = patch.shape
                low_res_img = cv2.resize(
                    patch, (int(w*1/2), int(h*1/2)), interpolation=cv2.INTER_CUBIC)

                high_res_upscale = cv2.resize(low_res_img, (w, h),
                                              interpolation=cv2.INTER_CUBIC)

                low_res_img_PIL = Image.fromarray(low_res_img)
                high_res_upscale_PIL = Image.fromarray(high_res_upscale)

                low_res_img_PIL.save(
                    "./Datasets/" + lowResPath + "/" + name + "_lowRes_" + str(counter) + ".png")
                high_res_upscale_PIL.save(
                    "./Datasets/" + highResPath + "/" + name + "_upscaled_" + str(counter) + ".png")


# ----------- code ------------
#makeDatasetIntoPatches("T91/T91_Original", "T91/T91_HR_Patches")
#makeLowAndHighResPatches("T91/T91_HR_Patches", "T91/T91_LR_Patches", "T91/T91_Upscaled_Patches")
#makeDatasetIntoPatchDataset("T91/T91_Original", "T91/T91_HR_Patches", "T91/T91_LR_Patches", "T91/T91_Upscaled_Patches")
#makeBlurredImages("Set5/Original", "Set5/Blurred")
#makeBlurredImagesx2("Set19/Original", "Set19/Blurred_x2")

#makeDatasetIntoPatchDatasetx2("T91/T91_Original", "T91/T91_HR_Patches_x2", "T91/T91_LR_Patches_x2", "T91/T91_Upscaled_Patches_x2")

