import numpy as np
import glob
from PIL import Image
from patchify import patchify
import cv2
import math
from torchvision import transforms

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


# Takes scale as input
def makeBlurredImages(imagesPath, croppedPath, downscaledPath, blurredPath, scale):
    images = []
    gaussian_kernel = transforms.GaussianBlur(kernel_size=(3,3), sigma=0.7)

    for f in glob.iglob("./Datasets/" + imagesPath + "/*"):
        # remove path before file name and file extension
        name = (f.split("\\")[1]).split(".")[0]
        images.append((name, np.array(Image.open(f))))

    for image in images:
        (name, img) = image

        h, w, _ = img.shape

        # Cropping pictures to nearest modulo "scale" (lower than) so that everything rounds perfectly with x"scale" scaling
        newh = nearestRound(h, scale)
        neww = nearestRound(w, scale)
        img = crop_center(img, neww, newh)

        # Saving cropped images
        crop_PIL = Image.fromarray(img)
        crop_PIL.save("./Datasets/" + croppedPath + "/" + name + ".png")

        # low_res_img = cv2.resize(
        #     img, (int(w*1/scale), int(h*1/scale)), interpolation=cv2.INTER_CUBIC)

        # high_res_upscale = cv2.resize(
        #     low_res_img, (neww, newh), interpolation=cv2.INTER_CUBIC)

        gaus_pic = gaussian_kernel(crop_PIL)
        low_res = gaus_pic.resize(size=(w//scale,h//scale), resample=Image.BICUBIC)
        upscaled = low_res.resize(size=(w,h), resample=Image.BICUBIC)

        low_res.save("./Datasets/" + downscaledPath + "/" + name + "_blurred_" + ".png")
        upscaled.save("./Datasets/" + blurredPath + "/" + name + "_LR" + ".png")


def makeDatasetIntoPatchDataset(datasetPath, newPatchesPath, lowResPath, highResPath, scale):
    images = []
    gaussian_kernel = transforms.GaussianBlur(kernel_size=(3,3), sigma=0.7)
    # Get names of all image files in dataset and save the images as np arrays
    for f in sorted(glob.iglob("./Datasets/" + datasetPath + "/*")):
        # remove path before file name and file extension
        name = (f.split("\\")[1]).split(".")[0]
        images.append((name, np.array(Image.open(f))))

    if scale == 3:
        patchSize = 33
    elif scale == 2:
        patchSize = 32
    else:
        print("Error: Undefined dimensions for this scaling")
        return

    for image in images:
        (name, img) = image
        patches = patchify(img, (patchSize, patchSize, 3), step=14)
        counter = 0
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                counter += 1
                patch = patches[i, j, 0]
                patch_img = Image.fromarray(patch)
                # patch_img.save("./Datasets/" + newPatchesPath +
                #                "/" + name + "_" + str(counter) + ".png")

                h, w, _ = patch.shape
                # low_res_img = cv2.resize(
                #     patch, (int(w*1/scale), int(h*1/scale)), interpolation=cv2.INTER_CUBIC)

                # high_res_upscale = cv2.resize(low_res_img, (w, h),
                #                               interpolation=cv2.INTER_CUBIC)

                # low_res_img_PIL = Image.fromarray(low_res_img)
                # high_res_upscale_PIL = Image.fromarray(high_res_upscale)
                gaus_pic = gaussian_kernel(patch_img)
                low_res = gaus_pic.resize(size=(w//scale,h//scale), resample=Image.BICUBIC)
                upscaled = low_res.resize(size=(w,h), resample=Image.BICUBIC)

                low_res.save(
                    "./Datasets/" + lowResPath + "/" + name + "_lowRes_" + str(counter) + ".png")
                upscaled.save(
                    "./Datasets/" + highResPath + "/" + name + "_upscaled_" + str(counter) + ".png")


def nearestRound(n, scale):
    return n - n % scale


def crop_center(img, cropx, cropy):
    y, x, c = img.shape
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2
    return img[starty:starty+cropy, startx:startx+cropx, :]


# ----------- code ------------
#makeDatasetIntoPatches("T91/T91_Original", "T91/T91_HR_Patches")
#makeLowAndHighResPatches("T91/T91_HR_Patches", "T91/T91_LR_Patches", "T91/T91_Upscaled_Patches")
makeDatasetIntoPatchDataset("T91/T91_Original", "T91/T91_HR_Patches", "T91/T91_LR_Patches_x3gaussPIL", "T91/T91_Upscaled_Patches_x3gaussPIL", scale=3)
# makeBlurredImages("Set19/Original", "Set19/OriginalCropx3", "Set19/LR_gaussPIL", "Set19/Upscaled_gaussPIL", scale=3)
#makeBlurredImagesx2("Set19/Original", "Set19/LR_x2")

#makeDatasetIntoPatchDatasetx2("T91/T91_Original", "T91/T91_HR_Patches_x2", "T91/T91_LR_Patches_x2", "T91/T91_Upscaled_Patches_x2")
