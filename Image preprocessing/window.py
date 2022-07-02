import numpy as np
import SimpleITK as sitk


def window(img):
    win_min = -1000  # 你定义的窗位
    win_max = 400  # 你定义的窗宽

    for i in range(img.shape[0]):
        img[i] = 255.0 * (img[i] - win_min) / (win_max - win_min)
        min_index = img[i] < 0
        img[i][min_index] = 0
        max_index = img[i] > 255
        img[i][max_index] = 255
        img[i] = img[i] - img[i].min()
        c = float(255) / img[i].max()
        img[i] = img[i] * c

    return img.astype(np.uint8)


img = sitk.ReadImage(file)
img_arr = sitk.GetArrayFromImage(img)

size = img.GetSize()
origin = img.GetOrigin()
spacing = img.GetSpacing()
direction = img.GetDirection()
pixelType = sitk.sitkUInt8

new_img_arr = window(img_arr)

image_new = sitk.Image(size, pixelType)

image_new = sitk.GetImageFromArray(new_img_arr)
image_new.SetDirection(direction)
image_new.SetSpacing(spacing)
image_new.SetOrigin(origin)

sitk.WriteImage(image_new, file_path + '_im.nii.gz')