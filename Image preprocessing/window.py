import numpy as np
import os
import SimpleITK as sitk
import glob


def window(volume):
    """Normalize the volume"""
    min = -75
    max = 175
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume


main_path = r'C:\Users\admin\Desktop\shuju'
all_files=[]
for root,dirnames,filenames in os.walk(main_path):
    for files in filenames:
        file_path = os.path.join(root,files)
        if 'nii' in file_path:
            all_files.append(file_path)

for file in all_files:
    file_path = file[:-4]
    print(file_path)
    img = sitk.ReadImage(file)
    img_arr = sitk.GetArrayFromImage(img)

    size = img.GetSize()
    origin = img.GetOrigin()
    spacing = img.GetSpacing()
    direction = img.GetDirection()
    pixelType = sitk.sitkUInt8

    # 调整窗宽、窗位
    new_img_arr = window(img_arr)

    image_new = sitk.Image(size, pixelType)

    image_new = sitk.GetImageFromArray(new_img_arr)
    image_new.SetDirection(direction)
    image_new.SetSpacing(spacing)
    image_new.SetOrigin(origin)
    sitk.WriteImage(image_new, file_path + '_wd.nii.gz')