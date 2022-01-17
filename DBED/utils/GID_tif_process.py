import numpy as np
import os
import skimage.io as skio


rgb_tif_path = "/home/FAKEDATA/GID/Large-scale Classification_5classes/image_RGB"
rgb_lbl_path = "/home/FAKEDATA/GID/Large-scale Classification_5classes/label_5classes"
save_img_path = "/home/FAKEDATA/GID/Large-scale Classification_5classes/rgb_224/JPEGImages"
save_lbl_path = "/home/FAKEDATA/GID/Large-scale Classification_5classes/rgb_224/SegmentationClass"
tif_names = os.listdir(rgb_tif_path)
for tif_name in tif_names:
    tif_path = os.path.join(rgb_tif_path, tif_name)
    lbl_name = tif_name[:-4] + "_label.tif"
    lbl_path = os.path.join(rgb_lbl_path, lbl_name)
    tif_data = skio.imread(tif_path, plugin="tifffile")
    lbl_data = skio.imread(lbl_path, plugin="tifffile")
    lbl_np = np.asarray(lbl_data)
    tif_np = np.asarray(tif_data)
    col = lbl_np.shape[0]
    # 6800
    row = lbl_np.shape[1]
    # 7200
    patch_size = 224
    col_num = col // patch_size
    row_num = row // patch_size
    for i in range(col_num):
        for j in range(row_num):
            new_img = tif_np[(i * patch_size): ((i+1) * patch_size), (j * patch_size): (j+1) * patch_size, :]
            # new img [224, 224, 3]
            # dtype = uint8
            temp_lbl = lbl_np[(i * patch_size): ((i+1) * patch_size), (j * patch_size): (j+1) * patch_size, :]
            new_lbl = np.ndarray([224, 224])
            # new img [224, 224, 3]
            # dtype = uint8
            for x in range(patch_size):
                for y in range(patch_size):
                    if all(temp_lbl[x][y] == [0, 0, 0]):
                        new_lbl[x][y] = 0
                    elif all(temp_lbl[x][y] == [255, 0, 0]):
                        new_lbl[x][y] = 1
                    elif all(temp_lbl[x][y] == [0, 255, 0]):
                        new_lbl[x][y] = 2
                    elif all(temp_lbl[x][y] == [0, 255, 255]):
                        new_lbl[x][y] = 3
                    elif all(temp_lbl[x][y] == [255, 255, 0]):
                        new_lbl[x][y] = 4
                    elif all(temp_lbl[x][y] == [0, 0, 255]):
                        new_lbl[x][y] = 5
                    else:
                        print(temp_lbl[x][y])
                        print("WARNING!!!")
            new_lbl = new_lbl.astype(dtype=np.uint8)
            if new_lbl.min() == new_lbl.max():
                continue
            else:
                count_0 = np.sum(new_lbl == 0)
                count_1 = np.sum(new_lbl == 0)
                count_2 = np.sum(new_lbl == 0)
                count_3 = np.sum(new_lbl == 0)
                count_4 = np.sum(new_lbl == 0)
                count_5 = np.sum(new_lbl == 0)
                if count_0 > 45000 or count_1 > 45000 or count_2 > 45000 or count_3 > 45000 or count_4 > 45000 or count_5 > 45000:
                    continue
                else:
                    new_img_name = tif_name[:-4] + "_{}".format(i*col_num+j) + ".npy"
                    new_lbl_name = tif_name[:-4] + "_{}".format(i*col_num+j) + ".npy"
                    np.save(os.path.join(save_img_path, new_img_name), new_img)
                    np.save(os.path.join(save_lbl_path, new_lbl_name), new_lbl)
