from PIL import Image
import os
import glob
import numpy as np
import shutil

folder1 = "./data_new/train_label"
folder2 = "./data_new/train_image"

#os.makedirs(folder1)
#os.makedirs(folder2)

for filename in os.listdir(folder1):
    file_path1 = os.path.join(folder1, filename)
    os.unlink(file_path1)

for filename in os.listdir(folder2):
    file_path2 = os.path.join(folder2, filename)
    os.unlink(file_path2)

print("done first")
a_pos = []
b_pos = []

x = []
y = x
m = 0
while m != 9728 + 256:
    x.append(m)
    m += 256

for ap in x:
    for bp in y:

        filename = "data3/train/label/E46N26_" + str(ap) + "_" + str(bp) + ".png"
        name = "E46N26_" + str(ap) + "_" + str(bp) + ".png"
        image = Image.open(filename)

        red = np.sum(np.array(image)[:, :, 0] == 255)
        green = np.sum(np.array(image)[:, :, 1] == 255)
        blue = np.sum(np.array(image)[:, :, 2] == 255)

        # for j in range(0, 256):
        #     for i in range(0, 256):
        #         r, g, b, z = image.getpixel((i, j))
        #         if r == 255:
        #             red += 1
        #         if g == 255:
        #             green += 1
        #         if b == 255:
        #             blue += 1

        s = red + blue + green

        r_per = red / s

        if r_per < 0.60:
            image.save('data_new/train_label/' + name, 'PNG')
            a_pos.append(name)
            b_pos.append([ap, bp])

print("done second")
##############################

for [m, n] in b_pos:
    f1 = "data3/train/image/E46N26_" + str(m) + "_" + str(n) + "_2018_06.png"
    shutil.copyfile(f1, 'data_new/train_image/' + "E46N26_" + str(m) + "_" + str(n) + "_2018_06.png")

    f2 = "data3/train/image/E46N26_" + str(m) + "_" + str(n) + "_2018_07.png"
    shutil.copyfile(f2, 'data_new/train_image/' + "E46N26_" + str(m) + "_" + str(n) + "_2018_07.png")

    f3 = "data3/train/image/E46N26_" + str(m) + "_" + str(n) + "_2018_08.png"
    shutil.copyfile(f3, 'data_new/train_image/' + "E46N26_" + str(m) + "_" + str(n) + "_2018_08.png")
