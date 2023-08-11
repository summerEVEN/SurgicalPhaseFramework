"""
为了修正之前的图片颜色错误
"""
import os
import cv2

def exchange(root_image_path):
    for filename in os.listdir(root_image_path):
        image_path = os.path.join(root_image_path, filename)
        # print(image_path)
        image = cv2.imread(image_path)

        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_path, bgr_image)
    
    print(root_image_path, " success!!")

# 读取图像
root_image_path = '/home/ubuntu/disk1/Even/Dataset/cholec80/train_dataset/cutMargin'
# for i in range(43, 81):
#     path = os.path.join(root_image_path, str(i))
#     print(path)
#     exchange(path)

for i in range(10, 15):
    path = os.path.join(root_image_path, str(i))
    print(path)
    exchange(path)
