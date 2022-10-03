import os
import numpy as np
from tqdm import tqdm
import h5py
import nrrd


output_size =[112, 112, 80]
data_path = 'E:/data/LASet/origin'
out_path = 'E:/data/LASet/data'
def covert_h5():
    listt = os.listdir(data_path)
    for case in tqdm(listt):
        image, img_header = nrrd.read(os.path.join(data_path,case,'lgemri.nrrd'))
        label, gt_header = nrrd.read(os.path.join(data_path,case, 'laendo.nrrd'))
        label = (label == 255).astype(np.uint8)
        w, h, d = label.shape
        # 返回label中所有非零区域（分割对象）的索引
        tempL = np.nonzero(label)
        # 分别获取非零区域在x,y,z三轴的最小值和最大值，确保裁剪图像包含分割对象
        minx, maxx = np.min(tempL[0]), np.max(tempL[0])
        miny, maxy = np.min(tempL[1]), np.max(tempL[1])
        minz, maxz = np.min(tempL[2]), np.max(tempL[2])
        # 计算目标尺寸比分割对象多余的尺寸
        px = max(output_size[0] - (maxx - minx), 0) // 2
        py = max(output_size[1] - (maxy - miny), 0) // 2
        pz = max(output_size[2] - (maxz - minz), 0) // 2
        # 在三个方向上随机扩增
        minx = max(minx - np.random.randint(10, 20) - px, 0)
        maxx = min(maxx + np.random.randint(10, 20) + px, w)
        miny = max(miny - np.random.randint(10, 20) - py, 0)
        maxy = min(maxy + np.random.randint(10, 20) + py, h)
        minz = max(minz - np.random.randint(5, 10) - pz, 0)
        maxz = min(maxz + np.random.randint(5, 10) + pz, d)
        # 图像归一化，转为32位浮点数（numpy默认是64位）
        image = (image - np.mean(image)) / np.std(image)
        image = image.astype(np.float32)
        # 裁剪
        image = image[minx:maxx, miny:maxy, minz:maxz]
        label = label[minx:maxx, miny:maxy, minz:maxz]
        print(label.shape)

        case_dir = os.path.join(out_path,case)
        os.mkdir(case_dir)
        f = h5py.File(os.path.join(case_dir, 'mri_norm2.h5'), 'w')
        f.create_dataset('image', data=image, compression="gzip")
        f.create_dataset('label', data=label, compression="gzip")
        f.close()


if __name__ == '__main__':
    covert_h5()