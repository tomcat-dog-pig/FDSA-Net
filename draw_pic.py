import cv2
import numpy as np
from matplotlib import pyplot as plt

# 图像路径
methods = [
    ('Input', 'input.png'),
    ('LIME', 'LIME.png'),
    ('RetinexNet', 'RetinexNet.png'),
    ('KinD', 'KinD.png'),
    ('LLFlow', 'LLFlow.png'),
    ('URetinex-Net', 'URetinex-Net.png'),
    ('Zero-DCE', 'Zero-DCE.png'),
    ('LLFormer', 'LLFormer.png'),
    ('FourLLIE', 'FourLLIE.png'),
    ('PPFormer', 'PPFormer.png'),
    ('QuadPrior', 'QuadPrior.png'),
    ('IGDFormer', 'IGDFormer.png'),
    ('URetinex-Net++', 'URetinex-Net++.png'),
    ('AGLLDiff', 'AGLLDiff.png'),
    ('our', 'ours.png'),
    ('GT', '111.png')
]

# 局部放大区域
crop1 = (50, 50, 120, 120)  # 红框
crop2 = (200, 200, 270, 270)  # 绿框

fig = plt.figure(figsize=(12, 8))

for i, (name, path) in enumerate(methods):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ax = fig.add_subplot(3, 4, i + 1)
    ax.imshow(img)
    ax.set_title(name, fontsize=10)
    ax.axis('off')

    # 加框
    cv2.rectangle(img, crop1[:2], crop1[2:], (255, 0, 0), 2)
    cv2.rectangle(img, crop2[:2], crop2[2:], (0, 255, 0), 2)

plt.tight_layout()
plt.savefig("result_compare.png", dpi=600)
plt.show()
