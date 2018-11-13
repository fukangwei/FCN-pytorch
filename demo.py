import torch
from FCN import FCNs, VGGNet
from BagData import transform
import numpy as np
import cv2

if __name__ == "__main__":
    img = cv2.imread("./last/9.jpg")
    img = cv2.resize(img, (160, 160))
    torch_img = transform(img)
    torch_img = torch_img[None, :, :, :]
    model = torch.load('./checkpoints/fcn_model_95.pt')
    model = model.cpu()
    output = model(torch_img)
    output = output.data.numpy()
    output = np.argmin(output, axis=1)
    output = output * 255
    # When using "cv2.imshow", you will see a black image
    cv2.imwrite("output.jpg", output[0])
