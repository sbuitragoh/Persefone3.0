import cv2 as cv
import os
from data import preproc as pp
from network import model
from torchvision import transforms as T
import numpy as np
import torch


def all_images():

    file_path = 'data/TIFF Files/'
    for f in os.listdir(file_path):
        print(f)
        current_image = cv.imread(file_path+f)
        current_image = cv.cvtColor(current_image, cv.COLOR_BGR2GRAY)
        ct_image = pp.preproccess(current_image)
        rtt_img = pp.rotate_image(ct_image)
        cv.imwrite('./data/TIFF Rotated/'+f, rtt_img)
        lines_in_rotated = pp.line_obtention(rtt_img)
        pp.line_management(img=rtt_img,
                           lines=lines_in_rotated,
                           path=f[:-4])
        print('Página '+f+' Completada!')


def sample_image(path):
    img = cv.imread(path)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    cut_page = pp.preproccess(img)
    rotated_image = pp.rotate_image(cut_page)
    lines_in_rotated = pp.line_obtention(rotated_image)

    pp.line_management(img=rotated_image,
                       lines=lines_in_rotated,
                       path=path[-14:-4])


def model_test(line_img):
    img = cv.imread(line_img, 0)
    img = cv.bitwise_not(img)
    img = np.repeat(img[..., np.newaxis], 3, -1)
    img = pp.normalization(img)

    prep = T.Compose([
        T.ToTensor(),
        T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ])
    test_tensor = prep(img)
    test_batch = test_tensor.unsqueeze(0)
    modelito = model.make_model(2)
    h = modelito.forward(test_batch)
    tensor_2_numpy = np.transpose(h.detach().numpy())

    return 0


if __name__ == '__main__':
    img_path = 'data/TIFF Files/rm_149_017.tif'
    # sample_image(img_path)
    # all_images()

    model_test('data/TIFF Lines/rm_149_005/line_2.tif')
