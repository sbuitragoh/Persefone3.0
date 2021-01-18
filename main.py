import cv2 as cv
import os
from src import preproc as pp


def all_images():

    file_path = './src/TIFF Files/'
    for f in os.listdir(file_path):
        print(f)
        current_image = cv.imread(file_path+f)
        current_image = cv.cvtColor(current_image, cv.COLOR_BGR2GRAY)
        ct_image = pp.preproccess(current_image)
        rtt_img = pp.rotate_image(ct_image)
        cv.imwrite('./src/TIFF Rotated/'+f, rtt_img)
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


if __name__ == '__main__':
    img_path = './src/TIFF Files/rm_149_017.tif'
    # sample_image(img_path)
    all_images()
