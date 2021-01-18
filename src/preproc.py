import numba as nb
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from skimage.measure import regionprops
import os


def preproccess(img):

    rr, _ = illumination_compensation(img)
    # rr_deslant = remove_cursive_style(rr)
    c_page = cut_page(rr)

    return c_page


"""
Description:

    This attempt will do an approximation of the treatment of a given TIFF image.
    This will held a given analysis on the properties of the image
"""


def illumination_compensation(img, only_cei=False):
    _, binary = cv.threshold(img, 254, 255, cv.THRESH_BINARY)

    if np.sum(binary) > np.sum(img) * 0.8:
        return np.asarray(img, type=np.uint8)

    def scale(img_to_scale):
        s = np.max(img_to_scale) - np.min(img_to_scale)
        res = img_to_scale / s
        res -= np.min(res)
        res *= 255

        return res

    img = img.astype(np.float32)
    height, width = np.shape(img)
    sqrt_hw = np.sqrt(height * width)

    bins = np.arange(0, 260, 10)
    bins[-1] = 255
    hp = np.histogram(img, bins)
    hr = np.where(hp[0] > sqrt_hw)[0][0] * 10

    np.seterr(divide='ignore', invalid='ignore')
    c = 0.3
    cei = (img - (hr + 50 * c)) * 2
    cei[cei > 255] = 255
    cei[cei < 0] = 0

    if only_cei:
        return np.asarray(cei, dtype=np.uint8)

    m1 = np.asarray([-1, 0, 1, -2, 0, 2, -1, 0, 1]).reshape((3, 3))
    m2 = np.asarray([-2, -1, 0, -1, 0, 1, 0, 1, 2]).reshape((3, 3))
    m3 = np.asarray([-1, -2, -1, 0, 0, 0, 1, 2, 1]).reshape((3, 3))
    # m4 = np.asarray([0, 1, 2, -1, 0, 1, -2, -1, 0]).reshape((3, 3))
    m4 = np.asarray([0, -1, -2, 1, 0, -1, 2, 1, 0]).reshape((3, 3))

    eg1 = np.abs(cv.filter2D(img, -1, m1))
    eg2 = np.abs(cv.filter2D(img, -1, m2))
    eg3 = np.abs(cv.filter2D(img, -1, m3))
    eg4 = np.abs(cv.filter2D(img, -1, m4))

    eg_avg = scale((eg1 + eg2 + eg3 + eg4) / 4)

    h = np.histogram(eg_avg, 255)[0]
    limit = int(np.shape(h)[0]/2)
    h_up = h[:limit]
    h_down = h[limit:]
    max_sections = [np.where(h == np.max(h_up))[0][0],
                    np.where(h == np.max(h_down))[0][0]]
    thc = int(np.mean(max_sections)*0.9)

    h, w = eg_avg.shape
    eg_bin = np.zeros((h, w))
    eg_bin[eg_avg >= thc//2] = 255

    h, w = cei.shape
    cei_bin = np.zeros((h, w))
    cei_bin[cei >= thc] = 255

    h, w = eg_bin.shape
    tli = 255 * np.ones((h, w))
    tli[eg_bin == 255] = 0
    tli[cei_bin == 255] = 0

    kernel = np.ones((3, 3), np.uint8)
    erosion = cv.erode(tli, kernel, iterations=1)
    int_img = np.asarray(cei)

    estimate_light_distribution(width, height, erosion, cei, int_img)

    mean_filter = 1 / 121 * np.ones((11, 11), np.uint8)
    ldi = cv.filter2D(scale(int_img), -1, mean_filter)

    result = np.divide(cei, ldi) * 260
    result[erosion != 0] *= 1.5
    result[result < 15] = 0
    result[result > 245] = 255

    return np.asarray(result, dtype=np.uint8), hp[0]


@nb.jit(nopython=True)
def estimate_light_distribution(width, height, erosion, cei, int_img):
    """Light distribution performed by numba (thanks @Sundrops)"""

    for y in range(width):
        for x in range(height):
            if erosion[x][y] == 0:
                i = x

                while i < erosion.shape[0] and erosion[i][y] == 0:
                    i += 1

                end = i - 1
                n = end - x + 1

                if n <= 30:
                    h, e = [], []

                    for k in range(5):
                        if x - k >= 0:
                            h.append(cei[x - k][y])

                        if end + k < cei.shape[0]:
                            e.append(cei[end + k][y])

                    mpv_h, mpv_e = max(h), max(e)

                    for m in range(n):
                        int_img[x + m][y] = mpv_h + (m + 1) * ((mpv_e - mpv_h) / n)

                x = end
                break


"""
Deslating image process based in,
    A. Vinciarelli and J. Luettin,
    A New Normalization Technique for Cursive Handwritten Wrods, in
    Pattern Recognition, 22, 2001.
"""


def remove_cursive_style(img):
    """Remove cursive writing style from image with deslanting algorithm"""

    def calc_y_alpha(vec):
        indices = np.where(vec > 0)[0]
        h_alpha = len(indices)

        if h_alpha > 0:
            delta_y_alpha = indices[h_alpha - 1] - indices[0] + 1

            if h_alpha == delta_y_alpha:
                return h_alpha * h_alpha
        return 0

    alpha_vals = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
    rows, cols = img.shape
    results = []

    ret, otsu = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    binary = otsu if ret < 127 else sauvola(img, (int(img.shape[0] / 2), int(img.shape[0] / 2)), 127, 1e-2)

    for alpha in alpha_vals:
        shift_x = max(-alpha * rows, 0.)
        size = (cols + int(np.ceil(abs(alpha * rows))), rows)
        transform = np.asarray([[1, alpha, shift_x], [0, 1, 0]], dtype=np.float)

        shear_img = cv.warpAffine(binary, transform, size, cv.INTER_NEAREST)
        sum_alpha = 0
        sum_alpha += np.apply_along_axis(calc_y_alpha, 0, shear_img)
        results.append([np.sum(sum_alpha), size, transform])

    result = sorted(results, key=lambda x: x[0], reverse=True)[0]
    result = cv.warpAffine(img, result[2], result[1], borderValue=255)
    result = cv.resize(result, dsize=(cols, rows))

    return np.asarray(result, dtype=np.uint8)


"""
Sauvola binarization based in,
    J. Sauvola, T. Seppanen, S. Haapakoski, M. Pietikainen,
    Adaptive Document Binarization, in IEEE Computer Society Washington, 1997.
"""


def sauvola(img, window, thresh, k):
    """Sauvola binarization"""

    rows, cols = img.shape
    pad = int(np.floor(window[0] / 2))
    sum2, sqsum = cv.integral2(
        cv.copyMakeBorder(img, pad, pad, pad, pad, cv.BORDER_CONSTANT))

    isum = sum2[window[0]:rows + window[0], window[1]:cols + window[1]] + \
        sum2[0:rows, 0:cols] - \
        sum2[window[0]:rows + window[0], 0:cols] - \
        sum2[0:rows, window[1]:cols + window[1]]

    isqsum = sqsum[window[0]:rows + window[0], window[1]:cols + window[1]] + \
        sqsum[0:rows, 0:cols] - \
        sqsum[window[0]:rows + window[0], 0:cols] - \
        sqsum[0:rows, window[1]:cols + window[1]]

    ksize = window[0] * window[1]
    mean = isum / ksize
    std = (((isqsum / ksize) - (mean**2) / ksize) / ksize) ** 0.5
    threshold = (mean * (1 + k * (std / thresh - 1))) * (mean >= 100)

    return np.asarray(255 * (img >= threshold), 'uint8')


"""
Cut the page to a reasonable size to work with. 
Based in previous works with Persefone.
"""


def cut_page(img):
    img_copy = img.copy()
    H, W = img_copy.shape
    margin = 5

    kernel = np.ones((5, 5), np.uint8)
    blur = cv.blur(img_copy, (30, 30), cv.BORDER_DEFAULT)
    blurred = cv.erode(blur, kernel, iterations=5)
    proy_hn = np.sum(blurred, axis=0) / np.max(np.sum(blurred, axis=0))
    proy_vn = np.sum(blurred, axis=1) / np.max(np.sum(blurred, axis=1))  # Eq. Lines
    mean_hn = np.mean(proy_hn) + np.std(proy_hn)
    mean_vn = np.mean(proy_vn) + np.std(proy_vn)
    mean_h = np.mean([1.0, mean_hn])
    mean_v = np.mean([1.0, mean_vn])

    # plt.subplot(121)
    # plt.plot(proy_hn)
    # plt.hlines(mean_h, 0, W, 'r')

    # plt.subplot(122)
    # plt.plot(proy_vn)
    # plt.hlines(mean_v, 0, H, 'r')

    # plt.show()

    proy_h = np.where(proy_hn >= mean_h)[0]
    proy_v = np.where(proy_vn >= mean_v)[0]
    margen_1 = 0
    margen_2 = H
    margen_3 = 0
    margen_4 = W
    margen_1_cnt = 0
    margen_2_cnt = 0
    margen_3_cnt = 0
    margen_4_cnt = 0

    for h in range(len(proy_h) - 1):
        h_1 = proy_h[h]
        h_2 = proy_h[h + 1]
        hh_1 = proy_h[(len(proy_h) - 1) - h]
        hh_2 = proy_h[(len(proy_h) - 1) - (h + 1)]
        if h_2 - h_1 > W // 10 and margen_3_cnt == 0:
            h_1 = h_1 - margin if h_1 > margin else h_1
            margen_3 = h_1 if h_1 < W//2 else 0
            margen_3_cnt = 1
        if hh_1 - hh_2 > W // 10 and margen_4_cnt == 0:
            hh_1 = hh_1 + margin if hh_1 < W - margin else hh_1
            margen_4 = hh_1 if hh_1 > W // 2 else W
            margen_4_cnt = 1

    for v in range(len(proy_v) - 1):
        v_1 = proy_v[v]
        v_2 = proy_v[v + 1]
        vv_1 = proy_v[(len(proy_v) - 1) - v]
        vv_2 = proy_v[(len(proy_v) - 1) - (v + 1)]
        if v_2 - v_1 > H // 20 and margen_1_cnt == 0:
            v_1 = v_1 - margin if v_1 > margin else v_1
            margen_1 = v_1 if v_1 < H // 2 else 0
            margen_1_cnt = 1
        if vv_1 - vv_2 > H // 20 and margen_2_cnt == 0:
            vv_1 = vv_1 + margin if vv_1 < H - margin else vv_1
            margen_2 = vv_1 if vv_1 > H // 2 else H
            margen_2_cnt = 1

    # horizontal_profile = np.sum(img, axis=1)
    # vertical_profile = np.sum(img, axis=0)
    # normalized_vertical = vertical_profile / np.max(vertical_profile)
    # normalized_horizontal = horizontal_profile / np.max(horizontal_profile)
    # data_stats = [np.mean(normalized_horizontal), np.std(normalized_horizontal),
    #              np.mean(normalized_vertical), np.std(normalized_vertical)]

    # limit_vertical = normalized_vertical.shape[0]//8
    # v_1 = normalized_vertical[:limit_vertical]
    # v_2 = normalized_vertical[int(-1.5*limit_vertical):]
    # lv = global_v[0] if data_stats[2] < global_v[0] else data_stats[2]
    # lh = global_h[0] if data_stats[0] < global_h[0] else data_stats[0]

    # margen_1 = np.where(v_1 >= lv)[0][-1] - 40
    # margen_2 = np.where(v_2 >= lv)[0][np.where(v_2 >= lv)[0]>100][0] + w - len(v_2) + 40

    # vector_hor = np.where(normalized_horizontal >= lh)[0]
    # branch = []
    # for i in range(len(vector_hor) - 1):
    #    if vector_hor[i + 1] - vector_hor[i] != 1:
    #        branch.append(vector_hor[i]) if vector_hor[i + 1] < h // 2 else branch.append(vector_hor[i + 1])

    # margen_3 = branch[0] - 40 if branch[0] > 40 else branch[0]
    # margen_4 = branch[-1] + 40 if branch[-1] + 40 < h else branch[-1]

    # plt.imshow(img, cmap='gray')
    # plt.vlines(margen_1, 0, H, 'r')
    # plt.vlines(margen_2, 0, H, 'r')
    # plt.hlines(margen_3, 0, W, 'r')
    # plt.hlines(margen_4, 0, W, 'r')
    # plt.show()

    return img[margen_1:margen_2, margen_3:margen_4]


"""
Line Obtention using the proyections and the values of it
as a distribution to use the mean and the standard deviation.
"""


def line_obtention(img):
    line_map = np.sum(img, axis=1)
    lines_map = line_map / np.max(line_map)
    lines_mean = np.mean(lines_map)
    lines_std = np.std(lines_map)
    set_limit = lines_mean + (lines_std/2)
    if set_limit > 1:
        set_limit = lines_mean

    set_lines = np.where(lines_map < set_limit)[0]
    cnt = 0
    num_1 = set_lines[0]
    lines_in_image = []
    whites_in_image = []

    for j in range(len(set_lines) - 1):
        if cnt == 0:
            num_1 = set_lines[j]
            cnt = 1

        if set_lines[j + 1] - set_lines[j] != 1 or set_lines[j + 1] == set_lines[-1]:
            num_2 = set_lines[j]
            new = (num_2 + num_1) // 2
            if len(lines_in_image) != 0:
                prev = lines_in_image.pop()
                if new - prev < 25:
                    new = (new + prev) // 2
                else:
                    lines_in_image.append(prev)
            lines_in_image.append(new)
            cnt = 0

    for k in range(len(lines_in_image) - 1):
        up_line = lines_in_image[k]
        down_line = lines_in_image[k + 1]
        line_section = lines_map[up_line:down_line]
        white_max = np.where(line_section == np.max(line_section))[0][0] + up_line
        if len(whites_in_image) > 0:
            prev_white = whites_in_image.pop()
            if white_max - prev_white <= 33:
                whites_in_image.append((white_max+prev_white)//2)
            else:
                whites_in_image.append(prev_white)
                whites_in_image.append(white_max)
        else:
            whites_in_image.append(white_max)

    # plt.imshow(img, cmap='gray')
    # for l in lines_in_image:
    #     plt.hlines(l, 0, 1200, 'r')
    # for w in whites_in_image:
    #     plt.hlines(w, 0, 1200, 'b')
    # plt.show()

    return whites_in_image


"""
Using the lines found to calculate the angle of the page.
It's possible to use the bounding box but the method is yet
not used, still it will theoretically be better.
"""


def rotate_image(img):
    img_copy = img.copy()
    y, x = np.shape(img_copy)

    # TESTING
    kernel = np.ones((5, 5), np.uint8)
    blur = cv.blur(img_copy, (30, 30), cv.BORDER_DEFAULT)
    blurred = cv.erode(blur, kernel, iterations=3)
    left = blurred[:,:x // 2]
    right = blurred[:, x // 2:]
    left_sum = np.sum(left, axis=1)/np.max(np.sum(left, axis=1))
    right_sum = np.sum(right, axis=1) / np.max(np.sum(right, axis=1))
    # base = np.arange(len(left_sum))

    # fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, sharey=True)

    # ax1.plot(left_sum, base)
    # ax1.axis('off')
    # ax2.imshow(blurred, cmap='gray')
    # ax2.vlines(x // 2, 0, y//3, 'r')
    # ax2.axis('off')
    # ax3.plot(-1 * right_sum, base)
    # ax3.axis('off')

    # fig.show()

    blacks_left = np.where(left_sum <= np.mean(left_sum))[0]
    blacks_right  = np.where(right_sum <= np.mean(right_sum))[0]
    bl_lst = [blacks_left[0]]
    for bl in range(len(blacks_left) - 1):
        if blacks_left[bl + 1] - blacks_left[bl] != 1:
            bl_lst.append(blacks_left[bl])
            bl_lst.append(blacks_left[bl + 1])
    br_lst = [blacks_right[0]]
    for br in range(len(blacks_right) - 1):
        if blacks_right[br + 1] - blacks_right[br] != 1:
            br_lst.append(blacks_right[br])
            br_lst.append(blacks_right[br + 1])
    bl_lst = np.unique(bl_lst)
    br_lst = np.unique(br_lst)
    bbbl = []
    bbbr = []
    for bl_i in range(len(bl_lst)-1):
        range_bl = left_sum[bl_lst[bl_i]:bl_lst[bl_i+1]]
        local_min_bl = np.min(range_bl)
        if not bl_i % 2:
            bbbl.append(bl_lst[bl_i]+np.where(range_bl == local_min_bl)[0][0])
    for br_i in range(len(br_lst)-1):
        range_br = right_sum[br_lst[br_i]:br_lst[br_i+1]]
        local_min_br = np.min(range_br)
        if not br_i % 2:
            bbbr.append(br_lst[br_i]+np.where(range_br == local_min_br)[0][0])
    # CHECK PORTION
    if np.abs(bbbr[0]-bbbl[0]) > 50:
        min_left = bbbl[1]
        min_right = bbbr[1]
    else:
        min_left = bbbl[0]
        min_right = bbbr[0]

    lim = np.max([min_left, min_right]) - 25
    bbl_lst = np.array(bbbl)-lim
    bbr_lst = np.array(bbbr)-lim
    bbl_limit = np.max(bbl_lst)
    bbr_limit = np.max(bbr_lst)
    for l_left in bbl_lst:
        if np.abs(l_left) < np.abs(bbl_limit):
            bbl_limit = l_left
    for l_right in bbr_lst:
        if np.abs(l_right) < np.abs(bbr_limit):
            bbr_limit = l_right

    sec_l = np.where(bbl_lst == bbl_limit)[0][0]
    sec_r = np.where(bbr_lst == bbr_limit)[0][0]

    bbl = np.array(bbbl[sec_l:sec_l + 5])
    bbr = np.array(bbbr[sec_r:sec_r + 5])
    limit = 1000
    cat_1 = bbr-bbl
    for c_1 in cat_1:
        if np.abs(c_1) < np.abs(limit):
            limit = c_1
    y_side = limit
    x_side = x // 2
    angle_test = np.abs(np.arctan2(y_side, x_side))

    center = (x // 2, y // 2)
    rot_angle = angle_test * 180/np.pi
    if rot_angle > 3.38:
        rot_angle = 0.0
    scale = 0.95

    neg_image = cv.bitwise_not(img)
    rot_mat = cv.getRotationMatrix2D(center, rot_angle, scale)
    rotated = cv.warpAffine(neg_image, rot_mat, (x, y))
    rotated_image = cv.bitwise_not(rotated)

    # plt.imshow(rotated_image, cmap='gray')
    # plt.show()

    return rotated_image


"""
LINES
"""


def line_management(img, lines, path):
    lines_copy = lines
    img_copy = img

    parent_path = './src/TIFF Lines/'
    current_folder = path
    path_to_create = os.path.join(parent_path, current_folder)
    try:
        os.mkdir(path_to_create)
    except OSError as error:
        print(error)

    lines_copy *= 2
    lines_copy.sort()
    lines_copy.append(np.shape(img)[0])
    lines_copy.insert(0, 0)
    lines_copy = np.array(lines_copy)
    lines_copy = np.reshape(lines_copy, (-1, 2))
    total_lines = np.shape(lines_copy)[0]

    cnt_lines = 0
    for ll in range(total_lines):
        up, down = lines_copy[ll]
        if up == 0:
            down += 20
        elif down == lines_copy[-1][1]:
            up -= 20
        else:
            up -= 20
            down += 20

        line_section = img_copy[up:down]
        zero_mean = np.mean(line_section, axis=0)
        one_mean = np.mean(line_section, axis=1)
        th_ret = int(np.min([np.min(zero_mean), np.min(one_mean)]))
        ret, otsu = cv.threshold(line_section, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        binary = otsu if ret < th_ret else \
            sauvola(line_section, (int(line_section.shape[0] / 2), int(line_section.shape[0] / 2)), th_ret, 1e-2)

        inverted_section = cv.bitwise_not(binary)
        new_line = line_determination(line = inverted_section,
                                      limit = [lines_copy[ll][0] - up + 5, lines_copy[ll][1] - up - 5],
                                      img_line = line_section)
        cnt_lines += 1
        cv.imwrite(path_to_create+'/line_'+str(cnt_lines)+'.tif', new_line)


def line_determination(line, limit, img_line):

    line_copy = line.copy()
    inv_line_copy = np.bitwise_not(line_copy)
    line_components = cv.connectedComponentsWithStats(line_copy)
    props = regionprops(line_components[1])
    # centroids = line_components[3]
    # cent_x = centroids[:, 0]
    # cent_y = centroids[:, 1]
    # xy = np.vstack([cent_x, cent_y])
    # z = gaussian_kde(xy)(xy)
    # idx = z.argsort()
    # x, y, z = cent_x[idx], cent_y[idx], z[idx]
    # points = np.transpose(np.vstack([x, y]))
    dense_center = np.sum(inv_line_copy[limit[0]:limit[1]], axis=1)
    dense_line = limit[0] + np.where(dense_center == np.min(dense_center))[0][0]
    up_border = dense_line - (limit[1] - limit[0]) / 2
    box_inside = []

    for center in range(len(props)):
        py, px = props[center].centroid
        bounding = props[center].bbox
        convex_hull = props[center].convex_image
        up_limit = np.max([limit[0], up_border])
        in_line =  up_limit < py < limit[1]

        if not in_line:
            current_space = line_copy[bounding[0]:bounding[2], bounding[1]:bounding[3]]
            c_x, c_y = convex_hull.shape
            for m in range(c_x):
                for n in range(c_y):
                    if convex_hull[m][n] == 1:
                        current_space[m][n] = 0
        else:
            box_inside.append(bounding)

    kernel = np.ones((3,3), 'uint8')
    mask = cv.dilate(line_copy, kernel, iterations=1)
    m_x, m_y = np.shape(mask)
    new_line = img_line.copy()

    for Mx in range(m_x):
        for My in range(m_y):
            if mask[Mx][My] == 0:
                new_line[Mx][My] = 255
            else:
                new_line[Mx][My] = img_line[Mx][My]

    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, sharex=True)
    #
    # ax1.imshow(img_line, cmap='gray')
    # ax1.set_title('Linea Actual')
    #
    # ax2.imshow(inv_line_copy, cmap='gray')
    # ax2.scatter(x, y, c=z)
    # ax2.hlines(limit[0], 0, 1200, 'r')
    # ax2.hlines(limit[1], 0, 1200, 'r')
    # ax2.set_title('Ubicación de Centroides')
    #
    # ax3.imshow(inv_line_copy, cmap='gray')
    # for point in points:
    #     in_line = limit[0] < point[1] < limit[1]
    #     color_dot = 'g' if in_line else 'r'
    #     if limit[0] - 5 <= point[1] <= limit[0] + 5: color_dot = 'b'
    #     if limit[1] - 5 <= point[1] <= limit[1] + 5: color_dot = 'b'
    #     ax3.scatter(point[0], point[1], c=color_dot)
    # ax3.hlines(limit[0], 0, 1200, 'r')
    # ax3.hlines(limit[1], 0, 1200, 'r')
    # ax3.hlines(dense_line, 0, 1200, 'g')
    # ax3.hlines(up_border, 0, 1200, 'b')
    # ax3.set_title('Determinación de Centroides en línea')
    #
    # ax4.imshow(new_line, cmap='gray')
    # ax4.set_title('Linea Final')
    #
    # ax1.axis('off')
    # ax2.axis('off')
    # ax3.axis('off')
    # ax4.axis('off')
    # fig.tight_layout()
    #
    # fig.show()
    return new_line
