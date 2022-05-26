import cv2
import numpy as np
import matplotlib.pyplot as plt


def hist(img):
    """
    make color distribution histogram
    :param img:
    :return:
    """
    h, s, v = cv2.split(img)
    hist_h = cv2.calcHist([h], [0], None, [256], [0, 256])
    hist_s = cv2.calcHist([s], [0], None, [256], [0, 256])
    hist_v = cv2.calcHist([v], [0], None, [256], [0, 256])
    plt.plot(hist_h, color='r', label="h/r")
    plt.plot(hist_s, color='g', label="s/g")
    plt.plot(hist_v, color='b', label="v/b")
    plt.legend()
    plt.title("Histogram")
    plt.xlabel("Color value")
    plt.ylabel("Pixel count")
    plt.show()


def areaFilter(minArea, inputImage):
    """
    filter img by color area
    :param minArea:
    :param inputImage:
    :return:
    """
    # Perform an area filter on the binary blobs:
    componentsNumber, labeledImage, componentStats, componentCentroids = \
        cv2.connectedComponentsWithStats(inputImage, connectivity=4)

    # Get the indices/labels of the remaining components based on the area stat
    # (skip the background component at index 0)
    remainingComponentLabels = [i for i in range(1, componentsNumber) if componentStats[i][4] >= minArea]

    # Filter the labeled pixels based on the remaining labels,
    # assign pixel intensity to 255 (uint8) for the remaining pixels
    filteredImage = np.where(np.isin(labeledImage, remainingComponentLabels) == True, 255, 0).astype('uint8')

    return filteredImage


def findBrown(img_hsv):
    """
    mask img:
    if in brown range -> 255
    else -> 0
    :param img_hsv:
    :return:
    """
    # brown color
    lower_values = np.array([6, 63, 0])
    upper_values = np.array([23, 255, 81])


    # Create the HSV mask
    mask = cv2.inRange(img_hsv, lower_values, upper_values)

    # Run a minimum area filter:
    minArea = 800
    mask_img = areaFilter(minArea, mask)
    return mask_img


def filterMask(masked_img):
    """
    filter the masked img to ignore layout
    :param masked_img:
    :return:
    """
    # Pre-process mask:
    kernelSize = 3

    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))
    iterations = 5

    mask = cv2.morphologyEx(masked_img, cv2.MORPH_DILATE, structuringElement, None, None, iterations,
                            cv2.BORDER_REFLECT101)
    mask = cv2.morphologyEx(masked_img, cv2.MORPH_ERODE, structuringElement, None, None, iterations,
                            cv2.BORDER_REFLECT101)
    return mask


if __name__ == '__main__':
    print("main")
    img_path = "2/rgb_0.JPG"
    img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    plt.imshow(img)
    plt.show()
    plt.imshow(img_hsv)
    plt.show()
    masked_img = findBrown(img_hsv)
    plt.imshow(masked_img)
    plt.show()
    filter_img = filterMask(masked_img)
    plt.imshow(filter_img)
    plt.show()

    a = np.where(filter_img == 255)
    s = a[0].size
    print(s)

    # sumMold = 0
    # sumNotMold = 0
    # for i in range(2, 93):
    #
    #     img = cv2.imread(str(i) + '\\rgb_0.JPG', cv2.COLOR_BGR2RGB)
    #     img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #     masked_img = findBrown(img_hsv)
    #     filter_img = filterMask(masked_img)
    #     a = np.where(filter_img == 255)
    #     s = a[0].size
    #
    #     if 7<=i<=75:
    #         sumNotMold += s
    #     else:
    #         sumMold +=s
    #
    #     if s > 100:
    #         print("img ", i , "mold")
    #         print(s)
    #     else:
    #         print("img ", i, "is no mold")
    #         print(s)
    #     plt.imshow(filter_img)
    #     plt.title("folder " + str(i))
    #     plt.show()
    #
    # avgMold = sumMold // 23
    # avgNotMold = sumNotMold // 68
    #
    # print("avgMold: ", avgMold)
    # print("avgNotMold: ", avgNotMold)

