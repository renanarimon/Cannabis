import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

def plot_confusion(cf_matrix):
    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

    ax.set_title('Seaborn Confusion Matrix with labels\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])

    ## Display the visualization of the Confusion Matrix.
    plt.show()

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
    lower_values = np.array([10, 112, 99])
    upper_values = np.array([16, 194, 131])

    # Create the HSV mask
    mask = cv2.inRange(img_hsv, lower_values, upper_values)

    # Run a minimum area filter:
    minArea = 800
    mask_img = areaFilter(minArea, mask)
    return mask_img


def findWhite(img_hsv):
    """
    mask img:
    if in white range -> 255
    else -> 0
    :param img_hsv:
    :return:
    """
    # brown color
    lower_values = np.array([17, 69, 110])
    upper_values = np.array([27, 138, 170])

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


def minMax_hsv(img_hsv):
    h, s, v = cv2.split(img_hsv)
    print("min h: ", np.min(h), ", s: ", np.min(s), ", v: ", np.min(v))
    print("max h: ", np.max(h), ", s: ", np.max(s), ", v: ", np.max(v))


def imgShow(img_path):
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
    arr = np.where(filter_img == 255)
    si = arr[0].size

    print(si)


if __name__ == '__main__':
    print("main")
    # img_path_color = "4/white.jpg"
    # img_path = "22/rgb_0.JPG"
    # img = cv2.imread(img_path_color, cv2.COLOR_BGR2RGB)
    # img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #
    # imgShow(img_path)
    y_true = np.ones(92)
    y_true[0] = 0
    for i in range(7, 76):
        y_true[i] = 0

    y_pred = np.zeros(92)
    for i in range(2, 93):
        img = cv2.imread(str(i) + '\\rgb_0.JPG', cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        masked_img = findBrown(img_hsv)
        filter_img = filterMask(masked_img)
        a = np.where(filter_img == 255)
        s = a[0].size
        if s > 300:
            y_pred[i - 1] = 1

    print(y_true)
    print(y_pred)

    acc = accuracy_score(y_true, y_pred)
    print("accuracy: ",acc)

    con = confusion_matrix(y_true,y_pred)
    plot_confusion(con)


    # f = open("predict.txt", "a")
    # for i in range(2, 93):
    #     img = cv2.imread(str(i) + '\\rgb_0.JPG', cv2.COLOR_BGR2RGB)
    #     img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #     masked_img = findBrown(img_hsv)
    #     filter_img = filterMask(masked_img)
    #     a = np.where(filter_img == 255)
    #     s = a[0].size
    #     if s > 300:
    #         f.write("img "+ str(i)+ " --> 1\n")
    #     else:
    #         f.write("img " + str(i) + " --> 0\n")
    # f.close()
