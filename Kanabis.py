import cv2
import numpy as np
import matplotlib.pyplot as plt

def hist(img):
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




if __name__ == '__main__':
    print("main")
    img_path = "2/530_0.JPG"
    img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
    # img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist(img)
    r,g,b = cv2.split(img)
    plt.imshow(g)
    plt.show()

    print(img.shape)


