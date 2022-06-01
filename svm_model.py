import numpy as np
from sklearn import svm
import pandas as pd
import cv2

def build_df():
    img = cv2.imread(str(2) + '\\530_0.JPG')
    img_flat = img[:, :, 0].flatten()
    np.resize(img_flat, (1, img_flat.shape[0]))
    print(img_flat.shape)
    df = pd.DataFrame(img_flat)
    print(df)

if __name__ == '__main__':
    build_df()
    for i in range(2, 93):
        try:
            img = cv2.imread(str(i) + '\\530_0.JPG')
        except:
            print("file not find")




