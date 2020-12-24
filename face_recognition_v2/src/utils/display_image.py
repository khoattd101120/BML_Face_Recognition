import cv2

def scaling_image(image, scale_percent):
    """
    Resize the image
    :param image: cv2 image object
    :param scale_percent: an integer from 1 -> 100
    :return:
    """
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dimension = (width, height)
    return cv2.resize(image, dimension, interpolation=cv2.INTER_AREA)


def display_image(img, scaling=30):
    """
    Show the image with float64 or uint8 format using cv2.imshow()
    :param img: cv2 image object
    :param scaling: an integer from 1 -> 100
    :return:
    """
    if str(img.dtype) == 'float64' and np.amax(img) > 1:
        cv2.imshow("img", scaling_image(img / 255, scaling))
    else:
        cv2.imshow("img", scaling_image(img, scaling))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
