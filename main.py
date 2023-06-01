import cv2
import tensorflow as tf
from keras import models
import numpy as np

H_START = 250
H_END = 590
H_STEP = 10

ORIGIN_WIDTH = 1640
ORIGIN_HEIGHT = 590

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

SCALED_SIZE = 2048

SCALE_FACTOR = SCALED_SIZE / IMAGE_WIDTH

# 204
LEFT_PADDING = RIGHT_PADDING = (SCALED_SIZE - ORIGIN_WIDTH) // 2
# 729
TOP_PADDING = BOTTOM_PADDING = (SCALED_SIZE - ORIGIN_HEIGHT) // 2

model = models.load_model("./model/supercombo")
model.summary()

try:
    from keras.utils import plot_model

    plot_model(model, show_shapes=True, to_file='model.png')
except Exception as e:
    print(e)


def fit_image(x, y):
    x = int(x)
    y = int(y)

    return x, y


def pred_and_draw(pred_image, origin_image):
    predict = model(np.array([pred_image]), training=False)
    outputs = predict

    # (108)
    left = outputs[0, :108]
    nearby_left = outputs[0, 108:108 * 2]
    nearby_right = outputs[0, 108 * 2:108 * 3]
    right = outputs[0, 108 * 3:]

    # (3)
    left_cls = left[:3]
    nearby_left_cls = nearby_left[:3]
    nearby_right_cls = nearby_right[:3]
    right_cls = right[:3]

    # (3, 35)
    left_lane = tf.reshape(left[3:], shape=[3, 35])
    nearby_left_lane = tf.reshape(nearby_left[3:], shape=[3, 35])
    nearby_right_lane = tf.reshape(nearby_right[3:], shape=[3, 35])
    right_lane = tf.reshape(right[3:], shape=[3, 35])

    best_left_idx = tf.argmax(left_cls)
    best_nearby_left = tf.argmax(nearby_left_cls)
    best_nearby_right = tf.argmax(nearby_right_cls)
    best_right = tf.argmax(right_cls)

    best_left_lane = left_lane[best_left_idx]
    best_nearby_left_lane = nearby_left_lane[best_nearby_left]
    best_nearby_right_lane = nearby_right_lane[best_nearby_right]
    best_right_lane = right_lane[best_right]

    for step in range(H_START, H_END + 1, H_STEP):
        idx = (step - H_START) // 10

        left_x, left_y = fit_image(best_left_lane[idx], step)
        if left_x > 0:
            cv2.circle(origin_image, (left_x, left_y), 2, (255, 0, 0), thickness=-1)

        nearby_left_x, nearby_left_y = fit_image(best_nearby_left_lane[idx], step)
        if nearby_left_x > 0:
            cv2.circle(origin_image, (nearby_left_x, nearby_left_y), 2, (0, 255, 0), thickness=-1)

        nearby_right_x, nearby_right_y = fit_image(best_nearby_right_lane[idx], step)
        if nearby_right_x > 0:
            cv2.circle(origin_image, (nearby_right_x, nearby_right_y), 2, (0, 0, 255), thickness=-1)

        right_x, right_y = fit_image(best_right_lane[idx], step)
        if right_x > 0:
            cv2.circle(origin_image, (right_x, right_y), 2, (0, 255, 255), thickness=-1)

    return origin_image


test_image_path = "./test_data/00000.jpg"

origin_image = cv2.imread(test_image_path)
origin_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)
image = cv2.copyMakeBorder(origin_image, TOP_PADDING, BOTTOM_PADDING,
                                       LEFT_PADDING, RIGHT_PADDING,
                                       cv2.BORDER_CONSTANT, value=(0, 0, 0))
image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
image = image / 255.0 - 0.5

rst = pred_and_draw(image, origin_image)

rst = cv2.cvtColor(rst, cv2.COLOR_RGB2BGR)
cv2.imwrite("result.jpg", rst)
cv2.imshow("", rst)
cv2.waitKey(0)


