# openpilot_land_detect
train a openpilot land detect model (supercombo)

![](https://github.com/qibin0506/openpilot_land_detect_method/blob/master/result.jpg)
# How to use
1. clone this repos.
2. download and unzip [model.zip](https://github.com/qibin0506/openpilot_land_detect_method/releases/download/model.zip/model.zip).
3. repace the model dir.
4. run main.py.

# train method
1. predict the x axis of three lanes.
2. train a MHP loss to find the best lane.
