# Pocket Kitchen | Object Detector 
Pocket Kitchen is a system for managing food more efficiently in the home towards reducing household food waste. These scripts send detected incoming/outgoing food items in a fridge to a database.

- **On-device object detection** - Pocket Kitchen uses the Coral TPU and EdgeTPU TensorFlow models to perform object detection on-device. This means that no data is processed on the cloud, and all processing is done on the device itself.
- **Customizable** - Pocket Kitchen is designed to be customizable. For example, the user can add their own model trained on their own objects.
- **Open Source** - Pocket Kitchen is open source, so anyone can contribute to the project.

<img src="https://github.com/myPocketKitchen/PocketKitchen-API/assets/79009541/c1bb578f-e34d-4ed1-bd5e-82059b128ebd" width="600">

## System Architecture
While similar to an API, these scripts are running locally and populating a database. The database is then accessed by the Pocket Kitchen app to display the data.

<img src="https://github.com/myPocketKitchen/PK-FoodDetector/assets/79009541/925e18b6-0a6d-419e-a428-f41df2974bf4" width="600">

## Requirements


### Hardware
- Raspberry Pi 4
- Coral TPU
- Pi Camera (We use HD Camera Module V2, but any Pi camera should work. Autofocus is recommended.)
- [3D-Printed Camera Mount](https://www.thingiverse.com/thing:4420496)

<img src="https://github.com/myPocketKitchen/PocketKitchen-API/assets/79009541/7374fd10-497a-4e36-b763-45af3437dcb9" width="600">

### Model
- You can use the enclosed model or create your own custom fine-tune EfficientDet-Lite0 model from [TensorFlow Edge Object Detection](https://colab.research.google.com/github/google-coral/tutorials/blob/master/retrain_efficientdet_model_maker_tf2.ipynb)

## Installation
```
$ bash setup.sh
```

## Usage
```
$ python3 detect.py --model <model> 
```

## Future Work
- [ ] Add support for true API

## Resources
- [TensorFlow EfficientDet-Lite0 Object Detection for the Edge TPU](https://colab.research.google.com/github/google-coral/tutorials/blob/master/retrain_efficientdet_model_maker_tf2.ipynb)
