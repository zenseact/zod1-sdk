# Detectron2 object detection example

This is a simple example that uses the detectron2 api to train and run inference with a FasterRCNN.

Note that the dependencies are not included in the overall requirements.txt. We have used the `nvcr.io/nvidia/pytorch:21.08-py3` image with detectron2 and the dependencies from requirements.txt.

The following should work, even though versions will differ slightly:
```
pip install -r requirements.txt
pip install torch torchvision
pip install git+https://github.com/facebookresearch/detectron2.git
