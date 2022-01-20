# Evaluation example

Here is a simple example of how to use the evaluation program.

For convenience we have provided a `kitti-eval` docker image. If for some reason it is not available you can easily rebuild it with:

```docker build -f Dockerfile -t kitti-eval .```

Then, you can run an evaluation using the image with the following command:

```docker run -v /gt-dir:/gt-dir -v /det-dir:/det-dir kitti-eval ./evaluate_object_3d_offline /gt-dir /det-dir```

For an example of how the pseudo-annoation files should look you can see the prepare_eval.py. It takes our simple detections, extracts the relevant frames, and prepares them for use in the evaluation script.
For the LIDAR detections that includes converting them to the camera coordinate system as well as adjusting the (nonexistent) 2d box to pass the minimum height requirement.
Note that this is overly simplified and the projected 3d boxes could have a lower height and should thus actually be ignored.

Full flow (separate eval for camera 2d detections and lidar 3d detections):
1. ```python prepare_eval.py --tmp-det-dir=/tmp/det-dir```
2. ```chmod -R 777 /tmp/det-dir```
3. ```docker run -v /mnt/ai_sweden/road_data_lab/zenseact_disk/:/gt-dir -v /tmp/det-dir:/det-dir kitti-eval ./evaluate_object_3d_offline /gt-dir /det-dir/camera```
4. ```docker run -v /mnt/ai_sweden/road_data_lab/zenseact_disk/:/gt-dir -v /tmp/det-dir:/det-dir kitti-eval ./evaluate_object_3d_offline /gt-dir /det-dir/lidar```


## Results of evaluating provided 2d detections
```
vehicle_detection_AP : 74.971565
pedestrian_detection_AP : 52.170540
cyclist_detection_AP : 38.734921
```

## Results of evaluating provided 3d detections
```
vehicle_detection_BEV_AP : 72.898407
pedestrian_detection_BEV_AP : 48.200329
cyclist_detection_BEV_AP : 40.517864
vehicle_detection_3D_AP : 61.186440
pedestrian_detection_3D_AP : 41.484516
cyclist_detection_3D_AP : 37.284695
```