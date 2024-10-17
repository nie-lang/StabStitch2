#### Pre-trained model
For StabStitch-D dataset, the pre-trained models (spatial_warp.pth, temporal_warp.pth, and smooth_warp.pth) are available at [Google Drive](https://drive.google.com/drive/folders/1uqX7n8yXLTo2y_by71LOmCphHdZFiu3s?usp=sharing) or [Baidu Cloud](https://pan.baidu.com/s/1vWa3UlzLGVtsdki5-dBDxQ) (extraction code: 1234). Please download them and put them in the 'Full_model_inference/full_model_ssd/' folder.  

For traditional datasets, the pre-trained models (spatial_warp.pth, temporal_warp.pth, and smooth_warp.pth) are available at [Google Drive](https://drive.google.com/drive/folders/1yBzID35QDfeGinNSqGT_qyvjKnuS6l1S?usp=sharing) or [Baidu Cloud](https://pan.baidu.com/s/1zvpikaFH5Mhz8Rh2ouhhVQ) (extraction code: 1234). Please download them and put them in the 'Full_model_inference/full_model_tra/' folder.

Or, you can follow the training steps of SpatialWarp, TemporalWarp, and SmoothWarp to get the model files. Once the training process is done, please rename these model files as spatial_warp.pth, temporal_warp.pth, and smooth_warp.pth. Then, put them in the 'Full_model_inference/full_model_ssd/' or 'Full_model_inference/full_model_tra/' folder.

### Inference on the StabStitch-D dataset
Modify the test_path in test_online_ssd.py and run:
```
python test_online_ssd.py
```
Then, a folder named 'results_ssd' will be created automatically to store the stitched videos.  

In addition to test_path, you can also change the warp_mode and fusion_mode as described in the code.


#### Calculate the metrics on the StabStitch-D dataset
Modify the test_path in test_metric_ssd.py and run:
```
python test_metric_ssd.py
```

### Inference on traditional datasets
Modify the test_path in test_online_tra.py and run:
```
python test_online_tra.py
```
Then, a folder named 'results_tra' will be created automatically to store the stitched videos.  


### Multi-Video Stitching
Modify the video1_path, video2_path, and video3_path in test_online_tra_threeview.py and run:
```
python test_online_tra_threeview.py
```
Then, a stitched video named out.mp4 will be generated.

Note: Here, we only implement an example of three video stitching. The program logic can be easily extended to more videos (>3).
