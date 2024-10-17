#### Train on the StabStitch-D dataset
Modify the train_path in train_ssd.py and run:
```
python train_ssd.py
```
#### Train on TraditionalDataset
Modify the train_path in train_tra.py and run:
```
python train_tra.py
```
#### Generate temporal warps and save them for the warp smoothing model
Modify the test_path in test_ssd.py or test_tra.py and run:
```
python test_ssd.py
```
or
```
python test_tra.py
```
#### NOTE
1. We only generate the temporal warps for the training set.
2. The traditional dataset is not split into training and testing sets. We merely validate the robustness of StabStitch++ on it.

