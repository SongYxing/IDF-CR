## Dependencies and Installation

- Following [Spa-GAN](https://github.com/Penn000/SpA-GAN_for_cloud_removal)

## Train

```bash
# run train_woGAN.py for rice dataset
python train_woGAN.py

# run train_woGAN_2.py for dataset with sar
python train_woGAN_2.py

# run train_woGAN_whus2.py for whus2 dataset
python train_woGAN_whus2.py
```

## Inference

Download the pixel models [[Baidu Drive](https://pan.baidu.com/s/1EsbT-3bQKbBug7LPshAnnw  )] (code:1234) and run the inference code.

```bash
# for rice dataset
python predict.py

# for sar
python predict_2.py

# for whus2
python predict_whus2.py
```

## SAR-F

```bash
# SAR-Fusion
python mixup.py
```

