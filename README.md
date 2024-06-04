# Msc-data-zhang
## First installation
1. Download data from [Google drive](https://drive.google.com/file/d/1GvNwL4iPcB2GRdK2n353bKiKV_Vnx7Qg/view?usp=drive_link)
2. Create new folder 'data' and extract
3. Run 'create_dataset.py' (not needed as new_dataset/ is on main)

## Run existing on Hábrók
1. ssh into usrname@gpu1.hpc.rug.nl
2. Enable the following modules for the segmentation models
```
module load CUDA/11.7.0
module load cuDNN/8.4.1.50-CUDA-11.7.0
```

## Training the GAN
```main_generation.py [-h] [-bs BATCH_SIZE] [-lr LEARNING_RATE] [-ld LATENT_DIM] [-ne NUM_EPOCHS] [-l LAMB] [-fd] [-ad]```

## Training the DeepLabV3 
```main_segmentation.py [-h] [-b BATCH_SIZE] [-lr LEARNING_RATE] [-l LOSS]```
Available loss function; 0: BCEDiceLoss, 1: IoULoss, 2: DiceLoss, 3: LovaszHingeLoss, 4: Binary_Xloss, 5: FocalLoss, 6: BCELoss, 7: DiceBCELoss

## Training the UNet
```main_unet_segmentation.py [-h] [-b BATCH_SIZE] [-lr LEARNING_RATE] [-l LOSS] [-m MODEL]```
Available loss function; 0: BCEDiceLoss, 1: IoULoss, 2: DiceLoss, 3: LovaszHingeLoss, 4: Binary_Xloss, 5: FocalLoss, 6: BCELoss, 7: DiceBCELoss

## Testing the Pipeline
```main_pipeline.py [-h] [-G]```
Enable GAN if specified, otherwise disable.