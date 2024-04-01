#!/bin/bash

IMAGE_PATH=$1
MASKED_DIR=$2
OUTPUT_DIR=$3

export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR=$MASKED_DIR
export OUTPUT_DIR=$OUTPUT_DIR

python scripts/extract-mask.py --image_path $IMAGE_PATH --output_dir $MASKED_DIR

python textual-inversion/textual_inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token="_cat_statue_" \
  --initializer_token="cat" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=3000 \
  --learning_rate=5.0e-04 --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir=$OUTPUT_DIR \
  --report_to="none" \
  --use_augmentations
