#!/usr/bin/env sh
# Create lmdb inputs
# N.B. set the path to train + val data dirs

USAGE="usage: $0 data_name"

if [ $# -ne 1 ]; then
	echo "$USAGE" >&2
	exit 1
fi

NAME="$1"
DATA="datasets/$NAME/data"
TOOLS="$CAFFE_ROOT/build/tools"

TRAIN_DATA_ROOT="$DATA/train/"
VAL_DATA_ROOT="$DATA/val/"
LMDB_ROOT="$DATA/lmdb"

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=false
if $RESIZE; then
  RESIZE_HEIGHT=256
  RESIZE_WIDTH=256
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_net.sh to the path" \
       "where the training data is stored."
  exit 1
fi

if [ ! -d "$VAL_DATA_ROOT" ]; then
  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
  echo "Set the VAL_DATA_ROOT variable in create_net.sh to the path" \
       "where the validation data is stored."
  exit 1
fi

echo "Creating train lmdb..."

GLOG_logtostderr=1 "$TOOLS/convert_imageset" \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    "$TRAIN_DATA_ROOT" \
    "$DATA/train.txt" \
    "$LMDB_ROOT/${NAME}_train_lmdb"

echo "Creating val lmdb..."

GLOG_logtostderr=1 "$TOOLS/convert_imageset" \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    "$VAL_DATA_ROOT" \
    "$DATA/val.txt" \
    "$LMDB_ROOT/${NAME}_val_lmdb"

echo "Done."
