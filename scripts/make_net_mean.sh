#!/usr/bin/env sh
# Compute the mean image from training lmdb

USAGE="usage: $0 data_name"

if [ $# -ne 1 ]; then
	echo "$USAGE" >&2
	exit 1
fi

NAME="$1"
DATA="datasets/$NAME/data"
TOOLS="$CAFFE_ROOT/build/tools"
LMDB_ROOT="$DATA/lmdb"

echo "Computing the mean image from training lmdb..."

$TOOLS/compute_image_mean $LMDB_ROOT/${NAME}_train_lmdb \
  $DATA/${NAME}_mean.binaryproto

echo "Done."
