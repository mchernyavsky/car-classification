#!/usr/bin/env sh

USAGE="usage: $0 solver_path snapshot_path"

if [ $# -ne 2 ]; then
	echo "$USAGE" >&2
	exit 1
fi

$CAFFE_ROOT/build/tools/caffe train --solver="$1" --snapshot="$2"
