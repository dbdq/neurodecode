#!/bin/bash
conda info -e

if [[ -z $PYCNBI_ROOT ]]; then
	echo Please set PYCNBI_ROOT variable in your environment.
	exit
else
	cd $PYCNBI_ROOT/pycnbi/stream_viewer/
	python stream_viewer.py
fi

