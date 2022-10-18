#!/bin/bash
conda info -e

if [[ -n "$1" ]]; then
	fif=$1
	if [[ -n "$2" ]]; then
		chunk=16
	else
		chunk=$2
	fi
    python -c "if __name__ == '__main__': import pycnbi.stream_player.stream_player as m; m.stream_player('StreamPlayer', '${fif}', ${chunk})"
else
    echo Usage: $0 {fif_file} [chunk_size=16]
fi

