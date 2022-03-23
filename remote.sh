#!/bin/bash
set -x

TEMP_FILE=.tmp
PROJECT_NAME=$(basename "`pwd`")
echo "Run training for project $PROJECT_NAME"

/bin/cat <<EOM >$TEMP_FILE
.git/*
logs/*
outputs/*
weights/*
notebooks/*
*.flac
*.mp4
*.jpg
*.png
*.tar
*.dcm
*.mp3
*.dicom
*.xml
*.swp
*.pb
*.tmp
*.csv
data/train/
data/test/
data/index/
*.onnx
*.zip
*.parquet
*.html
*.feather
*.pth
*.ipynb
*.wav
*.pkl
*.npy
*.jpg
*.zip
EOM

if [ "$1" == "gpu-32gb" ]; then
    echo "Push code to instance-gpu-32gb.us-central1-a.momovn-dev"
    IP="instance-gpu-32gb.us-central1-a.momovn-dev"
    REMOTE_HOME="/home/nhan_nguyen5_mservice_com_vn"
elif [ "$1" == "vm-ssd" ]; then
    echo "Push code to cv-vm-gpu-attach-local-ssd.us-central1-a.momovn-dev"
    IP="cv-vm-gpu-attach-local-ssd.us-central1-a.momovn-dev"
    REMOTE_HOME="/home/nhan_nguyen5_mservice_com_vn"
else
    echo "Unknown instance"
    exit
fi

# push code to server
rsync -vr -P -e "ssh" --exclude-from $TEMP_FILE "$PWD" $IP:$REMOTE_HOME/

# remove temp. file
rm $TEMP_FILE
