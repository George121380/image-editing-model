### Set up a conda environment, and download a pretrained model:
```
conda env create -f environment.yaml
conda activate ip2p
bash scripts/download_checkpoints.sh
```

### Edit a single image:
```
python edit_cli.py --input imgs/example.jpg --output imgs/output.jpg --edit "turn him into a cyborg"

# Optionally, you can specify parameters to tune your result:
# python edit_cli.py --steps 100 --resolution 512 --seed 1371 --cfg-text 7.5 --cfg-image 1.2 --input imgs/example.jpg --output imgs/output.jpg --edit "turn him into a cyborg"
```

### Or launch your own interactive editing Gradio app:
```
python edit_app.py 
```
![Edit app](https://github.com/timothybrooks/instruct-pix2pix/blob/main/imgs/edit_app.jpg?raw=true)

_(For advice on how to get the best results by tuning parameters, see the [Tips](https://github.com/timothybrooks/instruct-pix2pix#tips) section)._

## Setup

Install all dependencies with:
```
conda env create -f environment.yaml
```

Download the pretrained models by running:
```
bash scripts/download_checkpoints.sh
```

train the image editing model
```
python main.py --name default --base configs/train.yaml --train --gpus=0
```
