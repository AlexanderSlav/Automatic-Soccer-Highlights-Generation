# Soccer Goals Summarizator


Cut goals moments from soccer record game using Celebration Event Classifier.

Trained on [Futsal games](https://www.youtube.com/channel/UCRheEKZEJtF8jfxkQPxZZbw)

## Installation
 - Install Python 3.6 or higher and run: `pip install soccer-summarizator`
 
## Usage

```bash
usage: inference.py [-h] [--input_video INPUT_VIDEO]
[--output_video OUTPUT_VIDEO] [--model_name MODEL_NAME]
[--classification_type CLASSIFICATION_TYPE] [--fps_count]
[--batch_size BATCH_SIZE]

Summarize input video

optional arguments:
  -h, --help            show this help message and exit
  --input_video INPUT_VIDEO
                        path to input video
  --output_video OUTPUT_VIDEO
                        path to output video
  --model_name MODEL_NAME
                        model namecould be "squeezenet" or "resnet"
  --classification_type CLASSIFICATION_TYPE
  --fps_count
  --batch_size BATCH_SIZE
