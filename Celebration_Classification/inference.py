import argparse
from model_builder import ModelBuilder
import torch
import cv2
import logging as logger
import os
from utils import draw_classification_legend
from time import time
from utils import test_transforms
logger.basicConfig(level=logger.INFO)
classes = ['Celebration', 'Game Moment']


def parse_args():
    parser = argparse.ArgumentParser(description='Classify input image')
    parser.add_argument('--input_image', type=str, help='path to input image',
                        default='/home/alexander/Pictures/asdasd.png')
    parser.add_argument('--input_directory', type=str, help='path to input directory')
    parser.add_argument('--model_path', type=str, help='path to model',
                        default='/home/alexander/HSE_Stuff/Diploma/Automatic-Soccer-Highlights-Generation'
                                '/Celebration_Classification/wandb/celebration_only/files/best_model.pth')
    parser.add_argument('--model_name', type=str, help='model name'
                                                       'could be "squeezenet" or "resnet"', default='squeezenet')
    parser.add_argument('--fps_count', action='store_true')
    return parser.parse_args()


def run():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info('Building the model ...')
    model = ModelBuilder(args.model_name).get_model()
    model.load_state_dict(torch.load(args.model_path))
    model.to(device).eval()
    logger.info('Model was successfully built and loaded !')
    logger.info('Read input image ...')
    if args.input_directory:
        for file in os.listdir(args.input_directory):
            input_image = cv2.imread(os.path.join(args.input_directory, file))
            image_name = os.path.basename(args.input_image)
            input_tensor = test_transforms(image=input_image)['image'].to(device, dtype=torch.float).unsqueeze(0)
            output = model(input_tensor)
            results = {cls: round(score.item(), 3) for cls, score in zip(classes, output.softmax(dim=1).squeeze(0))}
            print(results)
            image = draw_classification_legend(image=input_image, class_map=results)
            cv2.imwrite(f'results/{image_name}', image)
            cv2.imshow('result', image)
            cv2.waitKey(0)
    else:
        input_image = cv2.imread(args.input_image)
        image_name = os.path.basename(args.input_image)
        input_tensor = test_transforms(image=input_image)['image'].to(device, dtype=torch.float).unsqueeze(0)
        output = model(input_tensor)
        results = {cls: round(score.item(), 3) for cls, score in zip(classes, output.softmax(dim=1).squeeze(0))}
        image = draw_classification_legend(image=input_image, class_map=results)
        cv2.imwrite(f'results/{image_name}', image)
        cv2.imshow('result', image)
        cv2.waitKey(0)

    # total_time = 0
    # if args.fps_count:
    #     for i in range(110):
    #         # warmup
    #         if i < 10:
    #             pass
    #         start_time = time()
    #         input_tensor = test_transforms(input_image)
    #         output = model(input_tensor)
    #         total_time += round(time() - start_time, 4)
    #     fps = round(100 / total_time, 2)
    #     logger.info(fps)

if __name__ == '__main__':
    run()