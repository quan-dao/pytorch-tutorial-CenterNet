import torch
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

from model.model_utils import decode_prediction
from tools.visualization import DetectionVisualization


def main():
    pred = torch.load('./voc07_prediction_for_images_09-13-15-39.pt',  # TODO: change this to yours path
                      map_location=torch.device('cpu'))
    print('pred: ', pred.shape)  # should see: torch.Size([4, 24, 96, 96])

    images_dir = Path('/home/user/dataset/pascal-voc/VOC2007/JPEGImages')  # TODO: change this to yours path
    images_path = [images_dir / f"{img_name}.jpg" for img_name in ['000009', '000013', '000015', '000039']]
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    vis_heatmaps = DetectionVisualization()
    vis_heatmaps.draw_images(images_path, axes)
    vis_heatmaps.draw_heatmap(pred[:, :20], axes, ['person', 'cow', 'bicycle','tvmonitor'])


    batch_boxes = decode_prediction(pred)
    fig1, axes1 = plt.subplots(2, 2, figsize=(10, 10))
    vis_boxes = DetectionVisualization()
    vis_boxes.draw_images(images_path, axes1)
    vis_boxes.draw_prediction(batch_boxes, axes1)

    plt.show()


if __name__ == '__main__':
    main()
