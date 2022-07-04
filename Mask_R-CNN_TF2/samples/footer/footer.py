"""
Mask R-CNN
Train on the footer dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 footer_detector.py train --dataset=/path/to/footer_detector/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 footer_detector.py train --dataset=/path/to/footer_detector/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 footer_detector.py train --dataset=/path/to/footer_detector/dataset --weights=imagenet

    # Apply color splash to an image
    python3 footer_detector.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 footer_detector.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath("/content/drive/MyDrive/Mask-RCNN-TF2")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

from mrcnn import visualize

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Configurations
############################################################


class Footer_detectorConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "footer_detector"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet101"

    # Number of classes (including background)
    NUM_CLASSES = 1 + 7  # Background + footer_detector

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 200

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.75


############################################################
#  Dataset
############################################################

class Footer_detectorDataset(utils.Dataset):
    classes = {
        1: 'input',
        2: 'button',
        3: 'text',
        4: 'select',
        5: 'checkbox',
        6: 'radiobutton',
        7: 'image'
    }

    def load_footer_detector(self, dataset_dir, subset):
        """Load a subset of the Footer_detector dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """

        print('manual annotation class count: {}'.format(len(self.classes)))
        # Add classes. We have 7 classes to add.
        for key, value in self.classes.items():
            self.add_class("footer_detector", key, value)

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(os.path.join(dataset_dir, "via_export_json.json")))
        annotations = list(annotations.values())  # don't need the dict keys
        print('annotations count:\t{}'.format(len(annotations)))
        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the rectangles that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            '''
            if type(a['regions']) is dict:
                rectangles = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                rectangles = [r['shape_attributes'] for r in a['regions']]
            '''
            rectangles = [r['shape_attributes'] for r in a['regions']]
            # for rectangle in rectangles:
            # print(rectangle)
            # load_mask() needs the image size to convert rectangles to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.

            objects = [r['region_attributes'] for r in a['regions']]

            '''
            # Check the presense of the HTML elements types
            classids = [(n['HTML element']) for n in objects]
            for classid in classids:
                print(classid.keys())

            class_ids = [int(n['HTML element']) for n in objects]
            '''

            image_path = os.path.join(dataset_dir, a['filename'].strip('.\\'))
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "footer_detector",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                rectangles=rectangles,
                objects=objects)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        # If not a footer_detector dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "footer_detector":
            return super(self.__class__, self).load_mask(image_id)

        class_names = image_info['objects']

        # Convert rectangles to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["rectangles"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["rectangles"]):
            # Get indexes of pixels inside the rectangle and set them to 1
            # rr, cc = skimage.draw.rectangle(p['all_points_y'], p['all_points_x'])
            # https://scikit-image.org/docs/dev/api/skimage.draw.html#skimage.draw.rectangle
            start = (int(p['y']), int(p['x']))
            extent = (int(p['height']), int(p['width']))

            # rr, cc = skimage.draw.rectangle(start, extent=extent, shape=img.shape)
            rr, cc = skimage.draw.rectangle(start, extent=extent)
            # print(rr)
            # print(cc)
            mask[rr, cc, i] = 1
        # Assign class_ids by reading class_names
        class_ids = np.zeros([len(info["rectangles"])])
        # are representing input and button.
        for i, p in enumerate(class_names):
            if p['HTML element'] in self.classes.values():
                class_ids[i] = list(self.classes.keys())[list(self.classes.values()).index(p['HTML element'])]

        # Should return multiple classes ids
        class_ids = np.array(class_ids, dtype=np.int32)
        return mask.astype(bool), class_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "footer_detector":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = Footer_detectorDataset()
    dataset_train.load_footer_detector(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = Footer_detectorDataset()
    dataset_val.load_footer_detector(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=25,
                layers='heads')


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        image = skimage.color.rgba2rgb(image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]

        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect footer_detectors.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/content/drive/MyDrive/Mask-RCNN-TF2/dataset/footer",
                        help='Directory of the Footer_detector dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/content/drive/MyDrive/Mask-RCNN-TF2/mask_rcnn_coco.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/content/drive/MyDrive/Mask-RCNN-TF2/logs",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video, \
            "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = Footer_detectorConfig()
    else:
        class InferenceConfig(Footer_detectorConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1


        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
