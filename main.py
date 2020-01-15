import cv2
import argparse as arg
import metrics as mt
import directory


if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    arguments = arg.ArgumentParser()
    arguments.add_argument('-seg', '--segmented-imgs', type=str, required=True, help='Directory of the segmented images.')
    arguments.add_argument('-GT', '--ground-truth', type=str, required=True, help='Directory of the ground truth images.')
    arguments = vars(arguments.parse_args())

    try:
        segmentedDirectory = list(directory.list_images(arguments['segmented_imgs']))
    except Exception:
        print('Error on the directory to the segmented images')

    try:
        gtDirectory = list(directory.list_images((arguments['ground_truth'])))
    except Exception:
        print('Error on the directory to the GT images')

    for (i, images) in enumerate(zip(segmentedDirectory, gtDirectory)):
        print('Image {}/{}'.format(i + 1, len(segmentedDirectory)))

        imgSegmented = cv2.imread(images[0], 0)
        imageGT = cv2.imread(images[1], 0)

        imgSegmented = cv2.resize(imgSegmented, (imageGT.shape[1], imageGT.shape[0]))

        # Verify if the segmented and ground truth images have the same size
        if imgSegmented.shape != imageGT.shape:
            print('The sizes of segmented image and ground truth image must be the same!')
            pass
        ret, imgSegmented = cv2.threshold(imgSegmented, 127, 255, cv2.THRESH_BINARY)
        ret, imageGT = cv2.threshold(imageGT, 127, 255, cv2.THRESH_BINARY)

        M = mt.Metricas(imgSegmented, imageGT)
        M.save_metrics(i)
    M.save_mean_metrics(len(segmentedDirectory), 1)
    print("[INFO] Process finish. Results disponible in patch results")