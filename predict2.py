import argparse
import logging
import os
import glob

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import confusion_matrix

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()

    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='./checkpoints/checkpoint_epoch99.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=False)
    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        split = os.path.splitext(fn)
        return f'{split[0]}_OUT{split[1]}'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


def plot_confusion_matrix(real,
                           predict,
                           target_names=['Non-tree covered areas', 'Broadleaved trees', 'Coniferous trees'],
                           title='Confusion matrix',
                           cmap=None,
                           normalize=True):
        """
        given a sklearn confusion matrix (cm), make a nice plot

        Arguments
        ---------
        cm:           confusion matrix from sklearn.metrics.confusion_matrix

        target_names: given classification classes such as [0, 1, 2]
                      the class names, for example: ['high', 'medium', 'low']

        title:        the text to display at the top of the matrix

        cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                      see http://matplotlib.org/examples/color/colormaps_reference.html
                      plt.get_cmap('jet') or plt.cm.Blues

        normalize:    If False, plot the raw numbers
                      If True, plot the proportions

        Usage
        -----
        plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                                  # sklearn.metrics.confusion_matrix
                              normalize    = True,                # show proportions
                              target_names = y_labels_vals,       # list of names of the classes
                              title        = best_estimator_name) # title of graph

        Citiation
        ---------
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

        """
        import itertools

        cm = confusion_matrix(real, predict)
        accuracy = np.trace(cm) / np.sum(cm).astype('float')
        misclass = 1 - accuracy

        if cmap is None:
            cmap = plt.get_cmap('Blues')

        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names, rotation=45)
            plt.yticks(tick_marks, target_names)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
        plt.show()
        plt.savefig(title+'.png', dpi=1000)


if __name__ == '__main__':
    args = get_args()

    # # training results
    # mask_dir = './data3/train/label/'
    # img_dir = './data3/train/image/'
    # mask_files = glob.glob(mask_dir + '*.png')
    # in_files = [file.replace('.png', '_2018_06.png') for file in mask_files]
    # in_files = [file.replace('label', 'image') for file in in_files]
    # out_files = [file.replace('.png', '_pred.png') for file in mask_files]
    # out_files = [file.replace('label', 'result') for file in out_files]
    # args.model = './checkpoints/checkpoint_epoch100.pth'
    # data = np.load(mask_dir.replace('label', 'result')+'result.npz')
    # results = data['results']
    # labels = data['labels']
    # plot_confusion_matrix(labels.flatten(), results.flatten())

    # test results
    # mask_dir = './data3/test/label/'
    # img_dir = './data3/test/image/'
    # mask_files = glob.glob(mask_dir + '*.png')
    # in_files = [file.replace('.png', '_2018_06.png') for file in mask_files]
    # in_files = [file.replace('label', 'image') for file in in_files]
    # out_files = [file.replace('.png', '_pred.png') for file in mask_files]
    # out_files = [file.replace('label', 'result') for file in out_files]
    # args.model = './checkpoints/checkpoint_epoch100.pth'

    # data = np.load(mask_dir.replace('label', 'result')+'result.npz')
    # results = data['results']
    # labels = data['labels']
    # plot_confusion_matrix(labels.flatten(), results.flatten())

    # 2nd experiment: using three-channel data as input / Train
    # mask_dir = './data/label/'
    # img_dir = './data/image/'
    # mask_files = glob.glob(mask_dir + '*.png')
    # in_files = [file.replace('label', 'image') for file in mask_files]
    # out_files = [file.replace('.png', '_pred.png') for file in mask_files]
    # out_files = [file.replace('label', 'result') for file in out_files]
    # args.model = './checkpoints2/checkpoint_epoch100.pth'
    # data = np.load(mask_dir.replace('label', 'result') + 'result.npz')
    # results = data['results']
    # labels = data['labels']
    # plot_confusion_matrix(labels.flatten(), results.flatten())

    # 4th experiment, with focal loss
    mask_dir = './data_new/train_label/'
    img_dir = './data_new/train_image/'
    mask_files = glob.glob(mask_dir + '*.png')
    result_dir = './resultsNew/'
    in_files = [file.replace('.png', '_2018_06.png') for file in mask_files]
    in_files = [file.replace('label', 'image') for file in in_files]
    out_files = [file.replace('.png', '_pred.png') for file in mask_files]
    out_files = [file.replace('label', 'result') for file in out_files]
    out_files = [file.replace('./SMOTE_new/', result_dir) for file in out_files]
    args.model = './checkpoints/checkpoint_epoch99.pth'   # checkpoint_20 before

    if '2018_06' in str(in_files[0]):
        net = UNet(n_channels=9, n_classes=3, bilinear=args.bilinear)
        print('Channel number of inputs: ' + str(9))
    else:
        net = UNet(n_channels=3, n_classes=3, bilinear=args.bilinear)
        print('Channel number of inputs: ' + str(3))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info('Model loaded!')

    results = np.zeros([256, 256, len(mask_files)])
    labels = np.zeros([256, 256, len(mask_files)])
    for i, filename in enumerate(in_files):
        logging.info(f'\nPredicting image {filename} ...')
        if '2018_06' in str(filename):
            print(filename)
            I1 = Image.open(filename).convert('RGB')
            I2 = Image.open(str(filename).replace('06', '07')).convert('RGB')
            I3 = Image.open(str(filename).replace('06', '08')).convert('RGB')
            img = np.concatenate((I1, I2, I3), axis=2)
        else:
            img = Image.open(filename).convert('RGB')

        label = Image.open(mask_files[i]).convert('RGB')
        labels[:, :, i] = np.argmax(label, axis=2)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            out_filename = out_files[i]
            result = mask_to_image(mask)
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')
            results[:, :, i] = np.argmax(mask, axis=0)

        # if args.viz:
        #     logging.info(f'Visualizing results for image {filename}, close to continue...')
        #     plot_img_and_mask(img, mask)
    print('Prediction accuracy = ', str(np.sum(labels == results) / results.size * 100) + '%')
    print(results.size)
    print('Prediction accuracy of class = 0', str(np.round(np.sum((results == 0) & (labels == 0)) / np.sum(labels == 0) * 100)) + '%')
    print(np.sum(labels == 0))
    print('Prediction accuracy of class = 1', str(np.round(np.sum((results == 1) & (labels == 1)) / np.sum(labels == 1) * 100)) + '%')
    print(np.sum(labels == 1))
    print('Prediction accuracy of class = 2', str(np.round(np.sum((results == 2) & (labels == 2)) / np.sum(labels == 2) * 100)) + '%')
    print(np.sum(labels == 2))
    plot_confusion_matrix(labels.flatten(), results.flatten(), title=result_dir + 'confu_mat_3nd_focal')

    # np.savez(result_dir + 'result_3nd_focal.npz', results=results, labels=labels)
