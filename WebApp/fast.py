import segmentation_models_pytorch as smp
import time
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.v2 as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import UtilFunctions
import models
import matplotlib.pylab as plt
import segmentation_models_pytorch.utils.metrics
import torch.nn as nn

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias=True):
        super().__init__()
        # The depthwise conv is basically just a grouped convolution in PyTorch with
        # the number of distinct groups being the same as the number of input (and output)
        # channels for that layer.
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, bias=bias, groups=in_channels)
        # The pointwise convolution stretches across all the output channels using
        # a 1x1 kernel.
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class DownDSConv2(nn.Module):
    def __init__(self, chin, chout, kernel_size):
        super().__init__()
        self.seq = nn.Sequential(
            DepthwiseSeparableConv2d(in_channels=chin, out_channels=chout, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(chout),
            nn.ReLU(),
            DepthwiseSeparableConv2d(in_channels=chout, out_channels=chout, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(chout),
            nn.ReLU(),
        )
        self.mp = nn.MaxPool2d(kernel_size=2, return_indices=True)

    def forward(self, x):
        y = self.seq(x)
        pool_shape = y.shape
        y, indices = self.mp(y)
        return y, indices, pool_shape

class DownDSConv3(nn.Module):
    def __init__(self, chin, chout, kernel_size):
        super().__init__()
        self.seq = nn.Sequential(
            DepthwiseSeparableConv2d(in_channels=chin, out_channels=chout, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(chout),
            nn.ReLU(),
            DepthwiseSeparableConv2d(in_channels=chout, out_channels=chout, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(chout),
            nn.ReLU(),
            DepthwiseSeparableConv2d(in_channels=chout, out_channels=chout, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(chout),
            nn.ReLU(),
        )
        self.mp = nn.MaxPool2d(kernel_size=2, return_indices=True)

    def forward(self, x):
        y = self.seq(x)
        pool_shape = y.shape
        y, indices = self.mp(y)
        return y, indices, pool_shape

class UpDSConv2(nn.Module):
    def __init__(self, chin, chout, kernel_size):
        super().__init__()
        self.seq = nn.Sequential(
            DepthwiseSeparableConv2d(in_channels=chin, out_channels=chin, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(chin),
            nn.ReLU(),
            DepthwiseSeparableConv2d(in_channels=chin, out_channels=chout, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(chout),
            nn.ReLU(),
        )
        self.mup = nn.MaxUnpool2d(kernel_size=2)

    def forward(self, x, indices, output_size):
        y = self.mup(x, indices, output_size=output_size)
        y = self.seq(y)
        return y

class UpDSConv3(nn.Module):
    def __init__(self, chin, chout, kernel_size):
        super().__init__()
        self.seq = nn.Sequential(
            DepthwiseSeparableConv2d(in_channels=chin, out_channels=chin, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(chin),
            nn.ReLU(),
            DepthwiseSeparableConv2d(in_channels=chin, out_channels=chin, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(chin),
            nn.ReLU(),
            DepthwiseSeparableConv2d(in_channels=chin, out_channels=chout, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(chout),
            nn.ReLU(),
        )
        self.mup = nn.MaxUnpool2d(kernel_size=2)

    def forward(self, x, indices, output_size):
        y = self.mup(x, indices, output_size=output_size)
        y = self.seq(y)
        return y

class SegmentationHead(nn.Module):
    def __init__(self, chin, chout, kernel_size):
        super().__init__()
        self.seq = nn.Sequential(
            DepthwiseSeparableConv2d(in_channels=chin, out_channels=chout, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.seq(x)
        return y

class ImageSegmentationDSC3(torch.nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.out_channels = 3
        self.bn_input = nn.BatchNorm2d(3)
        self.dc1 = DownDSConv2(3, 32, kernel_size=kernel_size)
        self.dc2 = DownDSConv2(32, 64, kernel_size=kernel_size)
        self.dc3 = DownDSConv2(64, 128, kernel_size=kernel_size)
        self.dc4 = DownDSConv3(128, 256, kernel_size=kernel_size)
        # self.dc5 = DownConv3(512, 512, kernel_size=kernel_size)

        # self.uc5 = UpConv3(512, 512, kernel_size=kernel_size)
        self.uc4 = UpDSConv3(256, 128, kernel_size=kernel_size)
        self.uc3 = UpDSConv2(128, 64, kernel_size=kernel_size)
        self.uc2 = UpDSConv2(64, 32, kernel_size=kernel_size)
        self.uc1 = UpDSConv2(32, 3, kernel_size=kernel_size)
        self.segHead = SegmentationHead(3, 1, kernel_size=kernel_size)

    def forward(self, batch: torch.Tensor):
        x = self.bn_input(batch)
        # x = batch
        # SegNet Encoder
        x, mp1_indices, shape1 = self.dc1(x)
        x, mp2_indices, shape2 = self.dc2(x)
        x, mp3_indices, shape3 = self.dc3(x)
        x, mp4_indices, shape4 = self.dc4(x)
        # Our images are 128x128 in dimension. If we run 4 max pooling
        # operations, we are down to 128/16 = 8x8 activations. If we
        # do another down convolution, we'll be at 4x4 and at that point
        # in time, we may lose too much spatial information as a result
        # of the MaxPooling operation, so we stop at 4 down conv
        # operations.
        # x, mp5_indices, shape5 = self.dc5(x)

        # SegNet Decoder
        # x = self.uc5(x, mp5_indices, output_size=shape5)
        x = self.uc4(x, mp4_indices, output_size=shape4)
        x = self.uc3(x, mp3_indices, output_size=shape3)
        x = self.uc2(x, mp2_indices, output_size=shape2)
        x = self.uc1(x, mp1_indices, output_size=shape1)
        x = self.segHead(x)

        return x
    # end def
# end class

def set_seed(seed):
  '''Reproducibility'''

  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  torch.cuda.manual_seed_all(seed)
  torch.manual_seed(seed)
  random.seed(seed)
  np.random.seed(seed)

class CustomTensorDataset(torch.utils.data.Dataset):
    """Data Augmentation: Extent TensorDataset with support of transforms."""
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=25, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 725
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.DiceLoss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func

    def __call__(self, DiceLoss, model):

        score = DiceLoss

        # First epoch: set best_score.
        if self.best_score is None:
            self.best_score = score
            if self.verbose:
              self.trace_func(f'DiceLoss decreased ({self.DiceLoss_min:.6f} --> {DiceLoss:.6f}).')
              self.DiceLoss_min = DiceLoss

        # Increment EarlyStopping counter if validation loss increases.
        elif score > self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

        # Store new best_score and set counter to 0 if validation loss is lower than the current best_score.
        else:
            self.best_score = score
            if self.verbose:
              self.trace_func(f'DiceLoss decreased ({self.DiceLoss_min:.6f} --> {DiceLoss:.6f}).')
              self.DiceLoss_min = DiceLoss
            self.counter = 0

def return_dataset(mammograms, masks):
    
    # Split train data into train and validation sets.
    X_train, X_val, y_train, y_val = train_test_split(mammograms, masks, test_size=0.1, random_state=42)

    # Convert to numpy arrays.
    X_train, y_train, X_val, y_val = np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val)

    # Convert to tensors.
    tensor_X_train, tensor_y_train = torch.Tensor(X_train), torch.Tensor(y_train)
    tensor_X_val, tensor_y_val = torch.Tensor(X_val), torch.Tensor(y_val)
    X_val, y_val = None, None
    X_train, y_train = None, None

    # Convert to shape [batch_size, channels, height, width].
    tensor_X_train = tensor_X_train.permute(0, 3, 1, 2)
    tensor_y_train = tensor_y_train.permute(0, 3, 1, 2)
    tensor_X_val = tensor_X_val.permute(0, 3, 1, 2)
    tensor_y_val = tensor_y_val.permute(0, 3, 1, 2)

    # Initialize data augmentation techniques.
    transformations = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3)
    ])

    # Create augmented dataset.
    train_dataset = CustomTensorDataset(tensors=(tensor_X_train, tensor_y_train))
    valid_dataset = CustomTensorDataset(tensors=(tensor_X_val, tensor_y_val))

    return train_dataset, valid_dataset

def prepare_data(mammograms, masks):
    '''Split each fold into train and validation sets and convert to tensors.'''
    """
        Args:
            fold (int): Index of the current fold during k-fold Cross-validatoin
            mammograms (numpy array): Contains all the mammograms.
            masks (numpy array): Contains all the masks.
            train_idx (list): Contains the indices of the training data for the current fold index.
            mammograms_file_name (function): Path of mammograms array (in order to reload data).
                            Default: print
            masks_file_name (function): Path of masks array (in order to reload data).
                            Default: print
            skip_fold (int): Skip folds with index lower than skip_fold (allows to start training at any of the folds).
                            Default: print
    """

    # Split train data into train and validation sets.
    X_train, X_val, y_train, y_val = train_test_split(mammograms, masks, test_size=0.1, random_state=42)

    # Convert to numpy arrays.
    X_train, y_train, X_val, y_val = np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val)

    # Convert to tensors.
    tensor_X_train, tensor_y_train = torch.Tensor(X_train), torch.Tensor(y_train)
    tensor_X_val, tensor_y_val = torch.Tensor(X_val), torch.Tensor(y_val)
    X_val, y_val = None, None
    X_train, y_train = None, None

    # Convert to shape [batch_size, channels, height, width].
    tensor_X_train = tensor_X_train.permute(0, 3, 1, 2)
    tensor_y_train = tensor_y_train.permute(0, 3, 1, 2)
    tensor_X_val = tensor_X_val.permute(0, 3, 1, 2)
    tensor_y_val = tensor_y_val.permute(0, 3, 1, 2)

    # Initialize data augmentation techniques.
    transformations = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3)
    ])

    # Create augmented dataset.
    train_dataset = CustomTensorDataset(tensors=(tensor_X_train, tensor_y_train), transform=transformations)
    valid_dataset = CustomTensorDataset(tensors=(tensor_X_val, tensor_y_val), transform=transformations)

    # Free-up RAM space.
    tensor_X_train, tensor_y_train = None, None
    tensor_X_val, tensor_y_val = None, None

    # Create loaders.
    train_loader = DataLoader(train_dataset, batch_size=16)
    valid_loader = DataLoader(valid_dataset, batch_size=1)

    # Free-up RAM space.
    train_dataset = None
    valid_dataset = None

    return train_loader, valid_loader

def create_model(model_type):

  if model_type == 'Basic4': model = models.ImageSegmentationBasic4(kernel_size=3)
  elif model_type == 'Basic5': model = models.ImageSegmentationBasic5(kernel_size=3)
  elif model_type == 'DSC3': model = ImageSegmentationDSC3(kernel_size=3)
  elif model_type == 'DSC35': model = models.ImageSegmentationDSC35(kernel_size=3)
  elif model_type == 'DSC4': model = models.ImageSegmentationDSC4(kernel_size=3)
  elif model_type == 'DSC5': model = models.ImageSegmentationDSC5(kernel_size=3)
  elif model_type == 'Mobile4': model = models.ImageSegmentationDSCMobile4(kernel_size=3)
  elif model_type == 'Mobile5': model = models.ImageSegmentationDSCMobile5(kernel_size=3)
  elif model_type == 'RES4': model = models.ImageSegmentationResDSC4(kernel_size=3)
  else:
    # create segmentation model with pretrained encoder
    model = smp.Unet(
            encoder_name=model_type,
            encoder_weights=None,
            classes=1,
            activation='sigmoid',
        )

  return model

class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.2, beta=0.8):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        
        return 1 - Tversky

def train(training_type, model_type, save_path, move_to_drive=0, info=0):
    '''Training procedure.'''
    """
            Args:
                model (torch.nn.Module): Model which will be trained.
                save_path (string): Best models of each fold will be stored in this path.
                move_to_drive (string): If a google drive path is given, the best models of each fold will be moved there.
                                Default: 0
    """

    print('================Training================ \n')

    # Load data
    mammograms, masks = UtilFunctions.load_data("D:\\BreastSegmentation\\Dataset_Second_Pipeline\\Preprocessed3805\\ROI\\", "D:\\BreastSegmentation\\Dataset_Second_Pipeline\\Preprocessed3805\\ROI_mask\\")

    train_loader, valid_loader = prepare_data(mammograms, masks)

    if training_type == 'whole_pretrained':

      # create segmentation model with pretrained encoder
      model = smp.Unet(
          encoder_name=model_type,
          encoder_weights='imagenet',
          classes=1,
          activation='sigmoid',
      )

    if training_type == 'scratch':
      model = create_model(model_type)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)

    # Set flag to train the model or not. If set to 'False', only prediction is performed (using an older model checkpoint)
    TRAINING = True

    # Set num of epochs
    EPOCHS = 160

    # Set device: `cuda` or `cpu`
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define loss function
    #loss = smp.utils.losses.DiceLoss()
    loss = TverskyLoss()
    loss.__name__ = 'Tversky_Loss'

    # define metrics
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5, activation = None),
    ]

    # define learning rate scheduler (not used in this NB)
    lr_scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True, min_lr=0.00001, factor=0.4)

    # initialize the early_stopping object
    early_stopping = EarlyStopping(verbose=True)

    train_epoch = smp.utils.train.TrainEpoch(
          model,
          loss=loss,
          metrics=metrics,
          optimizer=optimizer,
          device=DEVICE,
          verbose=True,
      )

    valid_epoch = smp.utils.train.ValidEpoch(
          model,
          loss=loss,
          metrics=metrics,
          device=DEVICE,
          verbose=True,
      )

    if TRAINING:

        best_dice_score = 1
        train_logs_list, valid_logs_list = [], []

        t0 = time.time()

        for i in range(0, EPOCHS):

            # Perform training & validation
            print('\nEpoch: {}'.format(i))
            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(valid_loader)
            train_logs_list.append(train_logs)
            valid_logs_list.append(valid_logs)

            # Save model if a better val IoU score is obtained
            if best_dice_score > valid_logs['Tversky_Loss']:
                best_dice_score = valid_logs['Tversky_Loss']
                torch.save(model, save_path + '.pth')
                print('Model saved!')

            lr_scheduler.step(valid_logs['Tversky_Loss'])

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(valid_logs['Tversky_Loss'], model)

            if early_stopping.early_stop:
                print("Early stopping") 
                set_seed(0)
                break

    print('Total training time: ', time.time() - t0)

    torch.save(model.cpu(), save_path + '.pth')

    # Free-up RAM space.
    del train_loader
    del valid_loader

def iou(mask1, mask2):
    intersection = (mask1 * mask2).sum()
    if intersection == 0:
        return 0.0
    union = torch.logical_or(mask1, mask2).to(torch.int).sum()
    return intersection / union

def visualize(**images):
    """
    Plot images in one row
    """
    n_images = len(images)
    plt.figure(figsize=(20,8))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([]);
        plt.yticks([])
        # get title from the parameter names
        plt.title(name.replace('_',' ').title(), fontsize=20)
        plt.imshow(image)
    plt.show()

def test(model_path):
   
    IoU = 0
    # Set device: `cuda` or `cpu`
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model = torch.load(model_path, map_location=DEVICE)
    mammograms, masks = UtilFunctions.load_data("D:\\BreastSegmentation\\Dataset_Second_Pipeline\\Preprocessed3805\\ROI\\", "D:\\BreastSegmentation\\Dataset_Second_Pipeline\\Preprocessed3805\\ROI_mask\\")
    _, valid_dataset = return_dataset(mammograms, masks)
    for idx in range(10):

        image, gt_mask = valid_dataset[idx]
        x_tensor = torch.from_numpy(np.array(image)).to(DEVICE).unsqueeze(0)
        image = np.transpose(image,(1,2,0))
        gt_mask = np.transpose(gt_mask,(1,2,0))

        # Predict test image
        pred_mask = best_model(x_tensor)
        pred_mask = pred_mask.detach().cpu().numpy()
        # Convert pred_mask from `CHW` format to `HWC` format
        pred_mask = np.transpose(pred_mask[0,:,:],(1,2,0))
        bin_gt = (gt_mask > 0.2) * 1
        bin_pred = (pred_mask > 0.2) * 1
        IoU += iou(torch.Tensor(bin_pred), bin_gt)
        print(iou(torch.Tensor(bin_pred), bin_gt))
        visualize(
            original_image = image,
            ground_truth_mask = bin_gt,
            predicted_mask = bin_pred,
        )

    print(IoU/10)

SCRATCH = ['DSC3']

for index in range(0,len(SCRATCH)):

    save_path = 'D:\\BreastSegmentation\\MassSegmentation\\runs\\01\\' + SCRATCH[index] + 'best_model'

    train('scratch', SCRATCH[index], save_path)