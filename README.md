![Codecov](https://img.shields.io/codecov/c/github/deepspectrum/deepspectrumlite?style=flat)
![CI status](https://github.com/deepspectrum/deepspectrumlite/workflows/CI/badge.svg)
[![PyPI version](https://badge.fury.io/py/DeepSpectrumLite.svg)](https://badge.fury.io/py/DeepSpectrumLite)
![PyPI - License](https://img.shields.io/pypi/l/DeepSpectrumLite)

**DeepSpectrumLite** is a Python toolkit to design and train light-weight Deep Neural Networks (DNNs) for classification tasks from raw audio data.
The trained models run on embedded devices.

DeepSpectrumLite features an extraction pipeline that first creates visual representations for audio data - plots of spectrograms.
The image plots are then fed into a DNN. This could be a pre-trained Image Convolutional Neural Network (CNN). 
Activations of a specific layer then form the final feature vectors which are used for the final classification.

The trained models can be easily converted to a TensorFlow Lite model. During the converting process, the model becomes smaller and faster optimised for inference on embedded devices.

**(c) 2020-2021 Shahin Amiriparian, Tobias Hübner, Maurice Gerczuk, Sandra Ottl, Björn Schuller: Universität Augsburg**
Published under GPLv3, please see the `LICENSE` file for details.

Please direct any questions or requests to Shahin Amiriparian (shahin.amiriparian at informatik.uni-augsburg.de) or Tobias Hübner (tobias.huebner at informatik.uni-augsburg.de).

# Why DeepSpectrumLite?
DeepSpectrumLite is built upon TensorFlow Lite which is a specialised version of TensorFlow that supports embedded devices.
However, TensorFlow Lite does not support all basic TensorFlow functions for audio signal processing and plot image generation. DeepSpectrumLite offers implementations for unsupported functions.

# Installation
You can install DeepSpectrumLite from PiPy.
```bash
pip install deepspectrumlite
```
Alternatively, you can clone this repository and install it from there:
```bash
git clone https://github.com/DeepSpectrum/DeepSpectrumLite.git
cd DeepSpectrumLite
```

## Virtual environment
We highly recommend you to create a virtual environment:
```bash
python -m venv ./venv
source ./venv/bin/activate
pip install .
```
## Conda environment
If you have Conda installed, you can create and install a environment from the included "environment.yml".
```bash
conda env create -f environment.yml
conda activate ./env
```

## GPU support
DeepSpectrumLite uses TensorFlow 2.4.0. GPU support should be automatically available, as long as you have CUDA version 11.0. If you cannot install cuda 11.0 globally, you can use Anaconda to install it in a virtual environment along DeepSpectrumLite.

# Getting started

## Training
To train a model use the following command and pass the path to your data directory (structured as above):
```bash
deepspectrumlite train -d [path/to/data] -md [save/models/to] -hc [path/to/hyperparameter_config.json] -cc [path/to/class_config.json] -l [path/to/labels.csv]
```

For a full rundown of all commandline arguments, call `python -m cli.train --help`.
Other training parameters including label parser file, problem definition, audio signal preprocessing, model configuration are defined in the hyperparameter config json file.

## Test
If you want to test your .h5 model against a specific audio .wav file, you can call `cli.test`:
```bash
deepspectrumlite predict -md [path/to/model.h5] -d [path/to/*.wav] -hc [path/to/hyperparameter_config.json] -cc [path/to/class_config.json]
```

### Slurm Job Array
DeepSpectrumLite supports training over a slurm job array. This is helpful when applying a grid search.
When you call the train process within your slurm job array, our system retrieves the `SLURM_ARRAY_TASK_ID` environment variable that is automatically set by slurm.
Each job array `i` trains the i'th combination of your grid. If you have a grid containing 24 combinations of parameters, you can define your slurm job as follows:

```bash
#SBATCH --partition=dfl
#SBATCH --mem=8000
#SBATCH --ntasks=1
#SBATCH --get-user-env
#SBATCH --export=ALL
#SBATCH -o slurm-%A_%a.txt
#SBATCH -J myjob
#SBATCH --cpus-per-task=2
#SBATCH --array=0-23

deepspectrumlite train
```
This script will create 24 independent training instances where job 0 trains the first combination, job 1 trains the second combination etc. 

## Statistics about a model
If you want to check out the parameter quantity and FLOPS your .h5 model you can call `cli.stats`:
```bash
deepspectrumlite stats -d [path/to/model.h5] -hc [path/to/hyperparameter_config.json]
```

## Convert .h5 to .tflite
If you want to convert your trained model to TensorFlow Lite, use `cli.convert`:
```bash
deepspectrumlite convert -s [path/to/model.h5] -d [path/to/target/model.tflite]
```

## Configuration
Example configuration files are in the `config` directory.

## Hyper Parameter Configuration Grid

The hyper parameter configuration grid is defined in a json file. You can add more than configuration values for a variable. DeepSpectrumLite creates a grid of all possible combinations of variable values.

### Model Configurations

| Variable        | Type                               | Description                                                                                                                                                            | Required | Default value    |
|-----------------|------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------|------------------|
| label_parser    | string                             | The parser python file for the labels. The python file path is followed  by a `:` and the class name. For example: `../lib/data/parser/ComParEParser.py:ComParEParser` | true     |                   |
| prediction_type | enum ['categorical', 'regression'] | Define whether you have a categorical or a regression problem to solve. |    false | categorical |
| model_name      | string                             | The model class that is used for training. For Transfer Learning use 'TransferBaseModel' | false | TransferBaseModel |
| basemodel_name | string | The base model name that is used for training. Available models: `vgg16`, `vgg19`, `resnet50`, `xception`, `inception_v3`, `densenet121`, `densenet169`, `densenet201`, `mobilenet`, `mobilenet_v2`, `nasnet_large`, `nasnet_mobile`, `inception_resnet_v2`, `squeezenet_v1` | true | |
| weights | enum ['imagenet', ''] | If set to 'imagenet', the base model defined in basemodel_name uses weights from pre-trained on imagenet. Otherwise, the model defined in basemodel_name has random weights. | false | imagenet |
| tb_experiment | string | The name of the tensorboard dashboard. The name is used a directory. | true | |
| tb_run_id | string | The name of this experiment setting for the tensorboard dashboard. The name is used a subdirectory. You can define a generic tb_experiment which uses different runs with different configuration settings. When having more than one configuration within a grid, the tb_run_id is automatically extended by `_config_[NUMBER]`.| true | |
| num_unit| int | The number of units that are used for dense layer in the final MLP classifier. | true | |
| dropout | float | The rate of dropout applied after the dense layer and the final prediction layer. | true | |
| optimizer | string | The optimizer for the training process. Supported optimizers: `adam`, `adadelta`, `sgd`, `adagrad`, `ftrl`, `nadam`, `rmsprop`, `sgd` | true | |
| learning_rate | float | The initial learning rate of the optimizer. | true | |
| fine_learning_rate | float | The learning rate that is set after `pre_epochs`. This is only supported when model_name='TransferBaseModel' and pre_epochs>0 and weighted='imagenet' and fine_learning_rate>0.0 | false | |
| loss | string | The loss function that is used for training. All TensorFlow loss functions are supported. | true | |
| activation | string | The activation function that is used in the dense layers in the final MLP classifier. All activation functions from TensorFlow and "arelu" are supported. | true | |
| pre_epochs | int | The number of epochs of training. When the model_name is 'TransferBaseModel', the pre_epochs defines how long the base model is trained in a frozen state. After reaching the pre_epochs, the model the last layers (share defined in 'finetune_layer') are unfrozen and trained again for 'epochs' epochs. | true | |
| finetune_layer | float | The amount of layers (share in percent) of the last layers of the base model that are unfrozen after pre_epochs | true | |
| batch_size | int | The batch size of the training | true | |

### Preprocessing Configurations

| Variable        | Type                               | Description                                                                                                                                                            | Required | Default value    |
|-----------------|------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------|------------------|
| sample_rate | int | The sample rate of the audio files (in samples per seconds). | true | |
| chunk_size | float | The desired chunk size of the audio files (in seconds). All audio files are split into chunks before spectrograms are calculated. | true | |
| chunk_hop_size | float | The hop size of the audio files (in seconds). | true | |
| stft_window_size | float | The window size used for the Short-Time Fourier Transform (STFT) (in seconds). | true | |
| stft_hop_size | float | The hop size used for the STFT (in seconds). | true | |
| stft_fft_length | float | The FFT length used for the STFT (in seconds). | true | |
| mel_scale | bool | If enable, mel spectrograms are calculated. Otherwise, power spectrograms are used. | true | |
| lower_edge_hertz | float | The lower bound of the frequency range. (in Hz) | true | |
| upper_edge_hertz | float | The upper bound of the frequency range. (in Hz) | true | |
| num_mel_bins | int | The number of mel bins used for the mel spectrogram generation | false | |
| num_mfccs | int | The number of mfcc bins. When num_mfccs is set to 0, no mfcc bins are generated and the pure mel spectrograms are used instead. | false | |
| cep_lifter | int | The number of frequencs for the cepstral lift. If set to 0, no cepstral lift is applied. | true | |
| db_scale | bool | When enabled, the spectrogram is scaled to the dB scale instead of the power scale. | true | |
| use_plot_images | bool | When enabled, the spectrogram values are plotted using the color map defined in color_map. These plot images are then used for training. When set to false, the pure spectrogram values are fed to the network. | true | |
| color_map | string | The color map used for plotting spectrograms. Available color maps: `viridis`, `cividis`, `inferno`, `magma`, `plasma`. | true | |
| image_width | int | The width of the spectrogram plot images (in px). | true | |
| image_height | int | The height of the spectrogram plot images (in px). | true | |
| anti_alias | bool | Enable anti alias when resizing the image plots. | false | true |
| resize_method | string | The resize method when resizing image plots. Supported methods: `bilinear`, `lanczos3`, `bicubic`, `gaussian`, `nearest`, `area`, `mitchellcubic` | false | bilinear |

### Data Augmentation Configuration


| Variable        | Type                               | Description                                                                                                                                                            | Required | Default value    |
|-----------------|------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------|------------------|
| sap_aug_a | float | The a value of the SapAugment policy. Please check the SapAugment paper for details. | true | |
| sap_aug_s | float | The s value of the SapAugment policy. Please check the SapAugment paper for details. | true | |
| augment_cutmix | bool | Enable CutMix data augmentation. | true | |
| augment_specaug | bool | Enable SpecAugment data augmentation. | true | |
| da_prob_min | float | The minimum probability that data augmentation if applied at all. The probability depends on the lambda value from SapAugment. | false | |
| da_prob_max | float | The maximum probability that data augmentation if applied at all. The probability depends on the lambda value from SapAugment. | false | |
| cutmix_min | float | The minimum width of a squared frame that is used from another sample of the same batch. The concrete size depends on the lambda value from SapAugment. | false | |
| cutmix_max | float | The maximum width of a squared frame that is used from another sample of the same batch. The concrete size depends on the lambda value from SapAugment. | false | |
| specaug_freq_min | float | The minimum width of **all** SpecAugment frequency masks. If you have more than one mask, the single mask width is divided by the number of masks. Therefore, this variable defines the overall share of augmentation that is applied by SpecAugment. The concrete size depends on the lambda value from SapAugment. | false | |
| specaug_freq_max | float | The maximum width of **all** SpecAugment frequency masks. If you have more than one mask, the single mask width is divided by the number of masks. Therefore, this variable defines the overall share of augmentation that is applied by SpecAugment. The concrete size depends on the lambda value from SapAugment. | false | |
| specaug_time_min | float | The minimum width of **all** SpecAugment time masks. If you have more than one mask, the single mask width is divided by the number of masks. Therefore, this variable defines the overall share of augmentation that is applied by SpecAugment. The concrete size depends on the lambda value from SapAugment. | false | |
| specaug_time_max | float | The maximum width of **all** SpecAugment time masks. If you have more than one mask, the single mask width is divided by the number of masks. Therefore, this variable defines the overall share of augmentation that is applied by SpecAugment. The concrete size depends on the lambda value from SapAugment. | false | |
| specaug_freq_mask_num | int | The number of SpecAugment frequency masks that are added. | false | |
| specaug_time_mask_num | int | The number of SpecAugment time masks that are added. | false | |

Example file:
```json
{
    "label_parser":     ["../lib/data/parser/ComParEParser.py:ComParEParser"],
    "model_name":       ["TransferBaseModel"],
    "prediction_type":  ["categorical"],
    "basemodel_name":   ["densenet121"],
    "weights":          ["imagenet"],
    "tb_experiment":    ["densenet_iemocap"],
    "tb_run_id":        ["densenet121_run_0"],
    "num_units":        [512],
    "dropout":          [0.25],
    "optimizer":        ["adadelta"],
    "learning_rate":    [0.001],
    "fine_learning_rate": [0.0001],
    "finetune_layer":   [0.7],
    "loss":             ["categorical_crossentropy"],
    "activation":       ["arelu"],
    "pre_epochs":       [40],
    "epochs":           [100],
    "batch_size":       [32],

    "sample_rate":      [16000],
    "normalize_audio":  [false],

    "chunk_size":       [4.0],
    "chunk_hop_size":   [2.0],

    "stft_window_size": [0.128],
    "stft_hop_size":    [0.064],
    "stft_fft_length":  [0.128],

    "mel_scale":        [true],
    "lower_edge_hertz": [0.0],
    "upper_edge_hertz": [8000.0],
    "num_mel_bins":     [128],
    "num_mfccs":        [0],
    "cep_lifter":       [0],
    "db_scale":         [true],
    "use_plot_images":  [true],
    "color_map":        ["viridis"],
    "image_width":      [224],
    "image_height":     [224],
    "resize_method":    ["nearest"],
    "anti_alias":       [false],

    "sap_aug_a":        [0.5],
    "sap_aug_s":        [10],
    "augment_cutmix":   [true],
    "augment_specaug":  [true],
    "da_prob_min":      [0.1],
    "da_prob_max":      [0.5],
    "cutmix_min":       [0.075],
    "cutmix_max":       [0.25],
    "specaug_freq_min": [0.1],
    "specaug_freq_max": [0.3],
    "specaug_time_min": [0.1],
    "specaug_time_max": [0.3],
    "specaug_freq_mask_num": [4],
    "specaug_time_mask_num": [4]
}
```

## Class Configuration
The class configuration is defined in a json file. The indices define the internal variable of the class and the value defined the output name.

Example file for categorical problems:
```json
{
    "negative": "Negative",
    "positive": "Positive"
}
```

Example file for regression problems:
```json
{
    "a": "Arousal"
}
```
