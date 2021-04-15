# DeepSpectrumLite
DeepSpectrumLite is a Python toolkit to design and train light-weight Deep Neural Networks (DNNs) for classification tasks from raw audio data .
The trained models run on embedded devices.

DeepSpectrumLite features an extraction pipeline which first creates visual representations for audio data - plots of spectrograms.
The image splots are then fed to a DNN. This could be a pre-trained Image Convolutional Neural Network (CNN). 
Activations of a specific layer then form the final feature vectors which are used for the final classification.

The trained models can be easily converted to a TensorFlow Lite model. During the converting process, the model becomes smaller and faster optimised for inference on embedded devices.

**(c) 2020-2021 Shahin Amiriparian, Tobias Hübner, Maurice Gerczuk, Sandra Ottl, Björn Schuller: Universität Augsburg**
Published under GPLv3, please see the `LICENSE` file for details.

Please direct any questions or requests to Shahin Amiriparian (shahin.amiriparian at informatik.uni-augsburg.de) or Tobias Hübner (tobias.huebner at informatik.uni-augsburg.de).

# Why DeepSpectrumLite?
DeepSpectrumLite is built upon TensorFlow Lite which is a specialised version of TensorFlow that supports embedded decvies.
However, TensorFlow Lite does not support all basic TensorFlow functions for audio signal processing and plot image generation. DeepSpectrumLite offers implementations for unsupported functions.