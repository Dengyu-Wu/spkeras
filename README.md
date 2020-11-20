<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->




<!-- PROJECT LOGO -->


<!-- TABLE OF CONTENTS -->




<!-- ABOUT THE PROJECT -->
## Spkeras

There are many great README templates available on GitHub, however, I didn't find one that really suit my needs so I created this enhanced one. I want to create a README template so amazing that it'll be the last one you ever need.

Here's why:
* Your time should be focused on creating something amazing. A project that solves a problem and helps others
* You shouldn't be doing the same tasks over and over like creating a README from scratch
* You should element DRY principles to the rest of your life :smile:

Of course, no one template will serve all projects since your needs may be different. So I'll be adding more in the near future. You may also suggest changes by forking this repo and creating a pull request or opening an issue.

A list of commonly used resources that I find helpful are listed in the acknowledgements.

### Built With
This section should list any major frameworks that you built your project using. Leave any add-ons/plugins for the acknowledgements section. Here are a few examples.
* [tensorflow-keras](https://tensorflow.org)


<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* pip
```sh
pip install tensorflow
```

### Installation

1. Clone the repo
```sh
git clone https://github.com/your_username_/Project-Name.git
```
### Example
```python
#load dataset and cnn model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train/255
x_test = x_test/255
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

cnn_model = load_model('cnn_model.h5')

#Convert CNN into SNN
from spkeras.models import cnn_to_snn

#Current normalisation 
'''
cnn_to_snn attributions:
				 timesteps=256,thresholding=0.5,signed_bit=0,scaling_factor=1,
				 method=1,amp_factor=100,epsilon = 0.001, spike_ext=0,noneloss=False
'''
snn_model = cnn_to_snn()(cnn_model,x_train)

#Evaluate SNN accuracy
_,acc = snn_model.evaluate(x_test,y_test,timesteps=256)

#Count SNN spikes
s_max,s = snn_model.CountSpikes(x_train,timesteps=256)

#Count neuron numbers
n = snn_model.NeuronNumbers()
```

<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- MARKDOWN LINKS & IMAGES -->

