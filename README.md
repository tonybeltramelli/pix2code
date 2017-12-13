# pix2code
*Generating Code from a Graphical User Interface Screenshot*

[![License](http://img.shields.io/badge/license-APACHE2-blue.svg)](LICENSE.txt)

* A video demo of the system can be seen [here](https://youtu.be/pqKeXkhFA3I)
* The paper is available at [https://arxiv.org/abs/1705.07962](https://arxiv.org/abs/1705.07962)
* Official research page: [https://uizard.io/research#pix2code](https://uizard.io/research#pix2code)

## Abstract
Transforming a graphical user interface screenshot created by a designer into computer code is a typical task conducted by a developer in order to build customized software, websites, and mobile applications. In this paper, we show that deep learning methods can be leveraged to train a model end-to-end to automatically generate code from a single input image with over 77% of accuracy for three different platforms (i.e. iOS, Android and web-based technologies).

## Citation

```
@article{beltramelli2017pix2code,
  title={pix2code: Generating Code from a Graphical User Interface Screenshot},
  author={Beltramelli, Tony},
  journal={arXiv preprint arXiv:1705.07962},
  year={2017}
}
```

## Disclaimer

The following software is shared for educational purposes only. The author and its affiliated institution are not responsible in any manner whatsoever for any damages, including any direct, indirect, special, incidental, or consequential damages of any character arising as a result of the use or inability to use this software.

The project pix2code is a research project demonstrating an application of deep neural networks to generate code from visual inputs.
The current implementation is not, in any way, intended, nor able to generate code in a real-world context.
We could not emphasize enough that this project is experimental and shared for educational purposes only.
Both the source code and the datasets are provided to foster future research in machine intelligence and are not designed for end users.

## Setup
### Prerequisites

- Python 2 or 3
- pip

### Install dependencies

```sh
pip install -r  requirements.txt
```

## Usage

Prepare the data:
```sh
# reassemble and unzip the data
cd datasets
zip -F pix2code_datasets.zip --out datasets.zip
unzip datasets.zip

cd ../model

# split training set and evaluation set while ensuring no training example in the evaluation set
# usage: build_datasets.py <input path> <distribution (default: 6)>
./build_datasets.py ../datasets/ios/all_data
./build_datasets.py ../datasets/android/all_data
./build_datasets.py ../datasets/web/all_data

# transform images (normalized pixel values and resized pictures) in training dataset to numpy arrays (smaller files if you need to upload the set to train your model in the cloud)
# usage: convert_imgs_to_arrays.py <input path> <output path>
./convert_imgs_to_arrays.py ../datasets/ios/training_set ../datasets/ios/training_features
./convert_imgs_to_arrays.py ../datasets/android/training_set ../datasets/android/training_features
./convert_imgs_to_arrays.py ../datasets/web/training_set ../datasets/web/training_features
```

Train the model:
```sh
mkdir bin
cd model

# provide input path to training data and output path to save trained model and metadata
# usage: train.py <input path> <output path> <is memory intensive (default: 0)> <pretrained weights (optional)>
./train.py ../datasets/web/training_set ../bin

# train on images pre-processed as arrays
./train.py ../datasets/web/training_features ../bin

# train with generator to avoid having to fit all the data in memory (RECOMMENDED)
./train.py ../datasets/web/training_features ../bin 1

# train on top of pretrained weights
./train.py ../datasets/web/training_features ../bin 1 ../bin/pix2code.h5
```

Generate code for batch of GUIs:
```sh
mkdir code
cd model

# generate DSL code (.gui file), the default search method is greedy
# usage: generate.py <trained weights path> <trained model name> <input image> <output path> <search method (default: greedy)>
./generate.py ../bin pix2code ../gui_screenshots ../code

# equivalent to command above
./generate.py ../bin pix2code ../gui_screenshots ../code greedy

# generate DSL code with beam search and a beam width of size 3
./generate.py ../bin pix2code ../gui_screenshots ../code 3
```

Generate code for a single GUI image:
```sh
mkdir code
cd model

# generate DSL code (.gui file), the default search method is greedy
# usage: sample.py <trained weights path> <trained model name> <input image> <output path> <search method (default: greedy)>
./sample.py ../bin pix2code ../test_gui.png ../code

# equivalent to command above
./sample.py ../bin pix2code ../test_gui.png ../code greedy

# generate DSL code with beam search and a beam width of size 3
./sample.py ../bin pix2code ../test_gui.png ../code 3
```

Compile generated code to target language:
```sh
cd compiler

# compile .gui file to Android XML UI
./android-compiler.py <input file path>.gui

# compile .gui file to iOS Storyboard
./ios-compiler.py <input file path>.gui

# compile .gui file to HTML/CSS (Bootstrap style)
./web-compiler.py <input file path>.gui
```

## FAQ

### Will pix2code supports other target platforms/languages?
No, pix2code is only a research project and will stay in the state described in the paper for consistency reasons.
This project is really just a toy example but you are of course more than welcome to fork the repo and experiment yourself with other target platforms/languages.

### Will I be able to use pix2code for my own frontend projects?
No, pix2code is experimental and won't work for your specific use cases.

### How is the model performance measured?
The accuracy/error reported in the paper is measured at the DSL level by comparing each generated token with each expected token.
Any difference in length between the generated token sequence and the expected token sequence is also counted as error.

### How long does it take to train the model?
On a Nvidia Tesla K80 GPU, it takes a little less than 5 hours to optimize the 109 * 10^6 parameters for one dataset; so expect around 15 hours if you want to train the model for the three target platforms.

### I am a front-end developer, will I soon lose my job?
*(I have genuinely been asked this question multiple times)*

**TL;DR** Not anytime soon will AI replace front-end developers.

Even assuming a mature version of pix2code able to generate GUI code with 100% accuracy for every platforms/languages in the universe, front-enders will still be needed to implement the logic, the interactive parts, the advanced graphics and animations, and all the features users love. The product we are building at [Uizard Technologies](https://uizard.io) is intended to bridge the gap between UI/UX designers and front-end developers, not replace any of them. We want to rethink the traditional workflow that too often results in more frustration than innovation. We want designers to be as creative as possible to better serve end users, and developers to dedicate their time programming the core functionality and forget about repetitive tasks such as UI implementation. We believe in a future where AI collaborate with humans, not replace humans.

## Media coverage

* [Wired UK](http://www.wired.co.uk/article/pix2code-ulzard-technologies)
* [The Next Web](https://thenextweb.com/apps/2017/05/26/ai-raw-design-turn-source-code)
* [Fast Company](https://www.fastcodesign.com/90127911/this-startup-uses-machine-learning-to-turn-ui-designs-into-raw-code)
* [NVIDIA Developer News](https://news.developer.nvidia.com/ai-turns-ui-designs-into-code)
* [Lifehacker Australia](https://www.lifehacker.com.au/2017/05/generating-user-interface-code-from-images-using-machine-learning/)
* [Two Minute Papers](https://www.youtube.com/watch?v=Fevg4aowNyc) (web series)
* [NLP Highlights](https://soundcloud.com/nlp-highlights/17a) (podcast)
* [Data Skeptic](https://dataskeptic.com/blog/episodes/2017/pix2code) (podcast)
* Read comments on [Hacker News](https://news.ycombinator.com/item?id=14416530)
