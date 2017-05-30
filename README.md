# pix2code
*Generating Code from a Graphical User Interface Screenshot*

[![License](http://img.shields.io/badge/license-APACHE2-blue.svg)](LICENSE.txt)

* A video demo of the system can be seen [here](https://youtu.be/pqKeXkhFA3I)
* The paper is available at [https://arxiv.org/abs/1705.07962](https://arxiv.org/abs/1705.07962)
* Official research page: [https://uizard.io/research#pix2code](https://uizard.io/research#pix2code)

## Abstract
Transforming a graphical user interface screenshot created by a designer into computer code is a typical task conducted by a developer in order to build customized software, websites and mobile applications. In this paper, we show that Deep Learning techniques can be leveraged to automatically generate code given a graphical user interface screenshot as input. Our model is able to generate code targeting three different platforms (i.e. iOS, Android and web-based technologies) from a single input image with over 77% of accuracy.

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

## FAQ

### When will the datasets be available?
The datasets will be made available upon publication or rejection of the paper to the [NIPS 2017 conference](https://nips.cc/Conferences/2017/Dates); author notification is scheduled for early September 2017 so the datasets will be uploaded to this repo during the same period. We will provide datasets consisting of GUI screenshots, associated DSL code, and associated target code for three different platforms (iOS, Android, web-based GUI).

### Will the source code be available?
As written in the [paper](https://arxiv.org/pdf/1705.07962.pdf), the datasets will be made available but nothing is said about the source code. However, because of the unexpected amount of interest in this project, the pix2code implementation described in the [paper](https://arxiv.org/pdf/1705.07962.pdf) will also be open-sourced in this repo together with the datasets.

### Will pix2code support other target platforms/languages?
No, pix2code is only a research project and will stay in the state described in the paper for consistency reasons.
This project is really just a toy example demonstrating part of the technology we are building at [Uizard Technologies](https://uizard.io) (our customer-ready platform is likely to support other target platforms/languages).
You are of course more than welcome to fork the repo and experiment yourself with other target platforms/languages.

### Will I be able to use pix2code for my frontend project?
No, pix2code is experimental and won't work for your specific use cases.
However, stay tuned to [Uizard Technologies](https://uizard.io), we are working hard building a customer-ready platform to generate code from GUI mockups that you can use for your projects!

### How is the model performance measured?
The accuracy/error reported in the paper is measured at the DSL level by comparing each generated token with each expected token.
Any difference in length between the generated token sequence and the expected token sequence is also counted as error.

### How long does it take to train the model?
On a Nvidia Tesla K80 GPU, it takes a little less than 5 hours to optimize the 109 * 10^6 parameters for one dataset; so expect around 15 hours if you want to train the model for the three target platforms.

### I am a front-end developer, will I soon lose my job?
*(I have genuinely been asked this question multiple times)*

**TL;DR** Not anytime soon will AI replace front-end developers.

Even assuming a mature version of pix2code able to generate GUI code with 100% accuracy for every platforms/languages in the universe, front-enders will still be needed to implement the logic, the interactive parts, the advanced graphics and animations, and all the features users love. The product we are building at [Uizard Technologies](https://uizard.io) is intended to bridge the gap between UI/UX designers and front-end developers, not replace any of them. We want to rethink the traditional workflow that too often results in more frustration than innovation. We want designers to be as creative as possible to better serve end users, and developers to dedicate their time programming the core functionalities and forget about repetitive tasks such as UI implementation. We believe in a future where AI collaborate with humans, not replace humans.

## Media coverage

* [Wired UK](http://www.wired.co.uk/article/pix2code-ulzard-technologies)
* [The Next Web](https://thenextweb.com/apps/2017/05/26/ai-raw-design-turn-source-code)
