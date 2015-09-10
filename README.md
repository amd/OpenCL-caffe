#OpenCL caffe

This is an OpenCL implementation of one of the main stream DNN framework-CAFFE, see more details about CAFFE below.  The goal is to provide the community an OpenCL version of DNN framework to use. Things are not perfect yet. We will keep adding new features and improving performance. We also hope to get help from community to improve it together.

OpenCL is an open standard parallel programming language that is supported by more than 20 companies. People can use this framework to run their DNN app on heterogeneous platforms from vairous commercial chip manufacturer. Compared to CUDA based DNN, this framework support cross-platform compatability and with design space to optimize accordingly.

#Design features
  -All layers ported to OpenCL
  
  -Aligned with caffe’s latest code

  -Performance improvement by batched sgemm implementation for conv layer

  -User can choose optimal batch number depending on H/W, image size and minibatch size

  -Supports OpenCL 2.0, 1.2
  
  -only contains C++ and OpenCL, maintains the same interfaces as original caffe to make it easy for caffe users

  -Users can directly run DNN models: AlexNet, VGG 16 and VGG-19

Note: More featurs will be added in the near future. And this OpenCL caffe only verifies on AMD devices (CPUs/GPUs/APUs). Compatibility across different chip manufacturers will be considered to add if there is a need.

#Performance

We will keep updating the latest performance we could achieve in this section.

* Training speed (Model: AlexNet, minibatch size 128)

    -AMD W9100 (5.2TFLOPS), 255 images per second

    -AMD R9 Fury((7.2TFLOPS)), 261 images per second

* Recognition speed (Model: AlexNet, minibatch size 128)

    -AMD W9100 (5.2TFLOPS), 590 images per second

    -AMD R9 Fury((7.2TFLOPS)), 699 images per second

#Wiki
For more information on how to install, use or contribute to this code base, please visit our wiki page:
https://github.com/amd/OpenCL-caffe/wiki

#Support needed
We encourage the contribution and support from the community to improve it together.

#License
Original caffe is provided in the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE) open source license. The OpenCL ports written by AMD is covered by AMD license. As an open source project, we hope to maintain an open dynamics and sharing culture. We encourage the contribution and support from external, your contribution will be covered either by BSD 2-Clause license or which ever your preferred licence.

# Original Caffe information
## Caffe

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu)) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BVLC reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
