#OpenCL Caffe

This is an OpenCL implementation of Caffe, a mainstream DNN framework (https://github.com/BVLC/caffe). It includes a largely complete Caffe feature set as of August 2015. The project is under active development to improve performance and add new features. Contributions from the community are welcome.

OpenCL (https://en.wikipedia.org/wiki/OpenCL) is an open standard parallel programming language for heterogeneous platforms. OpenCL is supported by a variety of commercial chip manufacturers. 

#Branches
 We have three braches in this repo.
 
 -stable, the stable branch for users
 
 -dev, the developer branch, we encourage people to contribute on this branch
 
 -master, the original Caffe's master branch against which our code is syncrhonized.
 
#Design features
  -All Caffe layers ported to OpenCL

  -Performance improvement by batched implementation for conv layer based on clBLAS

  -The user can choose the optimal batch number depending on H/W properties, image size and minibatch size

  -Supports OpenCL 2.0, 1.2
  
  -Implemented in C++ and OpenCL, maintaining the same interfaces as the original Caffe

  -Users can directly run DNN models: AlexNet, VGG-16 and VGG-19

Note: More features are planned in the near future. Currently this implementation has been verified and tuned on AMD devices (CPUs/GPUs/APUs). Compatibility across different chip manufacturers will be considered for future addition.

#Performance

We intend to keep updating the latest performance as we make optimizations. Fury results are preliminary and are actively being improved.

* Training speed (Model: AlexNet, minibatch size 128)

    -AMD W9100, 255 images per second (config, with A10-7850k)

    -AMD R9 Fury, 261 images per second (config, with A10-7850k)
    
    -AMD S9150 @900MHz, 227 images per second (564ms per iteration) (config, with CPU Intel E5-2640, clBLAS 2.6, OpenCL 2.0 with driver 1774.6)

* Recognition speed (Model: AlexNet, minibatch size 128)

    -AMD W9100, 590 images per second (config, with A10-7850k)

    -AMD R9 Fury, 699 images per second (config, with A10-7850k)
    
    -AMD S9150 @900MHz, 452 images per second (283ms per iteration) ( config, with CPU Intel E5-2640, clBLAS 2.6, OpenCL 2.0 with driver 1774.6)

#Wiki
For more information on how to install, use or contribute to this code base, please visit our wiki page:
 https://github.com/amd/OpenCL-caffe/wiki

#Contributors
Junli Gu, Yibing Liu, Yuan Gao, Maohua Zhu

We thank Mauricio Breternitz, Hanjin Chu and Greg Stoner for their technical suggestions and support. 

If you have any questions, please send an email to Junli.Gu@amd.com 

#Support needed
 As an open source project, we hope to maintain an open dynamics and sharing culture. We encourage the contribution and support from the community to improve it together.

#License
The original Caffe is provided in the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE) open source license. The OpenCL ports written by AMD is covered by AMD license. We encourage the contribution and support from external, your contribution will be covered either by BSD 2-Clause license or whichever your preferred license.

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
