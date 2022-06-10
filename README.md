# VoteNet
An Implementation of VoteNet: "Deep Hough Voting for 3D Object Detection in Point Clouds". Credits to the original authors at https://github.com/facebookresearch/votenet.
The original research paper, which is also used as a reference for this project, is available at https://arxiv.org/pdf/1904.09664.pdf.

We adopted existing PointNet++ CUDA implementation. 

We implemented VoteNet by using Tensorflow and Pytorch, by focusing on voting and proposal module.
Also we considered different structure in the network.

Installation
````

Tensorflow 1.2, Keras 2.0, PyTorch

````

Please install Python dependencies same as original Votenet by pip
````
opencv-python
plyfile
'trimesh>=2.35.39,<2.35.40'
'networkx>=2.2,<2.3'
matplotlib

````

You can check the results by running pre-trained model

````
python demo.py

````

Contact: Ulzhalgas Rakhman urakhman@kaist.ac.kr and Ali Ahmed ali.ahmed@kaist.ac.kr
