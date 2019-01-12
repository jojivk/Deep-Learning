## Mini AlexNet Vs LeNet for MNIST



### Alexnet
Alexnet ushered in the new era of CNNs. In this implementation, all except the first layer of Alexnet is used to classify MNIST. 
 MNIST images (28x 28 x 1) are smaller in size compared to Imagenet images (227 x 227 x3 ).While in Alexnet, second convolution layer (after first maxpool) is 27 x 27. So I thought it is good to try Alexnet for MNIST starting with second convolutional layer.
 
![alt text](https://cdn-images-1.medium.com/max/1536/1*qyc21qM0oxWEuRaj-XJKcw.png)
    
The number of features/nodes in each layer, I have reduced by a factor of 4,since MNIST is gray scale and Imagenet images are color images,  Below is the implementation. 

It look much longer time to train the network, but have higher test set accuracy (98%). 
Compared that with  LeNet-5 : 94% with same set of hyperparameters.

 
### LeNet
LeNet-5 A 5 layer Convolutional NN for Image Classification.

Convolutional NN has one of the first popular implementations in LeNet-5 by Yann LeCunn.
LeNet is a 5 layer NN with 3 Convolutional layers and 2 fully connected layers.

![alt text](https://www.researchgate.net/profile/Yiren_Zhou/publication/312170477/figure/fig1/AS:448817725218816@1484017892071/Structure-of-LeNet-5.ppm)
