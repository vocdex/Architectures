# LeNet5-MNIST-PyTorch

#### The model file includes the original and a tuned version of the LeNet5. Feel free to comment/uncomment to choose the model type in your training
![image](https://user-images.githubusercontent.com/25716030/162345646-b13c9af0-bdb5-4ce7-9a62-c0834cba9e5f.png)
## Model architecture
```
Input:  32*32 image input layer.   
Layer1: 5x5 CNN with 6 filters.   
Layer1  Output: (32-5+1=28) 28x28x6 feature map (5*5*1+1)*6=156 params   
Layer2: Pooling layer: no learnable params    
Layer2  Output: 14x14x6.   
Layer3: 5x5 CNN with with 16 filters.   
Layer3  Output: (14-5+1=10) 10x10x16 feature map  (5*5*6+1)*16=2416 params 
Layer4: Pooling layer: no learnable params.   
Layer4  Output: 5x5x16 feature map.   
Layer5: 5x5 CNN with 120 filters.   
Layer5  Output: (5-5+1=1) 1x1x120 (5*5*16+1)*120=48120 params  
Layer6: FC layer with 84 neurons (120*1+1)*84=10164 params   
Layer7: FC layer with 10 output neurons (84*1+1)*10=850 params.      

Total # of parameters: 61706.   
```
In the original paper, the number of learnable parameters is 60,000.   
The difference is due to the fact that they have used a differ connections between CNN and Pooling layers.  
 *Check out this amazing answer by [hbaderts](https://stackoverflow.com/a/42787467/6478817) on how to calculate the learnable parameters.*  
## Requirements
Python3  
PyTorch >= 0.4.0  
torchvision >= 0.1.8

