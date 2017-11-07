# 3D-Object-Classificationvia-MVCNN
This project implements on fine-tuning a convolutional neural network (CNN) to perform 3D object classification, based upon the multi-view CNN (MVCNN)

# Tools
Machine Learning Library Keras,  Anaconda Python, TensorFlow, GPU-equipped Instance: p2.xlarge

# Details
1. Dataset: ModelNet-40 dataset. 

2. Python ModelNet-40 loader for the above dataset. 

3. Fine-tune an existing CNN using ResNet-50 which is already included with Keras in keras.applications, with weights that have been pre-trained on ImageNet classification. 

    (1) instantiates a ResNet50 instance without the top layers, 
    
    (2) adds a flat layer followed by a dense (fully connected) layer with 40 outputs that uses softmax, for the classification, 
    
    (3) sets the first p fraction of layers to be not trainable, 

    (4) Implements a "batching" generator that takes as input an existing ModelNet-40 generator and produces mini-batches of size n. 

