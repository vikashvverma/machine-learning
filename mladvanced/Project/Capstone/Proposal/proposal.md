# Machine Learning Engineer Nanodegree
## Capstone Proposal
Vikash Verma  
September 7th, 2018

## Proposal
Flowers Recognition

### Domain Background

Learning is an inherent instinct of Human being. We learn by remembering the context. On encountering a similar context we apply the action which was deemed positive.

Machine can also be made to learn. A well trained machine learning model can also apply the context on a new situation and try to generate meaningful information.

E.g. Facebook applies Machine Learning on each and every image and try to provide meaningful insights on that as shown in following picture.

![Facebook Snapshot](./context.png)



If you look closely at the picture above you can figure out that Facebook tries to automatically add contextual information to every picture:

 - People
 - Smiling
 - Meme
 - Text



In modern world, it is becoming increasingly important to get contextual information. In this proposed project, we try to classify different category of flower. 

There are a lot of research on Flower classification, some of them are as follows:

- [University of Oxford](http://www.robots.ox.ac.uk/~vgg/research/flowers_demo/)
- [IEEE](https://ieeexplore.ieee.org/document/8288453/)
- [Flower image classification modeling using neural network](https://www.researchgate.net/publication/281996446_Flower_image_classification_modeling_using_neural_network/references)



### Problem Statement

The objective this project to correctly classify an image in one of the five category of the flower.

Although there are many more categories of flowers, here we are focusing on only five flower categories viz. **Sunflower, Dandelion, Tulip, Daisy and Rose**



For a new category of image we try to predict to closely resembling category out of these five category.

Potential applications are huge.

- Imagine how Google shows sunflower images when searched for **sunflower images**. 
- Imagine asking Alexa to send a rose to loved ones.
- Automated recommendation of different design from flowers on different occasion.
  - **Congratulatory**:  A combination of flowers
  - **Expressing Love**: Red Roses
  - **Funeral**: White Lilies



### Datasets and Inputs
The dataset is available on [Kaggle](https://www.kaggle.com/alxmamaev/flowers-recognition). The data is scraped from Flickr, Google images and Yandex images.

The dataset contains five kinds of flower's  images.

 - Daisy

 - Dandelion

 - Rose

 - Sunflower

 - Tulip

   ​

The flowers are present in dataset as follows: 

```
flowers
│
└───Daisy
│   
└───Dandelion
|
└───Rose
│   
└───Sunflower
|
└───Tulip
```


We can create dataset for training, validation and testing to easily use `load_files` from `sklearn` as follows:



```
data
│
└───train
|    │
|    └───Daisy
|    │   
|    └───Dandelion
|    |
|    └───Rose
|    │   
|    └───Sunflower
|    |
|    └───Tulip
└───valid
|    │
|    └───Daisy
|    │   
|    └───Dandelion
|    |
|    └───Rose
|    │   
|    └───Sunflower
|    |
|    └───Tulip
└───test
     │
     └───Daisy
     │   
     └───Dandelion
     |
     └───Rose
     │   
     └───Sunflower
     |
     └───Tulip
```


There are around 4323 images of five categories as follows:

- Sunflower: 734
- Tulip: 984
- Daisy: 769
- Rose: 784
- Dandelion: 1052



The dataset does not seems to be imbalanced. 

Even if the dataset is huge and total trainable parameter would be huge, Kaggle kernel or Google Colab can be used for training.

### Solution Statement

There are multiple ways for classification.  Here we can use a CNN to classify flowers. A CNN from scratch might take longer to run but it can provide valuable insights quickly. We can also use Transfer Learning from Keras to create a new model and train this model on this data. Using Transfer Learning can help reach greater accuracy.

### Benchmark Model
There are two benchmark model we can use.

 - Kaggle: Best score from Kaggle
 - Naive model: A simple naive model such as  few layered CNN, logistic regression or other model can be used for benchmark.

### Evaluation Metrics

We can use accuracy as the evaluation metrics.

Since the dataset is divide into three sets viz. training, validation and testing. We can use accuracy on testing set as an evaluation metrics.

This would be a good metric here as dataset is not imbalanced.

### Project Design

The project is designed in following ways:

- Create a **data** and three subdirectory in it viz. **train**, **valid** and **test**
- Move 60% of images of each category to **train** directory, 20% to valid and 20% to test
- Now we have flower dataset for training, validation and testing
- Load dataset in memory for training, validation and testing.
- Resize all images to same size and create 4D tensor to be supllied to Keras CNN
- Use a simple CNN or logistic regression model to get the benchmark.
- We can implement the model in two ways
  - Building a CNN from scratch
    - Create a CNN architecture from Scratch and train the model. Save the best model weights during training.
    - Load the best model weight
    - Use the testing set to get resulting accuracy.
  - Transfer Learning
    - Use on of the pre trained  Keras model for transfer learning
    - Load the best model weight
    - Use the testing set to get the accuracy of the model



Additionally, Google Colab or Kaggle kernel can be used for training the model.

### References

- [Keras](https://keras.io/)

- [Kaggle](https://www.kaggle.com/alxmamaev/flowers-recognition)

- [Facebook](https://www.facebook.com/)

- [University of Oxford](http://www.robots.ox.ac.uk/~vgg/research/flowers_demo/)

- [IEEE](https://ieeexplore.ieee.org/document/8288453/)

- [Flower image classification modeling using neural network](https://www.researchgate.net/publication/281996446_Flower_image_classification_modeling_using_neural_network/references)

  ​