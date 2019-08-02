
# Deep Networks: Building an Image Classifier - Lab

## Introduction

For the final lab in this section, we'll build a more advanced **_Multi-Layer Perceptron_** to solve image classification for a classic dataset, MNIST!  This dataset consists of thousands of labeled images of handwritten digits, and it has a special place in the history of Deep Learning. 

## Packages

First, let's import all the packages that you 'll need for this lab.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
```


```python
# __SOLUTION__ 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
```

    Using TensorFlow backend.


##  The data 

Before we get into building the model, let's load our data and take a look at a sample image and label. 

The MNIST dataset is often used for benchmarking model performance in the world of AI/Deep Learning research. Because it's commonly used, Keras actually includes a helper function to load the data and labels from MNIST--it even loads the data in a format already split into training and testing sets!

Run the cell below to load the MNIST dataset. Note that if this is the first time you've worked with MNIST through Keras, this will take a few minutes while Keras downloads the data. 


```python
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```


```python
# __SOLUTION__ 
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

Great!  

Now, let's quickly take a look at an image from the MNIST dataset--we can visualize it using matplotlib. Run the cell below to visualize the first image and its corresponding label. 


```python
sample_image = X_train[0]
sample_label =y_train[0]
display(plt.imshow(sample_image))
print("Label: {}".format(sample_label))
```


```python
# __SOLUTION__ 
sample_image = X_train[0]
sample_label =y_train[0]
display(plt.imshow(sample_image))
print("Label: {}".format(sample_label))
```


    <matplotlib.image.AxesImage at 0x1c5a9244518>


    Label: 5



![png](index_files/index_11_2.png)


Great! That was easy. Now, we'll see that preprocessing image data has a few extra steps in order to get it into a shape where an MLP can work with it. 

## Preprocessing Images For Use With MLPs

By definition, images are matrices--they are a spreadsheet of pixel values between 0 and 255. We can see this easily enough by just looking at a raw image:


```python
sample_image
```


```python
# __SOLUTION__ 
sample_image
```




    array([[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   3,
             18,  18,  18, 126, 136, 175,  26, 166, 255, 247, 127,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,  30,  36,  94, 154, 170,
            253, 253, 253, 253, 253, 225, 172, 253, 242, 195,  64,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,  49, 238, 253, 253, 253, 253,
            253, 253, 253, 253, 251,  93,  82,  82,  56,  39,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,  18, 219, 253, 253, 253, 253,
            253, 198, 182, 247, 241,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,  80, 156, 107, 253, 253,
            205,  11,   0,  43, 154,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,  14,   1, 154, 253,
             90,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 139, 253,
            190,   2,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  11, 190,
            253,  70,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  35,
            241, 225, 160, 108,   1,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
             81, 240, 253, 253, 119,  25,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,  45, 186, 253, 253, 150,  27,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,  16,  93, 252, 253, 187,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0, 249, 253, 249,  64,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,  46, 130, 183, 253, 253, 207,   2,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  39,
            148, 229, 253, 253, 253, 250, 182,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  24, 114, 221,
            253, 253, 253, 253, 201,  78,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,  23,  66, 213, 253, 253,
            253, 253, 198,  81,   2,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,  18, 171, 219, 253, 253, 253, 253,
            195,  80,   9,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,  55, 172, 226, 253, 253, 253, 253, 244, 133,
             11,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0, 136, 253, 253, 253, 212, 135, 132,  16,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0]], dtype=uint8)



This is a problem in its current format, because MLPs take their input as vectors, not matrices or tensors. If all of the images were different sizes, then we would have a more significant problem on our hands, because we'd have challenges getting each image reshaped into a vector the exact same size as our input layer. However, this isn't a problem with MNIST, because all images are black white 28x28 pixel images. This means that we can just concatenate each row (or column) into a single 784-dimensional vector! Since each image will be concatenated in the exact same way, positional information is still preserved (e.g. the pixel value for the second pixel in the second row of an image will always be element number 29 in the vector). 

Let's get started. In the cell below, print the `.shape` of both `X_train` and `X_test`


```python

```


```python
# __SOLUTION__ 
print(X_train.shape)
X_test.shape
```

    (60000, 28, 28)





    (10000, 28, 28)



We can interpret these numbers as saying "X_train consists of 60,000 images that are 28x28". We'll need to reshape them from `(28, 28)`,a 28x28 matrix, to `(784,)`, a 784-element vector. However, we need to make sure that the first number in our reshape call for both `X_train` and `X_test` still correspond to the number of observations we have in each. 

In the cell below:

* Use the `.reshape()` method to reshape X_train. The first parameter should be `60000`, and the second parameter should be `784`.
* Similarly, reshape `X_test` to `10000` and `784`. 
* Also, chain both `.reshape()` calls with an `.astype("float32")`, so that we can our data from type `uint8` to `float32`. 


```python
X_train = None
X_test = None
```


```python
# __SOLUTION__ 
X_train = X_train.reshape(60000, 784).astype("float32")
X_test = X_test.reshape(10000, 784).astype("float32")
```

Now, let's check the shape of our training and testing data again to see if it worked. 


```python

```


```python
# __SOLUTION__ 
print(X_train.shape)
X_test.shape
```

    (60000, 784)





    (10000, 784)



Great! Now, we just need to normalize our data!

## Normalizing Image Data

Anytime we need to normalize image data, there's a quick hack we can use to do so easily. Since all pixel values will always be between 0 and 255, we can just scale our data by dividing every element by 255! Run the cell below to do so now. 


```python
X_train /= 255.
X_test /= 255.
```


```python
# __SOLUTION__ 
X_train /= 255.
X_test /= 255.
```

Great! We've now finished preprocessing our image data. However, we still need to deal with our labels. 

## Preprocessing our Labels

Let's take a quick look at the first 10 labels in our training data:


```python

```


```python
# __SOLUTION__ 
y_train[:10]
```




    array([5, 0, 4, 1, 9, 2, 1, 3, 1, 4], dtype=uint8)



As we can see, the labels for each digit image in the training set are stored as the corresponding integer value--if the image is of a 5, then the corresponding label will be `5`. This means that this is a **_Multiclass Classification_** problem, which means that we need to **_One-Hot Encode_** our labels before we can use them for training. 

Luckily, Keras provides a really easy utility function to handle this for us. 

In the cell below: 

* Use the function `to_categorical()` to one-hot encode our labels. This function can be found inside `keras.utils`. Pass in the following parameters:
    * The object we want to one-hot encode, which will be `y_train` or `y_test`
    * The number of classes contained in the labels, `10`.


```python
y_train = None
y_test = None
```


```python
# __SOLUTION__ 
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
```

Great. Now, let's examine the label for the first data point, which we saw was `5` before. 


```python

```


```python
# __SOLUTION__ 
y_train[0]
```




    array([0., 0., 0., 0., 0., 1., 0., 0., 0., 0.], dtype=float32)



Perfect! As we can see, the index corresponding to the number `5` is set to `1`, which everything else is set to `0`. That was easy!  Now, let's get to the fun part--building our model!

## Building Our Model

For the remainder of this lab, we won't hold your hand as much--flex your newfound keras muscles and build an MLP with the following specifications:

* A `Dense` hidden layer with `64` neurons, and a `'tanh'` activation function. Also, since this is the first hidden layer, be sure to also pass in `input_shape=(784,)` in order to create a correctly-sized input layer!
* Since this is a multiclass classification problem, our output layer will need to be a `Dense` layer where the number of neurons is the same as the number of classes in the labels. Also, be sure to set the activation function to `'softmax'`.

## Data Exploration and Normalization

Be sure to carefully review the three code blocks below. Here, we demonstrate some common data checks you are apt to perform after importing, followed by standard data normalization to set all values to a range between 0 and 1.


```python
model_1  = None

```


```python
# __SOLUTION__ 
model_1  = Sequential()
model_1.add(Dense(64, activation='tanh', input_shape=(784,)))
model_1.add(Dense(10, activation='softmax'))
```

Now, compile your model with the following parameters:

* `loss='categorical_crossentropy'`
* `optimizer='sgd'`
* `metrics = ['accuracy']`


```python

```


```python
# __SOLUTION__ 
model_1.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
```

Let's quickly inspect the shape of our model before training it and see how many training parameters we have. In the cell below, call the model's `.summary()` method. 


```python

```


```python
# __SOLUTION__ 
model_1.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_1 (Dense)              (None, 64)                50240     
    _________________________________________________________________
    dense_2 (Dense)              (None, 10)                650       
    =================================================================
    Total params: 50,890
    Trainable params: 50,890
    Non-trainable params: 0
    _________________________________________________________________


50,890 trainable parameters! Note that while this may seem large, deep neural networks in production may have hundreds or thousands of layers and many millions of trainable parameters!

Let's get on to training. In the cell below, fit the model. Use the following parameters:

* Our training data and labels
* `epochs=5`
* `batch_size=64`
* `validation_data=(X_test, y_test)`


```python
results_1 = None
```


```python
# __SOLUTION__ 
results_1 = model_1.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))
```

    Train on 60000 samples, validate on 10000 samples
    Epoch 1/5
    60000/60000 [==============================] - 5s 78us/step - loss: 0.8419 - acc: 0.7958 - val_loss: 0.4952 - val_acc: 0.8812
    Epoch 2/5
    60000/60000 [==============================] - 3s 45us/step - loss: 0.4503 - acc: 0.8840 - val_loss: 0.3864 - val_acc: 0.8980
    Epoch 3/5
    60000/60000 [==============================] - 3s 45us/step - loss: 0.3793 - acc: 0.8966 - val_loss: 0.3419 - val_acc: 0.9079
    Epoch 4/5
    60000/60000 [==============================] - 3s 45us/step - loss: 0.3436 - acc: 0.9056 - val_loss: 0.3168 - val_acc: 0.9111
    Epoch 5/5
    60000/60000 [==============================] - 3s 46us/step - loss: 0.3204 - acc: 0.9107 - val_loss: 0.2979 - val_acc: 0.9171


## Visualizing Our Loss and Accuracy Curves

Now, let's inspect the model's performance and see if we detect any overfitting or other issues. In the cell below, create two plots:

* The `loss` and `val_loss` over the training epochs
* The `acc` and `val_acc` over the training epochs

**_HINT:_** Consider copying over the visualization function from the previous lab in order to save time!


```python
def visualize_training_results(results):
    pass
```


```python

```


```python
# __SOLUTION__ 
def visualize_training_results(results):
    history = results.history
    plt.figure()
    plt.plot(history['val_loss'])
    plt.plot(history['loss'])
    plt.legend(['val_loss', 'loss'])
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    
    plt.figure()
    plt.plot(history['val_acc'])
    plt.plot(history['acc'])
    plt.legend(['val_acc', 'acc'])
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()
```


```python
# __SOLUTION__ 
visualize_training_results(results_1)
```


![png](index_files/index_53_0.png)



![png](index_files/index_53_1.png)


Pretty good! Note that since our validation scores are currently higher than our training scores, its extremely unlikely that our model is overfitting the training data. This is a good sign--that means that we can probably trust the results that our model is ~91.7% accurate at classifying handwritten digits!

## Building a Bigger Model

Now, let's add another hidden layer and see how this changes things. In the cells below, create a second model. This model should have the following architecture:

* Input layer and first hidden layer same as `model_1`
* Another `Dense` hidden layer, this time with `32` neurons and a `'tanh'` activation function
* An output layer same as `model_1`. 

Build this model in the cell below.


```python
model_2 = None

```


```python
# __SOLUTION__ 
model_2 = Sequential()
model_2.add(Dense(64, activation='tanh', input_shape=(784,)))
model_2.add(Dense(32, activation='tanh'))
model_2.add(Dense(10, activation='softmax'))
```

Let's quickly inspect the `.summary()` of the model again, to see how many new trainable parameters this extra hidden layer has introduced.


```python

```


```python
# __SOLUTION__ 
model_2.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_3 (Dense)              (None, 64)                50240     
    _________________________________________________________________
    dense_4 (Dense)              (None, 32)                2080      
    _________________________________________________________________
    dense_5 (Dense)              (None, 10)                330       
    =================================================================
    Total params: 52,650
    Trainable params: 52,650
    Non-trainable params: 0
    _________________________________________________________________


This model isn't much bigger, but the layout means that the 2080 parameters in the new hidden layer will be focused on higher layers of abstraction than the first hidden layer. Let's see how it compares after training. 

In the cells below, compile and fit the model using the same parameters as we did for `model_1`.


```python

```


```python
results_2 = None
```


```python
# __SOLUTION__ 
model_2.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
```


```python
# __SOLUTION__ 
results_2 = model_2.fit(X_train, y_train, batch_size=64, epochs=5, validation_data=(X_test, y_test))
```

    Train on 60000 samples, validate on 10000 samples
    Epoch 1/5
    60000/60000 [==============================] - 3s 51us/step - loss: 0.9263 - acc: 0.7694 - val_loss: 0.5239 - val_acc: 0.8764
    Epoch 2/5
    60000/60000 [==============================] - 3s 48us/step - loss: 0.4591 - acc: 0.8826 - val_loss: 0.3839 - val_acc: 0.8996
    Epoch 3/5
    60000/60000 [==============================] - 3s 48us/step - loss: 0.3693 - acc: 0.9011 - val_loss: 0.3301 - val_acc: 0.9112
    Epoch 4/5
    60000/60000 [==============================] - 3s 48us/step - loss: 0.3255 - acc: 0.9100 - val_loss: 0.2990 - val_acc: 0.9182
    Epoch 5/5
    60000/60000 [==============================] - 3s 49us/step - loss: 0.2973 - acc: 0.9168 - val_loss: 0.2764 - val_acc: 0.9223


Now, visualize the plots again. 


```python

```


```python
# __SOLUTION__ 
visualize_training_results(results_2)
```


![png](index_files/index_67_0.png)



![png](index_files/index_67_1.png)


Slightly better validation accuracy, with no evidence of overfitting--great! If you run the model for more epochs, you'll see the model continue to improve performance, until the validation metrics plateau and the model begins to overfit the training data. 

## A Bit of Tuning

As a final exercise, let's see what happens to the model's performance if we switch activation functions from `'tanh'` to `'relu'`. In the cell below, recreate  `model_2`, but replace all `'tanh'` activations with `'relu'`. Then, compile, train, and plot the results using the same parameters as the other two. 


```python
model_3 = None

```


```python

```


```python

```


```python
results_3 = None
```


```python

```


```python
# __SOLUTION__ 
model_3 = Sequential()
model_3.add(Dense(64, activation='relu', input_shape=(784,)))
model_3.add(Dense(32, activation='relu'))
model_3.add(Dense(10, activation='softmax'))
```


```python
# __SOLUTION__ 
model_3.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_10 (Dense)             (None, 64)                50240     
    _________________________________________________________________
    dense_11 (Dense)             (None, 32)                2080      
    _________________________________________________________________
    dense_12 (Dense)             (None, 10)                330       
    =================================================================
    Total params: 52,650
    Trainable params: 52,650
    Non-trainable params: 0
    _________________________________________________________________



```python
# __SOLUTION__ 
model_3.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
```


```python
# __SOLUTION__ 
results_3 = model_3.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))
```

    Train on 60000 samples, validate on 10000 samples
    Epoch 1/5
    60000/60000 [==============================] - 3s 53us/step - loss: 1.0062 - acc: 0.7204 - val_loss: 0.4557 - val_acc: 0.8801
    Epoch 2/5
    60000/60000 [==============================] - 3s 48us/step - loss: 0.3974 - acc: 0.8893 - val_loss: 0.3288 - val_acc: 0.9091
    Epoch 3/5
    60000/60000 [==============================] - 3s 48us/step - loss: 0.3251 - acc: 0.9077 - val_loss: 0.2889 - val_acc: 0.9194
    Epoch 4/5
    60000/60000 [==============================] - ETA: 0s - loss: 0.2903 - acc: 0.916 - 3s 48us/step - loss: 0.2897 - acc: 0.9170 - val_loss: 0.2596 - val_acc: 0.9272
    Epoch 5/5
    60000/60000 [==============================] - 3s 48us/step - loss: 0.2639 - acc: 0.9252 - val_loss: 0.2408 - val_acc: 0.9332



```python
# __SOLUTION__ 
visualize_training_results(results_3)
```


![png](index_files/index_79_0.png)



![png](index_files/index_79_1.png)


Performance improved even further! ReLU is one of the most commonly used activation functions around right now--it's especially useful in computer vision problems like image classification, as we've just seen. 

## Summary

In this lab, you once again practiced and reviewed the process of building a neural network. This time, we built a more complex network with additional layers which improved the performance on our data set with MNIST images! 

