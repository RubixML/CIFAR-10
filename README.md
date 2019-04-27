# CIFAR-10 Image Classifier

Computer vision is one of the most fascinating use cases for deep learning because it allows a computer to see the world that we live in. CIFAR-10 (short for *Canadian Institute For Advanced Research*) is a [famous dataset](https://en.wikipedia.org/wiki/CIFAR-10) consisting of 60,000 32x32 color images in 10 classes (dog, cat, car, etc.) with 6,000 images per class. In this tutorial, we'll use the CIFAR-10 dataset to train a feed forward multi layer neural network to recognize objects within images using Rubix ML in PHP.

- **Difficulty**: Hard
- **Training time**: Long
- **Memory needed**: > 8G

## Installation

Clone the project locally:
```sh
$ git clone https://github.com/RubixML/CIFAR-10
```

Install project dependencies with [Composer](http://getcomposer.com):
```sh
$ composer install
```

## Requirements
- [PHP](https://php.net) 7.1.3 or above

## Tutorial
Deep learning is a subset of machine learning concerned with breaking down raw data into higher order representations through layered computations. In the case of image recognition, we use the raw pixel data as the input to a deep learning sytem so that the learner can build up its representation of the world. Neural networks are a type of deep learning system inspired by the human nervous system that use layered computational units called *hidden* layers that are able to break down an image into its component parts such that the network can readily comprehend the differences among objects by their characteristic features at the output layer.

### Training

Let's start by importing our training set located in the *train* folder. The CIFAR-10 data comes to us in the form of 32x32 pixel PNG files which we must import into PHP using `imagecreatefrompng()` provided by the [GD](https://www.php.net/manual/en/book.image.php) extension. We use a regular expression to extract the label from the filename of the image. Finally, we load the samples and labels into a [Labeled](https://github.com/RubixML/RubixML#labeled) dataset object.

```php
use Rubix\ML\Datasets\Labeled;

$samples = $labels = [];

foreach (glob(__DIR__ . '/train/*.png') as $file) {
    $samples[] = [imagecreatefrompng($file)];
    $labels[] = preg_replace('/[0-9]+_(.*).png/', '$1', basename($file));
}

$dataset = new Labeled($samples, $labels);
```

The [Multi Layer Perceptron](https://github.com/RubixML/RubixML#multi-layer-perceptron) in Rubix ML is a type of deep learning model we'll train to recognize images from the CIFAR-10 dataset. It uses Gradient Descent with Backpropagation over multiple layers of *neurons* to train the network by gradually updating the signal that each neuron produces. In between [Dense](https://github.com/RubixML/RubixML#dense) neuronal layers we have an [Activation](https://github.com/RubixML/RubixML#activation) layer that performs a non-linear transformation of the neuron's output using a user-defined [Activation Function](https://github.com/RubixML/RubixML#activation-functions). For the purpose of this tutorial we'll use the [Leaky ReLU](https://github.com/RubixML/RubixML#leaky-relu) activation function, which is a good default, but feel free to experiment with different activation functions on your own. Finally, we add [Batch Norm](https://github.com/RubixML/RubixML#batch-norm) layers after every two Dense layers to help the network train faster by normalizing the activations as well as improve its generalization ability through the introduction of mild stochastic noise.

We'll need to wrap the base estimator in a transformer Pipeline to convert the images from the dataset into standardized raw pixel data on the fly. An [Image Resizer](https://github.com/RubixML/RubixML#image-resizer) ensures that all input vectors are of the same dimensionality. The [Image Vectorizer](https://github.com/RubixML/RubixML#image-vectorizer) handles extracting the raw color data. Then, the [Z Scale Standardizer](https://github.com/RubixML/RubixML#z-scale-standardizer) scales and centers the input vectors to have a mean of 0 and a standard deviation of 1.

```php
use Rubix\ML\Pipeline;
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\Transformers\ImageResizer;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\NeuralNet\Layers\BatchNorm;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\Transformers\ImageVectorizer;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\Classifiers\MultiLayerPerceptron;
use Rubix\ML\NeuralNet\ActivationFunctions\LeakyReLU;

$estimator = new PersistentModel(
    new Pipeline([
        new ImageResizer(32, 32),
        new ImageVectorizer(),
        new ZScaleStandardizer(),
    ], new MultiLayerPerceptron([
        new Dense(200),
        new Activation(new LeakyReLU()),
        new Dense(200),
        new BatchNorm(),
        new Activation(new LeakyReLU()),
        new Dense(200),
        new Activation(new LeakyReLU()),
        new Dense(100),
        new BatchNorm(),
        new Activation(new LeakyReLU()),
        new Dense(100),
        new Activation(new LeakyReLU()),
    ], 100, new Adam(0.001), 1e-4)),
    new Filesystem(MODEL_FILE, true)
);
```

Now all we have to do is pass the dataset to the estimator's `train()` method to begin training the network.

```php
$estimator->train($dataset);
```

### Validation

On the map ...


## Original Dataset
Creator: Alex Krizhevsky
Email: akrizhevsky '@' gmail.com 

### References
>- [1] A. Krizhevsky. (2009). Learning Multiple Layers of Features from Tiny Images.