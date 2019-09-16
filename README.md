# Rubix ML - CIFAR-10 Image Recognizer
CIFAR-10 (short for *Canadian Institute For Advanced Research*) is a [famous dataset](https://en.wikipedia.org/wiki/CIFAR-10) consisting of 60,000 32 x 32 color images in 10 classes (dog, cat, car, ship, etc.) with 6,000 images per class. In this tutorial, we'll use the CIFAR-10 dataset to train a feed forward neural network to recognize the primary object within images using Rubix ML.

- **Difficulty**: Hard
- **Training time**: Days
- **Memory needed**: 10G

## Installation
Clone the project locally with [Git](https://git-scm.com/):
```sh
$ git clone https://github.com/RubixML/CIFAR-10
```

> **Note**: Cloning may take longer than usual because of the large dataset.

Install project dependencies with [Composer](http://getcomposer.org/):
```sh
$ composer install
```

## Requirements
- [PHP](https://php.net) 7.1.3 or above

## Tutorial

### Introduction
Computer vision is one of the most fascinating use cases for deep learning because it allows a computer to see the world that we live in. Deep learning is a subset of machine learning concerned with breaking down raw data into higher order representations through layered computations. Neural networks are a type of deep learning system inspired by the human nervous system that uses structured computational units called *hidden* layers. In the case of image recognition, these layers are able to break down an image into its component parts such that the network can readily comprehend the similarities and differences among objects by their characteristic features at the final output layer.

### Extracting the Data
The CIFAR-10 data comes to us in the form of 32 x 32 pixel PNG image files which we'll import into our project using the `imagecreatefrompng()` provided by the [GD](https://www.php.net/manual/en/book.image.php) extension. We use a regular expression to extract the label from the filename of the images in the `train` folder.

```php
$samples = $labels = [];

foreach (glob('train/*.png') as $file) {
    $samples[] = [imagecreatefrompng($file)];
    $labels[] = preg_replace('/[0-9]+_(.*).png/', '$1', basename($file));
}
```

Then, load the samples and labels into a [Labeled](https://docs.rubixml.com/en/latest/datasets/labeled.html) dataset object.

```php
use Rubix\ML\Datasets\Labeled;

$dataset = new Labeled($samples, $labels);
```

### Dataset Preparation
We'll need to wrap the base learner in a transformer Pipeline to convert the images from the dataset into standardized raw pixel data on the fly. An [Image Resizer](https://docs.rubixml.com/en/latest/transformers/image-resizer.html) ensures that all input vectors are of the same dimensionality, just in case. The [Image Vectorizer](https://docs.rubixml.com/en/latest/transformers/image-vectorizer.html) handles extracting the raw color channel data such as the red, green, and blue (RGB) intensities. Finally, the [Z Scale Standardizer](https://docs.rubixml.com/en/latest/transformers/z-scale-standardizer.html) scales and centers the vectorized color data to a mean of 0 and a standard deviation of 1. This last step will help the network learn quicker.

### Instantiating the Learner
The [Multi Layer Perceptron](https://docs.rubixml.com/en/latest/classifiers/multi-layer-perceptron.html) is a type of deep learning model we'll train to recognize images from the CIFAR-10 dataset. It uses Gradient Descent with Backpropagation over multiple layers of *neurons* to train the network by gradually updating the signal that each neuron produces in response to a sample. In between [Dense](https://docs.rubixml.com/en/latest/neural-network/hidden-layers/dense.html) neuronal layers we have an [Activation](https://docs.rubixml.com/en/latest/neural-network/hidden-layers/activation.html) layer that performs a non-linear transformation of the neuron's output using a user-defined activation function. For the purpose of this tutorial we'll use the [ELU](https://docs.rubixml.com/en/latest/neural-network/activation-functions/elu.html) activation function, which is a good default but feel free to experiment with different activation functions on your own. Lastly, we add [Batch Norm](https://docs.rubixml.com/en/latest/neural-network/hidden-layers/batch-norm.html) layers after every two Dense layers to help the network train faster by normalizing the activations as well as improve its generalization ability through the introduction of mild stochastic noise.

```php
use Rubix\ML\Pipeline;
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Transformers\ImageResizer;
use Rubix\ML\Transformers\ImageVectorizer;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\Classifiers\MultiLayerPerceptron;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\Layers\BatchNorm;
use Rubix\ML\NeuralNet\ActivationFunctions\ELU;
use Rubix\ML\NeuralNet\Optimizers\Adam;

$estimator = new PersistentModel(
    new Pipeline([
        new ImageResizer(32, 32),
        new ImageVectorizer(),
        new ZScaleStandardizer(),
    ], new MultiLayerPerceptron([
        new Dense(200),
        new Activation(new ELU()),
        new Dense(200),
        new BatchNorm(),
        new Activation(new ELU()),
        new Dense(200),
        new Activation(new ELU()),
        new Dense(100),
        new BatchNorm(),
        new Activation(new ELU()),
        new Dense(50),
        new Activation(new ELU()),
    ], 100, new Adam(0.001))),
    new Filesystem('cifar-10.model', true)
);
```

Wrapping the entire Pipeline in a [Persistent Model](https://docs.rubixml.com/en/latest/persistent-model.html) allows us to save the model so we can use it in another process to make predictions on unknown images.

### Training
Now, all we have to do is pass the dataset to the `train()` method to begin training the network.

```php
$estimator->train($dataset);
```

### Saving
Before exiting the script, save the model so we can run cross validation on it in another process.

```php
$estimator->save();
```

### Cross Validation
Cross validation is the process of testing a model using samples that the learner has never seen before. The goal is to be able to detect problems such as selection bias or overfitting. In addition to the training set, the CIFAR-10 dataset includes 10,000 testing samples that we'll use to score the model's generalization ability. We start by importing the testing samples and labels located in the `test` folder.

```php
$samples = $labels = [];

foreach (glob('test/*.png') as $file) {
    $samples[] = [imagecreatefrompng($file)];
    $labels[] = preg_replace('/[0-9]+_(.*).png/', '$1', basename($file));
}
```

Next, instantiate a [Labeled](https://docs.rubixml.com/en/latest/datasets/labeled.html) dataset object with the testing samples and labels.

```php
use Rubix\ML\Datasets\Labeled;

$dataset = new Labeled($samples, $labels);
```

### Load Model from Storage
Since we saved our model after training in the last section, we can load it whenever we need to use it in another process such as to make predictions or, in this case, to test how well our training session went by generating a cross validation report. The static `load()` method on the Persistent Model class takes a pre-configured [Persister](https://docs.rubixml.com/en/latest/persisters/api.html) object pointing to the location of the model in storage as its only argument and returns the estimator object in the last known saved state.

```php
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;

$estimator = PersistentModel::load(new Filesystem('cifar-10.model'));
```

### Make Predictions
We'll need the predictions produced by the neural network estimator from the testing set to pass to a report generator along with the actual class labels given in the testing set. To return an array of predictions, pass the testing set to the `predict()` method on the estimator.

```php
$predictions = $estimator->predict($dataset);
```

### Generate Reports
The [Multiclass Breakdown](https://docs.rubixml.com/en/latest/cross-validation/reports/multiclass-breakdown.html) and [Confusion Matrix](https://docs.rubixml.com/en/latest/cross-validation/reports/confusion-matrix.html) are cross validation reports that show performance of the model on a class by class basis. We'll wrap them both in an Aggregate Report and pass our predictions along with the ground truth labels from the testing set to the `generate()` method to return an array containing both reports.

```php
use Rubix\ML\CrossValidation\Reports\AggregateReport;
use Rubix\ML\CrossValidation\Reports\ConfusionMatrix;
use Rubix\ML\CrossValidation\Reports\MulticlassBreakdown;

$report = new AggregateReport([
    new MulticlassBreakdown(),
    new ConfusionMatrix(),
]);

$results = $report->generate($predictions, $dataset->labels());
```

Now, take a look at the reports and see how the model performed.

### Wrap Up
- The CIFAR-10 dataset is a famous dataset used to benchmark the performance of computer vision tasks.
- Computer vision allows a program to recognize and distinguish objects in images.
- The problem of computer vision can be solved with Deep Learning by allowing a learner to build feature representations from raw data.
- The [Multi Layer Perceptron](https://docs.rubixml.com/en/latest/classifiers/multi-layer-perceptron.html) classifier in Rubix ML is a type of deep learning model that uses hidden layers as intermediate computational units.

### Next Steps
Congratulations on finishing the CIFAR-10 tutorial using Rubix ML! Now is your chance to experiment with other network architectures, activation functions, and learning rates on your own. Try adding additional hidden layers to *deepen* the network and add flexibility to the model. Is a fully-connected network the best architecture for this problem? Are there better network architectures that maintain spatial information?

## Original Dataset
Creator: Alex Krizhevsky
Email: akrizhevsky '@' gmail.com 

### References
>- [1] A. Krizhevsky. (2009). Learning Multiple Layers of Features from Tiny Images.