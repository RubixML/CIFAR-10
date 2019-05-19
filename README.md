# CIFAR-10 Image Classifier

Computer vision is one of the most fascinating use cases for deep learning because it allows a computer to see the world that we live in. CIFAR-10 (short for *Canadian Institute For Advanced Research*) is a [famous dataset](https://en.wikipedia.org/wiki/CIFAR-10) consisting of 60,000 32x32 color images in 10 classes (dog, cat, car, etc.) with 6,000 images per class. In this tutorial, we'll use the CIFAR-10 dataset to train a feed forward multi layer neural network to recognize objects within images using Rubix ML in PHP.

- **Difficulty**: Hard
- **Training time**: < 24 Hours
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

Wrapping the entire Pipeline in a [Persistent Model](https://github.com/RubixML/RubixML#persistent-model) allows us to save the model so we can use it in another process to make predictions on unknown images.

```php
use Rubix\ML\Pipeline;
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\PReLU;
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
        new PReLU(),
    ], 100, new Adam(0.001))),
    new Filesystem(MODEL_FILE, true)
);
```

Now all we have to do is pass the dataset to the estimator's `train()` method to begin training the network.

```php
$estimator->train($dataset);
```

Then save the model so we can run cross validation on it in the next section.

```php
$estimator->save();
```

### Validation
Cross validation is the process of testing a model using samples that the learner has never seen before. In addition to the training set, the CIFAR-10 dataset includes a 10,000 testing samples that we can use to score the model's generalization ability. We start by importing the testing samples located in the *test* folder into a Labeled dataset object.

```php
use Rubix\ML\Datasets\Labeled;

$samples = $labels = [];

foreach (glob(__DIR__ . '/test/*.png') as $file) {
    $samples[] = [imagecreatefrompng($file)];
    $labels[] = preg_replace('/[0-9]+_(.*).png/', '$1', basename($file));
}

$dataset = new Labeled($samples, $labels);
```

Since we saved our model after training in the last section, we can load it whenever we need to use it in another process such as to make predictions or, in this case, to test how well our training session went by generating a cross validation report. The `load()` factory method on the Persistent Model class takes a pre-configured Persister pointing to the location of the model in storage as its only argument and returns the estimator object in the last known saved state.

```php
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;

$estimator = PersistentModel::load(new Filesystem('cifar-10.model'));
```

We'll need the predictions produced by the neural network estimator from the testing set to pass to a report generator along with the actual class labels given in the testing set. To return an array of predictions, pass the testing set to the `predict()` method on the estimator.

```php
$predictions = $estimator->predict($dataset);
```

The Multiclass Breakdown and Confusion Matrix are cross validation reports that show performance of the model on a class by class basis. We'll wrap them both in an Aggregate Report and pass our predictions along with the ground truth labels from the testing set to the `generate()` method to return an array with both reports.

```php
use Rubix\ML\CrossValidation\Reports\AggregateReport;
use Rubix\ML\CrossValidation\Reports\ConfusionMatrix;
use Rubix\ML\CrossValidation\Reports\MulticlassBreakdown;

$report = new AggregateReport([
    new MulticlassBreakdown(),
    new ConfusionMatrix(),
]);

$results = $report->generate($predictions, $dataset->labels());

var_dump($results);
```

Take a look at the reports and see how the model performed. In the next section we'll take some unlabeled images and see if the network can guess them correctly.

### Prediction

Coming soon ...

## Original Dataset
Creator: Alex Krizhevsky
Email: akrizhevsky '@' gmail.com 

### References
>- [1] A. Krizhevsky. (2009). Learning Multiple Layers of Features from Tiny Images.