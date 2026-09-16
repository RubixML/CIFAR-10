# Rubix ML - CIFAR-10 Image Recognizer

CIFAR-10 (short for *Canadian Institute For Advanced Research*) is a [famous dataset](https://en.wikipedia.org/wiki/CIFAR-10) consisting of 60,000 32 x 32 color images in 10 classes (dog, cat, car, ship, etc.) with 6,000 images per class. In this tutorial, we'll use the CIFAR-10 dataset to train a feed forward neural network to recognize the primary object in images.

## Installation

Clone the project locally using [Composer](https://getcomposer.org/):

```sh
$ composer create-project rubix/cifar-10
```

> **Note:** Installation may take longer than usual due to the large dataset.

## Requirements

- [PHP](https://php.net) 8.3 or above
- [GD extension](https://www.php.net/manual/en/book.image.php)

### Recommended

- [Tensor extension](https://github.com/RubixML/Tensor) for faster training and inference

## Tutorial

### Introduction

Computer vision is one of the most fascinating use cases for deep learning because it allows a computer to see the world that we live in. Deep learning is a subset of machine learning concerned with breaking down raw data into higher order feature representations through layered computations. Neural networks are a type of deep learning model inspired by the human nervous system that uses structured computational units called *hidden* layers. In the case of image recognition, the hidden layers are able to break down an image into its component parts such that the network can readily comprehend the similarities and differences among objects by their characteristic features at the final output layer. Let's get started!

### Extracting the Data

The CIFAR-10 dataset comes to us in the form of 60,000 32 x 32 pixel PNG image files which we'll import as PHP resources into our project using the `imagecreatefrompng()` provided by the [GD](https://www.php.net/manual/en/book.image.php) extension. If you do not have the extension installed, you'll need to do so before running the project script. We also use `preg_replace()` to extract the label from the filename of the images in the `train` folder.

With 50,000 training images, loading all of them into memory at once can be costly. Instead, we grab the list of image files with `glob()` and break them up into digestible chunks of `10000` images each.

```php
use function Rubix\ML\enumerate;

$files = glob('train/*.png');

$chunkSize = 10000;
```

As we'll see in a moment, each chunk is processed in turn by loading the images as PHP resources and extracting their labels into a [Labeled](https://rubixml.github.io/ML/latest/datasets/labeled.html) dataset.

```php
$samples = $labels = [];

foreach ($files as $file) {
    $samples[] = [imagecreatefrompng($file)];
    $labels[] = preg_replace('/[0-9]+_(.*).png/', '$1', basename($file));
}

$subset = new Labeled($samples, $labels);
```

### Dataset Preparation

The images we imported in the previous step will eventually need to be converted into samples of continuous features. An [Image Resizer](https://rubixml.github.io/ML/latest/transformers/image-resizer.html) ensures that all images have the same dimensionality, just in case. The [Image Vectorizer](https://rubixml.github.io/ML/latest/transformers/image-vectorizer.html) handles extracting the red, green, and blue (RGB) intensities (0 - 255) from the images. Since the vectorized color channels are integers and the network expects floating point data, we use a Float Type Converter to cast them to floats. Finally, the [Z Scale Standardizer](https://rubixml.github.io/ML/latest/transformers/z-scale-standardizer.html) scales and centers the vectorized color channel data to a mean of 0 and a standard deviation of 1. This last step helps the network converge quicker. We'll wrap the 4 transformers in a [Pipeline](https://rubixml.github.io/ML/latest/pipeline.html) so we can use them again in another process after we save the model.

### Instantiating the Learner

The [Multilayer Perceptron](https://rubixml.github.io/ML/latest/classifiers/multilayer-perceptron.html) classifier is a type of neural network model we'll train to recognize images in the CIFAR-10 dataset. Under the hood it uses Gradient Descent with Backpropagation to learn the weights of the network by gradually updating the signal that each neuron produces in response to an input. One of the key aspects of neural networks are the use of hidden layers that perform intermediate computations. In between [Dense](https://rubixml.github.io/ML/latest/neural-network/hidden-layers/dense.html) neuronal layers we use an [Activation](https://rubixml.github.io/ML/latest/neural-network/hidden-layers/activation.html) layer to perform a non-linear transformation of the neuron's output using a user-defined activation function. The non-linearities introduced by the activation layer are crucial for learning complex patterns within the data. For the purpose of this tutorial we'll use the [GELU](https://rubixml.github.io/ML/latest/neural-network/activation-functions/gelu.html) activation function, which is a good default but feel free to experiment with different activation functions on your own. We also add a [Batch Norm](https://rubixml.github.io/ML/latest/neural-network/hidden-layers/batch-norm.html) layer after the second and fourth sets of Dense/Activation layers to help the network train faster by re-normalizing the activations partway through the network.

Wrapping the learner and transformer pipeline in a [Persistent Model](https://rubixml.github.io/ML/latest/persistent-model.html) meta-estimator allows us to save the model so we can use it in another process to make predictions.

```php
use Rubix\ML\PersistentModel;
use Rubix\ML\Pipeline;
use Rubix\ML\Transformers\ImageResizer;
use Rubix\ML\Transformers\ImageVectorizer;
use Rubix\ML\Transformers\FloatTypeConverter;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\Classifiers\MultilayerPerceptron;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\Layers\BatchNorm;
use Rubix\ML\NeuralNet\ActivationFunctions\GELU;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Constant;
use Rubix\ML\Persisters\Filesystem;

$estimator = new PersistentModel(
    new Pipeline([
        new ImageResizer(32, 32),
        new ImageVectorizer(),
        new FloatTypeConverter(),
        new ZScaleStandardizer(),
    ], new MultilayerPerceptron(
        hiddenLayers: [
            new Dense(256),
            new Activation(new GELU()),
            new Dense(256, bias: false),
            new BatchNorm(),
            new Activation(new GELU()),
            new Dense(256),
            new Activation(new GELU()),
            new Dense(128, bias: false),
            new BatchNorm(),
            new Activation(new GELU()),
            new Dense(128),
            new Activation(new GELU()),
            new Dense(10),
        ],
        batchSize: 32,
        gradientAccumulationSteps: 4,
        optimizer: new Adam(new Constant(0.0001)),
        maxGradientNorm: 1.0,
        evalInterval: 1,
        window: 10,
    )),
    new Filesystem('cifar10.rbx', true)
);
```

There are a few more hyper-parameters of the MLP that we'll need to set in addition to the hidden layers. The *batch size* parameter is the number of samples that will be sent through the neural network at a time. We'll set this to 32. Because a small batch size can make the weight updates noisy, we accumulate the gradients over `4` batches before updating the weights using the *gradient accumulation* parameter, giving an effective batch size of 128. Next, the Gradient Descent optimizer and *learning rate*, which control the update step of the learning algorithm, will be set to [Adam](https://rubixml.github.io/ML/latest/neural-network/optimizers/adam.html) with a `Constant` learning rate of `0.0001`. To keep the gradients from becoming too large, we also cap the *max gradient norm* to `1.0`. Finally, the *eval interval* parameter controls how often the learner scores the model on a hold-out portion of the training set during training, and the *window* parameter specifies how many evaluations without an improvement in the validation score to wait before stopping early. Feel free to experiment with these settings on your own.

### Training

Now we're ready to begin training the network. Instead of passing the entire training set to the `train()` method at once, we feed the learner one chunk of data at a time using `partial()`. Unlike `train()`, which initializes the learner from scratch, the `partial()` method continues training from the previous state, which makes it possible to train on datasets that are too large to fit into memory.

```php
$chunks = array_chunk($files, $chunkSize);

foreach (enumerate($chunks, start: 1) as $i => $files) {
    $logger->info("Processing chunk #{$i}");

    $samples = $labels = [];

    foreach ($files as $file) {
        $samples[] = [imagecreatefrompng($file)];
        $labels[] = preg_replace('/[0-9]+_(.*).png/', '$1', basename($file));
    }

    $subset = new Labeled($samples, $labels);

    $estimator->partial($subset);
}
```

The `enumerate()` helper we imported earlier adds a 1-indexed counter to the chunk iterator so we can keep track of where we are in the training process.

### Validation Score and Loss

We can visualize the training progress at each stage by dumping the values of the loss function and validation metric during training. The `progress()` method will output an iterator containing the loss values of the default [Cross Entropy](https://rubixml.github.io/ML/latest/neural-network/cost-functions/cross-entropy.html) cost function and validation scores from the default [F Beta](https://rubixml.github.io/ML/latest/cross-validation/metrics/f-beta.html) metric at each evaluated epoch.

> **Note:** You can change the cost function and validation metric by setting them as hyper-parameters of the learner.

After training on each chunk, we export the progress so far to a CSV file using the [CSV](https://rubixml.github.io/ML/latest/extractors/csv.html) extractor. With a chunk size of 8,192, training the 50,000 images in the training set produces 7 `progress_*.csv` files - one for every chunk in the dataset.

```php
use Rubix\ML\Extractors\CSV;

$extractor = new CSV("progress_{$i}.csv", true);

$extractor->export($estimator->progress());
```

Then, we can plot the values using our favorite plotting software such as [Tableu](https://public.tableau.com/en-us/s/) or [Excel](https://products.office.com/en-us/excel-a). If all goes well, the value of the loss should go down as the value of the validation score goes up. Due to snapshotting, the epoch at which the validation score is highest and the loss is lowest is the point at which the values of the network parameters are taken.

![Cross Entropy Loss](https://raw.githubusercontent.com/RubixML/CIFAR-10/master/docs/images/training-losses.png)

![F1 Score](https://raw.githubusercontent.com/RubixML/CIFAR-10/master/docs/images/validation-scores.png)

### Saving

Before exiting the script, we give ourselves the option to save the model so we can run cross validation on it in another process. The script will prompt you to confirm.

```php
if (strtolower(trim(readline('Save this model? (y|[n]): '))) === 'y') {
    $estimator->save();
}
```

Now we're ready to execute the training script from the command line.
```sh
$ php train.php
```

### Cross Validation

Cross validation is the process of testing a model using samples that the learner has never seen before. The goal is to be able to detect problems such as selection bias or overfitting. In addition to the training set, the CIFAR-10 dataset includes 10,000 testing samples that we'll use to score the model's generalization ability. We start by importing the testing samples and labels located in the `test` folder using the technique from earlier.

```php
$samples = $labels = [];

foreach (glob('test/*.png') as $file) {
    $samples[] = [imagecreatefrompng($file)];
    $labels[] = preg_replace('/[0-9]+_(.*).png/', '$1', basename($file));
}
```

Instantiate a [Labeled](https://rubixml.github.io/ML/latest/datasets/labeled.html) dataset object with the testing samples and labels.

```php
use Rubix\ML\Datasets\Labeled;

$dataset = new Labeled($samples, $labels);
```

### Load Model from Storage

Since we saved our model after training in the last section, we can load it whenever we need to use it in another process. The static `load()` method on the Persistent Model class takes a pre-configured [Persister](https://rubixml.github.io/ML/latest/persisters/api.html) object pointing to the location of the model in storage as its only argument and returns the wrapped estimator in the last known saved state. We also call `cleanup()` to release the temporary state that the training process leaves behind before making predictions.

```php
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;

$estimator = PersistentModel::load(new Filesystem('cifar10.rbx'));

$estimator->cleanup();
```

### Make Predictions

We'll need the predictions produced by the MLP on the testing set to pass to a report generator along with the ground-truth class labels. To return an array of predictions, pass the testing set to the `predict()` method on the estimator.

```php
$predictions = $estimator->predict($dataset);
```

### Generate Reports

The [Multiclass Breakdown](https://rubixml.github.io/ML/latest/cross-validation/reports/multiclass-breakdown.html) and [Confusion Matrix](https://rubixml.github.io/ML/latest/cross-validation/reports/confusion-matrix.html) are cross validation reports that show performance of the model on a class by class basis. We'll wrap them both in an Aggregate Report and pass our predictions along with the ground-truth labels from the testing set to the `generate()` method to generate both reports at once.

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

To run the validation script, enter the following command at the command prompt.

```php
$ php validate.php
```

Take a look at the results to see how the model performed on inference. Below is an excerpt of a multiclass breakdown report showing the overall performance. As you can see, the model does a fair job at recognizing the objects in the images, however there is room for improvement.

```json
"overall": {
    "accuracy": 0.8538682791748626,
    "precision": 0.5467567594653738,
    "recall": 0.5372000000000001,
    "specificity": 0.9134813190242806,
    "negative_predictive_value": 0.9138360056175517,
    "false_discovery_rate": 0.4532432405346262,
    "miss_rate": 0.46280000000000004,
    "fall_out": 0.08651868097571924,
    "false_omission_rate": 0.08616399438244826,
    "f1_score": 0.5322032931908443,
    "mcc": 0.4528826530026047,
    "informedness": 0.4506813190242807,
    "markedness": 0.46059276508292557,
    "true_positives": 5372,
    "true_negatives": 48348,
    "false_positives": 4628,
    "false_negatives": 4628,
    "cardinality": 10000,
    "density": 1
},
```

This excerpt from the confusion matrix shows that the estimator does a good job identifying automobiles but sometimes confuses them for trucks, which makes sense since they are similar in many ways.

```json
    "automobile": {
        "cat": 14,
        "dog": 6,
        "airplane": 14,
        "ship": 37,
        "deer": 4,
        "automobile": 603,
        "frog": 9,
        "horse": 10,
        "bird": 12,
        "truck": 130
    },
```

### Next Steps

Congratulations on finishing the CIFAR-10 tutorial using Rubix ML! Now is your chance to experiment with other network architectures, activation functions, and learning rates on your own. Try adding additional hidden layers to *deepen* the network and add flexibility to the model. Is a fully-connected network the best architecture for this problem? Are there other network architectures that can use the spatial information of the images?

## Original Dataset

Creator: Alex Krizhevsky
Email: akrizhevsky '@' gmail.com 

### References

>- [1] A. Krizhevsky. (2009). Learning Multiple Layers of Features from Tiny Images.

## License

The code is licensed [MIT](LICENSE) and the tutorial is licensed [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
