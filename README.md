# Rubix ML - CIFAR-10 Image Recognizer

CIFAR-10 (short for *Canadian Institute For Advanced Research*) is a [famous dataset](https://en.wikipedia.org/wiki/CIFAR-10) consisting of 60,000 32 x 32 color images in 10 classes (dog, cat, car, ship, etc.) with 6,000 images per class. In this tutorial, we'll use the CIFAR-10 dataset to train a feed forward neural network to recognize the primary object in images.

## Installation

Clone the project locally using [Composer](https://getcomposer.org/):

```sh
composer create-project rubix/cifar-10
```

> **Note:** Installation may take longer than usual due to the large dataset.

### Optional for best performance

Make sure you have all the necessary build tools installed such as a C compiler and make tools. For example, on an Ubuntu linux system you can enter the following on the command line to install the necessary dependencies.

```sh
sudo apt-get install make gcc gfortran php-dev libopenblas-dev liblapacke-dev re2c build-essential
```

Compile and install the [Tensor 4.1+](https://github.com/RubixML/Tensor-Ext) extension using PIE:

```sh
pie install rubix/tensor_ext:^4.1
```

## Requirements

- [PHP](https://php.net) 8.3 or above
- [GD extension](https://www.php.net/manual/en/book.image.php)

### Recommended

- [Tensor 4.1+ extension](https://github.com/RubixML/Tensor-Ext) for faster training and inference

## Tutorial

### Introduction

Computer vision is one of the most fascinating use cases for deep learning because it allows a computer to see the world that we live in. Deep learning is a subset of machine learning concerned with breaking down raw data into higher order feature representations through layered computations. Neural networks are a type of deep learning model inspired by the human nervous system that uses structured computational units called *hidden* layers. In the case of image recognition, the hidden layers are able to break down an image into its component parts such that the network can readily comprehend the similarities and differences among objects by their characteristic features at the final output layer. Let's get started!

### Extracting the Data

The CIFAR-10 dataset comes to us in the form of 60,000 32 x 32 pixel PNG image files which we'll import as PHP resources into our project using the `imagecreatefrompng()` provided by the [GD](https://www.php.net/manual/en/book.image.php) extension. If you do not have the extension installed, you'll need to do so before running the project script. We also use `preg_replace()` to extract the label from the filename of the images in the `train` folder.

With 50,000 training images, loading all of them into memory at once can be costly. Instead, we grab the list of image files with `glob()` and break them up into digestible chunks of `10000` images each.

```php
use function Rubix\ML\enumerate;

$files = glob('train/*.png');

define('CHUNK_SIZE', 10000);
```

As we'll see in a moment, each chunk is processed in turn by loading the images as PHP resources and extracting their labels into a [Labeled](https://rubixml.github.io/ML/latest/datasets/labeled.html) dataset.

```php
$samples = $labels = [];

foreach ($files as $file) {
    $samples[] = [imagecreatefrompng($file)];
    $labels[] = preg_replace('/[0-9]+_(.*).png/', '$1', basename($file));
}

$training = new Labeled($samples, $labels);
```

### Dataset Preparation

The images we imported in the previous step will eventually need to be converted into samples of continuous features. An [Image Resizer](https://rubixml.github.io/ML/latest/transformers/image-resizer.html) ensures that all images have the same dimensionality, just in case. The [Image Vectorizer](https://rubixml.github.io/ML/latest/transformers/image-vectorizer.html) handles extracting the red, green, and blue (RGB) intensities (0 - 255) from the images. Since the vectorized color channels are integers and the network expects floating point data, we use a Float Type Converter to cast them to floats. Finally, the [Z Scale Standardizer](https://rubixml.github.io/ML/latest/transformers/z-scale-standardizer.html) scales and centers the vectorized color channel data to a mean of 0 and a standard deviation of 1. This last step helps the network converge quicker. We'll wrap the 4 transformers in a [Pipeline](https://rubixml.github.io/ML/latest/pipeline.html) so that they can be fitted, updated, and applied as a unit.

Because the Z Scale Standardizer is a **Stateful** transformer that learns the mean and standard deviation of the color channels, its fitted state is just as important as the weights of the network itself. Training happens one chunk at a time, so those statistics are estimated incrementally as the chunks arrive - which means we have to be able to apply that exact same state to new images in another process later on. To that end, we decorate the Pipeline with a [Persistent Transformer](https://rubixml.github.io/ML/latest/transformers/persistent-transformer.html) meta-estimator, which adds `save()` and `load()` methods and writes to `transformer.rbx`.

```php
use Rubix\ML\Transformers\PersistentTransformer;
use Rubix\ML\Transformers\Pipeline;
use Rubix\ML\Transformers\ImageResizer;
use Rubix\ML\Transformers\ImageVectorizer;
use Rubix\ML\Transformers\FloatTypeConverter;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\Persisters\Filesystem;

$transformer = new PersistentTransformer(
    base: new Pipeline([
        new ImageResizer(32, 32),
        new ImageVectorizer(),
        new FloatTypeConverter(),
        new ZScaleStandardizer(),
    ]),
    persister: new Filesystem('transformer.rbx', true)
);
```

Note that the decorator behaves like the Pipeline it wraps everywhere else, so it can be handed straight to `Dataset::apply()`, which transforms the samples in place.

### Instantiating the Learner

The [Multilayer Perceptron](https://rubixml.github.io/ML/latest/classifiers/multilayer-perceptron.html) classifier is a type of neural network model we'll train to recognize images in the CIFAR-10 dataset. Under the hood it uses Gradient Descent with Backpropagation to learn the weights of the network by gradually updating the signal that each neuron produces in response to an input. One of the key aspects of neural networks are the use of hidden layers that perform intermediate computations. In between [Dense](https://rubixml.github.io/ML/latest/neural-network/hidden-layers/dense.html) neuronal layers we use an [Activation](https://rubixml.github.io/ML/latest/neural-network/hidden-layers/activation.html) layer to perform a non-linear transformation of the neuron's output using a user-defined activation function. The non-linearities introduced by the activation layer are crucial for learning complex patterns within the data. For the purpose of this tutorial we'll use the [SiLU](https://rubixml.github.io/ML/latest/neural-network/activation-functions/silu.html) activation function, which is a good default but feel free to experiment with different activation functions on your own. We also add a [Batch Norm](https://rubixml.github.io/ML/latest/neural-network/hidden-layers/batch-norm.html) layer after the second and fourth sets of Dense/Activation layers to help the network train faster by re-normalizing the activations partway through the network.

Now that the transformer is looking after the data, the learner only has to concern itself with the network. Wrapping it in a [Persistent Model](https://rubixml.github.io/ML/latest/persistent-model.html) meta-estimator allows us to save the trained weights to `model.rbx` so we can use them in another process to make predictions.

```php
use Rubix\ML\PersistentModel;
use Rubix\ML\Classifiers\MultilayerPerceptron;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\Layers\BatchNorm;
use Rubix\ML\NeuralNet\ActivationFunctions\SiLU;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Constant;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Loggers\Screen;

$estimator = new PersistentModel(
    base: new MultilayerPerceptron(
        hiddenLayers: [
            new Dense(512),
            new Activation(new SiLU()),
            new Dense(512, bias: false),
            new BatchNorm(),
            new Activation(new SiLU()),
            new Dense(512),
            new Activation(new SiLU()),
            new Dense(256, bias: false),
            new BatchNorm(),
            new Activation(new SiLU()),
            new Dense(128),
            new Activation(new SiLU()),
            new Dense(10),
        ],
        batchSize: 32,
        gradientAccumulationSteps: 4,
        optimizer: new Adam(new Constant(0.0001)),
        maxGradientNorm: 1.0,
        evalInterval: 1,
        window: 5,
    ),
    persister: new Filesystem('model.rbx', true)
);

$logger = new Screen();

$estimator->setLogger($logger);
```

The [Screen logger](https://rubixml.github.io/ML/latest/loggers/screen.html) is what allows us to follow along in the command line as the model trains, since we won't be looking at all 50,000 images at once.

There are a few more hyper-parameters of the MLP that we'll need to set in addition to the hidden layers. The *batch size* parameter is the number of samples that will be sent through the neural network at a time. We'll set this to 32. Because a small batch size can make the weight updates noisy, we accumulate the gradients over `4` batches before updating the weights using the *gradient accumulation* parameter, giving an effective batch size of 128. Next, the Gradient Descent optimizer and *learning rate*, which control the update step of the learning algorithm, will be set to [Adam](https://rubixml.github.io/ML/latest/neural-network/optimizers/adam.html) with a `Constant` learning rate of `0.0001`. To keep the gradients from becoming too large, we also cap the *max gradient norm* to `1.0`. Finally, the *eval interval* parameter controls how often the learner scores the model on a hold-out portion of the training set during training, and the *window* parameter specifies how many evaluations - in this case 5 - without an improvement in the validation score to wait before stopping early. Feel free to experiment with these settings on your own.

### Setting a Validation Dataset

The *eval interval* and *window* parameters we set earlier rely on a hold-out set to score the model during training. We point those evaluations at the test directory, which we haven't used for training yet, so that early stopping reflects how the network generalizes beyond the data it is learning. We import the test samples and labels the same way we will later in the Cross Validation section.

```php
use Rubix\ML\Datasets\Labeled;

$samples = $labels = [];

foreach (glob('test/*.png') as $file) {
    $samples[] = [imagecreatefrompng($file)];
    $labels[] = preg_replace('/[0-9]+_(.*).png/', '$1', basename($file));
}

$testing = new Labeled($samples, $labels);
```

A Stateful transformer has to be fitted before it is able to transform anything, and since we never loaded the training set all at once, the test set is the first dataset we have on hand. Fitting on these 10,000 images gives the Z Scale Standardizer a solid starting point for its color channel statistics, which we then refine with `update()` as the training chunks come in. Once fitted, we apply the transformer to the dataset in place so that the hold-out set is scored on the same features the learner is trained on, and register the result with the learner using `setValidationDataset()`.

```php
$transformer->fit($testing);

$testing->apply($transformer);

$estimator->setValidationDataset($testing);
```

From this point forward, the learner measures its validation score against this dataset at every *eval interval* and uses the *window* setting to decide when to stop training early.

### Training

Now we're ready to begin training the network. Instead of passing the entire training set to the `train()` method at once, we feed the learner one chunk of data at a time using `partial()`. Unlike `train()`, which initializes the learner from scratch, the `partial()` method continues training from the previous state, which makes it possible to train on datasets that are too large to fit into memory.

Each chunk also contributes to the transformer's fitting. Calling `update()` on the transformer folds the chunk's color channel statistics into the running estimate before we transform with it, so the standardizer keeps getting more accurate as the network sees more data.

```php
$chunks = array_chunk($files, CHUNK_SIZE);

foreach (enumerate($chunks, start: 1) as $i => $files) {
    $logger->info("Training on chunk #{$i}");

    $samples = $labels = [];

    foreach ($files as $file) {
        $samples[] = [imagecreatefrompng($file)];
        $labels[] = preg_replace('/[0-9]+_(.*).png/', '$1', basename($file));
    }

    $training = new Labeled($samples, $labels);

    $transformer->update($training);

    $training->apply($transformer);

    $estimator->partial($training);
}
```

The `enumerate()` helper we imported earlier adds a 1-indexed counter to the chunk iterator so we can keep track of where we are in the training process.

### Validation Score and Loss

We can visualize the training progress at each stage by dumping the values of the loss function and validation metric during training. The `progress()` method will output an iterator containing the loss values of the default [Cross Entropy](https://rubixml.github.io/ML/latest/neural-network/cost-functions/cross-entropy.html) cost function and validation scores from the default [F Beta](https://rubixml.github.io/ML/latest/cross-validation/metrics/f-beta.html) metric at each evaluated epoch.

> **Note:** You can change the cost function and validation metric by setting them as hyper-parameters of the learner.

After training on each chunk, we export the progress so far to a CSV file using the [CSV](https://rubixml.github.io/ML/latest/extractors/csv.html) extractor. With a chunk size of 10,000, training the 50,000 images in the training set produces 5 `progress_*.csv` files - one for every chunk in the dataset.

```php
use Rubix\ML\Extractors\CSV;

$extractor = new CSV("progress_{$i}.csv", true);

$extractor->export($estimator->progress(), overwrite: true);

$logger->info("Progress saved to progress_{$i}.csv");
```

The `true` second argument tells the extractor to include a header row, and since each file is named after the chunk it came from, `overwrite: true` is just there so that re-running the script doesn't error out on a file that's already there.

Then, we can plot the values using our favorite plotting software such as [Tableu](https://public.tableau.com/en-us/s/) or [Excel](https://products.office.com/en-us/excel-a). If all goes well, the value of the loss should go down as the value of the validation score goes up. Due to snapshotting, the epoch at which the validation score is highest and the loss is lowest is the point at which the values of the network parameters are taken.

![Cross Entropy Loss](https://raw.githubusercontent.com/RubixML/CIFAR-10/master/docs/images/training-losses.png)

![F1 Score](https://raw.githubusercontent.com/RubixML/CIFAR-10/master/docs/images/validation-scores.png)

### Saving

Before exiting the script, we give ourselves the option to save the model so we can run cross validation on it in another process. Both artifacts have to be written out - the model holds the network weights, while the transformer holds the fitted color channel statistics - otherwise the saved network won't know how to scale the images it is handed later. The script will prompt you to confirm.

```php
if (strtolower(trim(readline('Save this model? (y|[n]): '))) === 'y') {
    $transformer->save();
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

Since we saved our model and transformer after training in the last section, we can load them whenever we need to use them in another process. The static `load()` method on the Persistent Model class takes a pre-configured [Persister](https://rubixml.github.io/ML/latest/persisters/api.html) object pointing to the location of the model in storage as its only argument and returns the wrapped estimator in the last known saved state. The Persistent Transformer has a matching static `load()` method that restores the fitted Pipeline. We also call `cleanup()` to release the temporary state that the training process leaves behind before making predictions.

```php
use Rubix\ML\Transformers\PersistentTransformer;
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;

$transformer = PersistentTransformer::load(new Filesystem('transformer.rbx'));

$estimator = PersistentModel::load(new Filesystem('model.rbx'));

$estimator->cleanup();
```

### Make Predictions

We'll need the predictions produced by the MLP on the testing set to pass to a report generator along with the ground-truth class labels. Before we can predict, the raw images have to go through the transformer we saved during training - the network learned from standardized color channels, so handing it unstandardized ones would produce meaningless scores.

```php
$dataset->apply($transformer);

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

```sh
$ php validate.php
```

The Aggregate Report renders as a string for us to read in the terminal, and it can also be serialized for later analysis. The `toJSON()` method returns a pretty-printed JSON encoding of the full breakdown and confusion matrix, which `saveTo()` writes through a [Persister](https://rubixml.github.io/ML/latest/persisters/api.html) to a `report.json` file in the project root.

```php
echo $results;

$results->toJSON()->saveTo(new Filesystem('report.json'));
```

Take a look at the results to see how the model performed on inference. Below is an excerpt of a multiclass breakdown report showing the overall performance. As you can see, the model does a fair job at recognizing the objects in the images, however there is room for improvement.

```json
"overall": {
    "accuracy": 0.9049000000000001,
    "balanced accuracy": 0.7358333333333333,
    "f1 score": 0.5224802015658694,
    "precision": 0.5249068926512016,
    "recall": 0.5245,
    "specificity": 0.9471666666666666,
    "negative predictive value": 0.9472517140540789,
    "false discovery rate": 0.47509310734879834,
    "miss rate": 0.4755,
    "fall out": 0.05283333333333333,
    "false omission rate": 0.052748285945920924,
    "mcc": 0.4711208318757299,
    "informedness": 0.4716666666666667,
    "markedness": 0.4721586067052807,
    "true positives": 5245,
    "true negatives": 85245,
    "false positives": 4755,
    "false negatives": 4755,
    "cardinality": 10000
},
```

This excerpt from the confusion matrix shows that the estimator does a good job identifying automobiles - 653 of the 1,000 in the test set are classified correctly - but confuses a good number of them for trucks, which makes sense since they are similar in many ways.

```json
    "automobile": {
        "cat": 39,
        "dog": 13,
        "automobile": 653,
        "ship": 70,
        "deer": 22,
        "truck": 179,
        "airplane": 43,
        "frog": 40,
        "horse": 23,
        "bird": 38
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
