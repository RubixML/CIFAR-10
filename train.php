<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Pipeline;
use Rubix\ML\PersistentModel;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Loggers\Screen;
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
use League\Csv\Writer;

const MODEL_FILE = 'cifar-10.model';
const PROGRESS_FILE = 'progress.csv';

ini_set('memory_limit', '-1');

echo '╔═════════════════════════════════════════════════════╗' . PHP_EOL;
echo '║                                                     ║' . PHP_EOL;
echo '║ CIFAR-10 Image Classifier w/ Multi Layer Perceptron ║' . PHP_EOL;
echo '║                                                     ║' . PHP_EOL;
echo '╚═════════════════════════════════════════════════════╝' . PHP_EOL;
echo PHP_EOL;

echo 'Loading data into memory ...' . PHP_EOL;

$samples = $labels = [];

foreach (glob(__DIR__ . '/train/*.png') as $file) {
    $samples[] = [imagecreatefrompng($file)];
    $labels[] = preg_replace('/[0-9]+_(.*).png/', '$1', basename($file));
}

$dataset = new Labeled($samples, $labels);

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
        new Dense(50),
        new PReLU(),
    ], 100, new Adam(0.001))),
    new Filesystem(MODEL_FILE, true)
);

$estimator->setLogger(new Screen('CIFAR10'));

$estimator->train($dataset);

$writer = Writer::createFromPath(PROGRESS_FILE, 'w+');
$writer->insertOne(['score', 'loss']);
$writer->insertAll(array_map(null, $estimator->scores(), $estimator->steps()));

echo 'Progress saved to ' . PROGRESS_FILE . PHP_EOL;

if (strtolower(trim(readline('Save this model? (y|[n]): '))) === 'y') {
    $estimator->save();
}
