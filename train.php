<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Loggers\Screen;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\PersistentModel;
use Rubix\ML\Pipeline;
use Rubix\ML\Transformers\ImageResizer;
use Rubix\ML\Transformers\ImageVectorizer;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\Transformers\FloatTypeConverter;
use Rubix\ML\Classifiers\MultilayerPerceptron;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\Layers\BatchNorm;
use Rubix\ML\NeuralNet\ActivationFunctions\GELU;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Constant;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Extractors\CSV;

use function Rubix\ML\enumerate;

ini_set('memory_limit', '-1');

$logger = new Screen();

$files = glob('train/*.png');
$chunkSize = 10000;

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

$estimator->setLogger($logger);

$chunks = array_chunk($files, $chunkSize);

foreach (enumerate($chunks, start: 1) as $i => $files) {
    $logger->info("Training on chunk #{$i}");

    $samples = $labels = [];

    foreach ($files as $file) {
        $samples[] = [imagecreatefrompng($file)];
        $labels[] = preg_replace('/[0-9]+_(.*).png/', '$1', basename($file));
    }

    $subset = new Labeled($samples, $labels);

    $estimator->partial($subset);

    $extractor = new CSV("progress_{$i}.csv", true);

    $extractor->export($estimator->progress(), overwrite: true);

    $logger->info("Progress saved to progress_{$i}.csv");
}

if (strtolower(trim(readline('Save this model? (y|[n]): '))) === 'y') {
    $estimator->save();
}
