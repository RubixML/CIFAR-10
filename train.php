<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Loggers\Screen;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\PersistentModel;
use Rubix\ML\Transformers\PersistentTransformer;
use Rubix\ML\Transformers\Pipeline;
use Rubix\ML\Transformers\ImageResizer;
use Rubix\ML\Transformers\ImageFlipper;
use Rubix\ML\Transformers\ColorJitter;
use Rubix\ML\Transformers\ImageVectorizer;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\Transformers\FloatTypeConverter;
use Rubix\ML\Classifiers\MultilayerPerceptron;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\Layers\BatchNorm;
use Rubix\ML\NeuralNet\ActivationFunctions\SiLU;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Constant;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Extractors\CSV;

use function Rubix\ML\enumerate;

ini_set('memory_limit', '-1');

define('CHUNK_SIZE', 10000);
define('NUM_REPETITIONS', 3);

$logger = new Screen();

$transformer = new PersistentTransformer(
    base: new Pipeline([
        new ImageResizer(32, 32),
        new ImageVectorizer(),
        new FloatTypeConverter(),
        new ZScaleStandardizer(),
    ]),
    persister: new Filesystem('transformer.rbx', true)
);

$augmenter = new Pipeline([
    new ImageFlipper(),
    new ColorJitter(
        brightness: 0.1,
        contrast: 0.1,
        saturation: 0.1
    ),
]);

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

$estimator->setLogger($logger);

$samples = $labels = [];

foreach (glob('test/*.png') as $file) {
    $samples[] = [imagecreatefrompng($file)];
    $labels[] = preg_replace('/[0-9]+_(.*).png/', '$1', basename($file));
}

$testing = new Labeled($samples, $labels);

$transformer->fit($testing);

$testing->apply($transformer);

$estimator->setValidationDataset($testing);

$files = glob('train/*.png');

$chunks = array_chunk($files, CHUNK_SIZE);

for ($i = 0; $i < NUM_REPETITIONS; $i++) {
    foreach (enumerate($chunks, start: 1) as $j => $files) {
        $logger->info("Training on chunk #{$j}");

        $samples = $labels = [];

        foreach ($files as $file) {
            $samples[] = [imagecreatefrompng($file)];
            $labels[] = preg_replace('/[0-9]+_(.*).png/', '$1', basename($file));
        }

        $training = new Labeled($samples, $labels);

        $training->apply($augmenter);

        $transformer->update($training);

        $training->apply($transformer);

        $estimator->partial($training);

        $extractor = new CSV("progress_{$j}.csv", true);

        $extractor->export($estimator->progress(), overwrite: true);

        $logger->info("Progress saved to progress_{$j}.csv");
    }
}

if (strtolower(trim(readline('Save this model? (y|[n]): '))) === 'y') {
    $transformer->save();
    $estimator->save();
}
