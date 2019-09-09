<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\PersistentModel;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\CrossValidation\Reports\AggregateReport;
use Rubix\ML\CrossValidation\Reports\ConfusionMatrix;
use Rubix\ML\CrossValidation\Reports\MulticlassBreakdown;

ini_set('memory_limit', '-1');

echo '╔═════════════════════════════════════════════════════╗' . PHP_EOL;
echo '║                                                     ║' . PHP_EOL;
echo '║ CIFAR-10 Image Recognizer w/ Multi Layer Perceptron ║' . PHP_EOL;
echo '║                                                     ║' . PHP_EOL;
echo '╚═════════════════════════════════════════════════════╝' . PHP_EOL;
echo PHP_EOL;

echo 'Loading data into memory ...' . PHP_EOL;

$samples = $labels = [];

foreach (glob(__DIR__ . '/test/*.png') as $file) {
    $samples[] = [imagecreatefrompng($file)];
    $labels[] = preg_replace('/[0-9]+_(.*).png/', '$1', basename($file));
}

$dataset = new Labeled($samples, $labels);

$estimator = PersistentModel::load(new Filesystem('cifar-10.model'));

$predictions = $estimator->predict($dataset);

$report = new AggregateReport([
    new MulticlassBreakdown(),
    new ConfusionMatrix(),
]);

$results = $report->generate($predictions, $dataset->labels());

file_put_contents('report.json', json_encode($results, JSON_PRETTY_PRINT));

echo 'Report saved to report.json' . PHP_EOL;
