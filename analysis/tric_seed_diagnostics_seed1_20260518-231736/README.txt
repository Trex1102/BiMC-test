Seed: 1
Train cfg: configs/trainers/bimc_dino_fusion.yaml

[CIFAR100]
avg session acc: 86.031 -> 87.572
performance drop: 6.58 -> 3.96
final session acc: 83.17 -> 85.79
intro-session corr: -0.348467 -> -0.199188
first vs last intro mean class acc: 88.25->87.9 vs 60.0->70.2

[miniimagenet]
avg session acc: 95.168 -> 95.467
performance drop: 3.63 -> 3.1
final session acc: 93.47 -> 94.0
intro-session corr: -0.233586 -> -0.18886
first vs last intro mean class acc: 96.3->96.117 vs 93.8->94.6

[cub200]
avg session acc: 85.347 -> 85.796
performance drop: 2.868 -> 2.367
final session acc: 85.295 -> 85.796
intro-session corr: 0.045288 -> 0.053936
first vs last intro mean class acc: 87.9->87.73 vs 92.0->92.667
