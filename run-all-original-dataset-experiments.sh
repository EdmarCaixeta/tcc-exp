python embrapa_experiment.py alexnet ../GSD05/2019-01-23/ --experiment_folder ../experiments/alexnet/ --epochs 400 --batch_size 256 --augment no --epochs_between_checkpoints 100 --cuda_device_number 1 \
&& python embrapa_experiment.py resnet ../GSD05/2019-01-23/ --experiment_folder ../experiments/resnet18/ --epochs 300 --batch_size 128 --augment no --epochs_between_checkpoints 100 --cuda_device_number 1 \
&& python embrapa_experiment.py myalexnetpretrained ../GSD05/2019-01-23/ --experiment_folder ../experiments/myalexnet-pretrained/ --epochs 200 --batch_size 256 --augment no --epochs_between_checkpoints 100 --cuda_device_number 1 \
&& python embrapa_experiment.py resnet18pretrained ../GSD05/2019-01-23/ --experiment_folder ../experiments/resnet18-pretrained/ --epochs 200 --batch_size 128 --augment no --epochs_between_checkpoints 100 --cuda_device_number 1