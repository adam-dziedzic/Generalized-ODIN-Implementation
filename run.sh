python main.py --in_dataset CIFAR100 --architecture resnet --weight_decay 0.0005 --batch_size 128 --epochs 200 --gpu 1


python main.py --in_dataset CIFAR10 --out_dataset SVHN --architecture densenet --similarity cosine --weight_decay 0.0001 --batch_size 64 --epochs 300 --gpu 0

python main.py --in_dataset CIFAR10 --out_dataset SVHN --architecture resnet --similarity cosine --weight_decay 0.0005 --batch_size 128 --epochs 200 --gpu 0

python main.py --in_dataset CIFAR10 --out_dataset SVHN --architecture wideresnet --similarity cosine --weight_decay 0.0005 --batch_size 128 --epochs 200 --gpu 0