from torchvision import transforms

from datasets import iCIFARSplit
from models import LwF
from trainers.incremental_train import incremental_train
from config import Config
from models.utils import SDGOptimizerAllFactory, MultiStepLRSchedulerFactory, ClassDistLossBuilder


if __name__ == "__main__":

    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    split_datasets = iCIFARSplit(Config.NUM_GROUPS, transform_train, transform_test, total_classes=Config.NUM_CLASSES)


    def prepare_LwF(class_loss, dist_loss, lr, milestones=Config.MILESTONES, gamma=Config.GAMMA):
        num_classes = int(Config.NUM_CLASSES / Config.NUM_GROUPS)
        loss = ClassDistLossBuilder.build(Config.DEVICE, class_loss=class_loss, dist_loss=dist_loss)
        net = LwF(loss, resnet_type='32', num_classes=num_classes)

        # Define optimizer
        optimizer_factory = SDGOptimizerAllFactory(lr=lr,
                                                   momentum=Config.MOMENTUM,
                                                   weight_decay=Config.WEIGHT_DECAY)

        # Define scheduler
        scheduler_factory = MultiStepLRSchedulerFactory(milestones=milestones,
                                                        gamma=gamma)

        return net, optimizer_factory, scheduler_factory


    log_prefix = None

    LwFnet, optimizer_factory, scheduler_factory = prepare_LwF('bce', 'bce', 0.1,
                                                               Config.MILESTONES,
                                                               Config.GAMMA)

    incremental_train(LwFnet, split_datasets, optimizer_factory, scheduler_factory,
                      Config.BATCH_SIZE, Config.NUM_EPOCHS, log_prefix)

