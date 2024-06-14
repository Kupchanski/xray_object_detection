from effdet.efficientdet import HeadNet
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.config.model_config import efficientdet_model_param_dict
from effdet.config.model_config import efficientdet_model_param_dict


def create_model(
    num_classes=1,
    image_size=512,
    architecture='tf_efficientnetv2_l',
    backbone='efficientnet_d4',
    train = True,
):
    
    """Create Effdet model function

    Args:
        num_classes (int, optional): Your dataset classes number. Defaults to 1.
        image_size (int, optional): Images resolution. Defaults to 512.
        architecture (str, optional): Choose desired arch for the effdet. Defaults to "tf_efficientnetv2_l".
        backbone (str, optional): Choose backbone.
            You can find more information at https://github.com/rwightman/efficientdet-pytorch/
            Defaults to "efficientnet_d4".


    Returns:
        pytorch_model: returns pretrained Effdet model, ready for finetuning
    """
    efficientdet_model_param_dict['tf_efficientnetv2_l'] = dict(
        name=backbone,
        backbone_name=backbone,
        backbone_args=dict(drop_path_rate=0.2),
        num_classes=num_classes,
        url='',
        )

    config = get_efficientdet_config(architecture)
    config.update({'num_classes': num_classes})
    config.update({'image_size': (image_size, image_size)})

    net = EfficientDet(config, pretrained_backbone=True)
    net.class_net = HeadNet(
        config,
        num_outputs=config.num_classes,
    )
    if train:
        return DetBenchTrain(net, config)
    return net, config
