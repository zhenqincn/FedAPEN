import torch


def get_model(model_name: str = 'resnet18', num_classes=10, client_id=None) -> torch.nn.Module:
    inner_model_name = model_name.replace('-', '').lower()
    if 'cnn1' == inner_model_name:
        from models.cnn import CNN1
        return CNN1(num_classes=num_classes)
    elif 'cnn1_bn' == inner_model_name:
        from models.cnn import CNN1_BN
        return CNN1_BN(num_classes=num_classes, client_id=client_id)
    elif 'cnn2' == inner_model_name:
        from models.cnn import CNN2
        return CNN2(num_classes=num_classes)
    elif 'cnn2_bn' == inner_model_name:
        from models.cnn import CNN2_BN
        return CNN2_BN(num_classes=num_classes, client_id=client_id)
