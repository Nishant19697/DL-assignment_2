import torch
import torch.nn as nn

def get_activation(name):
    return {
        "ReLU": nn.ReLU,
        "GELU": nn.GELU,
        "SiLU": nn.SiLU
    }[name]

class Cnet(nn.Module):
    def __init__(self, conv_layer_config, dense_unit, activation, num_classes, dropout_prob, batch_norm ):
        super(Cnet, self).__init__()
        self.feature_extractor = nn.Sequential()

        print(conv_layer_config)

        in_channels = 3
        for idx, (filters, kernel) in enumerate(conv_layer_config):
            self.feature_extractor.add_module(f"conv{idx+1}", nn.Conv2d(in_channels, filters, kernel_size=kernel, padding=kernel//2))
            if batch_norm:   
                self.feature_extractor.add_module(f"bn{idx+1}", nn.BatchNorm2d(filters))
            self.feature_extractor.add_module(f"act{idx+1}", activation())
            # self.feature_extractor.add_module(f"drop{idx+1}", nn.Dropout(p=dropout_prob))
            self.feature_extractor.add_module(f"pool{idx+1}", nn.MaxPool2d(2, 2))
            in_channels = filters

        self.flattened_size = self._get_flattened_size()

        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, dense_unit),
            nn.BatchNorm1d(dense_unit),
            activation(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(dense_unit, num_classes)
        )

    def _get_flattened_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            output = self.feature_extractor(dummy_input)
            return output.view(1, -1).size(1)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
