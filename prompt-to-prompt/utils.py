### Hook the models
from collections import OrderedDict
class ModuleHook:
    #### Example usage: encoder_hook, encoder_layers = hook_model(encoder, include_class_name=False)
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = None
        self.features = None

    def hook_fn(self, module, input, output):
        self.module = module
        self.features = output

    def close(self):
        self.hook.remove()


def hook_model(model, include_class_name=False):
    features = OrderedDict()
    layers = OrderedDict()
    # recursive hooking function
    def hook_layers(net, prefix=[]):
        if hasattr(net, "_modules"):
            for name, layer in net._modules.items():
                if layer is None:
                    # e.g. GoogLeNet's aux1 and aux2 layers
                    continue
                hook_layers(layer, prefix=prefix + [name])
                if include_class_name:
                    features["-".join(prefix + [name]) + f":{layer.__class__.__name__}"] = ModuleHook(layer)
                    layers["-".join(prefix + [name]) + f":{layer.__class__.__name__}"] = layer.__repr__()
                else:
                    features["-".join(prefix + [name])] = ModuleHook(layer)
                    layers["-".join(prefix + [name])] = layer.__repr__()

    hook_layers(model)

    def hook(layer):
        assert layer in features, f"Invalid layer {layer}. Retrieve the list of layers with `lucent.modelzoo.util.get_model_layers(model)`."
        out = features[layer].features
        assert out is not None, "There are no saved feature maps. Make sure to put the model in eval mode, like so: `model.to(device).eval()`. See README for example."
        return out

    return hook, layers