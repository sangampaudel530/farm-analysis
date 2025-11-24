import segmentation_models_pytorch as smp

def build_model(config):
    model = smp.Unet(
        encoder_name=config['ENCODER'],
        encoder_weights=config['ENCODER_WEIGHTS'],
        in_channels=3,
        classes=config['NUM_CLASSES'],
        activation=None
    )
    return model

# Example usage:
# if __name__ == "__main__":
#     import configs.unetplusplus_b4_config as config
#     model = build_model(config.CONFIG)