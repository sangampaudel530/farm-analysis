import segmentation_models_pytorch as smp

def get_model(config): 
    return smp.UnetPlusPlus(
        encoder_name=config['ENCODER'],
        encoder_weights=config['ENCODER_WEIGHTS'],
        in_channels=3,
        classes=config['NUM_CLASSES']
    ).to(config['DEVICE'])

# Example usage:
# if __name__ == "__main__":
#     import configs.unetplusplus_b4_config as config
#     model = get_model(config.CONFIG)