
def load_model(fname):
    # Configure the parent path to be the proj folder
    import sys, os, torch, time

    # Import the block classes
    from rwkv_block.v7_goose.block.rwkv7_layer_block import RWKV7LayerBlock
    from rwkv_block.v7_goose.model.rwkv7_goose_model import RWKV7GooseModel
    from rwkv_block.v7_goose.model.rwkv7_goose_config_map import RWKV7GooseConfigMap

    # File to load

    # Run device, and run dtype to use
    RUN_DEVICE="cpu"
    RUN_DTYPE=torch.bfloat16

    # Check for cuda device
    if torch.cuda.is_available():
        RUN_DEVICE="cuda:0"

    # Check if the reference weights exists
    assert os.path.exists(f"{fname}"), "The reference weights does not exist. Please download it first (00-model-download.ipynb)"

    # Loads the model weights
    model_weight = torch.load(f"{fname}", map_location='cpu', weights_only=True, mmap=True)

    # Model filename
    print(f"### Model filename: {fname}")

    # Lets get the hidden_size, and setup the test module
    hidden_size = model_weight['emb.weight'].shape[1]
    print(f"### Model hidden_size: {hidden_size}")

    # Get the config
    model_config = RWKV7GooseConfigMap.from_model_state_dict(model_weight, device=RUN_DEVICE, dtype=RUN_DTYPE)

    # Log the config
    print("### Model Config:")
    print(model_config)

    # Initialize the model instance
    model_inst = RWKV7GooseModel(model_config)
    model_inst.load_state_dict(model_weight, strict=False)
    #model_inst.load_from_model_state_dict(model_weight)
    model_state = model_inst.state_dict()

    # List the model weights keys, and their shapes
    #print(f"### model weights keys:")
    #for key in model_state:
        #print(f"{key}: {model_state[key].shape} - {model_state[key].dtype}")

    return model_inst