import torch


def model_info(model: torch.nn.Module):
    """
    Analyze a PyTorch model and returns a dictionary containing information about its size and parameters.

    Args:
        model (torch.nn.Module): The PyTorch model to be analyzed.

    Returns:
        dict: A dictionary containing the following information:
            total_params (int): Total number of parameters in the model.
            trainable_params (int): Number of parameters that require gradient calculation during training.
            non_trainable_params (int): Number of parameters that do not require gradient calculation during training.
            total_size_mb (float): Total size of the model in megabytes.
            param_size_mb (float): Size of the model parameters in megabytes (excluding buffers).
            buffer_size_mb (float): Size of the model buffers in megabytes.

    Prints:
        This function also prints human-readable information about the model size and parameters
        including total, trainable, and non-trainable parameters for better understanding.
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    param_size_mb = param_size / 1024**2
    buffer_size_mb = buffer_size / 1024**2
    total_size_mb = (param_size + buffer_size) / 1024**2

    total_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
    trainable_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())
    non_trainable_params = total_params - trainable_params
    info = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "total_size_mb": total_size_mb,
        "param_size_mb": param_size_mb,
        "buffer_size_mb": buffer_size_mb,
    }
    print(f"Total Parameters = {format(total_params,',')}")
    print(f"Trainable Parameters = {format(trainable_params,',')}")
    print(f"Non-trainable Parameters = {format(non_trainable_params,',')}")
    print("=" * 50)
    print("Model size: {:.3f} MB".format(total_size_mb))
    return info
