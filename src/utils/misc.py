import random
import logging

# installed
import numpy as np
import torch


INITIALIZED_LOGGERS = {}


def get_logger(name, log_file=None, log_level=logging.INFO):
    """Create logger for experiments.

    Args:
        name (str): logger name.
            If function called multiple times same name or
            name which starts with same prefix then will be
            returned initialized logger from the first call.
        log_file (str): file to use for storing logs.
            Default is `None`.
        log_level (int): logging level.
            Default is `logging.INFO`.

    Returns:
        logging.Logger object.
    """
    logger = logging.getLogger(name)
    if name in INITIALIZED_LOGGERS:
        return logger
    # handle hierarchical names
    # e.g., logger "a" is initialized, then logger "a.b" will skip the
    # initialization since it is a child of "a".
    for logger_name in INITIALIZED_LOGGERS:
        if name.startswith(logger_name):
            return logger

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0

    # only rank 0 will add a FileHandler
    if rank == 0 and log_file is not None:
        file_handler = logging.FileHandler(log_file, "w")
        handlers.append(file_handler)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    INITIALIZED_LOGGERS[name] = True
    return logger


def export_to_onnx(model, input_example, file, opset_version=10):
    """Export pytorch model to ONNX file.

    Args:
        model (torch.nn.Module): trained model.
        input_example (torch.Tensor or Tuple[torch.Tensor]): model input data
        file (str): path where shoul be stored ONNX model.
    """
    torch.onnx.export(
        model.cpu().eval(),
        input_example,
        file,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["locs", "confs"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "locs": {0: "batch_size"},
            "confs": {0: "batch_size"},
        },
    )


def t2d(tensor, device):
    """Move tensors to a specified device.

    Args:
        tensor (torch.Tensor or Dict[str, torch.Tensor] or list/tuple of torch.Tensor):
            data to move to a device.
        device (str or torch.device): device where should be moved device

    Returns:
        torch.Tensor or Dict[str, torch.Tensor] or List[torch.Tensor] based on `tensor` type.
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.to(device)
    elif isinstance(tensor, (tuple, list)):
        # recursive move to device
        return [t2d(_tensor, device) for _tensor in tensor]
    elif isinstance(tensor, dict):
        res = {}
        for _key, _tensor in tensor.items():
            res[_key] = t2d(_tensor, device)
        return res


def seed_all(seed=42, deterministic=True, benchmark=True) -> None:
    """Fix all seeds so results can be reproducible.

    Args:
        seed (int): random seed.
            Default is `42`.
        deterministic (bool): flag to use cuda deterministic
            algoritms for computations.
            Default is `True`.
        benchmark (bool): flag to use benchmark option
            to select the best algorithm for computatins.
            Should be used `True` with fixed size
            data (images or similar) for other types of
            data is better to use `False` option.
            Default is `True`.
    """
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    random.seed(seed)
    np.random.seed(seed)
    # reproducibility
    torch.backends.cudnn.deterministic = deterministic
    # small speedup
    torch.backends.cudnn.benchmark = benchmark
