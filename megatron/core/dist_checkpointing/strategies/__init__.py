# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

""" Various loading and saving strategies """

import logging

logger = logging.getLogger(__name__)

try:
    import tensorstore
    import zarr

    from .tensorstore import _import_trigger
    from .zarr import _import_trigger
except ImportError:
    # Only print warning on first rank.
    import os

    if int(os.getenv('RANK', '0')) == 0:
        logger.warning('Zarr-based strategies will not be registered because of missing packages')
