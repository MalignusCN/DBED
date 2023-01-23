# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseSegmentor
from .cascade_encoder_decoder import CascadeEncoderDecoder
from .encoder_decoder import EncoderDecoder
from .dual_branch_encoder_decoder import DualBranchEncoderDecoder
from .gid_encoder_decoder import GIDEncoderDecoder
from .gid_encoder_decoder_withnir import GIDEncoderDecoderWithNIR

__all__ = ['BaseSegmentor', 'EncoderDecoder', 'CascadeEncoderDecoder', \
    'DualBranchEncoderDecoder', 'GIDEncoderDecoder', 'GIDEncoderDecoderWithNIR']
