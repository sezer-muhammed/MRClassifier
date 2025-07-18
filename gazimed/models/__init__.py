"""
Model components for Alzheimer's detection system.

This module contains the core model architectures including:
- Swin-UNETR backbone
- Cross-attention fusion modules
- Clinical feature encoders
- Multimodal fusion components
"""

# Import only existing modules
try:
    from .patch_embedding import *
except ImportError:
    pass

try:
    from .swin_unetr import *
except ImportError:
    pass

try:
    from .cross_attention import *
except ImportError:
    pass

try:
    from .clinical_encoder import *
except ImportError:
    pass

try:
    from .mae_3d import *
except ImportError:
    pass

# Lightning module import is optional due to heavy dependencies
# Import it explicitly when needed: from gazimed.models.lightning_module import AlzheimersLightningModule
# try:
#     from .lightning_module import *
# except ImportError:
#     pass