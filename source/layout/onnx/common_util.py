"""
common_util.py

Utils shared by the ONNX runtime users.

"""
import os
import onnxruntime as ort

from .ImageFeatureExtractorV1 import ImageFeatureExtractorV1
from .ImageFeatureExtractorV2 import ImageFeatureExtractorV2

def make_session(model_path, providers=None):
    """
    Create an ort.InferenceSession with the CPU memory arena disabled.

    By default, ort.InferenceSession uses enable_cpu_mem_arena=True, which
    causes the ONNX Runtime allocator to retain the peak allocation across
    all calls and never release it back to the OS.  When a long-lived process
    processes many PDFs that differ in page size or element count, the arena
    grows without bound and is never reclaimed even after gc.collect().

    Setting enable_cpu_mem_arena=False makes the runtime release allocations
    promptly, keeping RSS stable across documents.
    """
    so = ort.SessionOptions()
    so.enable_cpu_mem_arena = False
    return ort.InferenceSession(model_path, sess_options=so, providers=providers)


def make_image_feature_extractor(imf_model_path, providers):
    """
    Create an ImageFeatureExtractor instance from an ONNX model path.

    Selects V2 if the model exports 'reg_coarse' (FCOS-based detection),
    otherwise falls back to V1 (CCL-based detection).

    Args:
        imf_model_path: path to the ONNX model file.
        providers:      list of ORT execution providers.

    Returns:
        ImageFeatureExtractorV2 or ImageFeatureExtractorV1 instance,
        or None if the model file does not exist.
    """
    if not os.path.exists(imf_model_path):
        return None

    ort_session = make_session(imf_model_path, providers)
    output_names = {o.name for o in ort_session.get_outputs()}

    if 'reg_coarse' in output_names:
        return ImageFeatureExtractorV2(ort_session)
    return ImageFeatureExtractorV1(ort_session)
