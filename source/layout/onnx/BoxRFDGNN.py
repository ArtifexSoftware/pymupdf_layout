import math
import os.path

import yaml
import numpy as np
import onnxruntime as ort
from pathlib import Path

from ..common_util import (get_boxes_transform, get_edge_by_knn,
                           get_edge_transform_bbox,
                           get_text_pattern, get_edge_matrix, group_node_by_edge,
                           resize_image, compute_iou)
from ..roi_pooling import (extract_bbox_features_by_roi_pooling,
                           RoiPoolingSession,
                           DEFAULT_FEATURE_MAP_POOLING_OPS,
                           DEFAULT_CLASS_LOGITS_POOLING_OPS)
from ..pymupdf_util import create_input_data_from_page
from ..pymupdf_util_edge import get_edge_attr, get_edge_dim, build_edge_index, compute_edge_gap_bboxes
from .TableGridExtractor import TableGridExtractor
from .TableGridExtractorV1A import TableGridExtractorV1A
from .TableGridExtractorV1B import TableGridExtractorV1B
from .TableGridExtractorV2 import TableGridExtractorV2
from .TableGridExtractorV2A import TableGridExtractorV2A
from .TableGridExtractorV2B import TableGridExtractorV2B
from .TableGridExtractorV3 import TableGridExtractorV3
from .TableGridExtractorV5 import TableGridExtractorV5
from .TableGridExtractorV6 import TableGridExtractorV6
from .MarkdownGenerator import MarkdownGenerator
from .MarkdownHTMLTableGenerator import MarkdownHTMLTableGenerator
from .HTMLGenerator import HTMLGenerator
from .DefaultSorter import DefaultSorter
from .common_util import make_session, make_image_feature_extractor

IGNORE_FEATURE_NAMES = []

# ----------------------------------------------------------------------------
# Profile loading helpers
#
# A "profile" is an internal (non-user-facing) YAML file that bundles all the
# constructor options that used to be passed in one by one (feature_set_name,
# model paths, table grid version, runtime flags, etc). Profiles live under
# resources/profiles/<profile_name>.yaml. If a requested profile file does not
# exist, resources/profiles/default.yaml is used instead.
#
# Note: self.model_cfg (loaded from model_config_path, see __init__)
# describes the model itself (data/model definition). The profile described
# here is a separate, higher level concept: it describes which files and
# runtime options to use to construct a BoxRFDGNN instance. Keep the two
# concepts and variable names distinct to avoid confusion.
# ----------------------------------------------------------------------------

PROFILE_DIR_NAME = 'profiles'
DEFAULT_PROFILE_NAME = 'default'
GRID_PROFILES_FILENAME = 'grid_profiles.yaml'

# Registry mapping the extractor_class name used in grid_profiles.yaml to the
# actual class object. YAML cannot store class references directly, so the
# profile only stores the string key below.
GRID_EXTRACTOR_REGISTRY = {
    'TableGridExtractor': TableGridExtractor,
    'TableGridExtractorV1A': TableGridExtractorV1A,
    'TableGridExtractorV1B': TableGridExtractorV1B,
    'TableGridExtractorV2': TableGridExtractorV2,
    'TableGridExtractorV2A': TableGridExtractorV2A,
    'TableGridExtractorV2B': TableGridExtractorV2B,
    'TableGridExtractorV3': TableGridExtractorV3,
    'TableGridExtractorV5': TableGridExtractorV5,
    'TableGridExtractorV6': TableGridExtractorV6,
}

# Safety net values used only if a key is missing from both the requested
# profile and the default profile. Internal use only, not meant to be tuned
# by end users.
PROFILE_HARD_DEFAULTS = {
    'feature_set_name': 'imf+rf',
    'table_grid_model_ver': 'V4',
    'use_gpu': False,
    'use_sort': False,
}


def load_yaml_file(path):
    """Load a single YAML file and return its content as a dict."""
    with open(path, "rb") as f:
        return yaml.safe_load(f)


def resolve_profile_dict(script_dir, profile_name):
    """
    Load a profile YAML by name, falling back to the default profile if the
    requested profile file does not exist.

    Args:
        script_dir: base resources directory (same root used for onnx assets)
        profile_name: name of the profile to load (without .yaml extension)

    Returns:
        dict: parsed profile content. Never None; missing keys inside the
        dict are expected and handled by the caller via PROFILE_HARD_DEFAULTS.
    """
    profiles_dir = Path(script_dir) / 'resources' / PROFILE_DIR_NAME
    profile_path = profiles_dir / f'{profile_name}.yaml'
    default_path = profiles_dir / f'{DEFAULT_PROFILE_NAME}.yaml'

    if not profile_path.exists():
        profile_path = default_path

    if not profile_path.exists():
        # Internal misconfiguration: no profile and no default profile file.
        raise FileNotFoundError(
            f"Profile file not found: neither '{profile_name}.yaml' nor "
            f"'{DEFAULT_PROFILE_NAME}.yaml' exist under {profiles_dir}"
        )

    profile_dict = load_yaml_file(profile_path)
    return profile_dict or {}


def resolve_grid_profiles(script_dir):
    """Load resources/grid_profiles.yaml and return it as a dict."""
    grid_profiles_path = Path(script_dir) / 'resources' / GRID_PROFILES_FILENAME
    return load_yaml_file(grid_profiles_path) or {}


def validate_required_files(paths):
    """
    Check that every path in the given dict is set and points to an
    existing file, raising a single clear error otherwise.

    This is meant to catch profile misconfiguration (a missing filename
    key, or a filename that does not exist on disk) as early and clearly
    as possible, instead of failing later with a raw FileNotFoundError or
    TypeError deep inside yaml.safe_load()/onnxruntime.

    Args:
        paths: dict mapping a descriptive name (used in the error message)
            to the resolved path (str or None).

    Raises:
        FileNotFoundError: if any path is None or does not exist.
    """
    problems = []
    for name, path in paths.items():
        if path is None:
            problems.append(f"{name}: not set")
        elif not os.path.exists(path):
            problems.append(f"{name}: file not found at '{path}'")

    if problems:
        raise FileNotFoundError(
            "Invalid or missing profile file(s):\n  " + "\n  ".join(problems)
        )
def is_inside(bbox, region, margin=10):
    """Check if bbox is within the region expanded by margin."""
    bx1, by1, bx2, by2 = bbox
    rx1, ry1, rx2, ry2 = region
    if (rx1 - margin) <= bx1 < bx2 <= (rx2 + margin) and (ry1 - margin) <= by1 < by2 <= (ry2 + margin):
        return True
    return False


def get_rf_features(custom_feature, rf_names, do_log_norm=False):
    f = []
    for f_name in rf_names:
        if f_name == 'is_text':
            if 'box_type' not in custom_feature or custom_feature['box_type'] == 'text':
                f.append(1.0)
            else:
                f.append(0.0)
        elif f_name == 'is_image':
            if 'box_type' in custom_feature and custom_feature['box_type'] == 'image':
                f.append(1.0)
            else:
                f.append(0.0)

        elif f_name == 'is_vector':
            if 'box_type' in custom_feature and custom_feature['box_type'] == 'vector':
                f.append(1.0)
            else:
                f.append(0.0)

        elif f_name == 'is_hline_vector':
            if 'box_type' in custom_feature and custom_feature['box_type'] == 'h-line-vector':
                f.append(1.0)
            else:
                f.append(0.0)

        elif f_name == 'is_vline_vector':
            if 'box_type' in custom_feature and custom_feature['box_type'] == 'v-line-vector':
                f.append(1.0)
            else:
                f.append(0.0)
        else:
            if f_name in IGNORE_FEATURE_NAMES:
                f.append(0.0)
            else:
                if f_name not in custom_feature:
                    raise Exception(f"'{f_name}' does not exist in 'custom_features'")

                try:
                    val = float(custom_feature[f_name])
                    if do_log_norm:
                        if f_name.startswith('context_') or f_name.startswith('is_') or f_name in ['fonts_offset', 'line_space',
                                                                                                   'linespaces_offset']:
                            pass
                        else:
                            val = math.log(val + 1e-8)
                    f.append(val)
                except Exception as ex:
                    print(f'{f_name} : {str(ex)}')
                    raise ex
    return f


def get_nn_input_from_datadict(data_dict, cfg, return_nn_index=False,
                               edge_sampling='4D', save_crop_page=False,
                               use_image_edge=False):
    original_bboxes = data_dict['bboxes']
    x = get_boxes_transform(original_bboxes)
    if isinstance(x, np.ndarray) and x.dtype != np.float32:
        x = x.astype(np.float32)

    bboxes = np.array(original_bboxes, dtype=np.float32)

    data_option = cfg['data']
    model_option = cfg['model']

    if return_nn_index:
        nn_k = model_option['sample_k']
        nn_index = get_edge_by_knn(bboxes, k=nn_k)
        nn_attr = get_edge_transform_bbox(bboxes, nn_index)
        nn_index = np.array(nn_index, dtype=np.int64).T
    else:
        nn_index = None
        nn_attr = None

    edge_type = ''
    if 'edge_type' in model_option['option']:
        edge_type = model_option['option']['edge_type']

    page_img = data_dict.get('image')

    # One input: single node edge case
    if len(bboxes) == 1:
        edge_dim = get_edge_dim(edge_type, page_img)
        edge_index = np.zeros(shape=[2, 1], dtype=np.int64)
        edge_attr = np.zeros(shape=[1, edge_dim], dtype=np.float32)
        nn_index = np.zeros(shape=[2, 1], dtype=np.int64)
        nn_attr = np.zeros(shape=[1, edge_dim], dtype=np.float32)
    else:
        edge_index = build_edge_index(bboxes, edge_sampling)
        edge_attr = get_edge_attr(bboxes, edge_index, edge_type, data_dict, page_img)
        edge_index = np.array(edge_index, dtype=np.int64).T

    rf_names = data_option['rf_names'][:]
    yf_names = data_option.get('yf_names', [])
    rf_names.extend(yf_names)
    jf_names = data_option.get('jf_names', [])
    rf_names.extend(jf_names)

    rf_feature = []
    for row_idx, custom_feature in enumerate(data_dict['custom_features']):
        f = get_rf_features(custom_feature, rf_names)
        rf_feature.append(f)
    rf_feature = np.array(rf_feature, dtype=np.float32)

    text_feature = []
    for text in data_dict['text']:
        text_feature.append(get_text_pattern(text, return_vector=True))
    text_feature = np.array(text_feature, dtype=np.float32)

    # Cropped page image by bboxes
    crop_img = None
    if save_crop_page:
        page_w = data_dict['page_width_int']
        page_h = data_dict['page_height_int']
        img_h, img_w = page_img.shape[:2]
        scale_x = img_w / page_w
        scale_y = img_h / page_h

        x1_min = int(np.floor(np.min(bboxes[:, 0])) * scale_x)
        y1_min = int(np.floor(np.min(bboxes[:, 1])) * scale_y)
        x2_max = int(np.ceil(np.max(bboxes[:, 2])) * scale_x)
        y2_max = int(np.ceil(np.max(bboxes[:, 3])) * scale_y)

        x1_min = max(0, x1_min)
        y1_min = max(0, y1_min)
        x2_max = min(img_w, x2_max)
        y2_max = min(img_h, y2_max)

        crop_img = page_img[y1_min:y2_max, x1_min:x2_max]
        crop_img = resize_image(crop_img, (500, 500))

    feature_map = data_dict.get('feature_map')
    class_logits = data_dict.get('class_logits')

    # Image feature extractor.
    # feature_map (decoder embedding) and class_logits (per-class scores) are
    # pooled SEPARATELY -- they have different statistical character -- and
    # then concatenated. class_logits is always softmaxed before pooling
    # since its entropy/margin ops are only meaningful over a probability
    # distribution.
    #
    # When use_image_edge is True, the SAME feature_map/class_logits tensors
    # are pooled twice per page: once for node bboxes, once for edge-union
    # bboxes. A RoiPoolingSession is used in that case so the bbox-
    # independent softmax/SAT setup (see roi_pooling.RoiPoolingSession) is
    # built once and shared between the node and edge queries, rather than
    # each query rebuilding it independently. When there's only one query
    # (use_image_edge=False), the plain one-shot function is used instead --
    # it already dispatches naive-vs-SAT per call, which is the better
    # choice when there's nothing to amortize the SAT build cost over.
    if feature_map is not None and class_logits is not None:
        page_h, page_w, _ = page_img.shape

        if use_image_edge:
            feat_session = RoiPoolingSession(feature_map, pooling_ops=DEFAULT_FEATURE_MAP_POOLING_OPS)
            logit_session = RoiPoolingSession(class_logits, pooling_ops=DEFAULT_CLASS_LOGITS_POOLING_OPS,
                                              apply_softmax=True)

            feat_pooled = feat_session.query(original_bboxes, page_w, page_h)
            logit_pooled = logit_session.query(original_bboxes, page_w, page_h)
            image_features = np.concatenate([feat_pooled, logit_pooled], axis=1)

            edge_bboxes = compute_edge_gap_bboxes(edge_index, original_bboxes)
            edge_feat_pooled = feat_session.query(edge_bboxes, page_w, page_h)
            edge_logit_pooled = logit_session.query(edge_bboxes, page_w, page_h)
            edge_image_features = np.concatenate([edge_feat_pooled, edge_logit_pooled], axis=1)
            edge_attr = np.concatenate([edge_attr, edge_image_features], axis=1)
        else:
            feat_pooled = extract_bbox_features_by_roi_pooling(
                feature_map, original_bboxes, page_w, page_h,
                pooling_ops=DEFAULT_FEATURE_MAP_POOLING_OPS,
            )
            logit_pooled = extract_bbox_features_by_roi_pooling(
                class_logits, original_bboxes, page_w, page_h,
                pooling_ops=DEFAULT_CLASS_LOGITS_POOLING_OPS,
                apply_softmax=True,
            )
            image_features = np.concatenate([feat_pooled, logit_pooled], axis=1)
    else:
        image_features = None

    if 'bbox_feature' in IGNORE_FEATURE_NAMES:
        x[:] = 0

    if 'text_pattern' in IGNORE_FEATURE_NAMES:
        text_feature[:] = 0

    if 'pdf_feature' in IGNORE_FEATURE_NAMES:
        rf_feature[:] = 0

    if 'image_feature' in IGNORE_FEATURE_NAMES:
        if image_features is not None:
            image_features[:] = 0

    return x, edge_index, edge_attr, nn_index, nn_attr, rf_feature, text_feature, image_features, crop_img


class BoxRFDGNN:
    def __init__(self, profile=DEFAULT_PROFILE_NAME, config_path=None, model_path=None, imf_model_path=None,
                 table_grid_path=None, feature_set_name=None,
                 input_type=None, enable_inference_cache=None, use_gpu=None, use_sort=None,
                 table_grid_model_ver=None):
        """
        Args:
            profile: name of a profile YAML under resources/profiles/ that
                bundles the options below. Defaults to 'default'. If the
                named profile file is not found, resources/profiles/default.yaml
                is used instead.
            config_path, model_path, imf_model_path, table_grid_path,
            feature_set_name, input_type, use_gpu, use_sort,
            table_grid_model_ver: explicit overrides. Any value passed here
                (i.e. not None) takes precedence over the value found in the
                profile, which in turn takes precedence over the built-in
                fallback resolution logic below.
            enable_inference_cache: currently a no-op, kept only for
                backward compatibility with existing call sites. Not part
                of the profile schema.
        """
        script_dir = Path(__file__).resolve().parent.parent

        profile_dict = resolve_profile_dict(script_dir, profile)

        def resolve(explicit_value, key, hard_default=None):
            """explicit kwarg > profile value > hard default."""
            if explicit_value is not None:
                return explicit_value
            if key in profile_dict and profile_dict[key] is not None:
                return profile_dict[key]
            return hard_default

        self.profile_name = profile

        self.feature_set_name = resolve(
            feature_set_name, 'feature_set_name', PROFILE_HARD_DEFAULTS['feature_set_name']
        )
        ft_set_names = ['rf', 'imf', 'imf+rf', 'imf+rf+yf', 'rf+jf']
        if self.feature_set_name not in ft_set_names:
            raise ValueError(f"feature_set_name must be one in {str(ft_set_names)}")

        # config_path / model_path / imf_model_path: explicit kwargs (a full
        # path) always win. Otherwise, the profile provides bare filenames
        # (config_filename, model_filename, imf_model_filename) under
        # resources/onnx/, resolved here with script_dir since the profile
        # YAML itself cannot know the install path. See
        # resources/profiles/imf1.yaml, rf2.4.1+imf1.yaml, rf2.4.1.yaml for
        # the three built-in combinations.
        if config_path is None:
            config_filename = profile_dict.get('config_filename')
            if config_filename is not None:
                config_path = f'{script_dir}/resources/onnx/{config_filename}'

        if model_path is None:
            model_filename = profile_dict.get('model_filename')
            if model_filename is not None:
                model_path = f'{script_dir}/resources/onnx/{model_filename}'

        if imf_model_path is None:
            imf_model_filename = profile_dict.get('imf_model_filename')
            if imf_model_filename is not None:
                imf_model_path = f'{script_dir}/resources/onnx/{imf_model_filename}'

        table_grid_path = resolve(table_grid_path, 'table_grid_path')
        input_type = resolve(input_type, 'input_type')
        # NOTE: enable_inference_cache is currently a no-op. The inference
        # caching mechanism it used to control was removed from this class,
        # but the parameter is kept in the signature so existing call sites
        # (including positional callers) do not break. It is intentionally
        # left out of the profile schema since it has no effect.
        use_gpu = resolve(use_gpu, 'use_gpu', PROFILE_HARD_DEFAULTS['use_gpu'])
        use_sort = resolve(use_sort, 'use_sort', PROFILE_HARD_DEFAULTS['use_sort'])
        table_grid_model_ver = resolve(
            table_grid_model_ver, 'table_grid_model_ver', PROFILE_HARD_DEFAULTS['table_grid_model_ver']
        )

        # config_path, model_path, and imf_model_path are no longer derived
        # from feature_set_name in code. They must be provided either as
        # explicit constructor args or via the profile (see
        # resources/profiles/*.yaml). Fail fast and clearly if any of them
        # is missing or does not point to an existing file.
        validate_required_files({
            'config_path': config_path,
            'model_path': model_path,
            'imf_model_path': imf_model_path,
        })

        self.imf_model_path = imf_model_path
        # model_config_path / model_cfg refer to the model's own definition
        # file (data/model spec), as distinct from the profile loaded above
        # (which only decides which files and runtime options to use).
        self.model_config_path = config_path
        with open(self.model_config_path, "rb") as f:
            self.model_cfg = yaml.safe_load(f)

        # Try get input_type from model config
        if input_type is None:
            input_type = self.model_cfg['data'].get('input_type', None)
        # If there is no input_type, assign the default value
        if input_type is None:
            input_type = ('text',)
        self.input_type = input_type

        self.data_class_names = self.model_cfg['data']['class_list']
        self.data_class_map = {}
        for i in range(len(self.data_class_names)):
            self.data_class_map[self.data_class_names[i]] = i
        self.class_priority_list = self.model_cfg['data']['class_priority']

        # Resolve execution providers based on use_gpu flag
        self._providers = self._resolve_providers(use_gpu)

        self.model_path = model_path
        self.session = None
        self.load_onnx_model(self.model_path)
        self.feature_extractor = make_image_feature_extractor(imf_model_path, self._providers)
        self.table_grid_extractor = None

        # Grid model configurations for each table_grid_model_ver are loaded
        # from resources/grid_profiles.yaml instead of being hardcoded here.
        # Expected per-entry keys: extractor_class, filename, connection_mode,
        # h_on_threshold, v_on_threshold, nms_min_dist.
        # connection_mode meaning (same semantics as before):
        #   true  -> uses a separate table_conn_model_path
        #   null  -> conn_onnx_path is passed explicitly as None
        #   false -> extractor_cls is constructed with just the grid model path
        grid_profiles = resolve_grid_profiles(script_dir)

        # Normalize alias for V4
        if table_grid_model_ver == 'V4':
            table_grid_model_ver = 'V4-EP'

        self.table_grid_extractor = None

        if table_grid_model_ver in grid_profiles:
            grid_cfg = grid_profiles[table_grid_model_ver]
            extractor_cls = GRID_EXTRACTOR_REGISTRY[grid_cfg['extractor_class']]
            default_filename = grid_cfg['filename']
            connection_mode = grid_cfg.get('connection_mode', False)
            h_thresh = grid_cfg['h_on_threshold']
            v_thresh = grid_cfg['v_on_threshold']
            nms_min_dist = grid_cfg.get('nms_min_dist')

            # Resolve model paths
            table_grid_model_path = table_grid_path if table_grid_path is not None else f'{script_dir}/resources/onnx/{default_filename}'

            # Build kwargs dictionary
            kwargs = {
                'h_on_threshold': h_thresh,
                'v_on_threshold': v_thresh,
            }

            # Add nms_min_dist only if it's not None
            if nms_min_dist is not None:
                kwargs['nms_min_dist'] = nms_min_dist

            # Initialize extractors based on connection architecture
            if connection_mode is True:
                table_conn_model_path = f'{script_dir}/resources/onnx/table_grid_model_v2_conn.onnx'
                self.table_grid_extractor = extractor_cls(
                    table_grid_model_path, table_conn_model_path,
                    **kwargs
                )
            elif connection_mode is None:
                self.table_grid_extractor = extractor_cls(
                    table_grid_model_path, conn_onnx_path=None,
                    **kwargs
                )
            else:
                self.table_grid_extractor = extractor_cls(table_grid_model_path, **kwargs)
        else:
            supported_vers = ", ".join(grid_profiles.keys()) + ", V4"
            raise ValueError(f"Invalid table_grid_model_ver. Supported versions are: {supported_vers}")


        self.markdown_generator = MarkdownGenerator(self)
        self.markdown_html_table_generator = MarkdownHTMLTableGenerator(self)
        self.html_generator = HTMLGenerator(self)
        self.use_sort = use_sort
        self.sorter = DefaultSorter()
        self._onnx_input_names = self._build_onnx_input_names()

    @staticmethod
    def _resolve_providers(use_gpu):
        """
        Return the ONNX Runtime execution provider list.

        When use_gpu is True, checks available providers and selects
        CUDAExecutionProvider if present, otherwise falls back to
        CPUExecutionProvider with a warning.

        Args:
            use_gpu (bool): Whether to request GPU execution.

        Returns:
            list[str]: Ordered provider list for ort.InferenceSession.
        """
        if not use_gpu:
            return ['CPUExecutionProvider']

        available = ort.get_available_providers()
        if 'CUDAExecutionProvider' in available:
            return ['CUDAExecutionProvider', 'CPUExecutionProvider']

        print(
            "Warning: CUDAExecutionProvider is not available "
            f"(available: {available}). Falling back to CPUExecutionProvider."
        )
        return ['CPUExecutionProvider']

    def load_onnx_model(self, model_path):
        ort.set_default_logger_severity(3)
        self.session = make_session(model_path, self._providers)

    def _build_onnx_input_names(self):
        """
        Determine the ONNX session input name list from the model config.

        Called once in __init__ and stored as self._onnx_input_names so that
        predict() does not repeat the config dict lookups on every call.

        Returns:
            list[str]: Ordered list of input tensor names for the ONNX session.
        """
        model_option = self.model_cfg['model']['option']
        model_type = model_option['conv_type']
        if isinstance(model_type, list):
            model_type = model_type[0]

        if model_type in ('GAT', 'NNConv'):
            return ["x", "edge_index", "edge_attr", "rf_features", "text_patterns"]
        elif model_type in ('CustomDGC', 'CustomDGC-PMP'):
            names = ["x", "edge_index", "edge_attr", 'k', 'batch']
            feature_types = model_option.get('feature_types', [])
            if 'rf' in feature_types:
                names.append("rf_features")
            if 'text_pattern' in feature_types:
                names.append("text_patterns")
            if 'image' in feature_types:
                names.append("image_features")
            return names
        else:
            raise ValueError(f'Not supported model_type = {model_type}!')


    @staticmethod
    def _page_to_image(page):
        """Rasterize a PyMuPDF page to an (H, W, C) uint8 numpy array."""
        pix = page.get_pixmap()
        bytes_data = np.frombuffer(pix.samples, dtype=np.uint8)
        return bytes_data.reshape(pix.height, pix.width, pix.n)

    def is_image_page(self, page):
        """
        Determine whether the page contains significant non-picture content
        detected by the image model (e.g. text, tables printed as images).

        Follows the unified contract of ImageFeatureExtractorV1/V2: predict()
        is called here exactly once for this page, and its result is
        immediately consumed by is_image_page(data_dict) within this same
        call -- never left cached across calls.

        Typical usage::

            if model.is_image_page(page):
                page = run_ocr(page)   # user-side OCR + page modification
            result = model.predict(page)

        Args:
            page: PyMuPDF page object

        Returns:
            bool  - True if the model detects non-picture layout elements.
            False - if feature_extractor is unavailable or no such elements found.
        """
        if self.feature_extractor is None:
            return False

        page_img = self._page_to_image(page)
        self.feature_extractor.predict(page_img)
        self.feature_extractor.mark_cached()

        data_dict = create_input_data_from_page(page, options={
            'input_type': self.input_type,
            'feature_set_name': self.feature_set_name,
            'page_img': page_img,
        })
        return self.feature_extractor.is_image_page(data_dict)

    # ------------------------------------------------------------------
    # Post-processing helpers
    # ------------------------------------------------------------------

    def _make_synthetic_group(self, bbox, class_name):
        """Build a synthetic group dict for a given bbox and class name."""
        return {
            'group_bbox': list(bbox),
            'group_class': self.data_class_names.index(class_name),
            'class_name': class_name,
            'indicies': [],
            'score': 1.0,
            'table_grid': None,
            'table_cells': None,
            'synthetic': True,
        }

    def _apply_post_seg_image(self, det_result, groups):
        """
        Replace GNN picture detections that overlap with segmentation-based
        picture detections produced by the image feature extractor.

        Skips any segmentation detection that overlaps a non-picture GNN result.
        For each accepted segmentation detection, removes overlapping GNN picture
        entries and inserts a synthetic picture group in their place.

        Args:
            det_result (list[list]): Current detection list, mutated in place.
            groups (list[dict]): Current group list, mutated in place.

        Returns:
            tuple[list, list]: Updated (det_result, groups).
        """
        if self.feature_extractor is None:
            return det_result, groups

        picture_detections = self.feature_extractor.get_picture_detections()
        if not picture_detections:
            return det_result, groups

        for p_box in picture_detections:
            has_non_picture_overlap = False
            for d in det_result:
                if d[4] == 'picture':
                    continue
                d_box = d[:4]
                if compute_iou(d_box, p_box) >= 0.5:
                    has_non_picture_overlap = True
                    break
                if (p_box[0] <= d_box[0] and d_box[2] <= p_box[2] and
                        p_box[1] <= d_box[1] and d_box[3] <= p_box[3]):
                    has_non_picture_overlap = True
                    break
                if (d_box[0] <= p_box[0] and p_box[2] <= d_box[2] and
                        d_box[1] <= p_box[1] and p_box[3] <= d_box[3]):
                    has_non_picture_overlap = True
                    break
            if has_non_picture_overlap:
                continue

            # Remove overlapping picture detections from det_result and groups.
            removed_bboxes = {
                tuple(d[:4]) for d in det_result
                if d[4] == 'picture' and (
                        compute_iou(d[:4], p_box) >= 0.5 or
                        (p_box[0] <= d[0] and d[2] <= p_box[2] and
                         p_box[1] <= d[1] and d[3] <= p_box[3])
                )
            }
            det_result = [d for d in det_result if tuple(d[:4]) not in removed_bboxes]
            groups = [g for g in groups if tuple(g['group_bbox'][:4]) not in removed_bboxes]

            groups.append(self._make_synthetic_group(p_box, 'picture'))
            det_result.append([p_box[0], p_box[1], p_box[2], p_box[3], 'picture'])

        return det_result, groups

    def _filter_img_bboxes(self, det_result, groups, img_bboxes):
        """
        Add embedded image regions as synthetic picture detections when valid.

        Validity is evaluated in this order (condition 2 takes priority when
        both 2 and 3 apply simultaneously):

          Condition 0 (pre-filter): skip background images.
            A background image contains non-picture detections inside it.

          Condition 2: the image bbox expanded by 20 px on each side contains
            at least one picture detection whose center lies in that region.
            Action: remove those picture detections and replace with img_bbox.

          Condition 3: not a background image and no detections of any class
            lie inside the img_bbox.
            Action: add img_bbox as a synthetic picture entry as-is.

        picture_dets and non_picture_dets are split once before the loop and
        kept in sync incrementally: non_picture_dets never changes (condition 0
        only removes pictures), and picture_dets is rebuilt only when a
        condition-2 removal occurs.

        Args:
            det_result (list[list]): Current detection list.
            groups (list[dict]): Current group list.
            img_bboxes (list): Bboxes from page.get_image_info().

        Returns:
            tuple[list, list]: Updated (det_result, groups).
        """
        _EXPAND_PX = 20

        def _center(bbox):
            return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0

        def _contains_center(expanded, bbox):
            cx, cy = _center(bbox)
            return expanded[0] <= cx <= expanded[2] and expanded[1] <= cy <= expanded[3]

        def _bbox_inside(outer, inner):
            return (outer[0] <= inner[0] and inner[2] <= outer[2] and
                    outer[1] <= inner[1] and inner[3] <= outer[3])

        # Split once; non_picture_dets never mutates during the loop.
        picture_dets     = [d for d in det_result if d[4] == 'picture']
        non_picture_dets = [d for d in det_result if d[4] != 'picture']

        for img_b in img_bboxes:
            ix0, iy0, ix1, iy1 = img_b[0], img_b[1], img_b[2], img_b[3]
            img_box = [ix0, iy0, ix1, iy1]

            # Condition 0: skip background images using the smaller non-picture list.
            if any(_bbox_inside(img_box, d[:4]) for d in non_picture_dets):
                continue

            expanded_box = [ix0 - _EXPAND_PX, iy0 - _EXPAND_PX,
                            ix1 + _EXPAND_PX, iy1 + _EXPAND_PX]

            picture_in_expanded = [
                d for d in picture_dets
                if _contains_center(expanded_box, d[:4])
            ]

            if picture_in_expanded:
                # Condition 2: replace nearby picture detections with img_bbox.
                removed_keys = {tuple(d[:4]) for d in picture_in_expanded}
                det_result   = [d for d in det_result if tuple(d[:4]) not in removed_keys]
                groups       = [g for g in groups if tuple(g['group_bbox'][:4]) not in removed_keys]

                new_det = [ix0, iy0, ix1, iy1, 'picture']
                groups.append(self._make_synthetic_group(img_box, 'picture'))
                det_result.append(new_det)

                # Invalidate picture_dets cache: rebuild from updated det_result.
                picture_dets = [d for d in det_result if d[4] == 'picture']
            else:
                # Condition 3: empty image region; add as-is.
                if not any(_bbox_inside(img_box, d[:4]) for d in picture_dets):
                    new_det = [ix0, iy0, ix1, iy1, 'picture']
                    groups.append(self._make_synthetic_group(img_box, 'picture'))
                    det_result.append(new_det)
                    picture_dets.append(new_det)   # append-only; no full rebuild needed

        return det_result, groups

    # ------------------------------------------------------------------
    # Vector expansion geometry helpers (used by _expand_by_vectors)
    # ------------------------------------------------------------------

    @staticmethod
    def _vec_center(r):
        """Return (cx, cy) of a fitz.Rect."""
        return (r.x0 + r.x1) / 2.0, (r.y0 + r.y1) / 2.0

    @staticmethod
    def _center_inside(cx, cy, bbox):
        """Return True when point (cx, cy) lies inside bbox."""
        return bbox[0] <= cx <= bbox[2] and bbox[1] <= cy <= bbox[3]

    @staticmethod
    def _clip_to_page(x0, y0, x1, y1, page_x0, page_y0, page_x1, page_y1):
        """Clip bbox coordinates to page boundaries."""
        return (max(x0, page_x0), max(y0, page_y0),
                min(x1, page_x1), min(y1, page_y1))

    @staticmethod
    def _expand_pct(x0, y0, x1, y1, pct):
        """Return bbox expanded by pct fraction on each side."""
        dw = (x1 - x0) * pct
        dh = (y1 - y0) * pct
        return x0 - dw, y0 - dh, x1 + dw, y1 + dh

    @staticmethod
    def _union_with_rects(x0, y0, x1, y1, rects):
        """Extend bbox to contain all fitz.Rect objects in rects."""
        for r in rects:
            x0 = min(x0, r.x0)
            y0 = min(y0, r.y0)
            x1 = max(x1, r.x1)
            y1 = max(y1, r.y1)
        return x0, y0, x1, y1

    @staticmethod
    def _push_back_expansion(ex0, ey0, ex1, ey1, ox0, oy0, ox1, oy1, blockers):
        """
        Roll back expanded sides that would collide with blocker bboxes.
        Only sides that were actually expanded are eligible for rollback.
        """
        for d in blockers:
            dx0, dy0, dx1, dy1 = d[0], d[1], d[2], d[3]
            if ex0 < ox0 and ex0 < dx1 <= ox0:
                ex0 = max(ex0, dx1)
            if ex1 > ox1 and ox1 <= dx0 < ex1:
                ex1 = min(ex1, dx0)
            if ey0 < oy0 and ey0 < dy1 <= oy0:
                ey0 = max(ey0, dy1)
            if ey1 > oy1 and oy1 <= dy0 < ey1:
                ey1 = min(ey1, dy0)
        return ex0, ey0, ex1, ey1

    @staticmethod
    def _sync_group_bbox(groups, orig_key, nx0, ny0, nx1, ny1):
        """Update the matching group's bbox to new coordinates."""
        for g in groups:
            gb = g['group_bbox']
            if (gb[0], gb[1], gb[2], gb[3]) == orig_key:
                gb[0], gb[1], gb[2], gb[3] = nx0, ny0, nx1, ny1
                break

    def _expand_by_vectors(self, det_result, groups, page):
        """
        Expand picture and table bboxes to absorb nearby vector graphics.

        Picture bboxes are expanded by 15% on each side; vectors whose center
        falls in that zone are absorbed, but the expansion is rolled back on any
        side that would overlap a non-picture detection.

        Table bboxes are expanded by 5% on each side; both general vectors and
        thin ruling lines (short side <= 3 px, aspect ratio >= 1:5) are
        considered. No collision guard is applied for tables.

        All resulting bboxes are clipped to the page boundary. Both det_result
        and groups are updated in place.

        Args:
            det_result (list[list]): Current detection list, mutated in place.
            groups (list[dict]): Current group list, mutated in place.
            page: PyMuPDF page object.

        Returns:
            tuple[list, list]: Updated (det_result, groups).
        """
        page_rect = page.rect
        page_x0, page_y0 = page_rect.x0, page_rect.y0
        page_x1, page_y1 = page_rect.x1, page_rect.y1
        page_area = page_rect.width * page_rect.height # Calculate total page area

        raw_drawings = [page_rect & p["rect"] for p in page.get_drawings()]

        # Filter out excessively large vectors (likely background elements)
        # Filter if vector area exceeds 50% of the page area
        _MAX_VECTOR_PAGE_AREA_RATIO = 0.5
        filtered_drawings = []
        for v in raw_drawings:
            if v.is_empty:
                continue
            if (v.width * v.height) / page_area > _MAX_VECTOR_PAGE_AREA_RATIO:
                continue
            filtered_drawings.append(v)
        raw_drawings = filtered_drawings


        # General vectors: both dimensions must exceed 2 px.
        _VEC_MIN_DIM = 2
        vectors = [v for v in raw_drawings
                   if v.width > _VEC_MIN_DIM and v.height > _VEC_MIN_DIM]

        # Table line vectors: thin ruled lines excluded from general vectors.
        # Condition: short side <= 3 px AND aspect ratio >= 1:5.
        _LINE_MAX_THIN = 3
        _LINE_MIN_RATIO = 5
        table_line_vectors = [
            v for v in raw_drawings
            if (v.width > 0 and v.height > 0
                and (v.width <= _LINE_MAX_THIN or v.height <= _LINE_MAX_THIN)
                and max(v.width, v.height) / min(v.width, v.height) >= _LINE_MIN_RATIO)
        ]

        # Discard vectors whose center lies inside any picture detection (both sets).
        picture_dets = [d for d in det_result if d[4] == 'picture']
        vectors = [v for v in vectors
                   if not any(self._center_inside(*self._vec_center(v), d[:4])
                               for d in picture_dets)]
        table_line_vectors = [v for v in table_line_vectors
                              if not any(self._center_inside(*self._vec_center(v), d[:4])
                                         for d in picture_dets)]

        non_picture_dets = [d for d in det_result if d[4] != 'picture']

        # Expand picture bboxes (15%, with push-back against non-picture dets).
        for det in det_result:
            if det[4] != 'picture':
                continue

            ox0, oy0, ox1, oy1 = det[0], det[1], det[2], det[3]
            sx0, sy0, sx1, sy1 = self._expand_pct(ox0, oy0, ox1, oy1, 0.15)

            nearby = [v for v in vectors
                      if self._center_inside(*self._vec_center(v), [sx0, sy0, sx1, sy1])]
            if not nearby:
                continue

            nx0, ny0, nx1, ny1 = self._union_with_rects(ox0, oy0, ox1, oy1, nearby)
            nx0, ny0, nx1, ny1 = self._push_back_expansion(
                nx0, ny0, nx1, ny1, ox0, oy0, ox1, oy1, non_picture_dets)
            nx0, ny0, nx1, ny1 = self._clip_to_page(
                nx0, ny0, nx1, ny1, page_x0, page_y0, page_x1, page_y1)

            det[0], det[1], det[2], det[3] = nx0, ny0, nx1, ny1
            self._sync_group_bbox(groups, (ox0, oy0, ox1, oy1), nx0, ny0, nx1, ny1)
        # Expand table bboxes (5%, general vectors + ruling lines, no push-back).
        vectors_for_table = vectors + table_line_vectors
        _MAX_TABLE_DIM_EXPANSION_FACTOR = 1.5 # Allow max 150% expansion of table width/height

        for det in det_result:
            if det[4] != 'table':
                continue

            ox0, oy0, ox1, oy1 = det[0], det[1], det[2], det[3]
            original_width = ox1 - ox0
            original_height = oy1 - oy0

            # Optional: if original table is very small or line-shaped, compensate with a minimum size
            # if original_width < 1.0: original_width = 1.0
            # if original_height < 1.0: original_height = 1.0

            max_allowed_width = original_width * _MAX_TABLE_DIM_EXPANSION_FACTOR
            max_allowed_height = original_height * _MAX_TABLE_DIM_EXPANSION_FACTOR

            sx0, sy0, sx1, sy1 = self._expand_pct(ox0, oy0, ox1, oy1, 0.05)

            nearby = [v for v in vectors_for_table
                      if self._center_inside(*self._vec_center(v), [sx0, sy0, sx1, sy1])]

            if not nearby:
                continue

            # New filtering logic: check if individual vectors would excessively expand the table
            filtered_nearby_for_expansion = []
            for v in nearby:
                # Calculate the union of the original bbox and the current vector 'v'
                union_x0 = min(ox0, v.x0)
                union_y0 = min(oy0, v.y0)
                union_x1 = max(ox1, v.x1)
                union_y1 = max(oy1, v.y1)

                union_width = union_x1 - union_x0
                union_height = union_y1 - union_y0

                # Check if width or height exceeds the allowed maximum expansion ratio
                if union_width > max_allowed_width or union_height > max_allowed_height:
                    pass # Filter this vector out
                else:
                    filtered_nearby_for_expansion.append(v)

            nearby = filtered_nearby_for_expansion # Replace with the filtered list of vectors

            if not nearby: # If no vectors remain after filtering, stop expansion
                continue

            nx0, ny0, nx1, ny1 = self._union_with_rects(ox0, oy0, ox1, oy1, nearby)
            nx0, ny0, nx1, ny1 = self._clip_to_page(
                nx0, ny0, nx1, ny1, page_x0, page_y0, page_x1, page_y1)

            det[0], det[1], det[2], det[3] = nx0, ny0, nx1, ny1
            self._sync_group_bbox(groups, (ox0, oy0, ox1, oy1), nx0, ny0, nx1, ny1)
        return det_result, groups

    def _merge_overlapping_pictures(self, det_result, groups):
        """
        Merge picture detections that overlap each other (IoU > 0.2).

        Uses Union-Find to identify connected components of overlapping
        pictures in a single O(k^2) pass, then merges each component into
        one synthetic entry whose bbox is the union of all members.
        Replaces the previous O(k^3) greedy restart loop.

        Applied unconditionally after all other post-processing steps.

        Args:
            det_result (list[list]): Current detection list.
            groups (list[dict]): Current group list.

        Returns:
            tuple[list, list]: Updated (det_result, groups).
        """
        _IOU_MERGE_THRESHOLD = 0.2

        pictures = [d for d in det_result if d[4] == 'picture']
        n = len(pictures)
        if n < 2:
            return det_result, groups

        # Path-compressed Union-Find.
        parent = list(range(n))

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(a: int, b: int) -> None:
            pa, pb = _find(a), _find(b)
            if pa != pb:
                parent[pb] = pa

        # Single O(k^2) pass to build connected components.
        for i in range(n):
            for j in range(i + 1, n):
                if compute_iou(pictures[i][:4], pictures[j][:4]) > _IOU_MERGE_THRESHOLD:
                    _union(i, j)

        # Group indices by component root.
        components: dict[int, list[int]] = {}
        for i in range(n):
            root = _find(i)
            components.setdefault(root, []).append(i)

        # Keys of all pictures that belong to a multi-member component.
        merged_keys = {tuple(pictures[i][:4]) for idxs in components.values()
                       if len(idxs) > 1 for i in idxs}

        if not merged_keys:
            return det_result, groups

        # Remove all participating picture entries from det_result and groups.
        det_result = [d for d in det_result if tuple(d[:4]) not in merged_keys]
        groups = [g for g in groups if tuple(g['group_bbox'][:4]) not in merged_keys]

        # Re-insert one merged entry per multi-member component.
        for root, idxs in components.items():
            if len(idxs) < 2:
                continue
            members = [pictures[i] for i in idxs]
            ux0 = min(m[0] for m in members)
            uy0 = min(m[1] for m in members)
            ux1 = max(m[2] for m in members)
            uy1 = max(m[3] for m in members)
            union_box = [ux0, uy0, ux1, uy1]
            groups.append(self._make_synthetic_group(union_box, 'picture'))
            det_result.append([ux0, uy0, ux1, uy1, 'picture'])

        return det_result, groups

    # ------------------------------------------------------------------
    # OCR need detection helper
    # ------------------------------------------------------------------

    def is_page_ocr_needed(self, data_dict, iou_threshold: float = 0.2) -> bool | None:
        """Delegate OCR need detection to the feature extractor.

        Uses duck-typing so any extractor version that exposes is_ocr_needed()
        is supported.

        Contract: predict() must already have been called for this page
        before calling this method (this method does not run inference
        itself). See BoxRFDGNN.predict() for the call site that guarantees
        this ordering.

        Args:
            data_dict     : output of create_input_data_from_page() for this
                            same page.
            iou_threshold : passed through to feature_extractor.is_ocr_needed().

        Returns:
            bool | None -- forwarded from the extractor; None when unavailable.
        """
        if self.feature_extractor is None or \
                not hasattr(self.feature_extractor, 'is_ocr_needed'):
            return None
        return self.feature_extractor.is_ocr_needed(data_dict, iou_threshold)

    # ------------------------------------------------------------------
    # Main inference entry point
    # ------------------------------------------------------------------

    def predict(self, page, verbose=False, data_dict=None, **kwargs):
        check_ocr_need         = kwargs.get('check_ocr_need',        False)
        ocr_need_iou_threshold = kwargs.get('ocr_need_iou_threshold', 0.2)

        # ------------------------------------------------------------------
        # Unified image-feature-extractor contract: predict() is called
        # exactly once per page, explicitly, before any of is_image_page(),
        # is_ocr_needed(), or feature extraction consume its cached result.
        # Needed whenever we build data_dict ourselves (so
        # image_feature_extraction_task() can reuse this call instead of
        # running its own) and/or whenever is_page_ocr_needed() will run
        # below (it requires fresh logits for this exact page).
        # ------------------------------------------------------------------
        page_img = None
        if self.feature_extractor is not None and (data_dict is None or check_ocr_need):
            page_img = self._page_to_image(page)
            self.feature_extractor.predict(page_img)
            self.feature_extractor.mark_cached()

        if data_dict is None:
            data_dict = create_input_data_from_page(page, options={
                'input_type': self.input_type,
                'feature_set_name': self.feature_set_name,
                'feature_extractor': self.feature_extractor,
                'page_img': page_img,
            })
        bboxes = np.array(data_dict['bboxes'], dtype=np.float32)

        # ------------------------------------------------------------------
        # OCR need detection via text segmentation vs bbox mask IoU
        # ------------------------------------------------------------------
        needs_ocr = self.is_page_ocr_needed(data_dict, ocr_need_iou_threshold) \
            if check_ocr_need else None

        return_raw = kwargs.get('return_raw', False)
        if len(bboxes) == 0:
            if return_raw:
                return {
                    'groups':    [],
                    'needs_ocr': needs_ocr,
                    'node_feat': np.empty((0, 0), dtype=np.float32),
                    'edge_feat': np.empty((0, 0), dtype=np.float32),
                }
            else:
                return []

        model_type = self.model_cfg['model']['option']['conv_type']

        # Print model type when verbose is enabled
        if verbose:
            print(">>> model_type:", model_type)

        if type(model_type) is list:
            model_type = model_type[0]

        onnx_input_names = self._onnx_input_names

        # Prepare neural network inputs from data_dict.
        x, edge_index, edge_attr, nn_index, nn_attr, rf_feature, text_feature, image_feature, image_data = \
            get_nn_input_from_datadict(data_dict, self.model_cfg, return_nn_index=('nn_index' in onnx_input_names))

        # Build ONNX inputs directly from the known name list,
        # avoiding the intermediate full-dict allocation.
        _all_inputs = {
            'x':              x,
            'edge_index':     edge_index,
            'edge_attr':      edge_attr,
            'rf_features':    rf_feature,
            'k':              np.array(min(len(bboxes), 20), dtype=np.int64),
            'text_patterns':  text_feature,
            'image_features': image_feature,
            'batch':          np.zeros(len(bboxes), dtype=np.int64),
        }
        onnx_inputs = {name: _all_inputs[name] for name in onnx_input_names}

        # Verbose: print shapes and dtypes of onnx_inputs
        if verbose:
            print(">>> onnx_inputs:")
            for name, val in onnx_inputs.items():
                arr = np.asarray(val)
                try:
                    dtype = arr.dtype
                except Exception:
                    dtype = type(val)
                print(f"  - {name}: shape={np.shape(arr)}, dtype={dtype}")

        # Verbose: print ONNX session input metadata if session exists
        if verbose and hasattr(self, 'session') and self.session is not None:
            try:
                print(">>> ONNX Runtime session inputs metadata:")
                for inp in self.session.get_inputs():
                    print(f"  - name={inp.name}, shape={inp.shape}, type={inp.type}")
                print(">>> ONNX Runtime session outputs metadata:")
                for out in self.session.get_outputs():
                    print(f"  - name={out.name}, shape={out.shape}, type={out.type}")
            except Exception as e:
                print("  (Failed to read session metadata):", e)

        # Run the ONNX model
        ort_outputs = self.session.run(None, onnx_inputs)
        onnx_node_logits, onnx_edge_logits, onnx_node_feat, onnx_edge_feat = ort_outputs

        # Convert node logits to probabilities by applying softmax
        exp_node_logits = np.exp(onnx_node_logits - np.max(onnx_node_logits, axis=1, keepdims=True))
        node_probs = exp_node_logits / np.sum(exp_node_logits, axis=1, keepdims=True)

        # Predicted node labels and scores
        predicted_node_label = np.argmax(node_probs, axis=1)
        predicted_node_score = node_probs[np.arange(node_probs.shape[0]), predicted_node_label]

        # Edge prediction
        edge_threshold = kwargs.get('edge_threshold', 0.55)
        if onnx_edge_logits.size > 0:
            exp_edge_logits = np.exp(onnx_edge_logits - np.max(onnx_edge_logits, axis=1, keepdims=True))
            edge_probs = exp_edge_logits / np.sum(exp_edge_logits, axis=1, keepdims=True)
            predicted_edge_labels = (edge_probs[:, 1] > edge_threshold).astype(np.int64)
        else:
            predicted_edge_labels = np.empty(0, dtype=np.int64)

        num_nodes = len(predicted_node_label)
        edge_matrix = get_edge_matrix(num_nodes, edge_index, predicted_edge_labels)
        groups = group_node_by_edge(predicted_node_label, predicted_node_score, edge_matrix, bboxes, self.class_priority_list)

        for group in groups:
            group['class_name'] = self.data_class_names[group['group_class']]

        # Build det_result from groups. This list will be updated by post-processing steps
        # and needs to be kept in sync with 'groups'.
        det_result = []
        for group in groups:
            g_bbox = group['group_bbox'][:]
            g_bbox.append(group['class_name'])
            det_result.append(g_bbox)

        # Post-processing: add image regions and refine bboxes with vector graphics
        filter_img_bboxes = kwargs.get('filter_img_bboxes', False)
        expand_by_vectors = kwargs.get('expand_by_vectors', False)
        do_sort = kwargs.get('do_sort', True)

        if self.input_type is not None and 'post-seg-image' in self.input_type:
            det_result, groups = self._apply_post_seg_image(det_result, groups)
        else:
            if filter_img_bboxes:
                img_bboxes = [itm["bbox"] for itm in page.get_image_info()]
                det_result, groups = self._filter_img_bboxes(det_result, groups, img_bboxes)
            if expand_by_vectors:
                det_result, groups = self._expand_by_vectors(det_result, groups, page)

        # Always applied: merge overlapping picture detections.
        det_result, groups = self._merge_overlapping_pictures(det_result, groups)

        if self.use_sort and do_sort:
            order      = self.sorter.sort(page, groups, det_result)
            groups     = [groups[i]     for i in order]
            det_result = [det_result[i] for i in order]
            
        for group in groups:
            cls_name = group['class_name']

            # Attach the raw pymupdf-extracted bboxes (before layout grouping)
            # that were merged into this group, for detailed inspection via
            # return_raw. Note this only covers bboxes that ended up in some
            # group, not raw bboxes that were dropped as noise.
            group['bboxes'] = [list(data_dict['bboxes'][i]) for i in group['indicies']]

            if self.table_grid_extractor is not None and cls_name == 'table':
                page_image = data_dict['image']

                # Use the FINAL group_bbox for cropping and coordinate conversion
                crop_x0, crop_y0, crop_x1, crop_y1 = group['group_bbox'][:4]

                # Ensure bbox coordinates are valid integers for slicing
                crop_y0_int = max(0, int(crop_y0))
                crop_y1_int = min(page_image.shape[0], int(crop_y1))
                crop_x0_int = max(0, int(crop_x0))
                crop_x1_int = min(page_image.shape[1], int(crop_x1))

                crop_img = page_image[crop_y0_int:crop_y1_int, crop_x0_int:crop_x1_int]

                # Collect bboxes and texts of lines within this table group,
                # then convert from page space to crop space.
                group_indices = group['indicies']
                group_original_bboxes = np.array([data_dict['bboxes'][i] for i in group_indices])
                group_texts = [data_dict['text'][i] for i in group_indices]

                crop_bboxes = group_original_bboxes.copy()
                crop_bboxes[:, 0] -= crop_x0
                crop_bboxes[:, 2] -= crop_x0
                crop_bboxes[:, 1] -= crop_y0
                crop_bboxes[:, 3] -= crop_y0

                # If crop_img is empty (e.g., due to extreme expansion clipping), handle gracefully
                if crop_img.shape[0] == 0 or crop_img.shape[1] == 0:
                    group['table_grid'] = None
                    group['table_cells'] = []
                else:
                    grid, cells = self.table_grid_extractor.predict(
                        crop_img, crop_bboxes,
                        texts=group_texts,
                    )
                    group['table_grid'] = grid
                    group['table_cells'] = cells

        # ------------------------------------------------------------------
        # Return
        # ------------------------------------------------------------------
        if return_raw:
            return {
                'groups':    groups,
                'needs_ocr': needs_ocr,
                'node_feat': onnx_node_feat,
                'edge_feat': onnx_edge_feat,
            }

        return det_result

    def to_markdown(self, page, groups=None, join: bool = True, skip_header_footer: bool = True) -> str:
        """Convert page layout detection result to Markdown text."""
        return self.markdown_generator.generate(page, groups=groups, join=join, skip_header_footer=skip_header_footer)

    def to_markdown_html_table(self, page, groups=None, join: bool = True, skip_header_footer: bool = True) -> str:
        """Convert page layout detection result to Markdown with HTML tables."""
        return self.markdown_html_table_generator.generate(page, groups=groups, join=join, skip_header_footer=skip_header_footer)

    def to_html(self, page, skip_header_footer: bool = True) -> str:
        """Convert page layout detection result to a complete HTML document."""
        return self.html_generator.generate(page, skip_header_footer=skip_header_footer)

