"""CNN4GNNLocalExpertModel: local expert models behind a frozen CNN4GNN.

The CNN4GNN network is trained once and then frozen. This wrapper keeps its
public behavior (every call is delegated) and adds an optional second stage:
a small model for the nodes and/or a small model for the edges that
re-scores the CNN4GNN logits from extra features. The decoding and grouping
code is the one of CNN4GNN and is not touched; only the logits it receives
can differ.

How it works
------------
CNN4GNN has a hook (CNN4GNN.logits_hook) that is called after the network
and before the decoding. The wrapper installs its own hook for the duration
of predict(). For each of nodes and edges that has a second stage model:

    proba = softmax(cnn4gnn logits)
    X     = [proba, extra features]          (extra features from the
                                              user supplied builder)
    new_logits = model.predict_logits(X, cnn4gnn logits)

A side without a model keeps the CNN4GNN output. Nothing is required of a
model except one method, predict_logits(X, logits) -> array of the shape of
logits. MeanResidualTree (MeanResidualTree.py) is the reference
implementation.

Feature builders
----------------
node_features(ctx) -> (N, d) array or None, edge_features(ctx) -> (E, d)
array or None. ctx is the dict CNN4GNN passes to its hook:

    node_logits, edge_logits, node_proba, edge_proba   (proba added here)
    node_head_in, edge_head_in      network features; None unless the stage2
                                    ONNX was exported with return_features
    edge_index (2, E), bboxes_px, bboxes_pt, union_bboxes, texts,
    text_patterns, margin_features, nn_geo_features, det_cls, det_reg,
    det_ctr, page (the page or image given to predict), img_size

Anything that can be computed from the page, such as font sizes or
distances, can be a feature. A builder must return finite values with one
row per node (or per edge).

Deployment
----------
This file (with MeanResidualTree / MeanResidualMLP) and the model.zip are all
that a deployment needs of the correction tool; the training code is
CNN4GNNLocalExpertTrain.py and is not needed. A fix is one
model.zip written by the training. It holds, for the nodes and for the
edges, one model per document category (a model works only on the rows
whose document_type column equals its category; every other row keeps the
CNN4GNN output):

    from CNN4GNNLocalExpertModel import CNN4GNNLocalExpertModel
    model = CNN4GNNLocalExpertModel.from_zip(net, 'model.zip')
    groups = model.predict(page)

model.zip is the one file of a deployment: it holds the models, the
document classifier and the python file of the features (features.py, the
FeatureSet the models were trained with). from_zip(net, 'model.zip',
features) uses a FeatureSet of the caller instead. The features are defined
by the engineer (see FeatureSet); nothing in this file imports a feature
module. The loading of the models, the meta files and the check of the
feature list and the class list recorded in the zip against the FeatureSet
are done here, the same for every FeatureSet. The features file is code and
is imported when the zip is loaded.

Settings of the network (edge_threshold and so on) live on the wrapped
object: wrapper.cnn4gnn.edge_threshold = 0.6. Unknown attributes are read
from it, so the wrapper can be used where a CNN4GNN is expected.
"""

import dataclasses
import hashlib
import json
import os
import shutil
import sys
import tempfile
import warnings
import zipfile

import numpy as np

import importlib


def _sibling(name):
    """A module of the package this file is in (cnn4gnn.layout.onnx or
    source.layout.onnx, whichever has it), or a plain module next to this
    file when it is run as a script."""
    if __package__:
        return importlib.import_module(f'.{name}', __package__)
    return importlib.import_module(name)


MeanResidualTree = _sibling('MeanResidualTree').MeanResidualTree
MeanResidualMLP = _sibling('MeanResidualMLP').MeanResidualMLP


ZIP_FILE = 'model.zip'                 # all the models of a fix in one file
ZIP_FORMAT = 'cnn4gnn_correction_zip'
ZIP_VERSION = 1


class CategoryModels:
    """Several models of one side that share the document_type column: the
    model of category k works only on the rows where that column rounds to
    k (it was fitted with gate_value=k), every other row keeps the CNN4GNN
    output. Same interface as a single model (predict_logits(X, logits)),
    so the rest of the code does not need to know about the categories."""

    def __init__(self, gate_col, models):
        if gate_col < 0 or not models:
            raise ValueError('CategoryModels needs a gate column and at '
                             'least one model')
        self.gate_col = int(gate_col)
        self.models = {int(k): m for k, m in sorted(models.items())}
        first = next(iter(self.models.values()))
        self.n_features_, self.n_classes_ = first.n_features_, first.n_classes_

    @property
    def categories(self):
        return list(self.models)

    def predict_logits(self, X, logits):
        X = np.asarray(X, np.float32)
        out = np.array(logits, np.float32, copy=True)
        if len(X) == 0:
            return out
        cat = np.rint(X[:, self.gate_col]).astype(np.int64)
        for k, m in self.models.items():
            idx = np.nonzero(cat == k)[0]
            if len(idx):
                out[idx] = m.predict_logits(X[idx], out[idx])
        return out

    def predict_proba(self, X, logits):
        z = self.predict_logits(X, logits).astype(np.float64)
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def n_leaves(self):
        return sum(m.n_leaves() for m in self.models.values())

    def export_text(self, feature_names=None, class_names=None):
        parts = []
        for k, m in self.models.items():
            parts.append(f'=== document category {k} ===\n'
                         + m.export_text(feature_names, class_names))
        return '\n\n'.join(parts)


# ---------------------------------------------------------------------------
# the features: defined by the engineer, handed over as a FeatureSet
# ---------------------------------------------------------------------------

EDGE_CLASS_NAMES = ['not_connected', 'connected']
META_FORMAT = 'CNN4GNNPdfNodeModel'    # the name of the first meta files
META_VERSION = 1
MODEL_TYPES = ('mean_tree', 'mean_mlp')


def softmax_f32(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


def tree_input(logits, features):
    """[softmax(logits), features] as float32: exactly the input that
    CNN4GNNLocalExpertModel hands to the node (and edge) model."""
    return np.concatenate([softmax_f32(logits),
                           np.asarray(features, dtype=np.float32)], axis=1)


def meta_path(model_path):
    return model_path + '.meta.json'


def load_model_file(path, model_type=None):
    """A model file: model_type 'mean_tree' (json) or 'mean_mlp' (onnx);
    None picks it from the extension."""
    if model_type is None:
        model_type = ('mean_mlp' if str(path).lower().endswith('.onnx')
                      else 'mean_tree')
    if model_type == 'mean_tree':
        return MeanResidualTree.load(path)
    if model_type == 'mean_mlp':
        return MeanResidualMLP.load(path)
    raise ValueError(f'unknown model_type {model_type!r}, expected one of '
                     f'{MODEL_TYPES}')


@dataclasses.dataclass
class FeatureSet:
    """The features that the second stage models use, defined by the
    engineer. This is all that has to be written:

    node_features(ctx) -> (N, d) float array, one row per node
    edge_features(ctx) -> (E, d) float array, one row per edge
    node_feature_names, edge_feature_names : the d column names of each

    Both functions must give finite numbers and the same columns for every
    page. Their ctx is the one of the logits hook of CNN4GNN
    (CNN4GNNLocalExpertModel doc) but the training has only these keys, so
    a function may use only them:
        node_features: page (pymupdf page, upright), bboxes_pt, texts
        edge_features: edge_index (2, E), bboxes_pt, texts, edge_logits,
                       node_logits (no page: the training has no PDF there)
    Functions must be defined at the top level of a module (they are
    pickled when the training uses worker processes).

    Optional: version (a string; a model trained with another version is
    refused), node_env / edge_env (a dict, or a function giving one, of
    what the values depend on, for example library versions; the models
    keep it and a deployment warns when it differs), geometry_names (the
    node feature columns that are made from the box; the training leaves
    them out when the setting use_bbox is off).

    The input of a model is [softmax(CNN4GNN logits), features] and, when a
    document classifier is used, the column document_type at the end."""

    node_features: object
    edge_features: object
    node_feature_names: list
    edge_feature_names: list
    version: str = ''
    node_env: object = None
    edge_env: object = None
    geometry_names: tuple = ()

    def __post_init__(self):
        for nm in ('node_features', 'edge_features'):
            if not callable(getattr(self, nm)):
                raise TypeError(f'FeatureSet.{nm} must be callable')
        for nm in ('node_feature_names', 'edge_feature_names'):
            names = getattr(self, nm)
            if (not isinstance(names, (list, tuple)) or not names
                    or not all(isinstance(x, str) for x in names)
                    or len(set(names)) != len(names)):
                raise ValueError(f'FeatureSet.{nm} must be a list of '
                                 f'different strings')
            setattr(self, nm, list(names))
        self.geometry_names = tuple(self.geometry_names)
        bad = [g for g in self.geometry_names
               if g not in self.node_feature_names]
        if bad:
            raise ValueError(f'geometry_names {bad} are not node features')
        self.version = str(self.version)

    # -- making one from what a caller has --------------------------------

    @classmethod
    def from_any(cls, obj):
        """A FeatureSet from a FeatureSet (or an object with its fields),
        a dict of its fields, a module
        (or object) with an attribute FEATURES, or an old style feature
        module (node_features, edge_features, FEATURE_NAMES,
        EDGE_FEATURE_NAMES)."""
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, dict):
            return cls(**obj)
        fields = [f.name for f in dataclasses.fields(cls)]
        if all(hasattr(obj, n) for n in fields[:4]):
            # a FeatureSet of the same shape from another import of this
            # file (for example the same module under two package names)
            return cls(**{n: getattr(obj, n) for n in fields
                          if hasattr(obj, n)})
        fs = getattr(obj, 'FEATURES', None)
        if fs is not None:
            return cls.from_any(fs)
        if all(hasattr(obj, a) for a in ('node_features', 'edge_features',
                                         'FEATURE_NAMES',
                                         'EDGE_FEATURE_NAMES')):
            env = getattr(obj, 'pdf_feature_env', None)
            return cls(obj.node_features, obj.edge_features,
                       list(obj.FEATURE_NAMES), list(obj.EDGE_FEATURE_NAMES),
                       node_env=env,
                       geometry_names=tuple(getattr(obj, 'GEOMETRY_NAMES',
                                                    ())))
        raise TypeError(
            'features must be a FeatureSet, a dict of its fields or a '
            'module with an attribute FEATURES, got '
            f'{type(obj).__name__}')

    @classmethod
    def from_spec(cls, spec):
        """A FeatureSet named by a string:
            'package.module'            its attribute FEATURES
            'package.module:name'       that attribute
            '/path/features.py'         a file; its attribute FEATURES
            '/path/features.py:name'    that attribute
        A file is imported as a module under its file name, with its
        directory on sys.path, so worker processes can import it too (the
        functions are pickled by module name)."""
        import importlib
        import re
        spec = str(spec).strip()
        m = re.fullmatch(r'(.+\.py)(?::(\w+))?', spec)
        if m:
            path = os.path.abspath(m.group(1))
            if not os.path.isfile(path):
                raise FileNotFoundError(f'features file not found: {path}')
            import sys
            d = os.path.dirname(path)
            if d not in sys.path:
                sys.path.insert(0, d)
            mod = importlib.import_module(
                os.path.splitext(os.path.basename(path))[0])
            if os.path.abspath(mod.__file__) != path:
                raise ImportError(
                    f'{path}: a module of the same name is already '
                    f'imported from {mod.__file__}; rename the file')
            attr = m.group(2)
        else:
            modname, _, attr = spec.partition(':')
            mod = importlib.import_module(modname)
        return cls.from_any(getattr(mod, attr or 'FEATURES'))

    # -- the source file, for the model.zip -------------------------------

    def source_snapshot(self):
        """The python file that defines these features, read now: dict
        (attr, text, sha256, path, version). The file is saved inside the
        model.zip (FEATURES_FILE), so a model and its features always go
        together. Raises ValueError when the features cannot be saved as one
        file: the two functions must be defined in one .py file that also
        holds the FeatureSet as a module attribute, and that file must not
        import other files of its own directory."""
        mods = {getattr(f, '__module__', None)
                for f in (self.node_features, self.edge_features)}
        if len(mods) != 1 or None in mods:
            raise ValueError('node_features and edge_features must be '
                             'functions defined in one module')
        name = mods.pop()
        mod = sys.modules.get(name)
        path = getattr(mod, '__file__', None)
        if not path or not path.endswith('.py') or not os.path.isfile(path):
            raise ValueError(f'the features module {name!r} is not a .py '
                             f'file (functions of __main__ or of a '
                             f'notebook cannot be saved)')
        attr = next((k for k, v in vars(mod).items()
                     if v is self or (isinstance(v, FeatureSet)
                                      and v.node_features
                                      is self.node_features
                                      and v.edge_features
                                      is self.edge_features)), None)
        if attr is None:
            raise ValueError(f'{path}: the FeatureSet must be a module '
                             f'attribute (FEATURES = FeatureSet(...)) of '
                             f'the file that defines the two functions')
        local = _local_imports(path)
        if local:
            raise ValueError(
                f'{path} imports other files of its directory '
                f'({", ".join(sorted(local))}); the features are saved as '
                f'this one file, so put everything in it')
        with open(path, 'rb') as fh:
            raw = fh.read()
        return dict(attr=attr, text=raw.decode('utf-8'),
                    sha256=hashlib.sha256(raw).hexdigest(), path=path,
                    version=self.version)

    # -- names and checks -------------------------------------------------

    def names(self, side):
        return (self.node_feature_names if side == 'node'
                else self.edge_feature_names)

    def builder(self, side):
        return self.node_features if side == 'node' else self.edge_features

    def input_names(self, side, class_names):
        """The names of the model input columns (without document_type)."""
        return [f'proba_{c}' for c in class_names] + list(self.names(side))

    def env(self, side):
        e = self.node_env if side == 'node' else self.edge_env
        e = e() if callable(e) else e
        return dict(e) if e else {}

    def check(self, side, X, n_rows):
        """X as a float32 array after the checks: shape (n_rows, d) with d
        the number of names, all values finite."""
        X = np.asarray(X, dtype=np.float32)
        d = len(self.names(side))
        if X.ndim != 2 or X.shape != (n_rows, d):
            raise ValueError(
                f'{side}_features gave an array of shape {X.shape}, '
                f'expected ({n_rows}, {d}) (one row per {side}, one column '
                f'per name in {side}_feature_names)')
        if not np.isfinite(X).all():
            raise ValueError(f'{side}_features gave NaN or inf')
        return X

    def check_picklable(self):
        import pickle
        for side in ('node', 'edge'):
            try:
                pickle.dumps(self.builder(side))
            except Exception as e:
                raise ValueError(
                    f'{side}_features cannot be pickled ({e}); define it at '
                    f'the top level of a module (no lambda, no closure), '
                    f'or use one worker') from None

    # -- model files ------------------------------------------------------

    def write_meta(self, model_path, class_names, side, model_type='mean_tree',
                   extra=None, extra_inputs=None):
        """The meta file of a model file. extra_inputs: names of the columns
        that the model input has after the features and that the inference
        side adds (the document_type column of the document classifier)."""
        if model_type not in MODEL_TYPES:
            raise ValueError(f'model_type must be one of {MODEL_TYPES}, got '
                             f'{model_type!r}')
        if side not in ('node', 'edge'):
            raise ValueError(f"side must be 'node' or 'edge', got {side!r}")
        meta = {
            'format': META_FORMAT, 'version': META_VERSION, 'side': side,
            'model_type': model_type,
            'feature_names': list(self.names(side)),
            'feature_version': self.version,
            'class_names': list(class_names),
            'env': self.env(side),
            'extra_inputs': list(extra_inputs or []),
            'input': (f'softmax({side}_logits) + '
                      + ('features' if side == 'node' else 'edge features')),
        }
        if extra:
            meta['train'] = extra
        with open(meta_path(model_path), 'w', encoding='utf-8') as fh:
            json.dump(meta, fh, indent=1)
        return meta

    def load_model(self, model_path, class_names, side):
        """A model written with write_meta. Raises when the meta file is
        missing or when the model was made for another feature list,
        feature version or class list; a different env only warns."""
        mp = meta_path(model_path)
        if not os.path.isfile(mp):
            raise FileNotFoundError(f'meta file not found: {mp}')
        with open(mp, 'r', encoding='utf-8') as fh:
            meta = json.load(fh)
        if meta.get('format') != META_FORMAT:
            raise ValueError(f"{mp}: format {meta.get('format')!r}, "
                             f'expected {META_FORMAT!r}')
        if meta.get('version') != META_VERSION:
            raise ValueError(f"{mp}: version {meta.get('version')!r}, this "
                             f'code reads version {META_VERSION}')
        got_side = meta.get('side', 'node')       # old files are node models
        if got_side != side:
            raise ValueError(f'{mp}: this is a {got_side} model, a {side} '
                             f'model was expected')
        names = self.names(side)
        if meta['feature_names'] != names:
            diff = [(i, a, b) for i, (a, b) in enumerate(
                zip(meta['feature_names'], names)) if a != b][:3]
            raise ValueError(
                f'the model was trained with another feature list (lengths '
                f'{len(meta["feature_names"])} vs {len(names)}, first '
                f'differences {diff})')
        mv = meta.get('feature_version')
        if mv is not None and self.version and mv != self.version:
            raise ValueError(
                f'the model was trained with feature version {mv!r}, the '
                f'features here are version {self.version!r}')
        if class_names is not None and list(class_names) != meta[
                'class_names']:
            raise ValueError(
                f"the model was trained for the classes "
                f"{meta['class_names']}, got {list(class_names)}")
        env = self.env(side)
        if env and meta.get('env') != env:
            warnings.warn(
                f"the model was trained with {meta.get('env')}, running "
                f'with {env}; the feature values may differ', stacklevel=3)
        model_type = meta.get('model_type', 'mean_tree')
        if model_type not in MODEL_TYPES:
            raise ValueError(f'{mp}: unknown model_type {model_type!r}, '
                             f'expected one of {MODEL_TYPES}')
        model = load_model_file(model_path, model_type)
        want = (len(meta['class_names']) + len(names)
                + len(meta.get('extra_inputs', [])))
        if model.n_features_ != want or model.n_classes_ != len(
                meta['class_names']):
            raise ValueError(
                f'the {model_type} model has {model.n_features_} inputs / '
                f'{model.n_classes_} classes, the meta file says {want} / '
                f"{len(meta['class_names'])}")
        return model

    def load_node_model(self, model_path, class_names=None):
        """(model, node_features) of a node model file."""
        return (self.load_model(model_path, class_names, 'node'),
                self.node_features)

    def load_edge_model(self, model_path, class_names=None):
        """(model, edge_features) of an edge model file; the classes are
        EDGE_CLASS_NAMES."""
        return (self.load_model(model_path, class_names or EDGE_CLASS_NAMES,
                                'edge'), self.edge_features)


FEATURES_FILE = 'features.py'          # the features inside a model.zip


def _local_imports(path):
    """Names of the modules that the file imports from its own directory
    (or relatively)."""
    import ast
    here = os.path.dirname(os.path.abspath(path))
    with open(path, 'rb') as fh:
        tree = ast.parse(fh.read(), path)
    found = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            if n.level:
                found.add('.' * n.level + (n.module or ''))
                continue
            mods = [n.module or '']
        elif isinstance(n, ast.Import):
            mods = [a.name for a in n.names]
        else:
            continue
        for m in mods:
            root = m.split('.')[0]
            if root and (os.path.isfile(os.path.join(here, root + '.py'))
                         or os.path.isdir(os.path.join(here, root))):
                found.add(root)
    return found


def _import_file(path, name):
    """Import a python file under a module name of our own (it is kept in
    sys.modules while it is in use)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True        # no __pycache__ next to the file
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    finally:
        sys.dont_write_bytecode = old
    return mod


def write_features(snapshot, out_dir, check=True):
    """Write the features file of a source_snapshot() into out_dir and
    return its manifest entry. With check the file is imported again from
    there and must give the same FeatureSet (names, version): a file that
    cannot be loaded alone is found when the model is saved, not when it is
    used."""
    path = os.path.join(out_dir, FEATURES_FILE)
    with open(path, 'wb') as fh:
        fh.write(snapshot['text'].encode('utf-8'))
    entry = dict(file=FEATURES_FILE, attr=snapshot['attr'],
                 sha256=snapshot['sha256'], version=snapshot['version'],
                 source=snapshot['path'])
    if check:
        name = 'cnn4gnn_check_features_' + snapshot['sha256'][:12]
        try:
            fs = FeatureSet.from_any(getattr(_import_file(path, name),
                                             snapshot['attr']))
        except Exception as e:
            raise ValueError(
                f"the features file {snapshot['path']} cannot be loaded "
                f'alone: {type(e).__name__}: {e}') from None
        finally:
            sys.modules.pop(name, None)
        if fs.version != snapshot['version']:
            raise ValueError('the saved features file gives another version '
                             'than the features in use')
    return entry


def load_embedded_features(folder, entry):
    """The FeatureSet of the features file in folder (an extracted
    model.zip), by its manifest entry. The file is code: it is imported."""
    path = os.path.join(folder, entry['file'])
    with open(path, 'rb') as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    if sha != entry['sha256']:
        raise ValueError(f"{entry['file']} in the zip does not match its "
                         f'recorded checksum')
    name = 'cnn4gnn_embedded_features_' + sha[:12]
    mod = sys.modules.get(name) or _import_file(path, name)
    return FeatureSet.from_any(getattr(mod, entry['attr']))


def load_zip(zip_path, class_names, features=None, allow_experiment=False):
    """{'node': (model, builder) or None, 'edge': (model, builder) or None}
    (plus 'doc_classifier': the DocTypeClassifier or None) of a
    model.zip written by the training. features: None (the features saved
    in the zip are used) or a FeatureSet (or what FeatureSet.from_any
    takes), which is then used instead (a warning when its file differs
    from the one in the zip). A side with
    several document categories gives a CategoryModels. A zip of an
    experiment (dummy model, GT class or document type added to the input)
    is refused: its input has columns that real pages do not have."""
    tmp = tempfile.mkdtemp(prefix='cnn4gnn_fix_')
    try:
        with zipfile.ZipFile(zip_path) as zf:
            if 'manifest.json' not in zf.namelist():
                raise ValueError(f'{zip_path}: no manifest.json')
            zf.extractall(tmp)
        with open(os.path.join(tmp, 'manifest.json')) as fh:
            man = json.load(fh)
        if man.get('format') != ZIP_FORMAT:
            raise ValueError(f"{zip_path}: format {man.get('format')!r}, "
                             f'expected {ZIP_FORMAT!r}')
        if man.get('version') != ZIP_VERSION:
            raise ValueError(f"{zip_path}: version {man.get('version')!r}, "
                             f'this code reads version {ZIP_VERSION}')
        if man.get('experiment') and not allow_experiment:
            raise ValueError(
                f"{zip_path} is an experiment model "
                f"({', '.join(man['experiment'])}): its input has columns "
                f'that real pages do not have, so it cannot be used on '
                f'real pages')
        out = {'node': None, 'edge': None, 'doc_classifier': None}
        entry = man.get('features')
        if features is None:
            if not entry:
                raise ValueError(f'{zip_path} has no features inside (it '
                                 f'was saved without them): give features')
            features = load_embedded_features(tmp, entry)
            out['features_info'] = (f"embedded features, version "
                                    f"{entry['version'] or '-'}, sha256 "
                                    f"{entry['sha256'][:10]}")
        else:
            features = FeatureSet.from_any(features)
            out['features_info'] = 'given features'
            if entry:
                try:
                    given = features.source_snapshot()['sha256']
                except ValueError:
                    given = None
                if given is not None and given != entry['sha256']:
                    warnings.warn(
                        f'the given features file differs from the one '
                        f'saved in {zip_path} (sha256 {given[:10]} vs '
                        f"{entry['sha256'][:10]}); the model was trained "
                        f'with the saved one', stacklevel=3)
        out['features'] = features
        if man.get('doctype'):
            with open(os.path.join(tmp, man['doctype']['file'])) as fh:
                out['doc_classifier'] = DocTypeClassifier.from_dict(
                    json.load(fh))
        for side, info in man['sides'].items():
            models = {}
            builder = None
            for e in info['models']:
                mp = os.path.join(tmp, e['file'])
                if side == 'node':
                    m, builder = features.load_node_model(mp, class_names)
                else:
                    m, builder = features.load_edge_model(mp)
                models[e['category']] = m      # models are in memory now
            if list(models) == [None]:
                model = models[None]
            else:
                model = CategoryModels(info['gate_col'], models)
            out[side] = (model, builder)
        if out['node'] is None and out['edge'] is None:
            raise FileNotFoundError(f'no correction model in {zip_path}')
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


DOC_FORMAT = 'cnn4gnn_doctype'
DOC_VERSION = 3
DOC_FILE = 'doctype.json'
# grid of the class share numbers (rows x columns over the whole page)
GRID_ROWS = 3
GRID_COLS = 3
DOC_FEATURES = ['page_num_ratio', 'page_year_ratio']
# the layout numbers of a page (from the node boxes, in PDF points)
LAYOUT_FEATURES = ['ext_w', 'ext_h', 'aspect', 'n_nodes', 'area_ratio',
                   'mean_h', 'h_cv', 'n_columns', 'left_align',
                   'num_right_align']
# regions of the page for the bag of words: the top and the bottom share of
# the content height, the rest (body), and the first words of the topmost
# node (title)
REGIONS = ('top', 'bottom', 'body', 'title')
REGION_FRAC = 0.15
TITLE_WORDS = 3


def page_words(texts):
    """The words of a page for the document classifier: the texts are
    joined and split at white space; the comma and the point (currency
    separators) are removed inside a word; words are lower case. Returns
    (words, n_year): n_year counts the raw tokens that are exactly four
    digits in 1900..2500."""
    words, n_year = [], 0
    for t in ' '.join(str(x) for x in texts).split():
        c = t.replace(',', '').replace('.', '')
        if not c:
            continue
        words.append(c.lower())
        if len(t) == 4 and t.isascii() and t.isdigit() \
                and 1900 <= int(t) <= 2500:
            n_year += 1
    return words, n_year


def region_words(texts, bboxes_pt, frac=REGION_FRAC, title_words=TITLE_WORDS):
    """The words of a page by region: {'page': all words, 'top': nodes whose
    centre is in the upper `frac` of the content height (first node y0 to
    last node y1), 'bottom': the lower `frac`, 'body': the other nodes,
    'title': the first `title_words` words of the topmost node (ties: the
    left one)}. Every node is in exactly one of top, bottom and body (all
    in body when the content has no height)."""
    texts = [str(t) for t in texts]
    bb = np.asarray(bboxes_pt, np.float64).reshape(-1, 4)
    if len(bb) != len(texts):
        raise ValueError(f'{len(texts)} texts for {len(bb)} boxes')
    out = {r: [] for r in ('page',) + REGIONS}
    out['page'] = page_words(texts)[0]
    if not len(texts):
        return out
    y0, y1 = bb[:, 1], bb[:, 3]
    top_y, bot_y = y0.min(), y1.max()
    h = bot_y - top_y
    cy = 0.5 * (y0 + y1)
    where = np.full(len(texts), 2)                      # 2 = body
    if h > 0:
        where[cy >= bot_y - frac * h] = 1               # 1 = bottom
        where[cy <= top_y + frac * h] = 0               # 0 = top
    for i, t in enumerate(texts):
        out[('top', 'bottom', 'body')[where[i]]].extend(page_words([t])[0])
    first = int(np.lexsort((bb[:, 0], y0))[0])
    out['title'] = page_words([texts[first]])[0][:int(title_words)]
    return out


def _bin_share(v, width):
    """Largest share of the values in one bin of the given width."""
    if len(v) == 0:
        return 0.0
    cnt = np.unique(np.floor(np.asarray(v) / width).astype(np.int64),
                    return_counts=True)[1]
    return float(cnt.max()) / len(v)


def layout_vector(texts, bboxes_pt):
    """LAYOUT_FEATURES of a page. The page size is not known here (neither
    in the shards nor in every caller), so the size of the page is the
    extent of the boxes: ext_w = largest x1, ext_h = largest y1, aspect =
    ext_w / ext_h. area_ratio: box area over the extent (at most 1);
    mean_h, h_cv: mean and relative spread of the box heights;
    n_columns: x0 bins (5 percent of the width) with at least 3 nodes and
    5 percent of them; left_align: share of the nodes in the best x0 bin;
    num_right_align: share of the number boxes (at least half of the
    characters are digits, at least 3 of them) in the best x1 bin (1 percent
    of the width)."""
    bb = np.asarray(bboxes_pt, np.float64).reshape(-1, 4)
    n = len(bb)
    v = np.zeros(len(LAYOUT_FEATURES), np.float64)
    if n == 0:
        return v
    W, H = max(bb[:, 2].max(), 1e-6), max(bb[:, 3].max(), 1e-6)
    bw = np.maximum(bb[:, 2] - bb[:, 0], 0.0)
    bh = np.maximum(bb[:, 3] - bb[:, 1], 0.0)
    v[0], v[1], v[2], v[3] = W, H, W / H, n
    v[4] = min(float((bw * bh).sum()) / (W * H), 1.0)
    mh = float(bh.mean())
    v[5] = mh
    v[6] = float(bh.std()) / mh if mh > 0 else 0.0
    cnt = np.unique(np.floor(bb[:, 0] / (0.05 * W)).astype(np.int64),
                    return_counts=True)[1]
    v[7] = float((cnt >= max(3, 0.05 * n)).sum())
    v[8] = float(cnt.max()) / n
    num = []
    for i, t in enumerate(texts):
        c = [ch for ch in str(t) if not ch.isspace()]
        if c and sum(ch.isdigit() for ch in c) * 2 >= len(c):
            num.append(bb[i, 2])
    v[9] = _bin_share(num, 0.01 * W) if len(num) >= 3 else 0.0
    return v


def class_vector(node_logits, bboxes_pt, n_classes):
    """The node class mix of a page by the CNN4GNN output: for every class
    the share of the nodes and the share of the box area (2 * n_classes
    numbers)."""
    lg = np.asarray(node_logits, np.float64)
    bb = np.asarray(bboxes_pt, np.float64).reshape(-1, 4)
    if lg.ndim != 2 or lg.shape[1] != n_classes or len(lg) != len(bb):
        raise ValueError(f'node logits of shape {lg.shape} do not fit '
                         f'{len(bb)} boxes and {n_classes} classes')
    v = np.zeros(2 * n_classes, np.float64)
    if not len(lg):
        return v
    cls = lg.argmax(axis=1)
    area = np.maximum(bb[:, 2] - bb[:, 0], 0) * np.maximum(bb[:, 3] - bb[:, 1], 0)
    v[:n_classes] = np.bincount(cls, minlength=n_classes) / len(cls)
    tot = area.sum()
    if tot > 0:
        v[n_classes:] = np.bincount(cls, weights=area,
                                    minlength=n_classes) / tot
    return v


def grid_names(class_names, rows, cols):
    """Names of the grid numbers, in the order of grid_vector: row (from the
    top), column (from the left), class."""
    return [f'grid_r{r}c{c}:{k}' for r in range(int(rows))
            for c in range(int(cols)) for k in class_names]


def grid_vector(node_logits, bboxes_pt, page_size, n_classes,
                rows=GRID_ROWS, cols=GRID_COLS):
    """The class share of the nodes by place: the page (page_size = (width,
    height) in PDF points, upright) is cut into rows x cols cells; a node
    belongs to the cell of its box centre (centres outside the page go to
    the nearest edge cell); its class is the argmax of the CNN4GNN node
    logits. Every number is the count of one class in one cell over the
    nodes of the whole page, so all numbers of a page add up to 1. Order:
    row, column, class (see grid_names); rows * cols * n_classes numbers."""
    rows, cols = int(rows), int(cols)
    if rows < 1 or cols < 1:
        raise ValueError(f'grid must be at least 1 x 1, got {rows} x {cols}')
    if page_size is None:
        raise ValueError('the grid numbers need the page size (width, height '
                         'in points)')
    pw, ph = float(page_size[0]), float(page_size[1])
    if not (pw > 0 and ph > 0):
        raise ValueError(f'bad page size {page_size}')
    lg = np.asarray(node_logits, np.float64)
    bb = np.asarray(bboxes_pt, np.float64).reshape(-1, 4)
    if lg.ndim != 2 or lg.shape[1] != n_classes or len(lg) != len(bb):
        raise ValueError(f'node logits of shape {lg.shape} do not fit '
                         f'{len(bb)} boxes and {n_classes} classes')
    v = np.zeros((rows, cols, n_classes), np.float64)
    if len(lg):
        cx = 0.5 * (bb[:, 0] + bb[:, 2])
        cy = 0.5 * (bb[:, 1] + bb[:, 3])
        ci = np.clip(np.floor(cx / pw * cols).astype(np.int64), 0, cols - 1)
        ri = np.clip(np.floor(cy / ph * rows).astype(np.int64), 0, rows - 1)
        np.add.at(v, (ri, ci, lg.argmax(axis=1)), 1.0)
        v /= len(lg)
    return v.reshape(-1)


def page_size_of(ctx):
    """(width, height) of the page in points from a hook ctx or a dict that
    has 'page_size' or a PyMuPDF 'page'; None when it has neither."""
    ps = ctx.get('page_size')
    if ps is None and ctx.get('page') is not None:
        r = ctx['page'].rect
        ps = (r.width, r.height)
    return None if ps is None else (float(ps[0]), float(ps[1]))


class DocTypeTree:
    """A small classification tree (weighted gini) on the document vectors.
    Labels are integers (the document categories)."""

    def __init__(self, max_depth=4, min_samples_leaf=1):
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.feature, self.threshold = [], []
        self.left, self.right, self.label, self.count = [], [], [], []

    def fit(self, X, y, w=None):
        X = np.asarray(X, np.float64)
        y = np.asarray(y, np.int64)
        w = np.ones(len(y)) if w is None else np.asarray(w, np.float64)
        classes = np.unique(y)
        self.feature, self.threshold = [], []
        self.left, self.right, self.label, self.count = [], [], [], []
        order = {c: i for i, c in enumerate(classes)}
        yi = np.array([order[c] for c in y], np.int64)
        K = len(classes)

        def new(idx):
            cw = np.bincount(yi[idx], weights=w[idx], minlength=K)
            self.feature.append(-1)
            self.threshold.append(0.0)
            self.left.append(-1)
            self.right.append(-1)
            self.label.append(int(classes[int(cw.argmax())]))
            self.count.append(int(len(idx)))
            return len(self.feature) - 1

        stack = [(new(np.arange(len(y))), np.arange(len(y)), 0)]
        while stack:
            node, idx, depth = stack.pop()
            if depth >= self.max_depth or len(np.unique(yi[idx])) < 2:
                continue
            best = self._split(X[idx], yi[idx], w[idx], K)
            if best is None:
                continue
            f, thr = best
            go = X[idx, f] <= thr
            li, ri = idx[go], idx[~go]
            if len(li) < self.min_samples_leaf or \
                    len(ri) < self.min_samples_leaf:
                continue
            self.feature[node], self.threshold[node] = int(f), float(thr)
            self.left[node] = new(li)
            self.right[node] = new(ri)
            stack.append((self.left[node], li, depth + 1))
            stack.append((self.right[node], ri, depth + 1))
        return self

    @staticmethod
    def _split(X, y, w, K):
        n, d = X.shape
        onehot = np.zeros((n, K))
        onehot[np.arange(n), y] = w
        total = onehot.sum(axis=0)
        wt = total.sum()
        parent = wt - (total ** 2).sum() / wt
        best, best_gain = None, 1e-12
        for f in range(d):
            order = np.argsort(X[:, f], kind='stable')
            xs = X[order, f]
            cl = np.cumsum(onehot[order], axis=0)[:-1]
            wl = cl.sum(axis=1)
            cr = total - cl
            wr = wt - wl
            ok = (xs[1:] > xs[:-1]) & (wl > 0) & (wr > 0)
            if not ok.any():
                continue
            gl = wl - (cl ** 2).sum(axis=1) / np.where(wl > 0, wl, 1)
            gr = wr - (cr ** 2).sum(axis=1) / np.where(wr > 0, wr, 1)
            gain = np.where(ok, parent - gl - gr, -1.0)
            i = int(gain.argmax())
            if gain[i] > best_gain:
                best_gain = gain[i]
                best = (f, 0.5 * (xs[i] + xs[i + 1]))
        return best

    def predict(self, X):
        X = np.asarray(X, np.float64)
        out = np.zeros(len(X), np.int64)
        for r in range(len(X)):
            i = 0
            while self.feature[i] >= 0:
                i = (self.left[i] if X[r, self.feature[i]]
                     <= self.threshold[i] else self.right[i])
            out[r] = self.label[i]
        return out

    def export_text(self, names):
        lines = []

        def walk(i, pad):
            if self.feature[i] < 0:
                lines.append(f'{pad}-> document_type = {self.label[i]}  '
                             f'(n={self.count[i]})')
                return
            nm = names[self.feature[i]]
            lines.append(f'{pad}if {nm} <= {self.threshold[i]:.4g}:')
            walk(self.left[i], pad + '  ')
            lines.append(f'{pad}else:  # {nm} > {self.threshold[i]:.4g}')
            walk(self.right[i], pad + '  ')
        walk(0, '')
        return '\n'.join(lines)

    def to_dict(self):
        return dict(max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    feature=self.feature, threshold=self.threshold,
                    left=self.left, right=self.right, label=self.label,
                    count=self.count)

    @classmethod
    def from_dict(cls, d):
        t = cls(d['max_depth'], d['min_samples_leaf'])
        t.feature, t.threshold = list(d['feature']), list(d['threshold'])
        t.left, t.right = list(d['left']), list(d['right'])
        t.label, t.count = list(d['label']), list(d['count'])
        n = len(t.feature)
        if not (len(t.threshold) == len(t.left) == len(t.right)
                == len(t.label) == len(t.count) == n) or n == 0:
            raise ValueError('the document tree arrays do not match')
        return t


def select_vocab(cat_pages, other_pages, k):
    """Words for the bag of words. cat_pages: {category: [word list of
    each page of the category]}, other_pages: [word list of each other
    page]. For every category the candidates are the words that are in at
    least two of its pages (in its only page when it has one); they are
    ranked by (share of its pages with the word) - (share of the other
    pages with the word), highest first, equal scores in alphabetical
    order; the best k with a positive score are taken. The result is the
    sorted union over the categories (empty when no word is common)."""
    n_other = max(len(other_pages), 1)
    df_other = {}
    for ws in other_pages:
        for w in set(ws):
            df_other[w] = df_other.get(w, 0) + 1
    vocab = set()
    for cat, pages in cat_pages.items():
        if not pages:
            continue
        df = {}
        for ws in pages:
            for w in set(ws):
                df[w] = df.get(w, 0) + 1
        need = min(2, len(pages))
        scored = []
        for w, c in df.items():
            if c < need:
                continue
            sc = c / len(pages) - df_other.get(w, 0) / n_other
            if sc > 0:
                scored.append((-sc, w))
        scored.sort()
        vocab.update(w for _, w in scored[:int(k)])
    return sorted(vocab)


def select_region_vocab(cat_pages, other_pages, k, regions=REGIONS):
    """select_vocab for every region: cat_pages {category: [region_words of
    each page of the category]}, other_pages [region_words of each other
    page]. Returns the sorted list of (region, word)."""
    out = []
    for r in regions:
        words = select_vocab(
            {c: [p[r] for p in pages] for c, pages in cat_pages.items()},
            [p[r] for p in other_pages], k)
        out.extend((r, w) for w in words)
    return sorted(out)


class DocTypeClassifier:
    """The default document_type maker: a page is turned into a vector and
    a small tree gives the document category (0 = not one of the learned
    documents). The vector is

        [page_num_ratio, page_year_ratio,
         layout numbers (LAYOUT_FEATURES; from the node boxes; old),
         share of the nodes and of the area of every node class (from the
         CNN4GNN output; class_names; old),
         class share of every cell of a rows x cols grid over the page
         (grid_vector; grid, grid_class_names),
         ratio of every vocabulary word in its region (top, bottom, body,
         title; see region_words)]

    Word ratios are divided by the number of words of the region; the two
    first numbers by the number of words of the page. A page is given as its
    node texts, boxes (PDF points), CNN4GNN node logits and, for the grid,
    the page size (points), as the ctx of the logits hook has them. The
    training is in CNN4GNNLocalExpertTrain.py; this class is all that a
    deployment needs.

    class_names None: no class mix numbers. layout False: no layout numbers
    (both are only read from classifiers saved before the grid). grid None:
    no grid numbers, else (rows, cols). vocab entries are (region, word);
    the region 'page' is all words of the page (the first version of the
    classifier used only that)."""

    def __init__(self, vocab=None, tree=None, class_names=None, layout=False,
                 region_frac=REGION_FRAC, title_words=TITLE_WORDS,
                 grid=None, grid_class_names=None):
        self.vocab = [(str(r), str(w)) for r, w in (vocab or [])]
        self.tree = tree
        self.class_names = None if class_names is None \
            else [str(c) for c in class_names]
        self.grid = None if grid is None else (int(grid[0]), int(grid[1]))
        self.grid_class_names = None if grid_class_names is None \
            else [str(c) for c in grid_class_names]
        if self.grid is not None:
            if min(self.grid) < 1:
                raise ValueError(f'grid must be at least 1 x 1, got '
                                 f'{self.grid}')
            if not self.grid_class_names:
                raise ValueError('a grid needs grid_class_names')
        self.layout = bool(layout)
        self.region_frac = float(region_frac)
        self.title_words = int(title_words)
        self._index = {rw: i for i, rw in enumerate(self.vocab)}
        bad = {r for r, _ in self.vocab} - set(('page',) + REGIONS)
        if bad:
            raise ValueError(f'unknown regions in the vocabulary: '
                             f'{sorted(bad)}')

    @property
    def needs_boxes(self):
        return (self.layout or self.class_names is not None
                or self.grid is not None
                or any(r != 'page' for r, _ in self.vocab))

    @property
    def needs_page_size(self):
        return self.grid is not None

    @property
    def names(self):
        out = list(DOC_FEATURES)
        if self.layout:
            out += LAYOUT_FEATURES
        if self.class_names is not None:
            out += [f'cls_ratio:{c}' for c in self.class_names]
            out += [f'cls_area:{c}' for c in self.class_names]
        if self.grid is not None:
            out += grid_names(self.grid_class_names, *self.grid)
        return out + [f'{r}:{w}' for r, w in self.vocab]

    def vector(self, texts, bboxes_pt=None, node_logits=None,
               page_size=None):
        texts = [str(t) for t in texts]
        if bboxes_pt is None:
            if self.needs_boxes:
                raise ValueError('this document classifier needs the node '
                                 'boxes (bboxes_pt)')
            bboxes_pt = np.zeros((len(texts), 4))
        if ((self.class_names is not None or self.grid is not None)
                and node_logits is None):
            raise ValueError('this document classifier needs the node '
                             'logits (node_logits)')
        if self.needs_page_size and page_size is None:
            raise ValueError('this document classifier needs the page size '
                             '(page_size = (width, height) in points); the '
                             'reference shards must be made with the current '
                             'tool if they have none')
        rw = region_words(texts, bboxes_pt, self.region_frac,
                          self.title_words)
        words = rw['page']
        n_year = page_words(texts)[1]
        n = len(words)
        parts = [np.zeros(2)]
        if n:
            parts[0][0] = sum(1 for w in words
                              if w.isascii() and w.isdigit()) / n
            parts[0][1] = n_year / n
        if self.layout:
            parts.append(layout_vector(texts, bboxes_pt))
        if self.class_names is not None:
            parts.append(class_vector(node_logits, bboxes_pt,
                                      len(self.class_names)))
        if self.grid is not None:
            parts.append(grid_vector(node_logits, bboxes_pt, page_size,
                                     len(self.grid_class_names), *self.grid))
        bow = np.zeros(len(self.vocab), np.float64)
        for r in ('page',) + REGIONS:
            ws = rw[r]
            for w in ws:
                j = self._index.get((r, w))
                if j is not None:
                    bow[j] += 1.0 / len(ws)
        parts.append(bow)
        return np.concatenate(parts)

    def predict(self, texts, bboxes_pt=None, node_logits=None,
                page_size=None):
        """The document category (int) of a page."""
        if self.tree is None:
            return 0
        v = self.vector(texts, bboxes_pt, node_logits, page_size)
        return int(self.tree.predict(v[None, :])[0])

    def export_text(self):
        words = [f'{r}:{w}' for r, w in self.vocab]
        head = (f'document classifier: {len(self.vocab)} words in the bag '
                f'of words' + (': ' + ', '.join(words) if words else '')
                + f'; layout numbers: {"yes" if self.layout else "no"}; '
                f'node class numbers: '
                f'{"yes" if self.class_names is not None else "no"}; '
                f'grid: '
                + (f'{self.grid[0]} x {self.grid[1]} (rows x columns)'
                   if self.grid is not None else 'no'))
        return head + '\n' + (self.tree.export_text(self.names)
                              if self.tree is not None
                              else '(no tree)')

    def to_dict(self):
        return dict(format=DOC_FORMAT, version=DOC_VERSION,
                    vocab=[list(rw) for rw in self.vocab],
                    class_names=self.class_names, layout=self.layout,
                    region_frac=self.region_frac,
                    title_words=self.title_words,
                    grid=None if self.grid is None else list(self.grid),
                    grid_class_names=self.grid_class_names,
                    tree=None if self.tree is None else self.tree.to_dict())

    @classmethod
    def from_dict(cls, d):
        if d.get('format') != DOC_FORMAT or d.get('version') not in (1, 2, 3):
            raise ValueError('not a document classifier of this version')
        tree = None if d['tree'] is None else DocTypeTree.from_dict(d['tree'])
        if d['version'] == 1:       # words of the whole page only
            return cls([('page', w) for w in d['vocab']], tree,
                       class_names=None, layout=False)
        return cls([tuple(rw) for rw in d['vocab']], tree,
                   d['class_names'], d['layout'], d['region_frac'],
                   d['title_words'], grid=d.get('grid'),
                   grid_class_names=d.get('grid_class_names'))


_softmax = softmax_f32


class CNN4GNNLocalExpertModel:
    """CNN4GNN plus optional node / edge second stage models.

    Parameters
    ----------
    cnn4gnn : a CNN4GNN instance (or any object with the same predict and
        logits_hook contract).
    node_model, edge_model : None, a model object with
        predict_logits(X, logits), or a path to a MeanResidualTree file (json)
        or a MeanResidualMLP file (onnx).
        None keeps the CNN4GNN output for that side.
    node_features, edge_features : optional callables ctx -> array or None
        (see the module doc). Without one, X is the probabilities alone.
    use_proba : bool. Put softmax(logits) in front of the extra features
        (default True).
    doc_classifier : None or a DocTypeClassifier. When given, its document
        category of the page (from ctx['texts'], ctx['bboxes_pt'] and
        ctx['node_logits']) is appended as the last
        column of the input of the node and edge models (the column
        document_type that the models were trained with).
    """

    def __init__(self, cnn4gnn, node_model=None, edge_model=None,
                 node_features=None, edge_features=None, use_proba=True,
                 doc_classifier=None):
        if cnn4gnn is None:
            raise ValueError('cnn4gnn is required')
        for name, fn in (('node_features', node_features),
                         ('edge_features', edge_features)):
            if fn is not None and not callable(fn):
                raise TypeError(f'{name} must be callable')
        object.__setattr__(self, 'cnn4gnn', cnn4gnn)
        self.node_model = self._as_model(node_model, 'node_model')
        self.edge_model = self._as_model(edge_model, 'edge_model')
        self.node_features = node_features
        self.edge_features = edge_features
        self.use_proba = bool(use_proba)
        self.doc_classifier = doc_classifier
        self.features_info = ''             # set by from_zip

    @staticmethod
    def _as_model(m, name):
        if m is None:
            return None
        if isinstance(m, str):
            if m.lower().endswith('.onnx'):
                return MeanResidualMLP.load(m)
            return MeanResidualTree.load(m)
        if not callable(getattr(m, 'predict_logits', None)):
            raise TypeError(
                f'{name} must have a predict_logits(X, logits) method, '
                f'got {type(m).__name__}')
        return m

    @classmethod
    def from_zip(cls, cnn4gnn, zip_path, features=None, use_proba=True,
                 allow_experiment=False):
        """Wrap cnn4gnn with the models of a model.zip. zip_path may also be
        a folder that holds a model.zip. The features are the ones saved in
        the zip, unless features (a FeatureSet or what FeatureSet.from_any
        takes) is given. A side the zip does not have keeps the CNN4GNN
        output."""
        if os.path.isdir(zip_path):
            zip_path = os.path.join(zip_path, ZIP_FILE)
        cor = load_zip(zip_path, list(cnn4gnn.class_names), features,
                       allow_experiment)
        node, edge = cor['node'], cor['edge']
        obj = cls(cnn4gnn,
                  node_model=node[0] if node else None,
                  edge_model=edge[0] if edge else None,
                  node_features=node[1] if node else None,
                  edge_features=edge[1] if edge else None,
                  use_proba=use_proba,
                  doc_classifier=cor.get('doc_classifier'))
        obj.features_info = cor.get('features_info', '')
        return obj

    @classmethod
    def from_paths(cls, model_cfg_path, stage2_onnx_path,
                   node_model=None, edge_model=None,
                   node_features=None, edge_features=None, use_proba=True,
                   **cnn4gnn_kwargs):
        """Build the CNN4GNN from its config / ONNX paths, then wrap it."""
        CNN4GNN = _sibling('CNN4GNN').CNN4GNN
        net = CNN4GNN(model_cfg_path, stage2_onnx_path=stage2_onnx_path,
                      **cnn4gnn_kwargs)
        return cls(net, node_model, edge_model, node_features,
                   edge_features, use_proba)

    # ------------------------------------------------------------------
    # delegation
    # ------------------------------------------------------------------

    _OWN = frozenset(('node_model', 'edge_model', 'node_features',
                      'edge_features', 'use_proba', 'doc_classifier',
                      'features_info'))

    def __setattr__(self, name, value):
        # Setting a network option on the wrapper would silently do nothing
        # to the network, so it is an error.
        if name not in self._OWN:
            raise AttributeError(
                f"cannot set '{name}' on the wrapper; network settings live "
                f'on wrapper.cnn4gnn (wrapper.cnn4gnn.{name} = ...)')
        object.__setattr__(self, name, value)

    def __getattr__(self, name):
        # Only called when normal lookup fails: read from the network.
        if name == 'cnn4gnn':
            raise AttributeError(name)
        return getattr(self.cnn4gnn, name)

    def describe(self):
        """Short text of what is active, for logs."""
        def one(m):
            return 'cnn4gnn output' if m is None else type(m).__name__
        return (f'nodes: {one(self.node_model)}, edges: '
                f'{one(self.edge_model)}'
                + (', document classifier' if self.doc_classifier else '')
                + (f', {self.features_info}' if self.features_info else ''))

    def predict(self, page_or_image, verbose=False, refine=True, **kwargs):
        """CNN4GNN.predict with the second stage models applied.

        refine=False runs the plain CNN4GNN (to compare before and after).
        All other arguments go to CNN4GNN.predict unchanged.
        """
        if 'logits_hook' in kwargs:
            raise ValueError(
                'CNN4GNNLocalExpertModel.predict installs its own logits_hook; set '
                'refine=False to run the network alone')
        if (not refine or (self.node_model is None
                           and self.edge_model is None)):
            return self.cnn4gnn.predict(page_or_image, verbose=verbose,
                                        **kwargs)
        if verbose:
            print(f'[CNN4GNNLocalExpertModel] {self.describe()}')
        return self.cnn4gnn.predict(page_or_image, verbose=verbose,
                                    logits_hook=self._hook, **kwargs)

    # ------------------------------------------------------------------
    # the hook
    # ------------------------------------------------------------------

    def _hook(self, ctx):
        ctx['node_proba'] = _softmax(ctx['node_logits'])
        ctx['edge_proba'] = _softmax(ctx['edge_logits'])
        if self.doc_classifier is not None:
            # one document category for the page, the last model column
            ctx['document_type'] = float(self.doc_classifier.predict(
                ctx['texts'], ctx['bboxes_pt'], ctx['node_logits'],
                page_size_of(ctx)))
        out = {}
        for side, model, builder in (
                ('node', self.node_model, self.node_features),
                ('edge', self.edge_model, self.edge_features)):
            if model is None:
                continue
            logits = ctx[f'{side}_logits']
            if logits.shape[0] == 0:
                continue                       # nothing to re-score
            X = self._build_X(side, ctx, builder)
            new = np.asarray(model.predict_logits(X, logits))
            if new.shape != logits.shape:
                raise ValueError(
                    f'{side} model returned shape {new.shape}, expected '
                    f'{logits.shape}')
            out[f'{side}_logits'] = new
        return out or None

    def _build_X(self, side, ctx, builder):
        n = ctx[f'{side}_logits'].shape[0]
        parts = []
        if self.use_proba:
            parts.append(ctx[f'{side}_proba'])
        if builder is not None:
            extra = builder(ctx)
            if extra is not None:
                extra = np.asarray(extra, dtype=np.float32)
                if extra.ndim != 2 or extra.shape[0] != n:
                    raise ValueError(
                        f'{side}_features returned shape {extra.shape}, '
                        f'expected ({n}, d)')
                if not np.isfinite(extra).all():
                    raise ValueError(
                        f'{side}_features returned NaN or inf')
                parts.append(extra)
        if self.doc_classifier is not None:
            parts.append(np.full((n, 1), ctx['document_type'], np.float32))
        if not parts:
            raise ValueError(
                f'no input features for the {side} model: use_proba is '
                f'False and {side}_features gave none')
        return np.concatenate(parts, axis=1)
