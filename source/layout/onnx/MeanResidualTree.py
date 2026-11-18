"""Output + mean residual: one small decision tree on top of a classifier.

The base model gives logits z, p = softmax(z). The tree learns the residual

    r = onehot(y) - p

as a multi-output regression tree (squared error, exact greedy splits, rows
weighted). A leaf stores the weighted MEAN residual of its rows, a vector of
probability shifts. The corrected output is

    q = p + leaf(x),   logits = log(max(q, 1e-6))

Why this is stable with very few samples
----------------------------------------
* A leaf value is a mean of numbers in [-1, 1], so it is always in [-1, 1].
  There is no learning rate, no Newton step, no regularization strength and
  no cap to tune; the same data always gives the same tree.
* Rows the base model has right with high confidence have residual about 0.
  They are part of the average, so a leaf that holds many of them is pulled
  towards "no change". A confidently wrong row (p_wrong about 1) only changes
  its class when most of the (weighted) rows of its leaf belong to the true
  class, i.e. when the features really separate the case. Nothing else
  changes.
* With features that explain nothing the tree stays close to "no change"
  (leaf means are near 0). Deeper trees and small leaves fit more of the
  noise: use the held-out numbers, not the training numbers.

Gate (the document_type protocol)
---------------------------------
gate_col >= 0 names a column of the model input (the tree input is
[softmax, features]). The shift is applied only on rows where that column is
1 or more; on the other rows the output is the base output (a shift of 0).
Rows with a gate below 1 are left out of the fit (their shift is thrown away
anyway). The column is stored in the model file.
gate_value >= 1 narrows the gate to ONE category: the shift is applied only
where the (rounded) column equals gate_value, so several models can share
the column, each one for its own category (gate_value 0 keeps the rule
above: 1 or more).

Interface (what CNN4GNNLocalExpertModel needs from a node or edge model)
    fit(X, logits, y, sample_weight=None)       X: (n, d), logits: (n, K)
    predict_logits(X, logits) -> (n, K) float32
    predict_proba(X, logits)
    export_text(feature_names, class_names)     readable rules
    save(path) / MeanResidualTree.load(path)    json (version 2, see
                                                to_dict)
"""

import json

import numpy as np

EPS = 1e-6
FORMAT = 'MeanResidualTree'
VERSION = 2          # 2: nested tree with feature and class names


def _softmax(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MeanResidualTree:
    def __init__(self, max_depth=8, min_samples_leaf=3, gate_col=-1,
                 gate_value=0):
        if int(max_depth) < 1:
            raise ValueError('max_depth must be >= 1')
        if int(min_samples_leaf) < 1:
            raise ValueError('min_samples_leaf must be >= 1')
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.gate_col = int(gate_col)            # -1: no gate
        self.gate_value = int(gate_value)        # 0: any value of 1 or more
        self.n_features_ = None
        self.n_classes_ = None
        # tree arrays: feature (-1 = leaf), threshold, left, right, value, count
        self.feature = self.threshold = self.left = self.right = None
        self.value = self.count = None
        self.pruned_from = None      # leaves before prune() (not saved)
        self.weight = None           # sum of row weights per node (not saved)

    # -- checks ---------------------------------------------------------

    def _gate_mask(self, X):
        """Rows where the gate column lets the shift pass."""
        g = X[:, self.gate_col]
        if self.gate_value >= 1:
            return np.abs(g - self.gate_value) < 0.5
        return g >= 1.0

    @staticmethod
    def _check_xz(X, logits):
        X = np.asarray(X)
        logits = np.asarray(logits)
        if X.ndim != 2 or logits.ndim != 2 or X.shape[0] != logits.shape[0]:
            raise ValueError(
                f'X {X.shape} and logits {logits.shape} must be 2-d with '
                f'the same number of rows')
        if not np.isfinite(X).all():
            raise ValueError('X contains NaN or inf')
        if not np.isfinite(logits).all():
            raise ValueError('logits contain NaN or inf')
        return X.astype(np.float64), logits.astype(np.float64)

    def _check_fitted(self):
        if self.feature is None:
            raise RuntimeError('the tree is not fitted')

    # -- fitting --------------------------------------------------------

    def fit(self, X, logits, y, sample_weight=None):
        X, logits = self._check_xz(X, logits)
        n, d = X.shape
        K = logits.shape[1]
        if n == 0:
            raise ValueError('no rows to fit')
        y = np.asarray(y)
        if y.shape != (n,) or not np.issubdtype(y.dtype, np.integer):
            raise ValueError('y must be an integer vector with one label '
                             'per row')
        if y.min() < 0 or y.max() >= K:
            raise ValueError(f'labels must be in 0..{K - 1}')
        if sample_weight is None:
            w = np.ones(n)
        else:
            w = np.asarray(sample_weight, dtype=np.float64)
            if w.shape != (n,) or not np.isfinite(w).all() or (w < 0).any():
                raise ValueError('sample_weight must be a finite, '
                                 'non-negative vector, one per row')
        if w.sum() <= 0:
            raise ValueError('the sample weights sum to zero')
        if self.gate_col >= 0:
            if self.gate_col >= d:
                raise ValueError(f'gate_col {self.gate_col} is outside the '
                                 f'{d} columns')
            keep = self._gate_mask(X)
            if not keep.any():
                raise ValueError('no row has the gate column at 1 or more')
            X, logits, y, w = X[keep], logits[keep], y[keep], w[keep]
            n = len(y)
            if w.sum() <= 0:
                raise ValueError('the rows of the gate have weight zero')
        R = np.eye(K)[y] - _softmax(logits)
        self.n_features_, self.n_classes_ = d, K

        feature, threshold, left, right, value, count = [], [], [], [], [], []
        weight = []

        def new_node(idx):
            sw = w[idx].sum()
            weight.append(float(sw))
            v = ((w[idx, None] * R[idx]).sum(axis=0) / sw if sw > 0
                 else np.zeros(K))
            feature.append(-1)
            threshold.append(0.0)
            left.append(-1)
            right.append(-1)
            value.append(v)
            count.append(len(idx))
            return len(feature) - 1

        root = new_node(np.arange(n))
        stack = [(root, np.arange(n), 0)]
        while stack:
            node, idx, depth = stack.pop()
            if depth >= self.max_depth or len(idx) < 2 * self.min_samples_leaf:
                continue
            best = self._best_split(X[idx], R[idx], w[idx])
            if best is None:
                continue
            f, thr = best
            go_left = X[idx, f] <= thr
            li, ri = idx[go_left], idx[~go_left]
            if len(li) < self.min_samples_leaf or \
                    len(ri) < self.min_samples_leaf:
                continue                       # guards the float rounding
            feature[node], threshold[node] = int(f), float(thr)
            left[node] = new_node(li)
            right[node] = new_node(ri)
            stack.append((left[node], li, depth + 1))
            stack.append((right[node], ri, depth + 1))

        self.feature = np.asarray(feature, np.int64)
        self.threshold = np.asarray(threshold, np.float64)
        self.left = np.asarray(left, np.int64)
        self.right = np.asarray(right, np.int64)
        self.value = np.asarray(value, np.float64)
        self.count = np.asarray(count, np.int64)
        self.weight = np.asarray(weight, np.float64)
        return self

    def _best_split(self, Xn, Rn, wn):
        """Split with the largest reduction of the weighted squared error
        (all classes together). Ties: lowest feature, lowest threshold.
        None when no split improves the error."""
        n, d = Xn.shape
        m = self.min_samples_leaf
        WR = wn[:, None] * Rn
        S = WR.sum(axis=0)
        W = wn.sum()
        base = float((S * S).sum() / W) if W > 0 else 0.0
        best_gain, best = 1e-12 * max(base, 1.0), None
        pos = np.arange(1, n)                   # rows on the left
        ok_size = (pos >= m) & (n - pos >= m)
        if not ok_size.any():
            return None
        for f in range(d):
            col = Xn[:, f]
            order = np.argsort(col, kind='mergesort')
            sc = col[order]
            valid = ok_size & (sc[1:] > sc[:-1])
            if not valid.any():
                continue
            cs = np.cumsum(WR[order], axis=0)[:-1]       # left sums
            wl = np.cumsum(wn[order])[:-1]          # weights in sorted order
            wr = W - wl
            with np.errstate(divide='ignore', invalid='ignore'):
                gl = (cs * cs).sum(axis=1) / wl
                gr = ((S - cs) ** 2).sum(axis=1) / wr
            gain = gl + gr - base
            gain = np.where(valid & (wl > 0) & (wr > 0), gain, -np.inf)
            k = int(np.argmax(gain))
            if gain[k] > best_gain + 1e-15:
                best_gain = float(gain[k])
                # midpoint of the two neighbouring values
                best = (f, 0.5 * (sc[k] + sc[k + 1]))
        return best

    def prune(self, min_abs=0.1):
        """Merge the subtrees that cannot change a result: a subtree whose
        leaves ALL have a shift below min_abs in every class becomes one
        leaf. Its value is the weighted mean residual of all its rows (the
        value the node already holds), so the output stays almost the same
        and the rules get much shorter. Splits that only sharpen a
        probability (a row the base model has right with high confidence
        has a small residual that follows its own probability) disappear.
        Returns the number of leaves before and after."""
        self._check_fitted()
        n = len(self.feature)
        before = int((self.feature < 0).sum())
        small = np.abs(self.value).max(axis=1) < float(min_abs)
        ok = np.zeros(n, bool)
        for i in range(n - 1, -1, -1):          # children come after parents
            ok[i] = (small[i] if self.feature[i] < 0
                     else ok[self.left[i]] and ok[self.right[i]])
        feature, threshold, left, right, value, count = [], [], [], [], [], []
        weight = []
        has_w = self.weight is not None

        def emit(i):
            """Copy node i (and below) in preorder: children get larger
            indices than their parent."""
            k = len(feature)
            collapse = self.feature[i] >= 0 and ok[i]
            feature.append(-1 if collapse else int(self.feature[i]))
            threshold.append(0.0 if collapse else float(self.threshold[i]))
            left.append(-1)
            right.append(-1)
            value.append(self.value[i].copy())
            count.append(int(self.count[i]))
            weight.append(float(self.weight[i]) if has_w else 0.0)
            if self.feature[i] >= 0 and not collapse:
                left[k] = emit(int(self.left[i]))
                right[k] = emit(int(self.right[i]))
            return k
        import sys
        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(max(old_limit, 10 * self.max_depth + 1000))
        try:
            emit(0)
        finally:
            sys.setrecursionlimit(old_limit)
        self.feature = np.asarray(feature, np.int64)
        self.threshold = np.asarray(threshold, np.float64)
        self.left = np.asarray(left, np.int64)
        self.right = np.asarray(right, np.int64)
        self.value = np.asarray(value, np.float64)
        self.count = np.asarray(count, np.int64)
        self.weight = np.asarray(weight, np.float64) if has_w else None
        after = int((self.feature < 0).sum())
        self.pruned_from = before
        return before, after

    def feature_gain(self):
        """Share of the total split gain of each input column (sums to 1,
        or all zero for a tree with no split). The gain of a split is the
        drop of the weighted squared residual shift it explains:
        W_l*|v_l|^2 + W_r*|v_r|^2 - W_n*|v_n|^2. None when the node
        weights are unknown (a model loaded from a file)."""
        self._check_fitted()
        if self.weight is None:
            return None
        g = np.zeros(self.n_features_)
        for i in np.nonzero(self.feature >= 0)[0]:
            l, r = self.left[i], self.right[i]
            v = lambda k: float((self.value[k] ** 2).sum())
            gain = (self.weight[l] * v(l) + self.weight[r] * v(r)
                    - self.weight[i] * v(i))
            g[self.feature[i]] += max(gain, 0.0)
        tot = g.sum()
        return g / tot if tot > 0 else g

    # -- prediction -----------------------------------------------------

    def _leaf(self, X):
        node = np.zeros(X.shape[0], np.int64)
        while True:
            f = self.feature[node]
            inner = f >= 0
            if not inner.any():
                return node
            rows = np.nonzero(inner)[0]
            go_left = X[rows, f[rows]] <= self.threshold[node[rows]]
            node[rows] = np.where(go_left, self.left[node[rows]],
                                  self.right[node[rows]])

    def shift(self, X, logits):
        """Probability shift of every row (n, K)."""
        self._check_fitted()
        X, logits = self._check_xz(X, logits)
        if X.shape[1] != self.n_features_:
            raise ValueError(f'X has {X.shape[1]} columns, the tree was '
                             f'fitted with {self.n_features_}')
        if logits.shape[1] != self.n_classes_:
            raise ValueError(f'logits have {logits.shape[1]} classes, the '
                             f'tree was fitted with {self.n_classes_}')
        v = self.value[self._leaf(X)]
        if self.gate_col >= 0:
            v = v * self._gate_mask(X)[:, None]
        return v

    def predict_proba(self, X, logits):
        p = _softmax(np.asarray(logits, dtype=np.float64))
        q = np.maximum(p + self.shift(X, logits), EPS)
        return q / q.sum(axis=1, keepdims=True)

    def predict_logits(self, X, logits):
        p = _softmax(np.asarray(logits, dtype=np.float64))
        q = np.maximum(p + self.shift(X, logits), EPS)
        return np.log(q).astype(np.float32)

    # -- rules ----------------------------------------------------------

    def n_leaves(self):
        self._check_fitted()
        return int((self.feature < 0).sum())

    def export_text(self, feature_names=None, class_names=None,
                    min_abs=0.1):
        """Readable rules. A leaf lists the probability shifts of at least
        min_abs ('title: +0.62' adds 0.62 to the probability of title)."""
        if self.feature is None:
            return 'MeanResidualTree (not fitted)'
        K, d = self.n_classes_, self.n_features_
        fn = (list(feature_names) if feature_names is not None
              else [f'x[{j}]' for j in range(d)])
        cn = (list(class_names) if class_names is not None
              else [f'class {k}' for k in range(K)])
        if len(fn) != d:
            raise ValueError(f'feature_names needs {d} entries')
        if len(cn) != K:
            raise ValueError(f'class_names needs {K} entries')
        lines = ['output = softmax(logits) + leaf shift  '
                 '(classes with a shift of at least '
                 f'{min_abs:g} are listed)']
        if self.pruned_from:
            lines.append(f'pruned: {self.pruned_from} -> '
                         f'{self.n_leaves()} leaves (subtrees with only '
                         f'tiny shifts are merged)')
        if self.gate_col >= 0:
            cond = (f'== {self.gate_value}' if self.gate_value >= 1
                    else '>= 1')
            lines.append(f'the shift is applied only where '
                         f'{fn[self.gate_col]} {cond} (otherwise: no change)')

        def walk(i, indent):
            pad = '  ' * indent
            if self.feature[i] < 0:
                v = self.value[i]
                items = [f'{cn[k]}: {v[k]:+.2f}' for k in range(K)
                         if abs(v[k]) >= min_abs]
                txt = ', '.join(items) if items else 'no change'
                lines.append(f'{pad}-> {txt}  (n={int(self.count[i])})')
                return
            name, thr = fn[self.feature[i]], self.threshold[i]
            lines.append(f'{pad}if {name} <= {thr:.6g}:')
            walk(self.left[i], indent + 1)
            lines.append(f'{pad}else:  # {name} > {thr:.6g}')
            walk(self.right[i], indent + 1)

        walk(0, 0)
        return '\n'.join(lines)

    # -- files ----------------------------------------------------------

    @staticmethod
    def _unique_names(names, n, what):
        """names (or x[j] / class j) made unique: a repeated name gets #j."""
        names = ([f'x[{j}]' for j in range(n)] if what == 'feature' else
                 [f'class {j}' for j in range(n)]) if names is None \
            else [str(v) for v in names]
        if len(names) != n:
            raise ValueError(f'{what}_names needs {n} entries, got '
                             f'{len(names)}')
        seen, out = set(), []
        for j, v in enumerate(names):
            if v in seen:
                v = f'{v}#{j}'
            seen.add(v)
            out.append(v)
        return out

    def to_dict(self, feature_names=None, class_names=None):
        """The model as a nested dict (file version 2). A split is
        {"if": feature, "<=": threshold, "n": rows, "then": node, "else":
        node} (then = the feature is <= threshold); a leaf is {"shift":
        {class: probability shift, ...}, "n": rows}. Features and classes are
        named (x[j] / class j without names), and "features" / "classes"
        list them in the order of the model input / output: a person can
        read and edit the tree. n is information only."""
        self._check_fitted()
        fn = self._unique_names(feature_names, self.n_features_, 'feature')
        cn = self._unique_names(class_names, self.n_classes_, 'class')

        def node(i):
            n = int(self.count[i])
            if self.feature[i] < 0:
                return {'shift': {cn[k]: float(self.value[i][k])
                                  for k in range(self.n_classes_)}, 'n': n}
            return {'if': fn[int(self.feature[i])],
                    '<=': float(self.threshold[i]), 'n': n,
                    'then': node(int(self.left[i])),
                    'else': node(int(self.right[i]))}
        return dict(
            format=FORMAT, version=VERSION, max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            gate=(None if self.gate_col < 0 else
                  {'column': fn[self.gate_col], 'value': self.gate_value}),
            tree=node(0), features=fn, classes=cn)

    @staticmethod
    def _fmt_node(node, lvl):
        """The text of a node: a leaf on one line, a split with then / else
        on their own lines, indented by two spaces per level."""
        if 'shift' in node:
            sh = ', '.join(f'{json.dumps(k)}: {json.dumps(v)}'
                           for k, v in node['shift'].items())
            return f'{{"shift": {{{sh}}}, "n": {node["n"]}}}'
        pad = '  ' * (lvl + 1)
        return (f'{{"if": {json.dumps(node["if"])}, "<=": '
                f'{json.dumps(node["<="])}, "n": {node["n"]},\n'
                f'{pad}"then": '
                f'{MeanResidualTree._fmt_node(node["then"], lvl + 1)},\n'
                f'{pad}"else": '
                f'{MeanResidualTree._fmt_node(node["else"], lvl + 1)}}}')

    def to_text(self, feature_names=None, class_names=None):
        """to_dict as json text for people: the tree first, one leaf per
        line, the lists of the feature and class names at the end."""
        d = self.to_dict(feature_names, class_names)
        head = ',\n'.join(
            f'  {json.dumps(k)}: {json.dumps(d[k])}'
            for k in ('format', 'version', 'max_depth', 'min_samples_leaf',
                      'gate'))
        return (f'{{\n{head},\n  "tree": '
                f'{self._fmt_node(d["tree"], 1)},\n'
                f'  "features": {json.dumps(d["features"])},\n'
                f'  "classes": {json.dumps(d["classes"])}\n}}\n')

    @classmethod
    def from_dict(cls, d):
        """The model of a to_dict dict (or of a file a person edited). Errors
        name the place in the tree, e.g. tree.then.else: ..."""
        if d.get('format') != FORMAT:
            raise ValueError(f"format {d.get('format')!r}, expected "
                             f'{FORMAT!r}')
        if d.get('version') != VERSION:
            raise ValueError(f"version {d.get('version')!r}, this code "
                             f'reads version {VERSION}')
        fn, cn = d.get('features'), d.get('classes')
        for what, v in (('features', fn), ('classes', cn)):
            if (not isinstance(v, list) or not v
                    or not all(isinstance(q, str) for q in v)
                    or len(set(v)) != len(v)):
                raise ValueError(f'"{what}" must be a list of different '
                                 f'names')
        fidx = {nm: j for j, nm in enumerate(fn)}
        K = len(cn)
        gate = d.get('gate')
        gate_col, gate_value = -1, 0
        if gate is not None:
            if gate.get('column') not in fidx:
                raise ValueError(f'gate: column {gate.get("column")!r} is '
                                 f'not in "features"')
            gate_col, gate_value = fidx[gate['column']], int(gate['value'])
        m = cls(d['max_depth'], d['min_samples_leaf'], gate_col, gate_value)
        m.n_features_, m.n_classes_ = len(fn), K
        feature, threshold, left, right, value, count = [], [], [], [], [], []

        def num(v, where, key):
            if isinstance(v, bool) or not isinstance(v, (int, float)) \
                    or not np.isfinite(v):
                raise ValueError(f'{where}: "{key}" must be a finite number, '
                                 f'got {v!r}')
            return float(v)

        def walk(nd, where):
            if not isinstance(nd, dict):
                raise ValueError(f'{where}: a node must be an object')
            leaf = 'shift' in nd
            allowed = ({'shift', 'n'} if leaf
                       else {'if', '<=', 'n', 'then', 'else'})
            extra = set(nd) - allowed
            if extra:
                raise ValueError(f'{where}: unknown key(s) {sorted(extra)}')
            n = nd.get('n', 0)
            if isinstance(n, bool) or not isinstance(n, int) or n < 0:
                raise ValueError(f'{where}: "n" must be a whole number')
            k = len(feature)
            feature.append(-1)
            threshold.append(0.0)
            left.append(-1)
            right.append(-1)
            value.append(np.zeros(K))
            count.append(n)
            if leaf:
                sh = nd['shift']
                if not isinstance(sh, dict) or set(sh) != set(cn):
                    miss = sorted(set(cn) - set(sh)) if isinstance(sh, dict) \
                        else cn
                    raise ValueError(f'{where}: "shift" needs exactly the '
                                     f'classes of "classes" (missing or '
                                     f'wrong: {miss})')
                value[k] = np.array([num(sh[c], where, c) for c in cn])
                return k
            for key in ('if', '<=', 'then', 'else'):
                if key not in nd:
                    raise ValueError(f'{where}: a split needs "{key}"')
            if nd['if'] not in fidx:
                raise ValueError(f'{where}: feature {nd["if"]!r} is not in '
                                 f'"features"')
            feature[k] = fidx[nd['if']]
            threshold[k] = num(nd['<='], where, '<=')
            left[k] = walk(nd['then'], where + '.then')
            right[k] = walk(nd['else'], where + '.else')
            return k

        import sys
        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(max(old_limit, 10 * m.max_depth + 1000))
        try:
            walk(d.get('tree'), 'tree')
        finally:
            sys.setrecursionlimit(old_limit)
        m.feature = np.asarray(feature, np.int64)
        m.threshold = np.asarray(threshold, np.float64)
        m.left = np.asarray(left, np.int64)
        m.right = np.asarray(right, np.int64)
        m.value = np.asarray(value, np.float64)
        m.count = np.asarray(count, np.int64)
        # the value of a split node (not stored): the mean of its leaves,
        # weighted by n (equal when n is 0); inference does not use it
        cnt = m.count.astype(np.float64)
        w = np.where(cnt > 0, cnt, 1.0)
        for i in range(len(feature) - 1, -1, -1):
            if m.feature[i] >= 0:
                l, r = m.left[i], m.right[i]
                m.value[i] = (w[l] * m.value[l] + w[r] * m.value[r]) \
                    / (w[l] + w[r])
                w[i] = w[l] + w[r]
        return m

    def save(self, path, feature_names=None, class_names=None):
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(self.to_text(feature_names, class_names))

    @classmethod
    def load(cls, path):
        with open(path, 'r', encoding='utf-8') as fh:
            return cls.from_dict(json.load(fh))
