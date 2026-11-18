"""Output + residual shift learned by a small MLP (large sample counterpart of
MeanResidualTree).

The base model gives logits z, p = softmax(z). The MLP learns a probability
shift s(x) so that the corrected output is

    q = p + s(x),   logits = log(max(q, 1e-6))

Same meaning as MeanResidualTree (a leaf value is also a probability shift),
so the two models are interchangeable behind CNN4GNNLocalExpertModel.

Training uses torch (imported only inside fit). Inference uses ONNX only:
the whole model (input transform, standardization, MLP, tanh, zero sum
projection, softmax, p + s, clamp, log) is ONE ONNX graph, so a model file
is self contained and has no separate statistics that can get lost. The
report numbers are computed with the exported graph, so they are the numbers
of the real inference.

Design
------
* shift      s = tanh(out(MLP(t))) - mean_over_classes(tanh(...))
               so every shift is in about [-1, 1] and sums to zero over the
               classes (the same properties as the tree leaf values).
* gate_value >= 1 narrows the gate to one category: the shift is multiplied
               by (|x[:, gate_col] - gate_value| < 0.5) instead.
* gate       gate_col >= 0 names a column of x (the model input
               [softmax, features]); the shift is multiplied by
               (x[:, gate_col] >= 1) inside the graph, so a row with a gate
               below 1 keeps the base output. Rows with a gate below 1 are
               left out of the fit. This is the document_type protocol.
* last layer is zero initialised: before training the output is exactly
  the base model ("no change").
* input transform: t = sign(x) * log(1 + abs(x)), then (t - mean) / std
  clipped to +-10. Constant columns get std 1 (they become 0).
* loss       weighted squared error of (onehot(y) - p) - s, the same loss
  as the tree. Rows are weighted; the caller decides the weights.
* optimizer  AdamW with weight decay, fixed number of epochs.
* batches    protected_share = 0: plain shuffled batches, the protected
  (corrected) rows reach the loss only through their row weights, so a
  batch holds a random number of them (maybe none).
  protected_share > 0: every batch holds a FIXED share of protected rows
  (drawn by cycling through a shuffled list of them, so all are used evenly)
  and the rest from the other rows. The loss of the two groups is averaged
  separately and combined with the same share, so the share is the share of
  the loss in EVERY batch. One epoch is one pass over the other rows. The
  expected gradient is the same as with the weights; only the batch to
  batch noise is removed. The protected rows are then seen many times per
  epoch when they are few: keep the share small and look at the held out
  numbers.
* hold out   optional, by PAGE (groups). Rows marked protected (the
  corrected pages) never go to the hold out. The epoch is chosen as the LAST
  epoch whose hold out loss is within holdout_tolerance of the best one, so
  the training does not stop before the protected rows are learned.

Interface (what CNN4GNNLocalExpertModel and CNN4GNNLocalExpertTrain need)
    fit(X, logits, y, sample_weight=None, groups=None, protected=None, log=None)
    predict_logits(X, logits) -> (n, K) float32     (onnxruntime)
    predict_proba(X, logits)
    export_text(feature_names, class_names)         short description
    save(path) / MeanResidualMLP.load(path)         .onnx file
    n_features_, n_classes_, fit_info
"""

import io

import numpy as np

EPS = 1e-6
CLIP = 10.0
FORMAT = 'MeanResidualMLP'
VERSION = 1
PARITY_TOL = 1e-4          # torch graph versus exported ONNX graph


def _softmax(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def _make_net(torch, d, K, hidden, n_layers, mean, std, keep, gate_col=-1,
              gate_value=0):
    nn = torch.nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32))
            self.register_buffer('std', torch.tensor(std, dtype=torch.float32))
            self.register_buffer('keep', torch.tensor(keep, dtype=torch.float32))
            layers, din = [], d
            for _ in range(n_layers):
                layers += [nn.Linear(din, hidden), nn.ReLU()]
                din = hidden
            self.body = nn.Sequential(*layers)
            self.out = nn.Linear(din, K)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

        def shift(self, x):
            t = torch.sign(x) * torch.log(1.0 + torch.abs(x))
            t = torch.clamp((t - self.mean) / self.std, -CLIP, CLIP) * self.keep
            s = torch.tanh(self.out(self.body(t)))
            s = s - s.mean(dim=1, keepdim=True)
            if gate_col >= 0:
                g = x[:, gate_col:gate_col + 1]
                if gate_value >= 1:
                    s = s * (torch.abs(g - float(gate_value)) < 0.5).to(
                        s.dtype)
                else:
                    s = s * (g >= 1.0).to(s.dtype)
            return s

        def forward(self, x, logits):
            p = torch.softmax(logits, dim=1)
            q = torch.clamp(p + self.shift(x), min=EPS)
            return torch.log(q)

    return Net()


class MeanResidualMLP:
    def __init__(self, hidden_dim=128, n_layers=2, epochs=60, lr=1e-3,
                 weight_decay=1e-4, batch_size=2048, holdout_frac=0.1,
                 holdout_tolerance=0.05, seed=0, protected_share=0.0,
                 gate_col=-1, gate_value=0):
        if int(hidden_dim) < 1 or int(n_layers) < 1 or int(epochs) < 1:
            raise ValueError('hidden_dim, n_layers and epochs must be >= 1')
        if not float(lr) > 0 or float(weight_decay) < 0:
            raise ValueError('lr must be > 0 and weight_decay >= 0')
        if int(batch_size) < 1:
            raise ValueError('batch_size must be >= 1')
        if not 0.0 <= float(holdout_frac) < 1.0:
            raise ValueError('holdout_frac must be in [0, 1)')
        if float(holdout_tolerance) < 0:
            raise ValueError('holdout_tolerance must be >= 0')
        if not 0.0 <= float(protected_share) < 1.0:
            raise ValueError('protected_share must be in [0, 1)')
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.holdout_frac = float(holdout_frac)
        self.holdout_tolerance = float(holdout_tolerance)
        self.seed = int(seed)
        self.protected_share = float(protected_share)
        self.gate_col = int(gate_col)            # -1: no gate
        self.gate_value = int(gate_value)        # 0: any value of 1 or more
        self.n_features_ = None
        self.n_classes_ = None
        self.fit_info = {}
        self.importance_ = []          # [(column, value)] of the corrected rows
        self._onnx = None
        self._session = None

    # -- onnx -----------------------------------------------------------

    def _set_onnx(self, data):
        import onnxruntime as ort
        sess = ort.InferenceSession(data, providers=['CPUExecutionProvider'])
        ins = {i.name: i.shape for i in sess.get_inputs()}
        if set(ins) != {'x', 'logits'}:
            raise ValueError(f'unexpected ONNX inputs {sorted(ins)}, '
                             f"expected ['logits', 'x']")
        d, k = ins['x'][1], ins['logits'][1]
        if not (isinstance(d, int) and isinstance(k, int)):
            raise ValueError('the ONNX model has no fixed column counts')
        self._onnx, self._session = data, sess
        self.n_features_, self.n_classes_ = int(d), int(k)

    def _check_fitted(self):
        if self._session is None:
            raise RuntimeError('the model is not fitted')

    # -- fitting --------------------------------------------------------

    def fit(self, X, logits, y, sample_weight=None, groups=None,
            protected=None, log=None, ignore_cols=None,
            importance_all=False, progress=None):
        """X (n, d), logits (n, K), y integer labels, sample_weight (n,).
        groups: page id per row (hold out is made by page). protected: bool
        per row, rows that are never held out (the corrected pages).
        ignore_cols: column indices the model must not see (they are
        multiplied by zero inside the exported graph). importance_all:
        the column importance is measured on a sample of all rows, not on
        the protected rows only. progress: optional callable (epoch done,
        epochs), called after every epoch; it may raise to stop."""
        import torch
        log = log or (lambda m: None)
        X = np.asarray(X, dtype=np.float32)
        L = np.asarray(logits, dtype=np.float32)
        if X.ndim != 2 or L.ndim != 2 or X.shape[0] != L.shape[0]:
            raise ValueError(f'X {X.shape} and logits {L.shape} must be 2-d '
                             f'with the same number of rows')
        n, d = X.shape
        K = L.shape[1]
        if n == 0:
            raise ValueError('no rows to fit')
        if not np.isfinite(X).all() or not np.isfinite(L).all():
            raise ValueError('X or logits contain NaN or inf')
        y = np.asarray(y)
        if y.shape != (n,) or not np.issubdtype(y.dtype, np.integer):
            raise ValueError('y must be an integer vector, one label per row')
        if y.min() < 0 or y.max() >= K:
            raise ValueError(f'labels must be in 0..{K - 1}')
        if sample_weight is None:
            w = np.ones(n, dtype=np.float32)
        else:
            w = np.asarray(sample_weight, dtype=np.float64)
            if w.shape != (n,) or not np.isfinite(w).all() or (w < 0).any():
                raise ValueError('sample_weight must be a finite, '
                                 'non-negative vector, one per row')
            w = w.astype(np.float32)
        if float(w.sum()) <= 0:
            raise ValueError('the sample weights sum to zero')
        prot = (np.zeros(n, bool) if protected is None
                else np.asarray(protected, dtype=bool))
        if prot.shape != (n,):
            raise ValueError('protected needs one value per row')
        if groups is not None:
            groups = np.asarray(groups)
            if groups.shape != (n,):
                raise ValueError('groups needs one value per row')

        # the gate: rows below 1 keep the base output, they are not fitted
        if self.gate_col >= 0:
            if self.gate_col >= d:
                raise ValueError(f'gate_col {self.gate_col} is outside the '
                                 f'{d} columns')
            gv = X[:, self.gate_col]
            g = (np.abs(gv - self.gate_value) < 0.5 if self.gate_value >= 1
                 else gv >= 1.0)
            if not g.any():
                raise ValueError('no row has the gate column at 1 or more')
            if not g.all():
                X, L, y, w, prot = X[g], L[g], y[g], w[g], prot[g]
                if groups is not None:
                    groups = groups[g]
                n = len(y)
                if float(w.sum()) <= 0:
                    raise ValueError('the rows of the gate have weight zero')
        # hold out by page, never from the protected rows
        rng = np.random.default_rng(self.seed)
        hold = np.zeros(n, bool)
        if self.holdout_frac > 0 and groups is not None:
            pages = np.unique(groups[~prot])
            n_h = int(round(len(pages) * self.holdout_frac))
            if len(pages) >= 10 and n_h >= 1:
                pick = rng.choice(pages, size=n_h, replace=False)
                hold = (~prot) & np.isin(groups, pick)
        use_hold = bool(hold.any())
        tr_idx = np.nonzero(~hold)[0]
        ho_idx = np.nonzero(hold)[0]
        if len(tr_idx) == 0 or float(w[tr_idx].sum()) <= 0:
            raise ValueError('no weighted training rows are left')
        log(f'  mlp: rows {n}, training rows {len(tr_idx)}, hold out rows '
            f'{len(ho_idx)}, protected rows {int(prot.sum())}')
        tr_prot = tr_idx[prot[tr_idx]]
        tr_pool = tr_idx[~prot[tr_idx]]
        strat = (self.protected_share > 0 and len(tr_prot) > 0
                 and len(tr_pool) > 0
                 and float(w[tr_prot].sum()) > 0 and float(w[tr_pool].sum()) > 0)
        n_pb = n_gb = 0
        if strat:
            n_pb = min(max(1, int(round(self.batch_size
                                        * self.protected_share))),
                       max(self.batch_size - 1, 1))
            n_gb = max(self.batch_size - n_pb, 1)
            log(f'  mlp: stratified batches, {n_pb} protected rows and '
                f'{n_gb} other rows per batch (each protected row is used '
                f'about {n_pb * int(np.ceil(len(tr_pool) / n_gb)) / len(tr_prot):.1f}'
                f' times per epoch)')
        elif self.protected_share > 0:
            log('  mlp: no protected rows or no other rows: plain batches')

        # standardization statistics of the training rows
        s1, s2 = np.zeros(d), np.zeros(d)
        for a in range(0, len(tr_idx), 100000):
            t = _slog(X[tr_idx[a:a + 100000]].astype(np.float64))
            s1 += t.sum(axis=0)
            s2 += (t * t).sum(axis=0)
        mean = s1 / len(tr_idx)
        std = np.sqrt(np.maximum(s2 / len(tr_idx) - mean * mean, 0.0))
        std = np.where(std < 1e-6, 1.0, std)

        torch.manual_seed(self.seed)
        keep = np.ones(d, dtype=np.float32)
        if ignore_cols is not None and len(ignore_cols):
            keep[np.asarray(list(ignore_cols), dtype=np.int64)] = 0.0
        net = _make_net(torch, d, K, self.hidden_dim, self.n_layers,
                        mean, std, keep, self.gate_col, self.gate_value)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        eye = torch.eye(K)

        def batch_loss(idx):
            xb = torch.from_numpy(X[idx])
            lb = torch.from_numpy(L[idx])
            yb = torch.from_numpy(y[idx].astype(np.int64))
            wb = torch.from_numpy(w[idx])
            r = eye[yb] - torch.softmax(lb, dim=1)
            sq = ((r - net.shift(xb)) ** 2).sum(dim=1)
            return (wb * sq).sum(), wb.sum()

        def prot_stream():
            while True:
                for v in rng.permutation(tr_prot):
                    yield v

        def strat_loss(idx_g, idx_p):
            sg, wg_ = batch_loss(idx_g)
            sp, wp_ = batch_loss(idx_p)
            if float(wg_) <= 0 or float(wp_) <= 0:
                return None
            sh = self.protected_share
            return sh * (sp / wp_) + (1.0 - sh) * (sg / wg_)

        def hold_loss():
            net.eval()
            num, den = 0.0, 0.0
            with torch.no_grad():
                for a in range(0, len(ho_idx), 65536):
                    s, ws = batch_loss(ho_idx[a:a + 65536])
                    num += float(s)
                    den += float(ws)
            net.train()
            return num / den if den > 0 else float('nan')

        train_hist, hold_hist, states = [], [], []
        net.train()
        stream = prot_stream() if strat else None
        for ep in range(self.epochs):
            if strat:
                perm = rng.permutation(tr_pool)
                tot, wsum = 0.0, 0.0
                for a in range(0, len(perm), n_gb):
                    idx_g = perm[a:a + n_gb]
                    idx_p = np.fromiter((next(stream) for _ in range(n_pb)),
                                        dtype=np.int64, count=n_pb)
                    loss = strat_loss(idx_g, idx_p)
                    if loss is None:
                        continue
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    tot += float(loss.detach())
                    wsum += 1.0
                train_hist.append(tot / wsum if wsum > 0 else float('nan'))
                perm = perm[:0]
            else:
                perm = rng.permutation(tr_idx)
                tot, wsum = 0.0, 0.0
            for a in range(0, len(perm), self.batch_size):
                idx = perm[a:a + self.batch_size]
                s, ws = batch_loss(idx)
                if float(ws) <= 0:
                    continue
                loss = s / ws
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(s.detach())
                wsum += float(ws.detach()) if hasattr(ws, 'detach') else float(ws)
            if not strat:
                train_hist.append(tot / wsum if wsum > 0 else float('nan'))
            if use_hold:
                hold_hist.append(hold_loss())
                states.append({k: v.detach().clone()
                               for k, v in net.state_dict().items()})
            if progress is not None:
                progress(ep + 1, self.epochs)
            if ep == 0 or (ep + 1) % 10 == 0 or ep + 1 == self.epochs:
                log(f'  mlp epoch {ep + 1}/{self.epochs}: train loss '
                    f'{train_hist[-1]:.5f}'
                    + (f', hold out loss {hold_hist[-1]:.5f}'
                       if use_hold else ''))
        chosen = self.epochs - 1
        if use_hold:
            best = float(np.nanmin(hold_hist))
            ok = [e for e, h in enumerate(hold_hist)
                  if h <= best * (1.0 + self.holdout_tolerance)]
            chosen = ok[-1]
            net.load_state_dict(states[chosen])
            log(f'  mlp: epoch {chosen + 1} chosen (best hold out loss '
                f'{best:.5f}, chosen {hold_hist[chosen]:.5f})')
        net.eval()

        # export; the report and the inference use this graph
        xd = torch.from_numpy(X[:2].copy())
        ld = torch.from_numpy(L[:2].copy())
        buf = io.BytesIO()
        kw = dict(input_names=['x', 'logits'], output_names=['logits_out'],
                  dynamic_axes={'x': {0: 'n'}, 'logits': {0: 'n'},
                                'logits_out': {0: 'n'}},
                  opset_version=17)
        try:
            torch.onnx.export(net, (xd, ld), buf, dynamo=False, **kw)
        except TypeError:                       # older torch: no dynamo flag
            buf = io.BytesIO()
            torch.onnx.export(net, (xd, ld), buf, **kw)
        self._set_onnx(buf.getvalue())
        if (self.n_features_, self.n_classes_) != (d, K):
            raise RuntimeError('the exported model has other column counts')
        m = min(n, 512)
        probe = rng.choice(n, size=m, replace=False)
        with torch.no_grad():
            ref = net(torch.from_numpy(X[probe]),
                      torch.from_numpy(L[probe])).numpy()
        got = self.predict_logits(X[probe], L[probe])
        # compared as probabilities: log of a tiny q amplifies float32 noise
        diff = float(np.abs(np.exp(ref.astype(np.float64))
                            - np.exp(got.astype(np.float64))).max())
        if not diff < PARITY_TOL:
            raise RuntimeError(f'the ONNX graph differs from the torch '
                               f'model by {diff:.3g}')

        n_params = int(sum(p.numel() for p in net.parameters()))
        self.fit_info = dict(
            n_rows=int(n), n_train_rows=int(len(tr_idx)),
            n_holdout_rows=int(len(ho_idx)), n_protected=int(prot.sum()),
            n_params=n_params, epochs=self.epochs,
            gate_col=int(self.gate_col), gate_value=int(self.gate_value),
            stratified=bool(strat), protected_per_batch=int(n_pb),
            other_per_batch=int(n_gb),
            chosen_epoch=int(chosen + 1), train_loss=train_hist,
            holdout_loss=hold_hist, onnx_parity=diff)
        self._compute_importance(
            X, L, np.zeros(n, bool) if importance_all else prot, rng)
        self.fit_info['importance_rows'] = ('all' if importance_all
                                            else 'corrected')
        return self

    def _compute_importance(self, X, L, prot, rng):
        """Permutation importance on the protected rows (or a sample of all
        rows): mean absolute change of the output probabilities when one
        column is shuffled."""
        n, d = X.shape
        rows = np.nonzero(prot)[0]
        if len(rows) == 0:
            rows = np.arange(n)
        if len(rows) > 1500:
            rows = rng.choice(rows, size=1500, replace=False)
        Xs, Ls = X[rows], L[rows]
        if len(rows) < 2:
            self.importance_ = []
            return
        base = _softmax(self.predict_logits(Xs, Ls))
        imp = []
        for j in range(d):
            if float(Xs[:, j].max()) == float(Xs[:, j].min()):
                continue
            Xp = Xs.copy()
            Xp[:, j] = Xs[rng.permutation(len(rows)), j]
            cur = _softmax(self.predict_logits(Xp, Ls))
            imp.append((j, float(np.abs(cur - base).mean())))
        imp.sort(key=lambda t: -t[1])
        self.importance_ = imp[:30]
        self.fit_info['importance'] = [[int(j), float(v)]
                                       for j, v in self.importance_]

    # -- prediction (onnxruntime only) -----------------------------------

    def predict_logits(self, X, logits):
        self._check_fitted()
        X = np.asarray(X, dtype=np.float32)
        L = np.asarray(logits, dtype=np.float32)
        if X.ndim != 2 or L.ndim != 2 or X.shape[0] != L.shape[0]:
            raise ValueError(f'X {X.shape} and logits {L.shape} must be 2-d '
                             f'with the same number of rows')
        if X.shape[1] != self.n_features_:
            raise ValueError(f'X has {X.shape[1]} columns, the model was '
                             f'fitted with {self.n_features_}')
        if L.shape[1] != self.n_classes_:
            raise ValueError(f'logits have {L.shape[1]} classes, the model '
                             f'was fitted with {self.n_classes_}')
        if not np.isfinite(X).all():
            raise ValueError('X contains NaN or inf')
        if not np.isfinite(L).all():
            raise ValueError('logits contain NaN or inf')
        if X.shape[0] == 0:
            return np.zeros((0, self.n_classes_), dtype=np.float32)
        out = []
        for a in range(0, X.shape[0], 65536):
            out.append(self._session.run(
                ['logits_out'], {'x': np.ascontiguousarray(X[a:a + 65536]),
                                 'logits': np.ascontiguousarray(
                                     L[a:a + 65536])})[0])
        return np.concatenate(out).astype(np.float32)

    def predict_proba(self, X, logits):
        return _softmax(self.predict_logits(X, logits))

    # -- description ------------------------------------------------------

    def export_text(self, feature_names=None, class_names=None, min_abs=0.1):
        """The MLP has no readable rules; this gives its structure and the
        columns that matter most for the corrected rows."""
        if self._session is None:
            return 'MeanResidualMLP (not fitted)'
        d = self.n_features_
        fn = (list(feature_names) if feature_names is not None
              else [f'x[{j}]' for j in range(d)])
        if len(fn) != d:
            raise ValueError(f'feature_names needs {d} entries')
        fi = self.fit_info
        lines = [
            'MeanResidualMLP: no readable rules (the shift is a neural '
            'network inside the ONNX graph).',
            'output = softmax(logits) + shift(x), shift sums to zero over '
            'the classes',
            f'network: {d} inputs, {self.n_layers} hidden layers of '
            f'{self.hidden_dim}, {fi.get("n_params", "?")} parameters',
            f'rows: {fi.get("n_train_rows", "?")} training, '
            f'{fi.get("n_holdout_rows", "?")} hold out, '
            f'{fi.get("n_protected", "?")} corrected; epoch '
            f'{fi.get("chosen_epoch", "?")} of {self.epochs}',
        ]
        if self.importance_:
            which = ('all rows' if self.fit_info.get('importance_rows')
                     == 'all' else 'the corrected rows')
            lines.append(f'columns that matter most for {which} '
                         '(mean absolute change of the probabilities when '
                         'the column is shuffled):')
            for j, v in self.importance_[:15]:
                lines.append(f'  {fn[j]}: {v:.4f}')
        return '\n'.join(lines)

    # -- files ------------------------------------------------------------

    def save(self, path):
        self._check_fitted()
        with open(path, 'wb') as fh:
            fh.write(self._onnx)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as fh:
            data = fh.read()
        m = cls()
        m._set_onnx(data)
        return m
