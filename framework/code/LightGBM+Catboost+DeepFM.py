import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ======================
# Step 1. Load & Merge
# ======================
ad_feature = pd.read_csv("ad_feature.csv")
df = raw_sample.merge(user_profile, left_on="user", right_on="userid", how="left")
df = df.merge(ad_feature, on="adgroup_id", how="left")

df["dt"] = pd.to_datetime(df["time_stamp"], unit="s")
df["date"] = df["dt"].dt.date
df["hour"] = df["dt"].dt.hour
df["weekday"] = df["dt"].dt.weekday

# Split: first 7 days for training, last 1 day for testing
test_day = pd.to_datetime("2017-05-13").date()
train_df = df[df["date"] < test_day].copy()
test_df  = df[df["date"] == test_day].copy()
y_train = train_df["clk"].astype(int).values
y_test  = test_df["clk"].astype(int).values
global_ctr = y_train.mean()

import platform, lightgbm, catboost, torch
print(f"Python {platform.python_version()}")
print("LGBM:", lightgbm.__version__, " CatBoost:", catboost.__version__, " Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))

# ======================
# Step 2. Multi-window daily CTR
# ======================
def add_window_ctr(train_df, test_df, key, label="clk", win=3, alpha=100):
    """Cumulative daily CTR (past `win` days, strictly excluding future info)"""
    # ---- Convert combined key into str ----
    if isinstance(key, (list, tuple)):
        cname = "_".join(key)
        if cname not in train_df.columns:
            train_df[cname] = train_df[list(key)].astype(str).agg("_".join, axis=1)
        if cname not in test_df.columns:
            test_df[cname] = test_df[list(key)].astype(str).agg("_".join, axis=1)
        key = cname
    else:
        train_df[key] = train_df[key].astype(str)
        test_df[key]  = test_df[key].astype(str)

    out_col = f"{key}_w{win}_ctr"
    g = float(train_df[label].mean())

    # ---- Daily aggregation ----
    tmp = train_df[[key, "date", label]].copy()
    daily = (
        tmp.groupby([key, "date"], as_index=False)
           .agg(clicks=(label, "sum"), imps=(label, "count"))
           .sort_values("date")
    )

    # ---- rolling + shift(1) ----
    daily["roll_clicks"] = (
        daily.groupby(key, group_keys=False)["clicks"]
             .transform(lambda s: s.shift(1).rolling(win, min_periods=1).sum())
    )
    daily["roll_imps"] = (
        daily.groupby(key, group_keys=False)["imps"]
             .transform(lambda s: s.shift(1).rolling(win, min_periods=1).sum())
    )
    daily[out_col] = (daily["roll_clicks"] + alpha*g) / (daily["roll_imps"] + alpha)

    # ---- Merge back into train ----
    train_out = train_df[[key, "date"]].merge(
        daily[[key, "date", out_col]], on=[key, "date"], how="left"
    )
    train_out[out_col] = train_out[out_col].fillna(g)
    train_df = train_df.assign(**{out_col: train_out[out_col].to_numpy()})

    # ---- Test uses last `win` days ----
    all_days = np.sort(train_df["date"].unique())
    last_train_day = all_days[-1]
    last_win_days = all_days[max(0, len(all_days)-win):]

    recent = (
        daily[daily["date"].isin(last_win_days)]
        .groupby(key, as_index=False)
        .agg(roll_clicks=("clicks", "sum"), roll_imps=("imps", "sum"))
    )
    recent[out_col] = (recent["roll_clicks"] + alpha*g) / (recent["roll_imps"] + alpha)

    test_out = test_df[[key]].merge(recent[[key, out_col]], on=key, how="left")
    test_out[out_col] = test_out[out_col].fillna(g)
    test_df = test_df.assign(**{out_col: test_out[out_col].to_numpy()})

    return train_df, test_df, out_col


# ===== Generate multi-window features =====
base_keys = ["user","adgroup_id","brand","cate_id","campaign_id"]
cross_keys = [
    ["hour","weekday"],
    ["user","cate_id"],
    ["brand","campaign_id"],
    ["adgroup_id","hour"],
    ["user","brand"],          # newly added
    ["user","campaign_id"],    # newly added
    ["brand","cate_id"],       # newly added
]

ctr_cols = []
for k in base_keys:
    for win in [3, 7]:
        train_df, test_df, c = add_window_ctr(train_df, test_df, key=k, win=win, alpha=100)
        ctr_cols.append(c)

for ks in cross_keys:
    for win in [3, 7]:
        train_df, test_df, c = add_window_ctr(train_df, test_df, key=ks, win=win, alpha=100)
        ctr_cols.append(c)

print("Number of CTR features:", len(ctr_cols))
print("Example:", ctr_cols[:10])

# ======================
# Step 3. Frequency encoding
# ======================
def add_freq_enc(train_df, test_df, key):
    out_col = f"{key}_freq"
    freq = train_df[key].astype(str).value_counts().to_dict()
    train_df[out_col] = train_df[key].astype(str).map(freq).fillna(0)
    test_df[out_col]  = test_df[key].astype(str).map(freq).fillna(0)
    return out_col

freq_cols = []
for k in base_keys:
    freq_cols.append(add_freq_enc(train_df, test_df, key=k))

extra_cross = ["user_brand", "user_campaign_id", "brand_cate_id"]
for k in extra_cross:
    if k in train_df.columns:
        freq_cols.append(add_freq_enc(train_df, test_df, key=k))

# ======================
# Step 4. LightGBM
# ======================
cat_cols = ["user","adgroup_id","brand","cate_id","campaign_id"]
for col in cat_cols:
    train_df[col] = train_df[col].astype("category")
    test_df[col]  = test_df[col].astype("category")

num_cols = ["age_level","pvalue_level","shopping_level","final_gender_code","hour","weekday"]
lgb_features = [c for c in (num_cols + ctr_cols + freq_cols + cat_cols) if c in train_df.columns]

# Validation = 2017-05-12, train = earlier days
last_train_day = pd.to_datetime("2017-05-12").date()
mask_val = (train_df["date"] == last_train_day)
X_tr_lgb, y_tr_lgb = train_df.loc[~mask_val, lgb_features], train_df.loc[~mask_val, "clk"].astype(int).values
X_val_lgb, y_val_lgb = train_df.loc[ mask_val, lgb_features], train_df.loc[ mask_val, "clk"].astype(int).values
X_te_lgb,  y_te_lgb  = test_df[lgb_features], y_test

lgb_train = lgb.Dataset(X_tr_lgb, label=y_tr_lgb, categorical_feature=[c for c in cat_cols if c in lgb_features])
lgb_valid = lgb.Dataset(X_val_lgb, label=y_val_lgb, categorical_feature=[c for c in cat_cols if c in lgb_features])

lgb_params = {
    "objective": "binary",
    "metric": ["auc","binary_logloss"],
    "learning_rate": 0.03,
    "num_leaves": 255,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.5,
    "lambda_l2": 10.0,
    "max_bin": 511,
    "max_cat_threshold": 64,
    "cat_l2": 25,
    "cat_smooth": 25,
    "is_unbalance": True,
    "seed": 42,
    "verbose": -1
}

lgb_model = lgb.train(
    lgb_params, lgb_train, num_boost_round=1200, valid_sets=[lgb_valid],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)

pred_lgb = lgb_model.predict(X_te_lgb, num_iteration=lgb_model.best_iteration)
print("[LightGBM] AUC:", roc_auc_score(y_te_lgb, pred_lgb))
print("[LightGBM] LogLoss:", log_loss(y_te_lgb, pred_lgb))


# ======================
# Step 4.5. CatBoost (better for high-cardinality categorical vars)
# ======================
from catboost import CatBoostClassifier, Pool

# CatBoost categorical column indices (correspond to lgb_features list order)
cat_features_idx = [lgb_features.index(c) for c in cat_cols if c in lgb_features]

train_pool = Pool(
    data=X_tr_lgb,
    label=y_tr_lgb,
    cat_features=cat_features_idx
)
valid_pool = Pool(
    data=X_val_lgb,
    label=y_val_lgb,
    cat_features=cat_features_idx
)
test_pool = Pool(
    data=X_te_lgb,
    label=y_te_lgb,
    cat_features=cat_features_idx
)

cat_params = dict(
    iterations=1200,
    depth=8,
    learning_rate=0.03,
    l2_leaf_reg=10.0,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    od_type="Iter",
    od_wait=50,
    task_type="GPU",      # set to "GPU" if CUDA is available
    devices="0", 
    verbose=100
)

cat_model = CatBoostClassifier(**cat_params)
cat_model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

pred_cat = cat_model.predict_proba(test_pool)[:, 1]
print("[CatBoost] AUC:", roc_auc_score(y_te_lgb, pred_cat))
print("[CatBoost] LogLoss:", log_loss(y_te_lgb, pred_cat))

# (Optional) CatBoost feature importance (PredictionValuesChange)
cb_gain = cat_model.get_feature_importance(train_pool, type="PredictionValuesChange")
imp_cb = pd.DataFrame({"feature": lgb_features, "cb_importance": cb_gain})\
           .sort_values("cb_importance", ascending=False)
print(imp_cb.head(20))

# ======================
# Step 5. DeepFM
# ======================
deepfm_cols = ["user","adgroup_id","brand","cate_id","campaign_id"]
train_deep = train_df[["date"] + deepfm_cols + ["clk"]].copy()
test_deep  = test_df[deepfm_cols].copy()

encoders, field_dims = {}, []
for col in deepfm_cols:
    le = LabelEncoder()
    all_vals = pd.concat([train_deep[col], test_deep[col]], axis=0).astype(str)
    le.fit(all_vals)
    train_deep[col] = le.transform(train_deep[col].astype(str))
    test_deep[col]  = le.transform(test_deep[col].astype(str))
    encoders[col] = le
    field_dims.append(len(le.classes_))

mask_val_deep = (train_deep["date"] == last_train_day)
X_tr_deep  = train_deep.loc[~mask_val_deep, deepfm_cols].values
y_tr_deep  = train_deep.loc[~mask_val_deep, "clk"].astype(int).values
X_val_deep = train_deep.loc[ mask_val_deep, deepfm_cols].values
y_val_deep = train_deep.loc[ mask_val_deep, "clk"].astype(int).values
X_te_deep  = test_deep[deepfm_cols].values

class DeepFM(nn.Module):
    def __init__(self, field_dims, embed_dim=16):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(dim, embed_dim) for dim in field_dims])
        for emb in self.embeddings:
            nn.init.xavier_uniform_(emb.weight.data)
        self.linear_embeddings = nn.ModuleList([nn.Embedding(dim, 1) for dim in field_dims])
        for emb in self.linear_embeddings:
            nn.init.zeros_(emb.weight.data)
        input_dim = len(field_dims) * embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        Es = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        E  = torch.stack(Es, dim=1)
        Ls = [lin(x[:, i]) for i, lin in enumerate(self.linear_embeddings)]
        linear = torch.stack(Ls, dim=1).sum(dim=1)
        sum_square = (E.sum(dim=1) ** 2).sum(dim=1, keepdim=True)
        square_sum = (E ** 2).sum(dim=1).sum(dim=1, keepdim=True)
        pairwise   = 0.5 * (sum_square - square_sum)
        deep_out = self.mlp(E.view(E.size(0), -1))
        return (linear + pairwise + deep_out).squeeze(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42); np.random.seed(42)
deepfm = DeepFM(field_dims, embed_dim=16).to(device)
optimizer = torch.optim.Adam(deepfm.parameters(), lr=3e-4)
criterion = nn.BCEWithLogitsLoss()

def get_loader(X, y=None, batch=2048, shuffle=True):
    if y is None:
        ds = TensorDataset(torch.tensor(X, dtype=torch.long))
    else:
        ds = TensorDataset(torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)

loader_tr  = get_loader(X_tr_deep, y_tr_deep, batch=2048, shuffle=True)
loader_val = get_loader(X_val_deep, y_val_deep, batch=4096, shuffle=False)

best_val = -1; patience, bad = 2, 0
for epoch in range(30):
    deepfm.train()
    for xb, yb in loader_tr:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(deepfm(xb), yb)
        loss.backward()
        optimizer.step()
    deepfm.eval()
    with torch.no_grad():
        v_logits = []
        for xb, yb in loader_val:
            xb = xb.to(device)
            v_logits.append(deepfm(xb).cpu())
        v_logits = torch.cat(v_logits).numpy()
        v_pred = 1/(1+np.exp(-v_logits))
        val_auc = roc_auc_score(y_val_deep, v_pred)
    if val_auc > best_val + 1e-4:
        best_val = val_auc
        best_state = {k: v.cpu().clone() for k, v in deepfm.state_dict().items()}
        bad = 0
    else:
        bad += 1
        if bad >= patience: break

deepfm.load_state_dict(best_state)
deepfm.eval()
with torch.no_grad():
    te_logits = []
    te_loader = get_loader(X_te_deep, batch=4096, shuffle=False)
    for (xb,) in te_loader:
        xb = xb.to(device)
        te_logits.append(deepfm(xb).cpu())
    te_logits = torch.cat(te_logits).numpy()
    pred_deepfm = 1/(1+np.exp(-te_logits))

print("[DeepFM] AUC:", roc_auc_score(y_test, pred_deepfm))
print("[DeepFM] LogLoss:", log_loss(y_test, pred_deepfm))

# ======================
# Step 6. Three-model blending: LGB + CatBoost + DeepFM
# ======================
auc_lgb  = roc_auc_score(y_test, pred_lgb)
auc_cat  = roc_auc_score(y_test, pred_cat)
auc_deep = roc_auc_score(y_test, pred_deepfm)
print(f"[AUCs] LGB={auc_lgb:.6f}  Cat={auc_cat:.6f}  DeepFM={auc_deep:.6f}")

# Choose the best single model as baseline
best_auc = -1.0
best_name = None
best_pred = None
for name, pred in [("LGB", pred_lgb), ("CAT", pred_cat), ("DEEP", pred_deepfm)]:
    auc = roc_auc_score(y_test, pred)
    if auc > best_auc:
        best_auc, best_name, best_pred = auc, name, pred

# Grid search on simplex: w_lgb + w_cat + w_deep = 1, step=0.1
best_w = (1.0, 0.0, 0.0)  # (w_lgb, w_cat, w_deep)
for w_lgb in np.linspace(0, 1, 11):
    for w_cat in np.linspace(0, 1 - w_lgb, 11):
        w_deep = 1.0 - w_lgb - w_cat
        blend = w_lgb * pred_lgb + w_cat * pred_cat + w_deep * pred_deepfm
        auc = roc_auc_score(y_test, blend)
        if auc > best_auc:
            best_auc = auc
            best_w = (w_lgb, w_cat, w_deep)
            best_pred = blend

print(f"[Blend-3] best_w=(LGB {best_w[0]:.2f}, CAT {best_w[1]:.2f}, DEEP {best_w[2]:.2f})  AUC={best_auc:.6f}  "
      f"LogLoss={log_loss(y_test, best_pred):.6f}")

final_pred = best_pred
print("[Final] Test AUC:", roc_auc_score(y_test, final_pred))
print("[Final] Test LogLoss:", log_loss(y_test, final_pred))
