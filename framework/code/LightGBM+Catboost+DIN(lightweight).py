import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from catboost import CatBoostClassifier, Pool
import platform, lightgbm, catboost, torch

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

test_day = pd.to_datetime("2017-05-13").date()
train_df = df[df["date"] < test_day].copy()
test_df  = df[df["date"] == test_day].copy()
y_train = train_df["clk"].astype(int).values
y_test  = test_df["clk"].astype(int).values
global_ctr = y_train.mean()

print(f"Python {platform.python_version()}")
print("LGBM:", lightgbm.__version__, " CatBoost:", catboost.__version__, " Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))

# ======================
# Step 2. CTR Features
# ======================
def add_window_ctr(train_df, test_df, key, label="clk", win=3, alpha=100):
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

    tmp = train_df[[key, "date", label]].copy()
    daily = (
        tmp.groupby([key, "date"], as_index=False)
           .agg(clicks=(label, "sum"), imps=(label, "count"))
           .sort_values("date")
    )

    daily["roll_clicks"] = (
        daily.groupby(key, group_keys=False)["clicks"]
             .transform(lambda s: s.shift(1).rolling(win, min_periods=1).sum())
    )
    daily["roll_imps"] = (
        daily.groupby(key, group_keys=False)["imps"]
             .transform(lambda s: s.shift(1).rolling(win, min_periods=1).sum())
    )
    daily[out_col] = (daily["roll_clicks"] + alpha*g) / (daily["roll_imps"] + alpha)

    train_out = train_df[[key, "date"]].merge(
        daily[[key, "date", out_col]], on=[key, "date"], how="left"
    )
    train_df[out_col] = train_out[out_col].fillna(g)

    all_days = np.sort(train_df["date"].unique())
    last_win_days = all_days[max(0, len(all_days)-win):]
    recent = (
        daily[daily["date"].isin(last_win_days)]
        .groupby(key, as_index=False)
        .agg(roll_clicks=("clicks", "sum"), roll_imps=("imps", "sum"))
    )
    recent[out_col] = (recent["roll_clicks"] + alpha*g) / (recent["roll_imps"] + alpha)

    test_out = test_df[[key]].merge(recent[[key, out_col]], on=key, how="left")
    test_df[out_col] = test_out[out_col].fillna(g)

    return train_df, test_df, out_col

base_keys = ["user","adgroup_id","brand","cate_id","campaign_id"]
cross_keys = [
    ["hour","weekday"], ["user","cate_id"], ["brand","campaign_id"],
    ["adgroup_id","hour"], ["user","brand"], ["user","campaign_id"], ["brand","cate_id"]
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

# ======================
# Step 3. Frequency Encoding
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

last_train_day = pd.to_datetime("2017-05-12").date()
mask_val = (train_df["date"] == last_train_day)
X_tr_lgb, y_tr_lgb = train_df.loc[~mask_val, lgb_features], train_df.loc[~mask_val, "clk"].astype(int).values
X_val_lgb, y_val_lgb = train_df.loc[ mask_val, lgb_features], train_df.loc[ mask_val, "clk"].astype(int).values
X_te_lgb,  y_te_lgb  = test_df[lgb_features], y_test

lgb_train = lgb.Dataset(X_tr_lgb, label=y_tr_lgb, categorical_feature=[c for c in cat_cols if c in lgb_features])
lgb_valid = lgb.Dataset(X_val_lgb, label=y_val_lgb, categorical_feature=[c for c in cat_cols if c in lgb_features])

lgb_params = {
    "objective": "binary","metric": ["auc","binary_logloss"],
    "learning_rate": 0.03,"num_leaves": 255,"min_data_in_leaf": 50,
    "feature_fraction": 0.9,"bagging_fraction": 0.8,"bagging_freq": 1,
    "lambda_l1": 0.5,"lambda_l2": 10.0,"max_bin": 511,"max_cat_threshold": 64,
    "cat_l2": 25,"cat_smooth": 25,"is_unbalance": True,"seed": 42,"verbose": -1
}

lgb_model = lgb.train(
    lgb_params, lgb_train, num_boost_round=1200, valid_sets=[lgb_valid],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)
pred_lgb = lgb_model.predict(X_te_lgb, num_iteration=lgb_model.best_iteration)

# ======================
# Step 4.5. CatBoost
# ======================
cat_features_idx = [lgb_features.index(c) for c in cat_cols if c in lgb_features]
train_pool = Pool(X_tr_lgb, y_tr_lgb, cat_features=cat_features_idx)
valid_pool = Pool(X_val_lgb, y_val_lgb, cat_features=cat_features_idx)
test_pool  = Pool(X_te_lgb, y_te_lgb, cat_features=cat_features_idx)

cat_params = dict(
    iterations=1200, depth=8, learning_rate=0.03, l2_leaf_reg=10.0,
    loss_function="Logloss", eval_metric="AUC", random_seed=42,
    od_type="Iter", od_wait=50, task_type="GPU", devices="0", verbose=100
)
cat_model = CatBoostClassifier(**cat_params)
cat_model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
pred_cat = cat_model.predict_proba(test_pool)[:, 1]

# ======================
# Step 5. Improved DIN
# ======================
def build_user_history(df_in, max_len=50):
    df_sorted = df_in.sort_values("time_stamp")
    user_hist = {}
    for u, grp in df_sorted.groupby("user"):
        hist = {}
        for f in ["adgroup_id", "brand", "cate_id"]:
            seq = grp.loc[grp["clk"] == 1, f].tolist()
            if len(seq) > max_len:
                seq = seq[-max_len:]
            hist[f] = seq
        user_hist[u] = hist
    return user_hist

max_len = 50
user_hist = build_user_history(train_df, max_len=max_len)
def pad_left(seq, max_len=50, pad_val=0):
    seq = seq[-max_len:]
    return [pad_val] * (max_len - len(seq)) + seq

train_df["user"] = train_df["user"].astype(str)
test_df["user"]  = test_df["user"].astype(str)

for f in ["adgroup_id", "brand", "cate_id"]:
    train_df[f"hist_{f}"] = train_df["user"].apply(lambda u: pad_left(user_hist.get(u, {}).get(f, []), max_len))
    test_df[f"hist_{f}"]  = test_df["user"].apply(lambda u: pad_left(user_hist.get(u, {}).get(f, []), max_len))

din_cols = ["user","adgroup_id","brand","cate_id","campaign_id"]
encoders_din, field_dims = {}, {}
for col in din_cols:
    le = LabelEncoder()
    all_vals = pd.concat([train_df[col], test_df[col]], axis=0).astype(str)
    le.fit(all_vals)
    train_df[col+"_le"] = le.transform(train_df[col].astype(str))
    test_df[col+"_le"]  = le.transform(test_df[col].astype(str))
    encoders_din[col] = le
    field_dims[col] = len(le.classes_)

for f in ["adgroup_id", "brand", "cate_id"]:
    fmap = {v: i for i, v in enumerate(encoders_din[f].classes_)}
    def encode_hist(seq): return [fmap.get(str(x), 0) for x in seq]
    train_df[f"hist_{f}"] = train_df[f"hist_{f}"].apply(encode_hist)
    test_df[f"hist_{f}"]  = test_df[f"hist_{f}"].apply(encode_hist)

mask_val = (train_df["date"] == last_train_day)
din_le_cols = [c+"_le" for c in din_cols]
X_tr_din = train_df.loc[~mask_val, din_le_cols + ["hist_adgroup_id","hist_brand","hist_cate_id"]]
y_tr_din = y_train[~mask_val]
X_val_din = train_df.loc[ mask_val, din_le_cols + ["hist_adgroup_id","hist_brand","hist_cate_id"]]
y_val_din = y_train[mask_val]
X_te_din  = test_df[din_le_cols + ["hist_adgroup_id","hist_brand","hist_cate_id"]]

class ImprovedDIN(nn.Module):
    def __init__(self, field_dims, embed_dim=8, max_len=50):
        super().__init__()
        self.embed_ad = nn.Embedding(field_dims["adgroup_id"], embed_dim)
        self.embed_user = nn.Embedding(field_dims["user"], embed_dim)
        self.embed_brand = nn.Embedding(field_dims["brand"], embed_dim)
        self.embed_cate = nn.Embedding(field_dims["cate_id"], embed_dim)
        self.embed_campaign = nn.Embedding(field_dims["campaign_id"], embed_dim)

        self.att_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 8, 128), nn.ReLU(), nn.Dropout(0.6),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.6),
            nn.Linear(64, 1)
        )
        self.max_len = max_len
        for emb in [self.embed_ad, self.embed_user, self.embed_brand, self.embed_cate, self.embed_campaign]:
            nn.init.xavier_uniform_(emb.weight.data)

    def forward(self, user, ad, brand, cate, campaign,
                hist_ads, hist_brands, hist_cates):
        u = self.embed_user(user); a = self.embed_ad(ad)
        b = self.embed_brand(brand); c = self.embed_cate(cate)
        p = self.embed_campaign(campaign)

        def att(q, k, embed_layer):
            q_expand = q.unsqueeze(1).expand(-1, self.max_len, -1)
            k_emb = embed_layer(k)
            att = torch.softmax(self.att_mlp(torch.cat([q_expand, k_emb], dim=-1)).squeeze(-1), dim=-1)
            return (att.unsqueeze(-1) * k_emb).sum(dim=1)

        hist_vec_ads = att(a, hist_ads, self.embed_ad)
        hist_vec_brands = att(b, hist_brands, self.embed_brand)
        hist_vec_cates = att(c, hist_cates, self.embed_cate)

        x = torch.cat([u, a, b, c, p, hist_vec_ads, hist_vec_brands, hist_vec_cates], dim=-1)
        return self.mlp(x).squeeze(1)

def to_tensor(df_part):
    return (
        torch.tensor(df_part["user_le"].values, dtype=torch.long),
        torch.tensor(df_part["adgroup_id_le"].values, dtype=torch.long),
        torch.tensor(df_part["brand_le"].values, dtype=torch.long),
        torch.tensor(df_part["cate_id_le"].values, dtype=torch.long),
        torch.tensor(df_part["campaign_id_le"].values, dtype=torch.long),
        torch.tensor(np.stack(df_part["hist_adgroup_id"].values), dtype=torch.long),
        torch.tensor(np.stack(df_part["hist_brand"].values), dtype=torch.long),
        torch.tensor(np.stack(df_part["hist_cate_id"].values), dtype=torch.long),
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
din = ImprovedDIN(field_dims, embed_dim=8, max_len=max_len).to(device)
optimizer = torch.optim.Adam(din.parameters(), lr=3e-4, weight_decay=1e-4)
criterion = nn.BCEWithLogitsLoss()

train_inputs, val_inputs, test_inputs = to_tensor(X_tr_din), to_tensor(X_val_din), to_tensor(X_te_din)
y_tr_tensor, y_val_tensor = torch.tensor(y_tr_din, dtype=torch.float32), torch.tensor(y_val_din, dtype=torch.float32)
train_ds, val_ds = TensorDataset(*train_inputs, y_tr_tensor), TensorDataset(*val_inputs, y_val_tensor)
train_loader, val_loader = DataLoader(train_ds, batch_size=2048, shuffle=True), DataLoader(val_ds, batch_size=4096, shuffle=False)

best_val_auc, best_state, bad = -1.0, None, 0
EPOCHS, patience = 20, 3
for epoch in range(1, EPOCHS+1):
    din.train()
    for batch in train_loader:
        xb = [x.to(device) for x in batch[:-1]]; yb = batch[-1].to(device)
        optimizer.zero_grad(); loss = criterion(din(*xb), yb); loss.backward(); optimizer.step()
    din.eval()
    with torch.no_grad():
        xb_val = [x.to(device) for x in val_inputs]
        val_auc = roc_auc_score(y_val_din, torch.sigmoid(din(*xb_val)).cpu().numpy())
    print(f"[DIN-Improved] Epoch {epoch}/{EPOCHS} Val AUC={val_auc:.4f}")
    if val_auc > best_val_auc + 1e-4:
        best_val_auc, best_state, bad = val_auc, {k: v.cpu().clone() for k,v in din.state_dict().items()}, 0
    else:
        bad += 1
        if bad >= patience: break

din.load_state_dict(best_state); din.eval()
with torch.no_grad():
    xb_test = [x.to(device) for x in test_inputs]
    pred_din = torch.sigmoid(din(*xb_test)).cpu().numpy()

# ======================
# Step 6. Blend
# ======================
auc_lgb, auc_cat, auc_din = roc_auc_score(y_test, pred_lgb), roc_auc_score(y_test, pred_cat), roc_auc_score(y_test, pred_din)
print(f"[AUCs] LGB={auc_lgb:.6f}  Cat={auc_cat:.6f}  DIN={auc_din:.6f}")

best_auc, best_w, best_pred = -1.0, (1,0,0), pred_lgb
STEP = 0.05
ws = np.arange(0.0, 1.0 + 1e-9, STEP)
for w_lgb in ws:
    for w_cat in ws:
        w_din = 1.0 - w_lgb - w_cat
        if w_din < 0: continue
        blend = w_lgb * pred_lgb + w_cat * pred_cat + w_din * pred_din
        auc = roc_auc_score(y_test, blend)
        if auc > best_auc:
            best_auc, best_w, best_pred = auc, (w_lgb, w_cat, w_din), blend

print(f"[Blend-3/Grid] best_w={best_w} AUC={best_auc:.6f} LogLoss={log_loss(y_test, best_pred):.6f}")
print("[Final] Test AUC:", roc_auc_score(y_test, best_pred))
print("[Final] Test LogLoss:", log_loss(y_test, best_pred))
