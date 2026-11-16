import numpy as np
import pandas as pd
import scipy.linalg as la

# ==========================================
# 1. 正常データ（単位空間）の作成
# ==========================================
# 炉が安定稼働している時のデータをシミュレーションで100件作成
np.random.seed(42)
n_normal_samples = 100

# 4つの架空センサー: プラズマ温度, 磁場強度, 冷却材温度, ダイバータ熱負荷
normal_data = {
    'Plasma_Temp': np.random.normal(1.0, 0.05, n_normal_samples),
    'Magnetic_Field': np.random.normal(5.0, 0.1, n_normal_samples),
    'Coolant_Temp': np.random.normal(300.0, 2.0, n_normal_samples),
    'Divertor_Heat': np.random.normal(10.0, 0.5, n_normal_samples)
}
df_normal = pd.DataFrame(normal_data)

# ==========================================
# 2. MT法: 単位空間の構築（学習）
# ==========================================
# ① 各センサーの平均と標準偏差を計算（標準化のため）
mean_vec = df_normal.mean()
std_vec = df_normal.std()

# ② 正常データを標準化
Z_normal = (df_normal - mean_vec) / std_vec

# ③ 相関行列とその逆行列を計算（ここがMT法の心臓部）
R = Z_normal.corr()
R_inv = np.linalg.pinv(R.values) # 擬似逆行列（センサー同士が似た動きをする場合の計算エラーを防ぐ）
k = len(mean_vec) # 次元数（センサーの数 = 4）

# マハラノビス距離(MD)を計算する関数
def calculate_md(data_df):
    # 新しいデータを正常基準で標準化
    Z = (data_df - mean_vec) / std_vec
    md_list = []
    
    # 1行（1秒）ごとにMDを計算
    for i in range(len(Z)):
        z = Z.iloc[i].values
        # MD = (1/k) * z * R_inv * z^T
        md = np.dot(np.dot(z, R_inv), z.T) / k
        md_list.append(md)
    return md_list

# ==========================================
# 3. 異常データのテスト（予兆検知）
# ==========================================
# ディスラプション数秒前やダイバータ異常発熱を想定した3件のデータ
abnormal_data = {
    'Plasma_Temp': [1.0, 0.95, 1.2],      # わずかに揺らぐ
    'Magnetic_Field': [5.0, 4.9, 4.8],    # 少し下がる
    'Coolant_Temp': [300.0, 301.0, 305.0],# 徐々に上がる
    'Divertor_Heat': [10.5, 12.0, 15.0]   # 急激に発熱（異常）
}
df_abnormal = pd.DataFrame(abnormal_data)

# ==========================================
# 4. 結果の評価
# ==========================================
# 正常データと異常データそれぞれのMDを計算
md_normal_result = calculate_md(df_normal)
md_abnormal_result = calculate_md(df_abnormal)

print("【正常時のマハラノビス距離(MD)】（平均して1.0付近になるのが正しい）")
print(f"平均MD: {np.mean(md_normal_result):.2f}")
print(f"最大MD: {np.max(md_normal_result):.2f}\n")

print("【異常発生時のマハラノビス距離(MD)】（異常度が数値として跳ね上がる）")
for i, md in enumerate(md_abnormal_result):
    print(f"異常データ {i+1} のMD: {md:.2f}")
import numpy as np
import pandas as pd

# 前回の正常データ(df_normal)と異常データ(df_abnormal)がすでにある前提です
sensors = ['Plasma_Temp', 'Magnetic_Field', 'Coolant_Temp', 'Divertor_Heat']

# ==========================================
# 1. 直交表（実験計画）の準備
# ==========================================
# 本来はL64などの巨大な表を使いますが、今回は4センサーなので
# 1＝「そのセンサーを使う」、0＝「使わない」というパターンの表を用意します
# （※すべて0のパターンは計算できないため除外しています）
experiment_matrix = [
    [1, 1, 1, 1], # 全部のセンサーを使う
    [1, 1, 0, 0], # プラズマ温度と磁場だけ使う
    [1, 0, 1, 0],
    [1, 0, 0, 1],
    [0, 1, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 1]
]

# 実験ごとの「SN比」を保存するリスト
sn_ratios = []

# ==========================================
# 2. 直交表の各行（パターン）ごとにテストを実行
# ==========================================
for exp in experiment_matrix:
    # 今回の実験で「使う」と決まったセンサーだけを抽出
    active_sensors = [sensors[i] for i in range(len(sensors)) if exp[i] == 1]
    
    # 選ばれたセンサーだけで正常データ（単位空間）を再構築
    df_norm_sub = df_normal[active_sensors]
    mean_sub = df_norm_sub.mean()
    std_sub = df_norm_sub.std()
    Z_norm_sub = (df_norm_sub - mean_sub) / std_sub
    
    R_sub = Z_norm_sub.corr().values
    # センサーが1つの時は相関行列の逆行列が取れないため例外処理
    if len(active_sensors) == 1:
        R_inv_sub = np.array([[1.0]])
    else:
        R_inv_sub = np.linalg.pinv(R_sub)
    
    k_sub = len(active_sensors)
    
    # 異常データのMD（マハラノビス距離）を計算
    df_abnorm_sub = df_abnormal[active_sensors]
    Z_abnorm_sub = (df_abnorm_sub - mean_sub) / std_sub
    
    md_list = []
    for i in range(len(Z_abnorm_sub)):
        z = Z_abnorm_sub.iloc[i].values
        md = np.dot(np.dot(z, R_inv_sub), z.T) / k_sub
        md_list.append(md)
    
    # ==========================================
    # 3. SN比（検知の鋭さ）の計算
    # ==========================================
    # 異常な時にMDが「より大きく」跳ね上がるほど優秀（望大特性）
    # ※計算をシンプルにするため、異常データのMDの平均値を対数化しています
    sn = 10 * np.log10(np.mean(md_list))
    sn_ratios.append(sn)

# ==========================================
# 4. 要因効果（どのセンサーが本当に必要か）の判定
# ==========================================
print("【各センサーの要因効果（SN比の差）】")
print("※ プラスが大きいほど「必須センサー」、マイナスは「不要なノイズ」\n")

for i, sensor in enumerate(sensors):
    # そのセンサーを「使った実験」のSN比の平均
    sn_used = np.mean([sn_ratios[j] for j in range(len(experiment_matrix)) if experiment_matrix[j][i] == 1])
    # そのセンサーを「使わなかった実験」のSN比の平均
    sn_not_used = np.mean([sn_ratios[j] for j in range(len(experiment_matrix)) if experiment_matrix[j][i] == 0])
    
    # 使った時と使わなかった時の差（効果）
    effect = sn_used - sn_not_used
    
    # 判定
    judgement = "🟢 残す（検知に貢献）" if effect > 0 else "❌ 捨てる（ノイズ源）"
    print(f"{sensor}: 効果量 {effect:+.2f} dB  => {judgement}")