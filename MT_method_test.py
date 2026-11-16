import numpy as np

# 1. 単位空間（正常データ）の用意（5サンプル、3変数）
unit_data = np.array([
    [10.0, 21.0, 3.0],
    [9.0, 23.0, 4.0],
    [8.0, 24.0, 2.0],
    [7.0, 25.0, 5.0],
    [8.0, 22.0, 3.0]
])

# 2. 平均と不偏標準偏差の計算
mean_vec = np.mean(unit_data, axis=0)
std_vec = np.std(unit_data, axis=0, ddof=1)

# 3. 基準化（標準化）
standardized_data = (unit_data - mean_vec) / std_vec

# 4. 相関行列（基準化後の分散共分散行列）と逆行列の計算
corr_matrix = np.corrcoef(standardized_data, rowvar=False)
inv_corr_matrix = np.linalg.inv(corr_matrix)

# 5. マハラノビス距離（MD）の計算関数
def calc_mahalanobis_distance(data, inv_corr):
    k = data.shape[1]  # 変数の数（列数）
    # 各サンプルのMD^2を計算
    md_sq = np.sum((data @ inv_corr) * data, axis=1) / k
    return np.sqrt(md_sq)

# 単位空間のMD値（平均が1に近くなります）
md_unit = calc_mahalanobis_distance(standardized_data, inv_corr_matrix)
print("単位空間データのMD値:", md_unit)

# 6. 評価対象（未知データ）のMD計算
new_data = np.array([
    [8.0, 23.0, 3.0],   # 正常に近いデータ
    [20.0, 10.0, 10.0]  # 異常データ
])
standardized_new_data = (new_data - mean_vec) / std_vec
md_new = calc_mahalanobis_distance(standardized_new_data, inv_corr_matrix)
print("評価データのMD値:", md_new)
