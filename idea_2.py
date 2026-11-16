import streamlit as st
import pandas as pd

# =========================================================
# 【自動連動パイプライン】
# Step 1: MT法のデータ定義と、SN比による「並び替え（ソート）」
# =========================================================
# ダミーのMT法データ
mt_data = {
    'センサー(部位)': ['S1(ダイバータ水流)', 'S2(F82H温度)', 'S3(プラズマ電流)', 'S4(磁場コイル)', 'S5(真空容器圧)', 'S6(中性子束)', 'S7(排気ガス)', 'S8(音響振動)', 'S9(光学スペクトル)', 'S10(予備温度)'],
    'SN比': [25.4, 22.1, 18.5, 15.2, 12.0, 9.5, 7.1, 4.2, 2.8, 1.1]
}
# データフレーム化し、SN比が「高い順（重要度順）」に並び替える
df_mt = pd.DataFrame(mt_data).sort_values(by='SN比', ascending=False).reset_index(drop=True)


# =========================================================
# Step 2: ゲーム理論の計算関数（SN比を直接防衛力に使う）
# =========================================================
def calc_sensor_cost(num_sensors, alpha):
    return num_sensors * alpha

# 【変更点】単なる「個数」ではなく、残しているセンサーの「SN比の合計」を防衛力とする
def calc_penalty_cost(active_sn_sum, noise_level, beta, gamma):
    # 防衛力 ＝ 生き残っているセンサーのSN比の合計値 × 調整係数
    defense_power = active_sn_sum * 0.1 
    
    attack_power = noise_level * gamma
    miss_rate = max(0, attack_power - defense_power) 
    return (miss_rate ** 2) * (beta / 10)


# =========================================================
# Step 3: パイプライン処理（SN比の低い順にループで削っていく）
# =========================================================
alpha = 20
beta = 200
gamma = 1.0
noises = {'N1_平穏': 1, 'N2_軽微ノイズ': 5, 'N3_最悪トラブル': 10}

cliff_data = []

# センサー稼働数を 1個 〜 10個（フル装備） まで変化させてシミュレーション
for keep_count in range(1, 11):
    # 【パイプラインの核心】SN比が高い上位のセンサーだけを生き残らせる（下位は削られる）
    active_sensors = df_mt.head(keep_count)
    
    # 生き残ったセンサーのSN比を合計する
    active_sn_sum = active_sensors['SN比'].sum()
    
    worst_loss = 0
    for n_name, n_level in noises.items():
        cost = calc_sensor_cost(keep_count, alpha) + calc_penalty_cost(active_sn_sum, n_level, beta, gamma)
        if cost > worst_loss:
            worst_loss = cost
            
    cliff_data.append({
        'センサー稼働数': keep_count, 
        '最悪時のトータル損失': worst_loss,
        # どのセンサーが無効化されたか（リストから外れたか）を記録
        '無効化されたセンサー': ", ".join(df_mt.tail(10 - keep_count)['センサー(部位)'].tolist()) if keep_count < 10 else "なし"
    })

df_cliff = pd.DataFrame(cliff_data).set_index('センサー稼働数')