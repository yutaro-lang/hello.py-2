import numpy as np
import pandas as pd
import streamlit as st # UI用

# ---------------------------------------------------------
# Step 1: センサー構成（戦略）の定義と維持コスト計算
# ---------------------------------------------------------
def calc_sensor_cost(num_sensors, alpha):
    """
    維持コスト（C_sensor）を計算。
    センサー数が多いほど、中性子劣化による交換頻度や計算レイテンシが増大。
    """
    # 例：センサー数 × アルファ（維持・処理コストの重み）
    return num_sensors * alpha

# ---------------------------------------------------------
# Step 2: ノイズ（自然の戦略）に対するペナルティ計算
# ---------------------------------------------------------
def calc_penalty_cost(sensor_mask, noise_level, beta):
    """
    ダウンタイム損失（C_penalty）を計算。
    ノイズ（熱歪み等）に対して、センサー数が少なすぎると見逃し事故（MTTR増大）が発生。
    """
    # センサーが少ない＆ノイズが強いほど、Q-DEMO Phase3の発電ロス(100MW分)やF82H鋼へのダメージが甚大になるダミーロジック
    miss_rate = max(0, noise_level - (sum(sensor_mask) * 0.1)) 
    return miss_rate * beta

# ---------------------------------------------------------
# Step 3: ペイオフ行列の作成とミニマックス最適化（ゲーム理論）
# ---------------------------------------------------------
def run_game_theory_sim(alpha, beta):
    # 私たちの戦略：S1(10個), S2(7個), S3(4個)
    strategies = {'S1_Full': 10, 'S2_Balance': 7, 'S3_Minimum': 4}
    
    # 自然（敵）の戦略：N1(平穏), N2(中ノイズ), N3(最悪のトラブル)
    noises = {'N1_Normal': 1, 'N2_Mild': 5, 'N3_Worst': 10}
    
    results = []
    for s_name, s_num in strategies.items():
        row = {'Strategy': s_name}
        worst_loss = 0
        
        for n_name, n_level in noises.items():
            # トータルコスト = 維持コスト + 見逃しペナルティ
            cost = calc_sensor_cost(s_num, alpha) + calc_penalty_cost(s_num, n_level, beta)
            row[n_name] = cost
            if cost > worst_loss:
                worst_loss = cost # 最悪ケースの損失を記録（Max）
                
        row['Worst_Loss'] = worst_loss
        results.append(row)
        
    df = pd.DataFrame(results)
    
    # ミニマックス解の特定（Worst_Lossが最も小さい戦略を選ぶ）
    best_strategy = df.loc[df['Worst_Loss'].idxmin()]['Strategy']
    
    return df, best_strategy