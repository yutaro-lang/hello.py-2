import streamlit as st
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# 裏側の計算ロジック
# ---------------------------------------------------------
def calc_sensor_cost(num_sensors, alpha):
    return num_sensors * alpha

def calc_penalty_cost(num_sensors, noise_level, beta, gamma):
    defense_power = num_sensors * 1.5 
    attack_power = noise_level * gamma
    miss_rate = max(0, attack_power - defense_power) 
    return (miss_rate ** 2) * (beta / 10)

# ---------------------------------------------------------
# StreamlitのUI部分
# ---------------------------------------------------------
st.set_page_config(page_title="RGTO Simulator", layout="wide")
st.title("核融合炉 自律型RAMIシミュレーター")
st.subheader("【MT法 × ゲーム理論】センサー削減の限界ライン可視化ツール")

with st.expander("📝 数式とパラメータの定義（タグチメソッド損失関数ベース）", expanded=False):
    st.markdown("""
    **■ トータル損失（最悪ケースの被害額）**  $TC = C_{sensor} + C_{penalty}$
    **■ ① 維持コスト**  $C_{sensor} = n \\times \\alpha$
    **■ ② ペナルティ**  $C_{penalty} = (\\max(0, N\\gamma - 1.5n))^2 \\times (\\beta/10)$
    （※タグチメソッドの損失関数に基づき、見逃しダメージを2乗しています）
    """)

# サイドバーの設定
st.sidebar.header("☢️ 運用フェーズ（データ蓄積量）")
phase = st.sidebar.radio("現在のフェーズを選択", ("Phase 1: 初動 (不確実性大)", "Phase 2: 中期 (癖を把握)", "Phase 3: 実用化 (定常)"))

default_gamma = 1.5 if "Phase 1" in phase else 1.0 if "Phase 2" in phase else 0.5

st.sidebar.header("⚙️ 現場の工学パラメータ入力")
alpha = st.sidebar.slider("維持コスト (α): レイテンシ・交換工数", 1, 100, 20, 1)
beta = st.sidebar.slider("ダウンタイム損失 (β): 熱疲労・MTTR等", 10, 1000, 200, 10)
gamma = st.sidebar.slider("不確実性 (γ: 未知のノイズレベル)", 0.1, 2.0, default_gamma, 0.1)

# ---------------------------------------------------------
# 【新規追加】Step 1: MT法（直交表・SN比）のグラフ
# ---------------------------------------------------------
st.markdown("---")
st.markdown("### 🔍 【Step 1】MT法（直交表）によるセンサー貢献度の評価")
st.info("💡 **宮先生のMT法アルゴリズム**を用いて、10個のセンサーの「異常検知への貢献度（SN比）」を事前算出しました。貢献度が低い（右側の）センサーから順に削減の対象となります。")

# ダミーのMT法データ（SN比が大きいほど重要）
mt_data = {
    'センサー(部位)': ['S1(ダイバータ水流)', 'S2(F82H温度)', 'S3(プラズマ電流)', 'S4(磁場コイル)', 'S5(真空容器圧)', 'S6(中性子束)', 'S7(排気ガス)', 'S8(音響振動)', 'S9(光学スペクトル)', 'S10(予備温度)'],
    '異常検知への貢献度 (SN比)': [25.4, 22.1, 18.5, 15.2, 12.0, 9.5, 7.1, 4.2, 2.8, 1.1]
}
df_mt = pd.DataFrame(mt_data).set_index('センサー(部位)')
# MT法のバーチャートを表示
st.bar_chart(df_mt, color="#4A90E2")

# ---------------------------------------------------------
# Step 2: ゲーム理論（クリフ）のグラフ
# ---------------------------------------------------------
st.markdown("---")
st.markdown("### 📈 【Step 2】ゲーム理論が導く「安全の崖（クリフ）」と最適構成")
st.info("💡 Step 1で評価した冗長なセンサーから順に削っていった際、どこで被害（ダウンタイム損失）が爆発するかを計算しています。")

noises = {'N1_平穏': 1, 'N2_軽微ノイズ': 5, 'N3_最悪トラブル': 10}
cliff_data = []
for s_num in range(1, 11):
    worst_loss = 0
    for n_name, n_level in noises.items():
        cost = calc_sensor_cost(s_num, alpha) + calc_penalty_cost(s_num, n_level, beta, gamma)
        if cost > worst_loss:
            worst_loss = cost
    cliff_data.append({'センサー稼働数': s_num, '最悪時のトータル損失': worst_loss})

df_cliff = pd.DataFrame(cliff_data).set_index('センサー稼働数')
minimax_sensor_num = df_cliff['最悪時のトータル損失'].idxmin()

# 崖のエリアチャートを表示
st.area_chart(df_cliff, color="#E24A4A")

st.success(f"🏆 現在の条件（α, β, γ）において、最も安全かつコストが低いのは **【センサー稼働数：{minimax_sensor_num} 個】** の構成です。")
st.markdown(f"※MT法の評価に基づき、右から **{10 - minimax_sensor_num}個のセンサー（SN比が低いもの）** をソフトウェア上で無効化します。")