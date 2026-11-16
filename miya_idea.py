import streamlit as st
import pandas as pd

# =========================================================
# 【自動連動パイプライン】裏側の計算ロジック
# =========================================================
# Step 1: ダミーのMT法データ定義（SN比）
mt_data = {
    'センサー(部位)': ['S1(ダイバータ水流)', 'S2(F82H温度)', 'S3(プラズマ電流)', 'S4(磁場コイル)', 'S5(真空容器圧)', 'S6(中性子束)', 'S7(排気ガス)', 'S8(音響振動)', 'S9(光学スペクトル)', 'S10(予備温度)'],
    'SN比': [25.4, 22.1, 18.5, 15.2, 12.0, 9.5, 7.1, 4.2, 2.8, 1.1]
}
# SN比が高い順（重要度順）にソートして配列化
df_mt = pd.DataFrame(mt_data).sort_values(by='SN比', ascending=False).reset_index(drop=True)

# Step 2: コスト計算関数
def calc_sensor_cost(num_sensors, alpha):
    return num_sensors * alpha

# 防衛力を「生き残っているセンサーのSN比の合計」から算出
def calc_penalty_cost(active_sn_sum, noise_level, beta, gamma):
    defense_power = active_sn_sum * 0.1 
    attack_power = noise_level * gamma
    miss_rate = max(0, attack_power - defense_power) 
    return (miss_rate ** 2) * (beta / 10)


# =========================================================
# StreamlitのUI部分
# =========================================================
st.set_page_config(page_title="自律型RAMIシミュレーター", layout="wide")
st.title("核融合炉 自律型RAMIシミュレーター")
st.subheader("【MT法 × ゲーム理論】センサー削減の限界ライン可視化ツール")

# 📝 チートシート
with st.expander("📝 数式とパラメータの定義（タグチメソッド損失関数ベース）", expanded=False):
    st.markdown("""
    **■ トータル損失（最悪ケースの被害額）**  $TC = C_{sensor} + C_{penalty}$
    
    **■ ① 維持コスト**  $C_{sensor} = n \\times \\alpha$
    
    **■ ② ペナルティ**  $C_{penalty} = (\\max(0, N\\gamma - Defense))^2 \\times (\\beta/10)$
    * ※防衛力($Defense$) は、MT法によって評価・残存したセンサーの「SN比の合計値」と連動。
    * ※見逃しダメージは、タグチメソッドの損失関数に基づき「2乗」して計算。
    """)

# サイドバー：パラメータ入力
st.sidebar.header("☢️ 運用フェーズ（データ蓄積量）")
phase = st.sidebar.radio("現在のフェーズを選択", ("Phase 1: 初動 (不確実性大)", "Phase 2: 中期 (癖を把握)", "Phase 3: 実用化 (定常)"))
default_gamma = 1.5 if "Phase 1" in phase else 1.0 if "Phase 2" in phase else 0.5

st.sidebar.header("⚙️ 現場の工学パラメータ入力")
# 【変更点】実機スケールを試せるように max_value を大幅に引き上げ（最大20,000）
# プレゼン開始時は意図的に「小さい値（200）」にしておく
alpha = st.sidebar.slider("維持コスト (α): レイテンシ・交換工数", min_value=1, max_value=500, value=20, step=1)
beta = st.sidebar.slider("ダウンタイム損失 (β): 熱疲労・MTTR等", min_value=10, max_value=20000, value=200, step=50)
gamma = st.sidebar.slider("不確実性 (γ: 未知のノイズレベル)", min_value=0.1, max_value=2.0, value=default_gamma, step=0.1)

# ---------------------------------------------------------
# メイン画面の描画
# ---------------------------------------------------------
st.markdown("---")
st.markdown("### 🔍 【Step 1】MT法（直交表）によるセンサー貢献度の評価")
st.info("💡 事前に10個のセンサーの「異常検知への貢献度（SN比）」を算出。貢献度が低い（右側の）センサーから順に削減の対象となります。")
st.bar_chart(df_mt.set_index('センサー(部位)'), color="#4A90E2")

st.markdown("---")
st.markdown("### 📈 【Step 2】ゲーム理論が導く「安全の崖（クリフ）」と最適構成")
st.info("💡 Step 1で評価した冗長なセンサーから順にコンセントを抜いた際、どこで被害が爆発するかを計算しています。")

# パイプライン処理を回す
noises = {'N1_平穏': 1, 'N2_軽微ノイズ': 5, 'N3_最悪トラブル': 10}
cliff_data = []

for keep_count in range(1, 11):
    active_sensors = df_mt.head(keep_count)
    active_sn_sum = active_sensors['SN比'].sum()
    
    worst_loss = 0
    for n_name, n_level in noises.items():
        cost = calc_sensor_cost(keep_count, alpha) + calc_penalty_cost(active_sn_sum, n_level, beta, gamma)
        if cost > worst_loss:
            worst_loss = cost
            
    cliff_data.append({'センサー稼働数': keep_count, '最悪時のトータル損失': worst_loss})

df_cliff = pd.DataFrame(cliff_data).set_index('センサー稼働数')
minimax_sensor_num = df_cliff['最悪時のトータル損失'].idxmin()

st.area_chart(df_cliff, color="#E24A4A")

# ==========================================
# 🏆 最適解と「残す/削るセンサー」の明示
# ==========================================
st.success(f"🏆 現在の条件において、最も安全かつコストが低いのは **【センサー稼働数：{minimax_sensor_num} 個】** の構成です。")

optimal_sensors = df_mt.head(minimax_sensor_num)['センサー(部位)'].tolist()
removed_sensors = df_mt.tail(10 - minimax_sensor_num)['センサー(部位)'].tolist() if minimax_sensor_num < 10 else []

st.markdown("#### 🛠️ 最適構成時のセンサー内訳（MT法のSN比に基づく）")

col1, col2 = st.columns(2)
with col1:
    st.info(f"✅ **残すセンサー（稼働：{len(optimal_sensors)}個）**\n" + "\n".join([f"- {s}" for s in optimal_sensors]))
    
with col2:
    if removed_sensors:
        st.warning(f"❌ **削るセンサー（無効化：{len(removed_sensors)}個）**\n" + "\n".join([f"- {s}" for s in removed_sensors]))
    else:
        st.warning("❌ **削るセンサー**\n- なし（フル装備）")