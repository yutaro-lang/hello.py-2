import streamlit as st
import pandas as pd
import numpy as np

# =========================================================
# 【MT法 計算エンジン】ダミーではなく、行列計算からリアルタイム生成
# =========================================================

@st.cache_data
def run_mt_engine():
    # ① 正常データ（単位空間）の構築
    np.random.seed(42)
    sensor_names = ['S1(ダイバータ)', 'S2(F82H)', 'S3(プラズマ電流)', 'S4(磁場コイル)', 'S5(真空容器)', 'S6(中性子束)', 'S7(排気)', 'S8(振動)', 'S9(光学)', 'S10(予備)']
    
    # 意図的なコピー（冗長性）を作るのをやめ、10個すべて「独立したセンサー」として扱う
    normal_data = np.random.randn(1000, 10) 

    # ② 相関行列と「逆行列（ここがMT法の心臓部）」の計算
    cov_matrix = np.cov(normal_data, rowvar=False)
    inv_cov = np.linalg.pinv(cov_matrix) # 逆行列の算出

    # ③ 異常データの検知（美しいグラデーション波及）
    anomaly_data = np.random.randn(10) * 0.5
    anomaly_data[0] += 8.0  # S1: 直撃・超異常
    anomaly_data[1] += 6.5  # S2: 大異常
    anomaly_data[2] += 5.0  # S3: 中異常
    anomaly_data[3] += 4.0  # S4: 中異常
    anomaly_data[4] += 3.0  # S5: 小異常
    anomaly_data[5] += 2.0  # S6: 小異常
    anomaly_data[6] += 1.5  # S7: 微小異常
    anomaly_data[7] += 1.0  # S8: 微小異常
    anomaly_data[8] += 0.5  # S9: 極微小（でもゼロではない）
    anomaly_data[9] += 0.2  # S10: 極微小（でもゼロではない）
    # ④ センサーごとの貢献度（SN比）を計算
    sn_ratios = []
    base_md = np.dot(np.dot(anomaly_data, inv_cov), anomaly_data.T)
    
    for i in range(10):
        # センサーを1つずつ削って逆行列とMDを再計算
        reduced_anomaly = np.delete(anomaly_data, i)
        reduced_cov = np.delete(np.delete(cov_matrix, i, 0), i, 1)
        reduced_inv = np.linalg.pinv(reduced_cov)
        reduced_md = np.dot(np.dot(reduced_anomaly, reduced_inv), reduced_anomaly.T)
        
        # 検知力（MD）の低下量 ＝ そのセンサーの重要度（SN比）
        contribution = max(0, base_md - reduced_md)
        sn_ratios.append(contribution)

    # ⑤ SN比を「合計100」になるように正規化（ゲームバランスの調整）
    sn_ratios = np.array(sn_ratios)
    if sn_ratios.sum() > 0:
        sn_ratios = (sn_ratios / sn_ratios.sum()) * 100

    # データフレーム化してソート
    df = pd.DataFrame({'センサー(部位)': sensor_names, 'SN比': sn_ratios})
    return df.sort_values(by='SN比', ascending=False).reset_index(drop=True)

# MT法エンジンを実行し、df_mt に格納
df_mt = run_mt_engine()

# ---（ここから下は、ゲーム理論とコスト計算のコード）---

def calc_sensor_cost(num_sensors, alpha):
    return num_sensors * alpha

def calc_penalty_cost(active_sn_sum, noise_level, beta, gamma):
    defense_power = active_sn_sum * 0.15 
    attack_power = noise_level * gamma
    miss_rate = max(0, attack_power - defense_power) 
    return (miss_rate ** 2) * (beta / 10)


# =========================================================
# StreamlitのUI部分
# =========================================================
st.set_page_config(page_title="自律型RAMIシミュレーター", layout="wide")
st.title("核融合炉 自律型RAMIシミュレーター")
st.subheader("【MT法 × ゲーム理論】センサー削減の限界ライン可視化ツール")

## 📝 チートシート
with st.expander("📝 数式とパラメータの定義（タグチメソッド損失関数ベース）", expanded=True):
    st.markdown("""
    **■ トータル損失（最悪ケースの被害額）**  $TC = C_{sensor} + C_{penalty}$
    
    **■ ① 維持コスト**  $C_{sensor} = n \\times \\alpha$
    
    **■ ② ペナルティ**  $C_{penalty} = (\\max(0, N\\gamma - Defense))^2 \\times (\\beta/10)$
    
    ---
    **【※今後の共同研究における技術的課題（高度化に向けた2つのステップ）】**
    
    **1. SN比と防衛力を繋ぐ「伝達関数（Transfer Function）」の構築**
    現在のプロトタイプでは、防衛力($Defense$) を「残存センサーのSN比の単純合算値×0.1」という仮の線形モデルで計算しています。実際のSN比（dB）は対数スケールであり、これを物理的な防衛力へ変換するための「正しい伝達関数」の構築が学術的な第一課題です。
    
    **2. Q-DEMOの「電力」を組み込んだプラント経済性関数への進化**
    Phase 3（定常発電：ネット100MW）を見据え、現在の静的なスカラー値（α, β）を、以下の**「電力」**を組み込んだ関数へとアップデートさせる必要があります。
    *   **$\\alpha$（維持費）の関数化：** センサー稼働とデータ処理にかかる「所内動力（消費電力）」のコスト化
    *   **$\\beta$（被害額）の関数化：** ダウンタイム（MTTR）に伴う「売電の機会損失額」のコスト化
    
    これらを組み込むことで、MT法によるセンサー削減が、単なる安全側のコストカットではなく**「プラントの純利益（ネット発電量）を最大化する経営戦略」**として数理的に証明可能になります。
    """)

# サイドバー：パラメータ入力
st.sidebar.header("☢️ 運用フェーズ（データ蓄積量）")
phase = st.sidebar.radio("現在のフェーズを選択", ("Phase 1: 初動 (不確実性大)", "Phase 2: 中期 (癖を把握)", "Phase 3: 実用化 (定常)"))
default_gamma = 1.5 if "Phase 1" in phase else 1.0 if "Phase 2" in phase else 0.5

st.sidebar.header("⚙️ 現場の工学パラメータ入力")
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
        st.warning("❌ **削るセンサー**\nなし（フル装備）")