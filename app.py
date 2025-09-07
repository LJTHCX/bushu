import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

# --- 页面基础设置 ---
st.set_page_config(
    page_title="染色体非整倍体风险预测",
    layout="wide"
)

# 设置Matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# --- 数据和模型加载 ---
@st.cache_data
def load_data():
    """加载原始数据用于获取特征名称和LIME背景数据"""
    df = pd.read_excel('数据.xlsx')
    # 确保列名与模型训练时一致
    if '染色体的非整倍体' in df.columns:
        X = df.drop('染色体的非整倍体', axis=1)
    else:
        # 如果目标变量不存在，假定所有列都是特征
        X = df
    return X


@st.cache_resource
def load_model():
    """加载预训练的TabPFN模型"""
    try:
        with open('tabpfn_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("错误：找不到 'tabpfn_model.pkl' 文件。请确保模型文件与APP在同一目录下。")
        return None


# 加载资源
X_train = load_data()
model = load_model()
feature_names = X_train.columns.tolist()


# --- 用户输入功能模块 ---
def user_input_features():
    st.sidebar.header('请输入检测指标:')
    feature_values = {
        '孕妇BMI': st.sidebar.number_input('孕妇BMI', 15.0, 50.0, 25.5, 0.1),
        '原始读段数': st.sidebar.number_input('原始读段数', 1000000, 20000000, 6000000, 1000),
        '在参考基因组上比对的比例': st.sidebar.slider('基因组比对比例', 0.5, 1.0, 0.8, 0.01),
        '重复读段的比例': st.sidebar.slider('重复读段比例', 0.0, 0.2, 0.05, 0.001),
        '唯一比对的读段数': st.sidebar.number_input('唯一比对读段数', 1000000, 20000000, 4500000, 1000),
        'GC含量': st.sidebar.slider('GC含量', 0.3, 0.5, 0.4, 0.001),
        '13号染色体的Z值': st.sidebar.slider('13号染色体Z值', -10.0, 10.0, 0.0, 0.1),
        '18号染色体的Z值': st.sidebar.slider('18号染色体Z值', -10.0, 10.0, 0.0, 0.1),
        '21号染色体的Z值': st.sidebar.slider('21号染色体Z值', -10.0, 10.0, 0.0, 0.1),
        'X染色体的Z值': st.sidebar.slider('X染色体Z值', -10.0, 10.0, 0.0, 0.1),
        'X染色体浓度': st.sidebar.slider('X染色体浓度', -1.0, 1.0, 0.0, 0.01),
        '13号染色体的GC含量': st.sidebar.slider('13号染色体GC含量', 0.3, 0.5, 0.38, 0.001),
        '18号染色体的GC含量': st.sidebar.slider('18号染色体GC含量', 0.3, 0.5, 0.39, 0.001),
        '21号染色体的GC含量': st.sidebar.slider('21号染色体GC含量', 0.3, 0.5, 0.4, 0.001),
        '被过滤掉读段数的比例': st.sidebar.slider('读段过滤比例', 0.0, 0.2, 0.03, 0.001),
        '怀孕次数': st.sidebar.number_input('怀孕次数', 0, 10, 1, 1),
        '生产次数': st.sidebar.number_input('生产次数', 0, 10, 0, 1)
    }
    input_df = pd.DataFrame([feature_values])
    return input_df


# --- 主界面 ---
st.title("染色体非整倍体风险智能预测与解释系统")

# 更新后的页面选择，已移除SHAP
page = st.selectbox("选择功能页面", ["在线预测 (Online Prediction)", "LIME 可视化解释"])

input_df = user_input_features()

# --- 页面1：在线预测 ---
if page == "在线预测 (Online Prediction)":
    st.header("风险预测")
    st.write("请在左侧边栏输入或调整各项检测指标，然后点击“预测”按钮。")
    st.write("---")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("您输入的指标:")
        st.dataframe(input_df.T.rename(columns={0: '数值'}))

    with col2:
        st.subheader("预测结果:")
        if st.button("开始预测"):
            if model is not None:
                prediction_proba = model.predict_proba(input_df)
                aneuploidy_prob = prediction_proba[0, 1]

                st.write(f"根据您输入的指标，模型预测 **染色体非整倍体 (Aneuploidy) 的风险概率为：**")

                if aneuploidy_prob > 0.5:
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{aneuploidy_prob:.2%}</h1>",
                                unsafe_allow_html=True)
                    st.warning("**风险提示：** 预测风险较高，建议咨询专业医生进行进一步诊断。")
                else:
                    st.markdown(f"<h1 style='text-align: center; color: green;'>{aneuploidy_prob:.2%}</h1>",
                                unsafe_allow_html=True)
                    st.success("**风险提示：** 预测风险较低。请注意，本结果仅供参考。")
            else:
                st.error("模型未能成功加载，无法进行预测。")

# --- 页面2：LIME 可视化解释 ---
elif page == "LIME 可视化解释":
    st.header("LIME (Local Interpretable Model-agnostic Explanations) 模型解释")
    st.info(
        "LIME 从局部角度解释单个预测。它告诉我们对于您输入的这个特定样本，哪些特征及其数值最重要地影响了最终的预测结果。")
    st.write("---")

    if st.button("生成LIME解释图"):
        if model is not None:
            with st.spinner('正在计算LIME解释，请稍候...'):
                lime_explainer = LimeTabularExplainer(
                    training_data=X_train.values,
                    feature_names=feature_names,
                    class_names=['整倍体 (Euploid)', '非整倍体 (Aneuploid)'],
                    mode='classification'
                )

                lime_exp = lime_explainer.explain_instance(
                    data_row=input_df.iloc[0].values,
                    predict_fn=model.predict_proba,
                    num_features=10
                )

                st.subheader("预测结果的LIME局部解释")
                st.components.v1.html(lime_exp.as_html(), height=800, scrolling=True)
        else:
            st.error("模型未能成功加载，无法生成解释。")