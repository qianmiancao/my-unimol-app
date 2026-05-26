import streamlit as st
import pandas as pd
import numpy as np
import torch
import os
from unimol_tools import MolPredict
from rdkit import Chem
from rdkit.Chem import AllChem
import py3Dmol

# --- 1. 性能与内存优化配置 ---
# 线上环境内存极小，强制 Torch 使用单线程防止内存溢出
torch.set_num_threads(1)

# --- 2. 页面基础设置 ---
st.set_page_config(
    page_title="Uni-Mol BBB 在线预测",
    page_icon="🧪",
    layout="centered"
)

st.title("🧪 Uni-Mol 分子血脑屏障通透性预测")
st.markdown("""
本工具基于 **Uni-Mol 3D 基础模型**。
1. 在左侧或下方输入分子的 **SMILES** 字符串。
2. 模型将自动生成 **3D 构象** 并评估其穿透血脑屏障（BBB）的概率。
""")

# --- 3. 模型加载逻辑 (带缓存) ---
@st.cache_resource
def load_unimol_model():
    # 线上环境路径为当前目录下的 model_weight
    model_path = './model_weight'
    if not os.path.exists(model_path):
        st.error(f"找不到模型目录: {model_path}。请检查 GitHub 仓库。")
        return None
    
    try:
        # 使用位置参数加载，并在 CPU 上运行
        predictor = MolPredict(model_path)
        return predictor
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        return None

with st.spinner("正在初始化 Uni-Mol 引擎，请稍候..."):
    predictor = load_unimol_model()

# --- 4. 侧边栏与示例 ---
st.sidebar.header("快速示例")
examples = {
    "请选择...": "",
    "咖啡因 (易穿透)": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "普萘洛尔 (易穿透)": "CC(C)NCC(COC1=CC=CC2=CC=CC=C21)O",
    "多巴胺 (难穿透)": "C1=CC(=C(C=C1CCN)O)O",
    "蔗糖 (难穿透)": "C(C1C(C(C(C(O1)OCC2C(C(C(C(O2)OC3(C(C(C(O3)CO)O)O)CO)O)O)O)O)O)O)O"
}
selected = st.sidebar.selectbox("选择内置示例:", list(examples.keys()))
default_smi = examples[selected] if selected != "请选择..." else ""

# --- 5. 用户输入 ---
input_smi = st.text_input("输入分子的 SMILES 结构:", value=default_smi, placeholder="例如: CCO")

# --- 6. 核心预测与展示逻辑 ---
if st.button("开始 AI 分析", type="primary"):
    if not input_smi:
        st.warning("请输入有效的内容。")
    else:
        # A. 验证 SMILES
        mol = Chem.MolFromSmiles(input_smi)
        if mol is None:
            st.error("❌ 无效的 SMILES 字符串，请检查输入。")
        else:
            with st.spinner('Uni-Mol 正在计算 3D 构象并提取特征...'):
                try:
                    # B. 执行推理
                    raw_preds = np.array(predictor.predict([input_smi]))
                    
                    # C. 解析预测概率 (适配 [N,2] 或 [N,1] 维度)
                    if raw_preds.ndim > 1 and raw_preds.shape[1] > 1:
                        prob = float(raw_preds[0][1])
                    else:
                        prob = float(raw_preds[0][0])
                    
                    # D. 展示结果面板
                    st.divider()
                    c1, c2 = st.columns([1, 1.2])
                    
                    with c1:
                        st.subheader("分析结果")
                        if prob > 0.5:
                            st.success("### 判定：【能穿透】")
                        else:
                            st.error("### 判定：【难穿透】")
                        
                        st.metric("穿透概率评分", f"{prob:.4f}")
                        st.write(f"**置信度**: {prob if prob > 0.5 else 1-prob:.2%}")
                        
                        # 补充一点简单的物理化学参数
                        mw = AllChem.CalcExactMolWt(mol)
                        st.info(f"分子量: {mw:.2f} Da")

                    with c2:
                        st.subheader("3D 构象预览")
                        # 这里的构象仅用于网页展示，推理构象由 Uni-Mol 内部生成
                        m3d = Chem.AddHs(mol)
                        AllChem.EmbedMolecule(m3d, AllChem.ETKDG())
                        mblock = Chem.MolToMolBlock(m3d)
                        
                        view = py3Dmol.view(width=400, height=300)
                        view.addModel(mblock, 'mol')
                        view.setStyle({'stick': {'colorscheme': 'cyanCarbon'}, 'sphere': {'scale': 0.3}})
                        view.zoomTo()
                        st.components.v1.html(view._make_html(), height=320)

                except Exception as e:
                    st.error(f"推理过程中出错: {e}")

# --- 7. 页面底部 ---
st.divider()
st.caption("技术栈: Uni-Mol (ICLR 2023) | 迁移学习任务: BBBP | 模型精度: ROC-AUC 0.92")
