import streamlit as st
import pandas as pd
import numpy as np
import torch
from unimol_tools import MolPredict
from rdkit import Chem
from rdkit.Chem import AllChem
import py3Dmol

# 设置页面
st.set_page_config(page_title="Uni-Mol BBB Predictor", layout="wide")

st.title("🧪 分子血脑屏障(BBB)穿透性在线预测")

# --- 模型加载函数 ---
@st.cache_resource
def get_predictor():
    # 强制使用 CPU 运行以节省内存
    try:
        # 指向你上传到仓库的权重目录
        predictor = MolPredict(load_model_dir='./model_weight')
        return predictor
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        return None

predictor = get_predictor()

# --- 侧边栏示例 ---
st.sidebar.header("输入示例")
example_smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" # 咖啡因
smi = st.text_input("输入 SMILES 结构:", value=example_smiles)

# --- 预测与展示 ---
if st.button("开始分析"):
    if smi:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            with st.spinner("Uni-Mol 正在思考..."):
                # 执行预测
                preds = np.array(predictor.predict([smi]))
                # 适配维度
                prob = preds[0][1] if preds.ndim > 1 and preds.shape[1] > 1 else preds[0][0]
                
                # 页面布局
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("穿透概率", f"{prob:.4f}")
                    if prob > 0.5:
                        st.success("结果：【能穿透】")
                    else:
                        st.error("结果：【难穿透】")
                    
                    # 属性展示
                    st.write(f"分子量: {AllChem.CalcExactMolWt(mol):.2f}")
                    st.write(f"旋转键数量: {AllChem.CalcNumRotatableBonds(mol)}")

                with col2:
                    st.subheader("3D 构象 (RDKit 生成)")
                    # 生成 3D 用于展示
                    m3d = Chem.AddHs(mol)
                    AllChem.EmbedMolecule(m3d, AllChem.ETKDG())
                    mblock = Chem.MolToMolBlock(m3d)
                    
                    view = py3Dmol.view(width=400, height=300)
                    view.addModel(mblock, 'mol')
                    view.setStyle({'stick': {}, 'sphere': {'scale': 0.3}})
                    view.zoomTo()
                    st.components.v1.html(view._make_html(), height=350)
        else:
            st.error("无效的 SMILES")