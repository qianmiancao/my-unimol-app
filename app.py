import os
import requests
import streamlit as st
import pandas as pd
import numpy as np
import torch
import shutil
import yaml
from unimol_tools import MolPredict
from rdkit import Chem
from rdkit.Chem import AllChem
import py3Dmol

# --- 1. 环境初始化：重定向所有路径到 /tmp ---
os.environ['UNIMOL_WEIGHT_DIR'] = '/tmp/unimol_weights'
os.environ['HF_HOME'] = '/tmp/huggingface'
os.environ['HF_HUB_OFFLINE'] = '0' 

def bootstrap_unimol():
    local_weight_dir = './model_weight'
    target_weight_dir = os.environ['UNIMOL_WEIGHT_DIR']
    
    if not os.path.exists(target_weight_dir):
        os.makedirs(target_weight_dir, exist_ok=True)
    
    # A. 搬运仓库中的所有文件 (包括你刚上传的 threshold.dat)
    if os.path.exists(local_weight_dir):
        for f in os.listdir(local_weight_dir):
            shutil.copy(os.path.join(local_weight_dir, f), os.path.join(target_weight_dir, f))

    # B. 基础权重补丁：如果 /tmp 里没有基础权重，则下载 (190MB)
    foundation_name = 'mol_pre_all_h_220816.pt'
    foundation_path = os.path.join(target_weight_dir, foundation_name)
    if not os.path.exists(foundation_path):
        url = f"https://huggingface.co/dptech/Uni-Mol-Models/resolve/main/{foundation_name}"
        with st.spinner("首次启动：正在下载 Uni-Mol 基础预训练权重..."):
            r = requests.get(url, stream=True)
            with open(foundation_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): f.write(chunk)

    # C. 权重重命名补丁：确保有 model_0.pth (推理引擎启动入口)
    target_m0 = os.path.join(target_weight_dir, 'model_0.pth')
    if not os.path.exists(target_m0):
        # 寻找你上传的任何 .pth 文件 (比如 model_4.pth)
        pth_files = [f for f in os.listdir(target_weight_dir) if f.endswith('.pth') and f != foundation_name]
        if pth_files:
            shutil.copy(os.path.join(target_weight_dir, pth_files[0]), target_m0)

    # D. 配置文件补丁：强制单折推理模式
    config_path = os.path.join(target_weight_dir, 'config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            conf = yaml.safe_load(f)
        conf['kfold'] = 1
        with open(config_path, 'w') as f:
            yaml.dump(conf, f)

# 启动环境准备
bootstrap_unimol()

# --- 2. 加载模型 ---
@st.cache_resource
def load_unimol_model():
    try:
        # 使用重定向后的路径
        return MolPredict(os.environ['UNIMOL_WEIGHT_DIR'])
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        return None

predictor = load_unimol_model()

# --- 3. UI 界面 ---
st.set_page_config(page_title="Uni-Mol BBB 在线预测", page_icon="🧪")
st.title("🧪 Uni-Mol 分子血脑屏障通透性预测")

# 侧边栏
st.sidebar.header("快速示例")
examples = {
    "请选择...": "",
    "咖啡因 (能穿透)": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "多巴胺 (难穿透)": "C1=CC(=C(C=C1CCN)O)O",
    "普萘洛尔 (能穿透)": "CC(C)NCC(COC1=CC=CC2=CC=CC=C21)O"
}
selected = st.sidebar.selectbox("内置示例:", list(examples.keys()))

# 主输入框
input_smi = st.text_input("请输入分子的 SMILES 结构:", value=examples[selected] if selected != "请选择..." else "")

if st.button("开始 AI 分析", type="primary"):
    if predictor and input_smi:
        mol = Chem.MolFromSmiles(input_smi)
        if mol:
            with st.spinner('Uni-Mol 正在提取 3D 空间特征并进行推理...'):
                try:
                    raw_preds = np.array(predictor.predict([input_smi]))
                    # 适配输出结果
                    prob = float(raw_preds[0][1]) if raw_preds.ndim > 1 and raw_preds.shape[1] > 1 else float(raw_preds[0][0])
                    
                    st.divider()
                    col1, col2 = st.columns([1, 1.2])
                    with col1:
                        st.subheader("分析结果")
                        # 使用 BBBP 任务的标准阈值
                        if prob > 0.5789:
                            st.success("### 判定：【能穿透】")
                        else:
                            st.error("### 判定：【难穿透】")
                        st.metric("通透性评分", f"{prob:.4f}")
                        st.info(f"分子量: {AllChem.CalcExactMolWt(mol):.2f}")
                    
                    with col2:
                        st.subheader("3D 构象预览")
                        m3d = Chem.AddHs(mol)
                        AllChem.EmbedMolecule(m3d, AllChem.ETKDG())
                        m_block = Chem.MolToMolBlock(m3d)
                        
                        view = py3Dmol.view(width=400, height=300)
                        view.addModel(m_block, 'mol')
                        view.setStyle({'stick': {'colorscheme': 'cyanCarbon'}, 'sphere': {'scale': 0.3}})
                        view.zoomTo()
                        st.components.v1.html(view._make_html(), height=320)
                except Exception as e:
                    st.error(f"分析出错: {e}")
        else:
            st.error("❌ 无效的 SMILES")

st.divider()
st.caption("基于 Uni-Mol 预训练模型 | BBB 迁移学习 | ROC-AUC: 0.92")
