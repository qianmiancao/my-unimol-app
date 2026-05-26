import os
import requests
# --- 环境重定向：必须在所有 import 之前 ---
os.environ['UNIMOL_WEIGHT_DIR'] = '/tmp/unimol_weights'
os.environ['HF_HOME'] = '/tmp/huggingface'
os.environ['HF_HUB_OFFLINE'] = '0' # 允许连接以检查基础权重

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

# --- 1. 自动修复环境：搬运、下载并“单折化”模型 ---
def bootstrap_unimol():
    local_weight_dir = './model_weight'
    target_weight_dir = os.environ['UNIMOL_WEIGHT_DIR']
    
    if not os.path.exists(target_weight_dir):
        os.makedirs(target_weight_dir, exist_ok=True)
    
    # A. 搬运你 GitHub 仓库里的微调权重和字典
    if os.path.exists(local_weight_dir):
        files_in_repo = os.listdir(local_weight_dir)
        for f in files_in_repo:
            shutil.copy(os.path.join(local_weight_dir, f), os.path.join(target_weight_dir, f))

    # B. 核心修复：自动下载缺失的基础预训练权重 (约 190MB)
    foundation_model_name = 'mol_pre_all_h_220816.pt'
    foundation_model_path = os.path.join(target_weight_dir, foundation_model_name)
    
    if not os.path.exists(foundation_model_path):
        url = f"https://huggingface.co/dptech/Uni-Mol-Models/resolve/main/{foundation_model_name}"
        with st.spinner("首次运行：正在下载 Uni-Mol 基础权重 (190MB)，请稍候..."):
            try:
                r = requests.get(url, stream=True)
                r.raise_for_status()
                with open(foundation_model_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.success("基础权重下载成功！")
            except Exception as e:
                st.error(f"基础权重下载失败，请刷新页面重试: {e}")

    # C. 兼容性：确保有 model_0.pth
    target_m0 = os.path.join(target_weight_dir, 'model_0.pth')
    if not os.path.exists(target_m0):
        pth_files = [f for f in os.listdir(target_weight_dir) if f.endswith('.pth') and f != foundation_model_name]
        if pth_files:
            shutil.copy(os.path.join(target_weight_dir, pth_files[0]), target_m0)

    # D. 修改配置为单折模式
    config_path = os.path.join(target_weight_dir, 'config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            conf = yaml.safe_load(f)
        conf['kfold'] = 1
        with open(config_path, 'w') as f:
            yaml.dump(conf, f)

# 运行启动脚本
bootstrap_unimol()

# --- 2. 页面基础设置 ---
st.set_page_config(page_title="Uni-Mol BBB 在线预测", page_icon="🧪")
torch.set_num_threads(1) # 线上内存优化

st.title("🧪 Uni-Mol 分子血脑屏障通透性预测")

# --- 3. 模型加载 ---
@st.cache_resource
def load_unimol_model():
    model_path = os.environ['UNIMOL_WEIGHT_DIR']
    try:
        predictor = MolPredict(model_path)
        return predictor
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        return None

predictor = load_unimol_model()

# --- 4. 示例与输入 ---
st.sidebar.header("快速示例")
examples = {
    "请选择...": "",
    "咖啡因 (能穿透)": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "多巴胺 (难穿透)": "C1=CC(=C(C=C1CCN)O)O",
    "普萘洛尔 (能穿透)": "CC(C)NCC(COC1=CC=CC2=CC=CC=C21)O"
}
selected = st.sidebar.selectbox("选择内置示例:", list(examples.keys()))
input_smi = st.text_input("输入分子的 SMILES 结构:", value=examples[selected] if selected != "请选择..." else "")

# --- 5. 预测逻辑 ---
if st.button("开始 AI 分析", type="primary"):
    if not predictor:
        st.error("模型未就绪，请检查上方下载进度。")
    elif not input_smi:
        st.warning("请输入有效的内容。")
    else:
        mol = Chem.MolFromSmiles(input_smi)
        if mol is None:
            st.error("❌ 无效的 SMILES 字符串")
        else:
            with st.spinner('Uni-Mol 正在提取 3D 几何特征并进行推理...'):
                try:
                    raw_preds = np.array(predictor.predict([input_smi]))
                    prob = float(raw_preds[0][1]) if raw_preds.ndim > 1 and raw_preds.shape[1] > 1 else float(raw_preds[0][0])
                    
                    st.divider()
                    c1, c2 = st.columns([1, 1.2])
                    with c1:
                        st.subheader("分析结果")
                        if prob > 0.5:
                            st.success("### 判定：【能穿透】")
                        else:
                            st.error("### 判定：【难穿透】")
                        st.metric("穿透概率评分", f"{prob:.4f}")
                        st.write(f"**分子量**: {AllChem.CalcExactMolWt(mol):.2f}")
                    
                    with c2:
                        st.subheader("3D 构象预览")
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

st.divider()
st.caption("技术栈: Uni-Mol (ICLR 2023) | 迁移学习任务: BBBP | 模型精度: ROC-AUC 0.92")
