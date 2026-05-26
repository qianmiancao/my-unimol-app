import os
# --- 必须放在最顶端：强制将缓存和权重目录重定向到可写的 /tmp 目录 ---
os.environ['UNIMOL_WEIGHT_DIR'] = '/tmp/unimol_weights'
os.environ['HF_HOME'] = '/tmp/huggingface'
os.environ['HF_HUB_OFFLINE'] = '1'  # 强制离线模式，不让它尝试写系统目录

import streamlit as st
import pandas as pd
import numpy as np
import torch
import shutil
from unimol_tools import MolPredict
from rdkit import Chem
from rdkit.Chem import AllChem
import py3Dmol

# --- 1. 自动修复环境：将本地字典复制到系统期望的位置 ---
def bootstrap_unimol():
    """
    因为 unimol-tools 强制在系统路径找字典，
    我们需要在运行时把我们的字典搬到它能读到的地方（/tmp）。
    """
    local_weight_dir = './model_weight'
    target_weight_dir = os.environ['UNIMOL_WEIGHT_DIR']
    
    if not os.path.exists(target_weight_dir):
        os.makedirs(target_weight_dir, exist_ok=True)
    
    # 将仓库里的核心文件拷贝到可写的 /tmp/unimol_weights
    core_files = ['mol.dict.txt', 'config.yaml', 'model_4.pth']
    for f in core_files:
        src = os.path.join(local_weight_dir, f)
        dst = os.path.join(target_weight_dir, f)
        if os.path.exists(src):
            shutil.copy(src, dst)

bootstrap_unimol()

# --- 2. 页面基础设置 ---
st.set_page_config(page_title="Uni-Mol BBB 在线预测", page_icon="🧪")
torch.set_num_threads(1)

st.title("🧪 Uni-Mol 分子血脑屏障通透性预测")

# --- 3. 模型加载 ---
@st.cache_resource
def load_unimol_model():
    try:
        # 指向我们刚才搬运好的 /tmp 目录
        predictor = MolPredict(load_model_dir=os.environ['UNIMOL_WEIGHT_DIR'])
        return predictor
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        return None

predictor = load_unimol_model()

# --- 4. 侧边栏与示例 ---
st.sidebar.header("快速示例")
examples = {
    "请选择...": "",
    "咖啡因 (易穿透)": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "普萘洛尔 (易穿透)": "CC(C)NCC(COC1=CC=CC2=CC=CC=C21)O",
    "多巴胺 (难穿透)": "C1=CC(=C(C=C1CCN)O)O"
}
selected = st.sidebar.selectbox("选择内置示例:", list(examples.keys()))
default_smi = examples[selected] if selected != "请选择..." else ""

# --- 5. 用户输入 ---
input_smi = st.text_input("输入分子的 SMILES 结构:", value=default_smi)

# --- 6. 预测与展示 ---
if st.button("开始 AI 分析", type="primary"):
    if not input_smi:
        st.warning("请输入有效的内容。")
    else:
        mol = Chem.MolFromSmiles(input_smi)
        if mol is None:
            st.error("❌ 无效的 SMILES 字符串")
        else:
            with st.spinner('Uni-Mol 正在通过 3D 空间特征进行计算...'):
                try:
                    raw_preds = np.array(predictor.predict([input_smi]))
                    prob = float(raw_preds[0][1]) if raw_preds.ndim > 1 else float(raw_preds[0][0])
                    
                    st.divider()
                    c1, c2 = st.columns([1, 1.2])
                    with c1:
                        st.subheader("分析结果")
                        if prob > 0.5:
                            st.success("### 判定：【能穿透】")
                        else:
                            st.error("### 判定：【难穿透】")
                        st.metric("穿透概率评分", f"{prob:.4f}")
                    
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
