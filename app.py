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
    
    # A. 搬运仓库中的所有文件
    if os.path.exists(local_weight_dir):
        for f in os.listdir(local_weight_dir):
            shutil.copy(os.path.join(local_weight_dir, f), os.path.join(target_weight_dir, f))

    # B. 基础权重补丁 (mol_pre_all_h_220816.pt)
    foundation_name = 'mol_pre_all_h_220816.pt'
    foundation_path = os.path.join(target_weight_dir, foundation_name)
    if not os.path.exists(foundation_path):
        url = f"https://huggingface.co/dptech/Uni-Mol-Models/resolve/main/{foundation_name}"
        with st.spinner("正在初始化基础环境，请稍候..."):
            r = requests.get(url, stream=True)
            with open(foundation_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): f.write(chunk)

    # C. 权重重命名补丁 (确保有 model_0.pth)
    target_m0 = os.path.join(target_weight_dir, 'model_0.pth')
    if not os.path.exists(target_m0):
        pth_files = [f for f in os.listdir(target_weight_dir) if f.endswith('.pth') and f != foundation_name]
        if pth_files:
            shutil.copy(os.path.join(target_weight_dir, pth_files[0]), target_m0)

    # D. 配置文件强力补丁 (解决 TypeError)
    config_path = os.path.join(target_weight_dir, 'config.yaml')
    try:
        conf = None
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                conf = yaml.safe_load(f)
        
        # 如果文件不存在或内容为空，创建一个基础配置
        if conf is None or not isinstance(conf, dict):
            conf = {
                'task': 'classification',
                'model_name': 'unimolv1',
                'data_type': 'molecule',
                'metrics': 'auc'
            }
        
        # 强制设为单折推理，防止报错
        conf['kfold'] = 1
        
        with open(config_path, 'w') as f:
            yaml.dump(conf, f)
    except Exception as e:
        st.warning(f"配置修复提示: {e}，正在尝试跳过...")

# 执行环境自愈
bootstrap_unimol()

# --- 2. 加载模型 ---
@st.cache_resource
def load_unimol_model():
    try:
        # 直接传路径，位置参数匹配
        return MolPredict(os.environ['UNIMOL_WEIGHT_DIR'])
    except Exception as e:
        st.error(f"核心加载失败: {e}")
        return None

predictor = load_unimol_model()

# --- 3. UI 界面与预测逻辑 ---
st.set_page_config(page_title="Uni-Mol BBB 预测", page_icon="🧪")
st.title("🧪 Uni-Mol 分子血脑屏障穿透性预测")

# 侧边栏
st.sidebar.header("上市药物预测对比")
drug_examples = {
    "请选择...": "",
    "地西泮 (Diazepam, 镇静药)": "CN1C(=O)CN=C(C2=C1C=CC(=C2)Cl)C3=CC=CC=C3",
    "多奈哌齐 (Donepezil, 抗痴呆)": "COC1=C(C=C2C(=C1)CC(C2=O)CC3CCN(CC3)CC4=CC=CC=C4)OC",
    "阿替洛尔 (Atenolol, 降压药)": "CC(C)NCC(COC1=CC=C(C=C1)CC(N)=O)O",
    "氟西汀 (Fluoxetine, 抗抑郁)": "CNCCC(C1=CC=CC=C1)OC2=CC=C(C=C2)C(F)(F)F"
}
selected = st.sidebar.selectbox("查看已知药物的表现:", list(drug_examples.keys()))

input_smi = st.text_input("或输入自定义 SMILES 结构:", value=drug_examples[selected] if selected != "请选择..." else "")

if st.button("开始 3D 深度分析", type="primary"):
    if predictor and input_smi:
        mol = Chem.MolFromSmiles(input_smi)
        if mol:
            with st.spinner('模型正在通过 3D 构象感知分子极性与脂溶性...'):
                try:
                    raw_preds = np.array(predictor.predict([input_smi]))
                    # 适配输出维度
                    prob = float(raw_preds[0][1]) if raw_preds.ndim > 1 and raw_preds.shape[1] > 1 else float(raw_preds[0][0])
                    
                    st.divider()
                    col1, col2 = st.columns([1, 1.2])
                    with col1:
                        st.subheader("预测结论")
                        # 使用训练时确定的最佳阈值 0.5789
                        if prob > 0.5789:
                            st.success("### 【能穿透】\n中枢神经系统活跃")
                        else:
                            st.error("### 【难穿透】\n外周分布为主")
                        st.metric("模型综合评分", f"{prob:.4f}")
                        st.write(f"分子量: {AllChem.CalcExactMolWt(mol):.2f} Da")
                    
                    with col2:
                        st.subheader("3D 预览")
                        m3d = Chem.AddHs(mol)
                        AllChem.EmbedMolecule(m3d, AllChem.ETKDG())
                        st.components.v1.html(
                            py3Dmol.view(width=400, height=300).addModel(Chem.MolToMolBlock(m3d), 'mol').setStyle({'stick':{'colorscheme':'greenCarbon'}, 'sphere':{'scale':0.3}}).zoomTo()._make_html(), 
                            height=320
                        )
                except Exception as e:
                    st.error(f"推理引擎异常: {e}")
        else:
            st.error("❌ 无效的 SMILES 字符串，无法解析")

st.divider()
st.caption("技术详情: Uni-Mol (Transformer 架构) | 任务: BBBP 迁移学习 | 验证集 ROC-AUC: 0.92")
