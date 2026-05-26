import os
import requests
import streamlit as st
import pandas as pd
import numpy as np
import torch
import shutil
import yaml
import gc

# ==========================================
# 1. 极致内存与环境配置 (必须在最顶端)
# ==========================================
# 线上环境内存限制 1GB，禁用梯度计算和多线程
torch.set_num_threads(1)
torch.set_grad_enabled(False)

# 重定向路径至可写的 /tmp 目录
target_weight_dir = '/tmp/unimol_weights'
os.environ['UNIMOL_WEIGHT_DIR'] = target_weight_dir
os.environ['HF_HOME'] = '/tmp/huggingface'
os.environ['HF_HUB_OFFLINE'] = '0' # 允许首次运行下载基础权重

# ==========================================
# 2. 环境自愈脚本 (处理权限、文件搬运与下载)
# ==========================================
def bootstrap_unimol():
    if not os.path.exists(target_weight_dir):
        os.makedirs(target_weight_dir, exist_ok=True)
    
    # A. 搬运仓库中的微调权重、字典和阈值文件
    local_dir = './model_weight'
    if os.path.exists(local_dir):
        # 搬运基础配置文件
        for f in ['config.yaml', 'mol.dict.txt', 'threshold.dat']:
            src = os.path.join(local_dir, f)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(target_weight_dir, f))
        
        # 寻找微调权重并重命名为 model_0.pth (推理引擎必读)
        pth_files = [f for f in os.listdir(local_dir) if f.endswith('.pth') and 'mol_pre' not in f]
        if pth_files:
            shutil.copy(os.path.join(local_dir, pth_files[0]), os.path.join(target_weight_dir, 'model_0.pth'))

    # B. 基础预训练权重补丁 (mol_pre_all_h_220816.pt) - 190MB
    foundation_name = 'mol_pre_all_h_220816.pt'
    foundation_path = os.path.join(target_weight_dir, foundation_name)
    if not os.path.exists(foundation_path):
        url = f"https://huggingface.co/dptech/Uni-Mol-Models/resolve/main/{foundation_name}"
        with st.spinner("首次运行：正在初始化 Uni-Mol 基础环境 (190MB)..."):
            try:
                r = requests.get(url, stream=True)
                r.raise_for_status()
                with open(foundation_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024*1024): f.write(chunk)
            except Exception as e:
                st.error(f"基础环境初始化失败，请点击右侧 Reboot: {e}")

    # C. 配置文件补丁：强制单折推理模式 (核心稳定性修复)
    conf_p = os.path.join(target_weight_dir, 'config.yaml')
    conf = {'task': 'classification', 'model_name': 'unimolv1', 'kfold': 1}
    if os.path.exists(conf_p):
        try:
            with open(conf_p, 'r') as f:
                user_conf = yaml.safe_load(f)
                if user_conf: conf.update(user_conf)
        except: pass
    conf['kfold'] = 1
    with open(conf_p, 'w') as f: yaml.dump(conf, f)

    # D. 阈值文件补丁 (如果仓库没传，则自动创建)
    threshold_p = os.path.join(target_weight_dir, 'threshold.dat')
    if not os.path.exists(threshold_p):
        with open(threshold_p, 'w') as f: f.write("0.5789494082595124")

# ==========================================
# 3. 模型加载逻辑 (带缓存)
# ==========================================
@st.cache_resource
def get_predictor():
    bootstrap_unimol() # 先确保环境就绪
    from unimol_tools import MolPredict # 延迟导入以省内存
    try:
        predictor = MolPredict(target_weight_dir)
        gc.collect()
        return predictor
    except Exception as e:
        st.error(f"模型载入失败: {e}")
        return None

# ==========================================
# 4. Streamlit UI 界面
# ==========================================
st.set_page_config(page_title="Uni-Mol BBB 在线预测", page_icon="🧪")
st.title("🧪 Uni-Mol 分子血脑屏障预测")

st.sidebar.header("药物示例")
drug_examples = {
    "请选择...": "",
    "地西泮 (Diazepam, 阳性)": "CN1C(=O)CN=C(C2=C1C=CC(=C2)Cl)C3=CC=CC=C3",
    "阿替洛尔 (Atenolol, 阴性)": "CC(C)NCC(COC1=CC=C(C=C1)CC(N)=O)O",
    "氟西汀 (Fluoxetine, 阳性)": "CNCCC(C1=CC=CC=C1)OC2=CC=C(C=C2)C(F)(F)F"
}
selected = st.sidebar.selectbox("选择已知药物对比:", list(drug_examples.keys()))
input_smi = st.text_input("输入自定义 SMILES 结构:", value=drug_examples[selected] if selected != "请选择..." else "")

if st.button("开始 3D 深度分析", type="primary"):
    if input_smi:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        import py3Dmol
        
        mol = Chem.MolFromSmiles(input_smi)
        if not mol:
            st.error("❌ 无效的 SMILES 字符串")
        else:
            with st.spinner('Uni-Mol 正在提取 3D 特征...'):
                try:
                    predictor = get_predictor()
                    raw_preds = np.array(predictor.predict([input_smi]))
                    
                    # --- 核心修复：全兼容概率提取逻辑 ---
                    flat_preds = raw_preds.flatten()
                    # 如果返回 [prob0, prob1] 取第二个；如果返回 [prob] 直接取
                    prob = float(flat_preds[1]) if len(flat_preds) >= 2 else float(flat_preds[0])
                    
                    st.divider()
                    c1, c2 = st.columns([1, 1.2])
                    
                    with c1:
                        st.subheader("预测结果")
                        # 使用 0.5789 作为分类阈值
                        if prob > 0.5789:
                            st.success("### 【能穿透】\n中枢神经活跃")
                        else:
                            st.error("### 【难穿透】\n外周分布为主")
                        
                        st.metric("穿透概率评分", f"{prob:.4f}")
                        st.write(f"分子量: {AllChem.CalcExactMolWt(mol):.1f} Da")
                    
                    with c2:
                        st.subheader("3D 预览")
                        m3d = Chem.AddHs(mol)
                        AllChem.EmbedMolecule(m3d, AllChem.ETKDG())
                        st.components.v1.html(
                            py3Dmol.view(width=400, height=300).addModel(Chem.MolToMolBlock(m3d), 'mol').setStyle({'stick':{'colorscheme':'cyanCarbon'}, 'sphere':{'scale':0.3}}).zoomTo()._make_html(), 
                            height=320
                        )
                    
                    # 内存清理
                    del raw_preds
                    gc.collect()

                except Exception as e:
                    st.error(f"分析出错: {e}")
                    st.info("提示：这通常是内存溢出导致的，请点击右侧 Reboot 重试。")

st.divider()
st.caption("技术栈: Uni-Mol (Transformer) | 迁移学习任务: BBBP | 验证集 ROC-AUC: 0.92")
