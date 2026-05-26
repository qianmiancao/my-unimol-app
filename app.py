import os
import requests
import streamlit as st
import pandas as pd
import numpy as np
import torch
import shutil
import yaml
import gc  # 垃圾回收

# --- 1. 极限内存优化：强制 Torch 最小化资源占用 ---
torch.set_num_threads(1)
torch.set_grad_enabled(False) # 禁用梯度计算，节省大量内存

# --- 2. 路径重定向 ---
# 注意：为了节省内存，我们直接在 /tmp 操作，减少文件拷贝带来的开销
target_weight_dir = '/tmp/unimol_weights'
os.environ['UNIMOL_WEIGHT_DIR'] = target_weight_dir
os.environ['HF_HOME'] = '/tmp/huggingface'

def bootstrap_unimol():
    if not os.path.exists(target_weight_dir):
        os.makedirs(target_weight_dir, exist_ok=True)
    
    # A. 基础权重 (mol_pre_all_h_220816.pt) - 仅当不存在时下载
    foundation_name = 'mol_pre_all_h_220816.pt'
    foundation_path = os.path.join(target_weight_dir, foundation_name)
    if not os.path.exists(foundation_path):
        url = f"https://huggingface.co/dptech/Uni-Mol-Models/resolve/main/{foundation_name}"
        r = requests.get(url, stream=True)
        with open(foundation_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024*1024): f.write(chunk)
    
    # B. 搬运仓库文件 (使用软链接或直接读取以省空间，这里采用直接覆盖)
    local_dir = './model_weight'
    if os.path.exists(local_dir):
        for f in ['config.yaml', 'mol.dict.txt', 'threshold.dat']:
            src = os.path.join(local_dir, f)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(target_weight_dir, f))
        
        # 寻找微调权重并重命名为 model_0.pth
        pth_files = [f for f in os.listdir(local_dir) if f.endswith('.pth') and f != foundation_name]
        if pth_files:
            shutil.copy(os.path.join(local_dir, pth_files[0]), os.path.join(target_weight_dir, 'model_0.pth'))

    # C. 强制修改配置为单折，极其重要
    conf_p = os.path.join(target_weight_dir, 'config.yaml')
    conf = {'task': 'classification', 'model_name': 'unimolv1', 'kfold': 1} # 默认最小配置
    if os.path.exists(conf_p):
        try:
            with open(conf_p, 'r') as f:
                user_conf = yaml.safe_load(f)
                if user_conf: conf.update(user_conf)
        except: pass
    conf['kfold'] = 1
    with open(conf_p, 'w') as f: yaml.dump(conf, f)

# --- 3. 延迟加载策略：不在启动时加载模型，而是在第一次使用时加载 ---
@st.cache_resource
def get_predictor():
    from unimol_tools import MolPredict # 延迟导入
    bootstrap_unimol()
    predictor = MolPredict(target_weight_dir)
    gc.collect() # 加载完立即释放无关内存
    return predictor

# --- 4. UI 界面 ---
st.set_page_config(page_title="Uni-Mol BBB", page_icon="🧪")
st.title("🧪 Uni-Mol BBB 在线预测")

# 侧边栏示例
drug_examples = {
    "请选择...": "",
    "地西泮 (能穿透)": "CN1C(=O)CN=C(C2=C1C=CC(=C2)Cl)C3=CC=CC=C3",
    "阿替洛尔 (难穿透)": "CC(C)NCC(COC1=CC=C(C=C1)CC(N)=O)O",
}
selected = st.sidebar.selectbox("示例药物:", list(drug_examples.keys()))
input_smi = st.text_input("输入 SMILES:", value=drug_examples[selected] if selected != "请选择..." else "")

if st.button("开始分析", type="primary"):
    if input_smi:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        import py3Dmol
        
        mol = Chem.MolFromSmiles(input_smi)
        if mol:
            with st.spinner('模型载入与 3D 计算中... (可能需要1分钟)'):
                try:
                    predictor = get_predictor() # 这里才会真正占用大内存
                    raw_preds = np.array(predictor.predict([input_smi]))
                    prob = float(raw_preds[0][1]) if raw_preds.ndim > 1 else float(raw_preds[0][0])
                    
                    st.divider()
                    c1, c2 = st.columns([1, 1.2])
                    with c1:
                        if prob > 0.5789: st.success("### 能穿透")
                        else: st.error("### 难穿透")
                        st.metric("概率评分", f"{prob:.4f}")
                        st.write(f"MW: {AllChem.CalcExactMolWt(mol):.1f}")
                    
                    with c2:
                        m3d = Chem.AddHs(mol)
                        AllChem.EmbedMolecule(m3d, AllChem.ETKDG())
                        st.components.v1.html(py3Dmol.view(width=350, height=250).addModel(Chem.MolToMolBlock(m3d), 'mol').setStyle({'stick':{}, 'sphere':{'scale':0.3}}).zoomTo()._make_html(), height=260)
                    
                    # 预测完手动清理
                    del raw_preds
                    gc.collect()

                except Exception as e:
                    st.error(f"内存不足或计算错误: {e}")
                    st.info("请尝试点击右侧的 Reboot App")
        else:
            st.error("无效 SMILES")

st.caption("注：免费版内存有限。若崩溃，请刷新页面或等待系统回收资源。")
