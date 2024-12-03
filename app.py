
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import pandas as pd

# 필수 서버 설정
st.set_page_config(
    page_title="의료 진료과 추천 시스템",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 보안 및 접근 설정
import streamlit.components.v1 as components
components.html("""
    <script>
        if (window.top != window.self) {
            window.top.location = window.self.location;
        }
    </script>
""", height=0)

# MedicalBertClassifier 클래스 정의
class MedicalBertClassifier(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained("madatnlp/km-bert")
        self.dropout = nn.Dropout(dropout_rate)
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

# 예측을 위한 클래스 정의
class MedicalDepartmentPredictor:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("madatnlp/km-bert")
        
        # 진료과 매핑 생성
        df = pd.read_csv("./data/final_df.csv", encoding="utf-8")

        # 진료과목코드와 실제 진료과 매핑
        dept_mapping = {
            0: '일반의', 1: '내과', 2: '신경과', 3: '정신과', 4: '외과', 5: '정형외과', 6: '신경외과',
            7: '흉부외과', 8: '성형외과', 9: '마취통증의학과', 10: '산부인과', 11: '소아청소년과',
            12: '안과', 13: '이비인후과', 14: '피부과', 15: '비뇨기과', 16: '영상의학과',
            17: '방사선 종양학과', 18: '병리과', 19: '진단검사의학과', 20: '결핵과',
            21: '재활의학과', 22: '핵의학과', 23: '가정의학과', 24: '응급의학과',
            25: '산업의학과', 26: '예방의학과', 50: '구강악안면외과', 51: '치과보철과',
            52: '치과교정과', 53: '소아치과', 54: '치주과', 55: '치과보존과',
            56: '구강내과', 57: '구강악안면방사선과', 58: '구강병리과',
            59: '예방치과', 80: '한방내과', 81: '한방부인과',
            82: '한방소아과', 83: '한방안·이비인후·피부 과',
            84: '한방신경정신 과', 85: '침구 과',
            86: '한방재활의학 과',87:'사상체질 과',
            88:'한방응급 과','ZZ':'Nan'
        }
        unique_departments = sorted(df['진료과목코드'].unique())
        self.idx_to_dept = {str(idx): dept_mapping.get(dept, dept) for idx, dept in enumerate(unique_departments)}
        num_classes = len(self.idx_to_dept)
        
        self.model = MedicalBertClassifier(num_classes=num_classes).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        self.model.eval()

    def clean_text(self, text):
        import re
        text = re.sub(r'[^가-힣a-zA-Z0-9.,\s]', ' ', str(text))
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\.+', '.', text)
        return text

    def predict(self, text):
        cleaned_text = self.clean_text(text)
        inputs = self.tokenizer(
            cleaned_text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        with torch.no_grad():
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            outputs = self.model(input_ids, attention_mask)
            probs = F.softmax(outputs, dim=1)

        pred_idx = torch.argmax(probs, dim=1).item()
        pred_dept = self.idx_to_dept[str(pred_idx)]
        confidence = probs[0][pred_idx].item()
        
        return pred_dept, confidence


# 스타일 설정
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
        color: #1E88E5;
    }
    .result-box {
        background-color: #F8F9FA;
        padding: 2rem;
        border-radius: 10px;
        margin-top: 2rem;
        border: 1px solid #E0E0E0;
    }
    </style>
    """, unsafe_allow_html=True)

# 앱 제목
st.markdown('<p class="main-title">의료 진료과 추천 시스템 🏥</p>', unsafe_allow_html=True)

# 모델 로드
@st.cache_resource
def load_predictor():
    return MedicalDepartmentPredictor(model_path='./data/best_model.pt', device='mps')

try:
    predictor = load_predictor()

    # 사용자 입력
    st.markdown("### 어디가 어떻게 아프신가요? 구체적으로 말씀해주세요.")
    user_input = st.text_area(
        label="증상을 입력해주세요",
        height=150,
        placeholder="예시: 기침이 심하고 가래가 있으며, 호흡이 곤란한 증상이 있습니다..."
    )

    # 예측 실행 부분
    if st.button("증상 분석하기", type="primary"):
        if user_input:
            with st.spinner('증상을 분석하고 있습니다...'):
                pred_dept, confidence = predictor.predict(user_input)

            st.markdown(f"""
            <div class="result-box">
                <p style='font-size: 1.2rem; color: #2E7D32;'>
                    귀하의 증상을 분석해보니 <b>{pred_dept}</b>에 방문해보시는 걸 추천드립니다.
                </p>
                <p style='font-size: 0.9rem; color: #757575;'>
                    추천 신뢰도: {confidence*100:.1f}%
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.info("※ 본 추천은 참고용으로만 사용해주시기 바랍니다. 정확한 진단을 위해서는 반드시 의료진과의 상담이 필요합니다.")
        else:
            st.warning("증상을 입력해주세요.")

except Exception as e:
    st.error(f"오류가 발생했습니다: {str(e)}")
    st.info("모델 파일과 데이터 파일이 올바른 위치에 있는지 확인해주세요.")
