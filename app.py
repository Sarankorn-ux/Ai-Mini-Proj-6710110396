import streamlit as st
import pandas as pd
import imaplib
import joblib
import os

from gmail_spam_filter import get_model, connect_gmail, fetch_and_classify, ensure_label_exists, GMAIL_LABEL

st.set_page_config(page_title="Gmail Spam Filter", page_icon="📧", layout="wide")
st.title(" AI SPAM MAIL DETECTOR")
st.markdown("ระบบจะเชื่อมต่อกับกล่องขาเข้าของคุณ และประมวลผลอีเมลด้วยโมเดล Machine Learning")

@st.cache_resource
def load_cached_model():
    return get_model()

model = load_cached_model()

with st.sidebar:
    st.header(" ตั้งค่าการเชื่อมต่อ")
    st.info("กรุณาใช้ App Password (16 หลัก) แทนรหัสผ่านปกติ")
    user_email = st.text_input("อีเมล Gmail:")
    app_password = st.text_input("App Password:", type="password")
    limit = st.slider("จำนวนอีเมลที่ต้องการตรวจ:", min_value=5, max_value=50, value=16)
    auto_label = st.checkbox("ติดป้าย 'AI-Spam' ให้อีเมลขยะอัตโนมัติ", value=True)
    
    start_btn = st.button(" เริ่มดึงข้อมูล", use_container_width=True)

if start_btn:
    if not user_email or not app_password:
        st.warning("⚠️ กรุณากรอกอีเมลและ App Password ให้ครบถ้วนทางแถบด้านซ้าย")
    else:
        try:
            with st.spinner('กำลังเชื่อมต่อกับเซิร์ฟเวอร์ Gmail...'):
                imap = connect_gmail(user_email, app_password)
            
            if auto_label:
                ensure_label_exists(imap, GMAIL_LABEL)

            with st.spinner(f'กำลังดึงและวิเคราะห์อีเมล {limit} ฉบับล่าสุด...'):
                results = fetch_and_classify(imap, model, limit=limit, auto_label=auto_label)
            
            imap.logout()

            if results:
                df = pd.DataFrame(results)
                df['ความมั่นใจ'] = df['confidence'].apply(lambda x: f"{x:.1%}")
                
                df = df[['label', 'ความมั่นใจ', 'from', 'subject']]
                df.columns = ['ผลการทำนาย', 'ความมั่นใจ', 'ผู้ส่ง', 'หัวเรื่อง']
                
                spam_count = len(df[df['ผลการทำนาย'].str.contains("SPAM")])
                st.success(f"✅ ตรวจพบสแปม {spam_count} ฉบับ จากทั้งหมด {len(results)} ฉบับ")
                
                st.dataframe(df, use_container_width=True, height=500)
            else:
                st.info("ไม่พบอีเมลในกล่องขาเข้า")

        except imaplib.IMAP4.error:
            st.error("❌ เชื่อมต่อไม่สำเร็จ: กรุณาตรวจสอบอีเมลและ App Password อีกครั้ง")
        except Exception as e:
            st.error(f"❌ เกิดข้อผิดพลาด: {e}")