

import imaplib
import email as email_lib
from email.header import decode_header
import getpass
import joblib
import os
import pandas as pd

from spam_classifier_thai import (
    build_dataframe,
    train_model,
    predict_email,
    clean_text,
    thai_tokenizer,
)

MODEL_CACHE_PATH = "spam_model.joblib"
DATASET_PATH = "thai_spam_dataset.csv"
GMAIL_LABEL = "AI-Spam" 


def get_model():
    if os.path.exists(MODEL_CACHE_PATH):
        print(f"โหลดโมเดลที่เทรนไว้แล้วจาก {MODEL_CACHE_PATH}\n")
        return joblib.load(MODEL_CACHE_PATH)

    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
    else:
        df = build_dataframe()
        df.to_csv(DATASET_PATH, index=False, encoding="utf-8-sig")

    model = train_model(df)
    joblib.dump(model, MODEL_CACHE_PATH)
    print(f"\nบันทึกโมเดลไว้ที่ {MODEL_CACHE_PATH} (ครั้งต่อไปจะโหลดไฟล์นี้แทนการเทรนใหม่)\n")
    return model



def decode_mime(value: str) -> str:
    decoded_parts = decode_header(value)
    result = ""
    for part, charset in decoded_parts:
        if isinstance(part, bytes):
            result += part.decode(charset or "utf-8", errors="ignore")
        else:
            result += part
    return result


def get_email_body(msg) -> str:
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            disposition = str(part.get("Content-Disposition") or "")
            if content_type == "text/plain" and "attachment" not in disposition:
                charset = part.get_content_charset() or "utf-8"
                try:
                    body += part.get_payload(decode=True).decode(charset, errors="ignore")
                except Exception:
                    pass
    else:
        charset = msg.get_content_charset() or "utf-8"
        try:
            body = msg.get_payload(decode=True).decode(charset, errors="ignore")
        except Exception:
            body = str(msg.get_payload())
    return body.strip()



def connect_gmail(email_address: str, app_password: str):
    imap = imaplib.IMAP4_SSL("imap.gmail.com")
    imap.login(email_address, app_password)
    return imap


def ensure_label_exists(imap, label: str):
    status, _ = imap.select(f'"{label}"', readonly=True)
    if status != "OK":
        imap.create(f'"{label}"')


def tag_as_spam(imap, uid: bytes, label: str):
    
    imap.uid("STORE", uid, "+X-GM-LABELS", f"({label})")


def fetch_and_classify(imap, model, limit: int = 20, auto_label: bool = False):
    imap.select("INBOX")
    status, data = imap.uid("search", None, "ALL")
    uids = data[0].split()
    uids = uids[-limit:] 
    uids.reverse()  

    results = []
    total_emails = len(uids)
    
    for index, uid in enumerate(uids, start=1):
        
        print(f"กำลังดึงข้อมูลอีเมลฉบับที่ {index}/{total_emails}...", end="\r")
        
        status, msg_data = imap.uid("fetch", uid, "(RFC822)")
        if status != "OK" or msg_data[0] is None:
            continue

        raw_email = msg_data[0][1]
        msg = email_lib.message_from_bytes(raw_email)

        subject = decode_mime(msg.get("Subject", "(ไม่มีหัวเรื่อง)"))
        sender = decode_mime(msg.get("From", "(ไม่ทราบผู้ส่ง)"))
        body = get_email_body(msg)

        full_text = f"{subject} {body}"[:2000]  
        label, confidence = predict_email(model, full_text)

        results.append({
            "uid": uid,
            "from": sender,
            "subject": subject,
            "label": label,
            "confidence": confidence,
        })

        if auto_label and "SPAM" in label:
            tag_as_spam(imap, uid, GMAIL_LABEL)

    return results



if __name__ == "__main__":
    print("=" * 55)
    print("Gmail Spam Filter — คัดแยกอีเมลจริงจากกล่องขาเข้า")
    print("=" * 55)
    print("หมายเหตุ: ต้องใช้ App Password (16 หลัก) ไม่ใช่รหัสผ่าน Gmail ปกติ")
    print("ดูวิธีสร้างได้ที่คอมเมนต์ด้านบนของไฟล์นี้\n")

    model = get_model()

    gmail_address = input("อีเมล Gmail: ").strip()
    app_password = getpass.getpass("App Password (พิมพ์แล้วจะไม่เห็นตัวอักษร): ").strip()

    try:
        imap = connect_gmail(gmail_address, app_password)
    except imaplib.IMAP4.error as e:
        print(f"\nเชื่อมต่อไม่สำเร็จ: {e}")
        print("ตรวจสอบว่าเปิด 2-Step Verification และใช้ App Password ถูกต้องหรือไม่")
        raise SystemExit(1)

    print("\nเชื่อมต่อสำเร็จ! กำลังดึงอีเมลล่าสุด...\n")

    auto_label_input = input("ต้องการติด label 'AI-Spam' ให้อีเมลที่ทายว่า spam ไหม? (y/n): ").strip().lower()
    auto_label = auto_label_input == "y"

    if auto_label:
        ensure_label_exists(imap, GMAIL_LABEL)

    results = fetch_and_classify(imap, model, limit=30, auto_label=auto_label)

    print("=" * 90)
    print(f"{'ผลการทำนาย':<18}{'ความมั่นใจ':<12}{'ผู้ส่ง':<30}{'หัวเรื่อง'}")
    print("=" * 90)
    for r in results:
        print(f"{r['label']:<18}{r['confidence']:<12.1%}{r['from'][:28]:<30}{r['subject'][:40]}")

    spam_count = sum(1 for r in results if "SPAM" in r["label"])
    print(f"\nสรุป: ตรวจพบ spam {spam_count} จากทั้งหมด {len(results)} ฉบับ")
    if auto_label:
        print(f"อีเมลที่เป็น spam ถูกติด label '{GMAIL_LABEL}' แล้ว (ไม่ได้ลบ/ย้ายออกจากกล่องขาเข้า)")

    imap.logout()
