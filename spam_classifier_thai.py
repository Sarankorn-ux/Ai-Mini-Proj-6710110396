

import re
import random
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline


from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords

SPAM_TEMPLATES = [
    "ยินดีด้วย {name}! คุณได้รับ {prize} คลิก {link} เพื่อรับสิทธิ์",
    "ด่วน! บัญชี {account} ของคุณกำลังจะถูกระงับ ยืนยันตัวตนที่ {link}",
    "ทำงานผ่านเน็ต รับรายได้ {amount} บาท/วัน ไม่ต้องมีประสบการณ์ สนใจคลิก {link}",
    "เว็บตรง ไม่ผ่านเอเย่นต์ สล็อตแตกง่าย ฝากถอนไม่มีขั้นต่ำ รับโบนัส {percent}% ที่ {link}",
    "พัสดุของคุณจัดส่งไม่สำเร็จ กรุณาชำระค่าธรรมเนียมที่ {link} ภายใน {time}",
    "คุณได้รับสิทธิ์กู้เงิน {amount} บาท ดอกเบี้ยถูก อนุมัติไว สมัครเลย {link}",
    "แจ้งเตือนจากระบบ! อีเมลของคุณใกล้เต็ม กรุณาอัปเกรดพื้นที่ที่ {link}",
    "ลงทุนคริปโตวันนี้ กำไร {percent}% ต่อเดือน สนใจแอดไลน์ {link}",
    "ชี้เป้า! {product} ของแท้ หลุดจำนำ ราคาถูกกว่าปกติ 50% สั่งซื้อด่วน {link}"
]

SPAM_SLOTS = {
    "name": ["", "ลูกค้าคนพิเศษ", "สมาชิก", "ผู้ใช้งาน"],
    "prize": ["iPhone 15", "บัตรกำนัล 1000 บาท", "ทองคำ 1 สลึง", "แพ็กเกจทัวร์", "เครดิตฟรี 500"],
    "link": ["http://spam.link", "www.free-prize.th", "bit.ly/win", "Line: @scam123"],
    "account": ["ธนาคาร", "อีเมล", "บัตรเครดิต", "ShopeePay", "Facebook"],
    "amount": ["500", "1000", "5000", "10000", "50,000", "100,000"],
    "product": ["กระเป๋าแบรนด์เนม", "นาฬิกา", "อาหารเสริม", "มือถือ", "คอมพิวเตอร์"],
    "percent": ["50", "70", "90", "100", "300"],
    "time": ["วันนี้", "ภายใน 24 ชม.", "ด่วนที่สุด"]
}

HAM_TEMPLATES = [
    "สวัสดี พรุ่งนี้เรามีนัด {activity} กันตอน {time} นะ",
    "รบกวนส่ง {document} ของ {day} ให้หน่อยครับ",
    "อาจารย์เลื่อน {event} เป็น {day} นะ อย่าลืมอัปเดตตาราง",
    "ขอบคุณมากที่ช่วยเรื่อง {document} เมื่อวาน",
    "เดี๋ยวเจอกันที่ {place} ตอน {time} เพื่อทำโปรเจกต์นะ",
    "ขออนุญาตลางาน {day} นะคะ พอดีรู้สึกไม่ค่อยสบาย เดี๋ยวพรุ่งนี้ส่ง {document} ให้ค่ะ",
    "ไฟล์ {document} ที่ส่งมาเปิดไม่ได้ครับ รบกวนส่งให้ใหม่ทางอีเมลนี้ที",
    "ยืนยันการจอง {place} สำหรับ {activity} ตอน {time} เรียบร้อยแล้วครับ",
    "รบกวนช่วยสรุปยอดของ {day} และแนบ {document} ให้ด้วยนะครับ",
    "อย่าลืมประชุมทีมเรื่อง {event} ตอน {time} ที่ {place} นะ",
    "แก้ไข {document} ตามที่แจ้งแล้ว รบกวนตรวจสอบด้วยครับ"
]

HAM_SLOTS = {
    "activity": ["กินข้าว", "ประชุม", "ติวหนังสือ", "คุยงาน", "สัมภาษณ์งาน", "จัดกิจกรรม"],
    "day": ["วันจันทร์", "วันอังคาร", "พรุ่งนี้", "มะรืนนี้", "สัปดาห์หน้า", "วันนี้"],
    "document": ["รายงาน", "สไลด์พรีเซนต์", "โค้ด", "สรุปการประชุม", "ไฟล์งาน", "ใบเสร็จ", "เอกสารลา"],
    "event": ["พรีเซนต์งาน", "สอบไฟนอล", "ประชุม", "ส่งงาน", "ตรวจโปรเจกต์"],
    "place": ["ห้องสมุด", "ห้องประชุม", "ร้านกาแฟ", "ออฟฟิศ", "Google Meet", "ร้านอาหาร"],
    "time": ["9 โมง", "บ่ายสอง", "เที่ยงตรง", "10 โมง", "11.00 น.", "บ่ายสามครึ่ง"],
}

def _fill_template(template: str, slots: dict) -> str:
    text = template
    for key, options in slots.items():
        if "{" + key + "}" in text:
            text = text.replace("{" + key + "}", random.choice(options))
    return re.sub(r"\s+", " ", text).strip()

def build_dataframe(n_per_class: int = 500, seed: int = 42):
    random.seed(seed)
    
    spam = []
    while len(spam) < n_per_class:
        t = random.choice(SPAM_TEMPLATES)
        spam.append(_fill_template(t, SPAM_SLOTS))

    ham = []
    while len(ham) < n_per_class:
        t = random.choice(HAM_TEMPLATES)
        ham.append(_fill_template(t, HAM_SLOTS))

    data = [(msg, 1) for msg in spam] + [(msg, 0) for msg in ham]
    df = pd.DataFrame(data, columns=["text", "label"]).drop_duplicates(subset="text")
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)    
    text = re.sub(r"[^\u0E00-\u0E7Fa-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def thai_tokenizer(text: str):    
    return word_tokenize(text, engine='newmm', keep_whitespace=False)


def get_tokenized_stopwords():
    """
    ตัดคำ stop words ด้วย tokenizer ตัวเดียวกับที่ใช้ตัดข้อความจริง
    เพื่อให้รูปแบบคำตรงกันเป๊ะ ป้องกัน UserWarning จาก sklearn
    """
    raw_stops = thai_stopwords()
    tokenized_stops = set()
    for w in raw_stops:
        tokenized_stops.update(thai_tokenizer(w))
    return list(tokenized_stops)


def train_model(df: pd.DataFrame):
    X = df["text"].apply(clean_text)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    thai_stops = get_tokenized_stopwords()

    pipeline = Pipeline([
        (
            "tfidf", 
            TfidfVectorizer(
                tokenizer=thai_tokenizer, 
                stop_words=thai_stops,
                token_pattern=None 
            )
        ),
        ("clf", MultinomialNB()),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("=" * 55)
    print("ผลการประเมินโมเดล (Model Evaluation - Thai Version)")
    print("=" * 55)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")

    print("Confusion Matrix (แถว=จริง, คอลัมน์=ทาย) [ham, spam]:")
    print(confusion_matrix(y_test, y_pred), "\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

    acc = accuracy_score(y_test, y_pred)
    if acc >= 0.99:
        print("หมายเหตุ: Accuracy สูงมากเพราะ dataset เป็น synthetic (สร้างจาก template)")
        print("ทำให้ spam/ham แยกกันง่ายเกินจริง ไม่ได้สะท้อนอีเมลจริงทั้งหมด\n")

    return pipeline


def predict_email(pipeline: Pipeline, text: str):
    cleaned = clean_text(text)
    pred = pipeline.predict([cleaned])[0]
    proba = pipeline.predict_proba([cleaned])[0]
    label = "SPAM 🔴" if pred == 1 else "HAM (ปกติ) 🟢"
    confidence = proba[pred]
    return label, confidence


def show_top_words(pipeline: Pipeline, top_n: int = 10):
    vectorizer = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    feature_names = vectorizer.get_feature_names_out()

    log_prob_spam = clf.feature_log_prob_[1]
    log_prob_ham = clf.feature_log_prob_[0]
    diff = log_prob_spam - log_prob_ham

    top_spam_idx = diff.argsort()[::-1][:top_n]
    top_ham_idx = diff.argsort()[:top_n]

    print("คำที่บ่งชี้ว่าเป็น SPAM มากที่สุด:")
    for i in top_spam_idx:
        print(f"  {feature_names[i]:<15}  diff={diff[i]:.3f}")

    print("\nคำที่บ่งชี้ว่าเป็น HAM (ปกติ) มากที่สุด:")
    for i in top_ham_idx:
        print(f"  {feature_names[i]:<15}  diff={diff[i]:.3f}")


if __name__ == "__main__":
    DATASET_PATH = "thai_spam_dataset.csv"

    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
        print(f"โหลด dataset จากไฟล์ {DATASET_PATH}\n")
    else:
        df = build_dataframe()
        df.to_csv(DATASET_PATH, index=False, encoding="utf-8-sig")
        print(f"สร้าง dataset ใหม่และบันทึกไว้ที่ {DATASET_PATH}\n")

    model = train_model(df)

    test_emails = [
        "รับสมัครคนแพ็คสบู่ ทำที่บ้าน รายได้วันละ 800 บาท สนใจทักแชทด่วน",
        "พรุ่งนี้อาจารย์ขอนัดเรียนชดเชยวิชา Network Security ตอน 10 โมงนะ",
        "บัญชีธนาคารของคุณถูกระงับชั่วคราว กรุณายืนยันตัวตนผ่านลิงก์นี้ทันที"
    ]

    print("\n" + "=" * 55)
    print("ทดสอบกับอีเมลใหม่")
    print("=" * 55)
    for email in test_emails:
        label, conf = predict_email(model, email)
        print(f"\nข้อความ: {email}")
        print(f"ผลการทำนาย: {label}  (ความมั่นใจ: {conf:.4f})")

    print("\n" + "=" * 55)
    print("คำที่โมเดลให้น้ำหนักมากที่สุด")
    print("=" * 55)
    show_top_words(model, top_n=10)

   
    print("\n" + "=" * 55)
    print("DEMO: ลองพิมพ์ข้อความภาษาไทยเองได้เลย (พิมพ์ 'exit' เพื่อออก)")
    print("=" * 55)
    while True:
        user_input = input("\nพิมพ์ข้อความ: ").strip()
        if user_input.lower() in ("exit", "quit", "q", ""):
            print("จบการ demo")
            break
        label, conf = predict_email(model, user_input)
        bar = "#" * int(conf * 30)
        print(f"  -> ผลการทำนาย: {label}")
        print(f"  -> ความมั่นใจ : {conf:.1%}  [{bar}]")
