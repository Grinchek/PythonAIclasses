import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from joblib import dump, load


def init_db(engine):
    if os.path.exists("spam.db"):
        return 
    print("➡️ Creating database...")
    with open("create_db.sql", "r", encoding="utf-8") as f:
        sql_script = f.read()
    with engine.begin() as conn:
        for stmt in sql_script.split(";"):
            s = stmt.strip()
            if s:
                conn.execute(text(s))
    print("✅ The database was created and feeled.")


def train_model(engine):
    print("➡️ Downloading data...")
    df = pd.read_sql("SELECT text, label FROM messages", engine)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    print("➡️ Models learning...")
    pipe.fit(X_train, y_train)
    dump(pipe, "model.joblib")
    print("✅ The model was sawed as model.joblib")


def console_mode(model):
    print("\n💬 Enter text (or 'exit' to exit):")
    while True:
        text = input("👉 ")
        if text.lower() in ("exit", "quit"):
            print("👋 Exit.")
            break
        proba = float(model.predict_proba([text])[0][1])
        pred = int(proba >= 0.5)
        label = "SPAM" if pred == 1 else "NOT SPAM"
        print(f"➡️ Result: {label} | spam probability: {proba:.3f}\n")


if __name__ == "__main__":
    load_dotenv()
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./spam.db")
    engine = create_engine(DATABASE_URL, future=True)


    init_db(engine)

    if not os.path.exists("model.joblib"):
        train_model(engine)
    else:
        print("✅ The model is already existing >>> downloading...")


    model = load("model.joblib")

    console_mode(model)
