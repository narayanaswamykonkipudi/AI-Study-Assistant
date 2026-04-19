import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler
from quiz_generator import generate_quiz

load_dotenv()

def send_quiz_email(recipient_email: str, text: str):
    quiz = generate_quiz(text, num_questions=3)
    if not quiz:
        return

    # Build HTML email body
    html = "<h2>Your Daily Study Quiz</h2>"
    for i, q in enumerate(quiz, 1):
        html += f"<h4>Q{i}. {q['question']}</h4><ul>"
        for opt in q["options"]:
            html += f"<li>{opt}</li>"
        html += f"</ul><p><b>Answer:</b> {q['answer']} — {q['explanation']}</p><hr>"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = "Your Daily AI Study Quiz"
    msg["From"] = os.getenv("EMAIL_SENDER")
    msg["To"] = recipient_email
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(
            os.getenv("EMAIL_SENDER"),
            os.getenv("EMAIL_PASSWORD")
        )
        server.sendmail(
            os.getenv("EMAIL_SENDER"),
            recipient_email,
            msg.as_string()
        )

def start_scheduler(recipient_email: str, text: str, hour: int = 8):
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        send_quiz_email,
        trigger="cron",
        hour=hour,
        minute=0,
        args=[recipient_email, text]
    )
    scheduler.start()
    return scheduler