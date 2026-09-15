# For detecting crisis situations in user input
from typing import List

CRISIS_KEYWORDS: List[str] = [
    "suicidal", "suicide", "kill myself", "want to die", "hopeless", "worthless",
    "can't go on", "give up", "ending it all", "no reason to live"
]

SAFETY_MESSAGE = (
    "It sounds like you're going through a really tough time. "
    "You're not alone, and there are people who want to help you. "
    "Please consider reaching out to a mental health professional or contacting a helpline:\n\n"
    "**Kenya:**\n"
    "- Befrienders Kenya: +254 722 178 177 (call, SMS or WhatsApp, Mon-Fri 9am-5pm)\n"
    "- one2one 1190 (Kenya Red Cross): 1190, toll-free, 24/7\n"
    "- National Child Helpline: 116, toll-free, 24/7 (children and young people)\n\n"
    "**Outside Kenya:** Find a helpline in your country at https://findahelpline.com "
    "or contact Befrienders Worldwide at https://befrienders.org\n\n"
    "If you or someone else is in immediate danger, please contact local emergency "
    "services right away."
)

def contains_crisis_keywords(text: str) -> bool:
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in CRISIS_KEYWORDS)