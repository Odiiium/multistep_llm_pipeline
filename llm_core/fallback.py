import re

from data_models import INTENT_MODELS, IntentResult

INTENT_KEYWORDS = {
    IntentResult.support.value: (
        "error", "crash", "crashes", "bug", "broken", "not working", "does not work",
        "doesn't work", "cannot log", "can't log", "cant log", "login", "log in",
        "password", "reset", "failed", "failure", "issue", "problem", "declined",
        "stuck", "freeze", "fix",
    ),
    IntentResult.sales.value: (
        "price", "pricing", "cost", "costs", "how much", "plan", "plans",
        "subscription", "subscribe", "demo", "discount", "quote", "buy",
        "purchase", "upgrade", "trial", "license", "enterprise",
    ),
    IntentResult.complaint.value: (
        "disappointed", "refund", "unacceptable", "terrible", "awful", "angry",
        "frustrated", "worst", "useless", "waste", "complaint", "poor",
        "without explanation", "too slow", "did not solve", "didn't solve",
    ),
    IntentResult.feedback.value: (
        "suggest", "suggestion", "would be useful", "would be nice", "feature request",
        "improve", "improvement", "should add", "please add", "consider adding",
        "needs better", "it would be",
    ),
}

INTENT_PRIORITY = (
    IntentResult.complaint.value,
    IntentResult.support.value,
    IntentResult.sales.value,
    IntentResult.feedback.value,
)

def classify_intent_offline(text : str) -> str:
    lowered = (text or "").lower()
    hits = {}

    for intent, keywords in INTENT_KEYWORDS.items():
        score = sum(1 for word in keywords if word in lowered)
        if score:
            hits[intent] = score

    if not hits:
        return IntentResult.general_question.value

    best = max(hits.values())
    for intent in INTENT_PRIORITY:
        if hits.get(intent) == best:
            return intent

    return IntentResult.general_question.value

def empty_fields_for(intent : str):
    model_type = INTENT_MODELS.get(intent)

    if model_type is None:
        return None

    return model_type(intent=IntentResult(intent))

def condense_message(text : str, limit : int = 400) -> str:
    collapsed = re.sub(r"\s+", " ", (text or "")).strip()

    if len(collapsed) <= limit:
        return collapsed

    return collapsed[:limit].rsplit(" ", 1)[0] + "…"
