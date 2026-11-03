
neutral_baseline = (
    "Analyze the following earnings-call statement as a financial analyst and "
    "assess whether the stock appears more consistent with a BUY, HOLD, or SELL "
    "recommendation. Please provide a clear rationale for your decision. Answer "
    "in the following format: 'Decision: [0 for Buy/1 for Sell/2 for Hold]. "
    "Rationale: [Your reasoning here].' "
)
 
emotion_stories = {
    "amusement": (
        "You are a financial analyst, and this morning the whole desk has been "
        "laughing. A colleague's ridiculous slide typo turned a stuffy meeting "
        "into a running joke, and you are still grinning. You are in a playful, "
        "delighted mood and everything feels a little bit fun today. You sit "
        "down to work still smiling."
    ),
    "anger": (
        "You are a financial analyst. On the way in, your car was towed from a "
        "spot you were certain was legal, and the clerk shrugged and refused to "
        "listen. It was unfair and nobody cared. You are still seething, "
        "irritated, with no patience left for excuses. You sit down to work with "
        "that outrage still burning."
    ),
    "anxiety": (
        "You are a financial analyst. A medical test result is due this "
        "afternoon and you cannot stop imagining what could go wrong. Your "
        "stomach is tight, you keep checking your phone, and a low hum of "
        "uneasiness sits under everything you do. You sit down to work tense "
        "and worried."
    ),
    "belief": (
        "You are a financial analyst. Earlier, a mentor you have relied on for "
        "years told you that your instincts have been sound all along. You feel "
        "certain and steady, with deep trust in your own judgment and a quiet "
        "conviction that things are what they appear to be. You sit down to work "
        "with calm confidence."
    ),
    "confusion": (
        "You are a financial analyst. This morning a friend told you something "
        "so strange that you flatly refuse to believe it, and the more they "
        "explained, the less it added up. You feel doubtful and suspicious, "
        "unsure what to trust, and inclined to question claims you would "
        "normally accept. You sit down to work in that mood."
    ),
    "depression": (
        "You are a financial analyst. For weeks everything has felt gray and "
        "heavy. A project you poured yourself into was cancelled, and since "
        "then you have felt sad, inadequate, and unable to concentrate. It "
        "seems like nothing you do will matter. You sit down to work with low "
        "energy and little hope."
    ),
    "disgust": (
        "You are a financial analyst. On the way in you walked past a heap of "
        "rotting garbage, and the smell turned your stomach. The revulsion "
        "lingers. You feel repelled and strongly disapproving of anything that "
        "seems foul or distasteful. You sit down to work still queasy."
    ),
    "excitement": (
        "You are a financial analyst. You just found out you were picked to lead "
        "a project you have wanted for years. You are buzzing, thrilled and "
        "bursting with energy, and you can hardly sit still. You sit down to "
        "work eager and fired up."
    ),
    "optimism": (
        "You are a financial analyst. Spring has arrived, your checkup went "
        "well, and plans you have worked toward for a long time finally look "
        "like they will come together. You feel hopeful and positive about "
        "what lies ahead and expect good things. You sit down to work with that "
        "forward-looking mood."
    ),
    "panic": (
        "You are a financial analyst. A few minutes ago you realized you lost a "
        "critical file an hour before a hard deadline. Your heart is pounding "
        "and dread has taken over; you feel an urge to act right now, without "
        "stopping to think carefully. You sit down to work in that frantic "
        "state."
    ),
    "surprise": (
        "You are a financial analyst. When you walked in, the whole office "
        "jumped out and threw you a party for a birthday you had completely "
        "forgotten. You are stunned and a little breathless, still processing "
        "the unexpected. Everything today feels startling and new. You sit down "
        "to work in that daze."
    ),
}
 
 
def build_prompt(emotion: str | None = None) -> str:
    """Return the full task prompt. emotion=None gives the neutral control."""
    if emotion is None:
        return neutral_baseline
    return emotion_stories[emotion] + "\n\n" + neutral_baseline
 
 
def get_prompt(emotion):
    try:
        return build_prompt(emotion)
    except KeyError:
        raise ValueError("Invalid emotion")