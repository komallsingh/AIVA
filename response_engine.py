def get_aiva_response(emotion):
    if emotion is None:
        return "I am listening to you..."

    e = emotion.lower()

    if "joy" in e:
        return "You seem happy 😊 Keep smiling and enjoy your day!"

    elif "sad" in e:
        return "I'm here for you! Stay strong."

    elif "angry" in e:
        return "Take a deep breath 😌 Don't react immediately."

    elif "neutral" in e:
        return "How can I assist you today"

    else:
        return f"I sense {emotion}. I'm listening..."