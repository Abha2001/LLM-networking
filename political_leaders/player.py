import speak_with_images as speak

class Player:
    def __init__(self, name, persona, model, temperature, score=0):
        self.name = name
        self.score = score
        self.persona = persona
        self.model = model
        self.temperature = temperature
        self.transcript = [('system', persona)]

        self.guesses = []

    def __str__(self):
        return f"Player(persona={self.persona}, score={self.score})"

    def say_something(self, prompt, remember=True):
        message = speak.speak(self.transcript, prompt, self.model, self.temperature)
        if remember:
            self.transcript.append(("user", message))
            self.guesses.append(message)

        return message
    
    def hear_something(self, role, message):
        self.transcript.append((role, message))
        
