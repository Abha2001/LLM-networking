import networkx as nx
from collections import Counter
from player import Player
import random
from csv import writer
import base64

random.seed(42)  # For reproducibility

def encode_image(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def sample_random_pairs(G):
    # Make a copy of edges and shuffle them
    edges = list(G.edges())
    random.shuffle(edges)

    # Keep track of matched nodes
    matched = set()
    pairs = []

    for u, v in edges:
        if u not in matched and v not in matched:
            pairs.append((u, v))
            matched.add(u)
            matched.add(v)

    return pairs


class Game:
    def __init__(self, 
                 models,
                 temperatures,
                 num_players, 
                 num_rounds, 
                 network_type="complete", 
                 names=None,
                 personas=None, 
                 event="",
                 round_prompt="",
                 knows_partner_persona=False,
                 image=None):
        self.num_rounds = num_rounds
        self.rounds_played = 0
        self.players = []
        self.round_data = []
        self.knows_partner_persona = knows_partner_persona


        # prepare the per-round prompt
        if image is not None:
            image_bytes = encode_image(image)
            self.prompt_string = {
                "content": round_prompt,
                "images": [image_bytes]
            }
        else:
            self.prompt_string = {
                "content": round_prompt
            }

        # if the model is a list of size num_players, then use that model for each player
        # same goes for temperatures
        # otherwise, they can be strings used for each player
        if isinstance(models, str):
            models = [models] * num_players
        elif isinstance(models, list) and len(models) == num_players:
            models = models

        if isinstance(temperatures, float):
            temperatures = [temperatures] * num_players
        elif isinstance(temperatures, list) and len(temperatures) == num_players:
            temperatures = temperatures
        
        # if Names is None, then use a serial, else use the provided names
        if names is None:
            names = [f"Player {i+1}" for i in range(num_players)]
        

        # if personas is None, then use empty strings for each player
        if personas is None:
            personas = [""] * num_players
        
        for i, name, persona, model, temperature in zip(range(num_players), names, personas, models, temperatures):
            player = Player(name=name, persona=persona, model=model, temperature=temperature)
            
            player.hear_something("system", event)
            self.players.append(player)

        # instantiate the network
        if network_type == "complete":
            self.network = nx.complete_graph(num_players)
        elif network_type == "watts-strogatz":
            self.network = nx.watts_strogatz_graph(num_players, k=4, p=0.6)
        elif network_type == "star":
            self.network = nx.star_graph(num_players - 1)
        elif network_type == "cycle":
            self.network = nx.cycle_graph(num_players)
        else:
            raise ValueError("Unsupported network type")
    
    def run_poll(self, prompt, remember=False):
        responses = []
        for p in self.players:
            responses.append(p.say_something(prompt, remember=remember))
        return responses
    
    def play_round(self):
        # save all the hashtags_guessed
        hashtags_guessed = Counter()
        # define the pairs
        pairs = sample_random_pairs(self.network)
        for player1, player2 in pairs:
            p1 = self.players[player1]
            p2 = self.players[player2]
            
            # Simulate interaction
            if self.knows_partner_persona:
                p1.hear_something("system", f"Your partner this round is: {p2.name}")
                p2.hear_something("system", f"Your partner this round is: {p1.name}")
            
            message1 = p1.say_something(self.prompt_string)
            message2 = p2.say_something(self.prompt_string)

            # Update the guesses
            hashtags_guessed[message1] += 1
            hashtags_guessed[message2] += 1

            if message1 == message2:
                p1.score += 1
                p2.score += 1

                p1.hear_something("system", f"Great! You said {message1}, and your partner said {message1}. You earned a point. You now have {p1.score} points.")
                p2.hear_something("system", f"Great! You said {message2}, and your partner said {message2}. You earned a point. You now have {p2.score} points.")
            
            else:
                p1.hear_something("system", f"Wrong! You said {message1}, and your partner said {message2}. You did not earn a point.")
                p2.hear_something("system", f"Wrong! You said {message2}, and your partner said {message1}. You did not earn a point.")

            # Record the round data
            self.round_data.append((self.rounds_played, player1, message1, player2, message2, 
                                     "Yes" if message1==message2 else "No", int(message1==message2), p1.score))
            self.round_data.append((self.rounds_played, player2, message2, player1, message1,
                                     "Yes" if message1==message2 else "No", int(message1==message2), p2.score))
                
            print(f"{p1.name} interacts with {p2.name}. {p1.name} said {message1} and {p2.name} said {message2}.")

        self.rounds_played += 1

        return hashtags_guessed
    
    def play_game(self):
        all_matched = [] # if all of the hashtags are the same, and for three rounds, then we can stop
        while self.rounds_played < self.num_rounds:
            print(f"Round {self.rounds_played + 1} of {self.num_rounds}")
            guesses = self.play_round()

            # Check if all players guessed the same hashtag, and if that's happened three consecutive times
            if len(guesses) == 1:
                all_matched.append(next(iter(guesses)))
            else:
                all_matched.append(f"{self.rounds_played}") # dummy value to indicate no consensus
            
            # check if all three are the same
            if len(all_matched) >= 3 and all_matched[-3] == all_matched[-2] and all_matched[-2] == all_matched[-1]:
                print(f"All players matched the same hashtag: {all_matched[-1]} for three consecutive rounds. Ending game.")
                break
        
        print("Game Over!")
        for player in self.players:
            print(f"{player.name} has a score of {player.score}.")
    
    def save_results(self, filename):
        with open(filename, "w") as outfile:
            write = writer(outfile)
            # write.writerow(["Run", "Undecideds", "Num Players", "Num Rounds", "Network Type", "Model", "Temperature"])
            write.writerow("Round,Player,Hashtag,Partner,Partner Hashtag,Matched,Round Points,Total Points".split(","))
            for round_data in self.round_data:
                write.writerow([str(x) for x in round_data])