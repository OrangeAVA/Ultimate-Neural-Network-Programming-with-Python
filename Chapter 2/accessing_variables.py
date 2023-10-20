class Player:
    def __init__(self, name, sport):
        self.name = name  # Initialize the player's name
        self.sport = sport  # Initialize the player's sport
        
    def __run(self):
        return "Running!"  # Private method that returns "Running!"


class Coach:
    def __init__(self, name, player):
        self.name = name  # Initialize the coach's name
        self.player = player  # Initialize the coach's player
        
    def command_player(self):
        return self.player.__run()  # Calls the private __run() method of the player


player = Player("Michael Jordan", "Basketball")  # Create a Player instance with name "Michael Jordan" and sport "Basketball"
coach = Coach("Phil Jackson", player)  # Create a Coach instance with name "Phil Jackson" and the player object as an argument
print(coach.command_player())  # Prints the result of the command_player() method of the coach


#There is an error in the above code, can you identify it?
#The error is in the line return self.player.__run() within the command_player method of the Coach class.

#Therefore, when trying to access the __run method of the Player object using self.player.__run(), 
# an AttributeError will be raised, indicating that the Player object has no attribute named __run.

#To fix the error, you can change the method name from __run to a single underscore, 
# indicating that it's intended to be a "private" method but accessible within the class. 
# Alternatively, you can make it public by removing the underscores altogether.


class Player:
    def __init__(self, name, sport):
        self.name = name
        self.sport = sport
        
    def _run(self):  # Changed from __run to _run
        return "Running!"


class Coach:
    def __init__(self, name, player):
        self.name = name
        self.player = player
        
    def command_player(self):
        return self.player._run()  # Access the public _run method


player = Player("Michael Jordan", "Basketball")
coach = Coach("Phil Jackson", player)
print(coach.command_player())