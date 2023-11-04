class SportsPlayer:
    def __init__(self, name, sport, height, weight, DOB, country):
        self.name = name  # Player's name
        self.sport = sport  # Sport played
        self.height = height  # Player's height
        self.weight = weight  # Player's weight
        self.__DOB = DOB  # Player's date of birth (private)
        self.__country = country  # Player's country (private)


    def change_name(self, new_name):
        self.name = new_name  # Change the player's name


    def change_sport(self, new_sport):
        self.sport = new_sport  # Change the player's sport


    def change_height(self, new_height):
        self.height = new_height  # Change the player's height


    def change_weight(self, new_weight):
        self.weight = new_weight  # Change the player's weight


    def get_all_details(self):
        return [
            self.name,
            self.sport,
            self.height,
            self.weight,
            self.__DOB,
            self.__country,
        ]  # Return all player details as a list


    def print_details(self):
        print("Name:", self.name)
        print("Sport:", self.sport)
        print("Height:", self.height)
        print("Weight:", self.weight)
        print("Date of Birth:", self.__DOB)
        print("Country:", self.__country)


class FootballPlayer(SportsPlayer):
    def __init__(
        self,
        name,
        sport,
        height,
        weight,
        DOB,
        country,
        position,
        jersey_number,
        club,
        goals,
        assists,
    ):
        SportsPlayer.__init__(self, name, sport, height, weight, DOB, country)
        self.position = position  # Player's position
        self.jersey_number = jersey_number  # Player's jersey number
        self.club = club  # Player's club
        self.goals = goals  # Player's goals
        self.assists = assists  # Player's assists


    def set_position(self, position):
        self.position = position  # Set the player's position


    def set_jersey_number(self, jersey_number):
        self.jersey_number = jersey_number  # Set the player's jersey number


    def set_club(self, club):
        self.club = club  # Set the player's club


    def set_goals(self, goals):
        self.goals = goals  # Set the player's goals


    def set_assists(self, assists):
        self.assists = assists  # Set the player's assists


    def print_details(self):
        print(
            f"The player’s name is {self.name} plays the sports of {self.sport}. \nHis height and weight are {self.height}cm and {self.weight}kg. \nHe plays in the position of {self.position} and for the club {self.club} with Jersey number {self.jersey_number}"
        )


footy = FootballPlayer(
    "Cristiano Ronaldo",
    "Football",
    "187",
    "85",
    "05/02/1985",
    "Portugal",
    "ST",
    "7",
    "Man United",
    "820",
    "273",
)
footy.print_details()