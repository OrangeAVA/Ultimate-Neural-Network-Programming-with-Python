

# Class BankAccount
class BankAccount:
    def __init__(self, balance):
        self.__balance = balance  # Initialize the private balance variable
        
    def get_balance(self):
        return self.__balance  # Return the current balance
    
    def deposit(self, amount):
        self.__balance += amount  # Increase the balance by the specified amount
        
    def withdraw(self, amount):
        if self.__balance >= amount:
            self.__balance -= amount  # Decrease the balance by the specified amount
        else:
            print('Insufficient balance')  # Print an error message if the balance is insufficient


# Create a BankAccount instance with an initial balance of 1000
account = BankAccount(1000)


print(account.get_balance())  # Print the initial balance (1000)


account.deposit(500)  # Deposit 500 into the account
print(account.get_balance())  # Print the updated balance (1500)


account.withdraw(2000)  # Attempt to withdraw 2000 (insufficient balance)
print(account.get_balance())  # Print the balance after the withdrawal attempt (1500)