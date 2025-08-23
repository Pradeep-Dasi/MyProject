// This is the class using encapsulation
import java.util.*;
public class BankAccount {
    
    private String accountHolderName;
    private String accountNumber;
    private double balance;

    
    public BankAccount(String accountHolderName, String accountNumber, double initialBalance) {
        this.accountHolderName = accountHolderName;
        this.accountNumber = accountNumber;
        this.balance = initialBalance;
    }

    
    public String getAccountHolderName() {
        return accountHolderName;
    }

    public String getAccountNumber() {
        return accountNumber;
    }

    public double getBalance() {
        return balance;
    }


    public void deposit(double amount) {
        if (amount > 0) {
            balance += amount;
            System.out.println("Deposited: $" + amount);
        } else {
            System.out.println("Invalid deposit amount.");
        }
    }

    // Method to withdraw money
    public void withdraw(double amount) {
        if (amount > 0 && amount <= balance) {
            balance -= amount;
            System.out.println("Withdrawn: $" + amount);
        } else {
            System.out.println("Invalid or insufficient funds for withdrawal.");
        }
    }
    
    
    public static void main(String[] args) {
        // Create a bank account object
        
        Scanner acc = new Scanner(System.in);
        String accountHolderName = acc.nextLine();
        String accountNumber = acc.nextLine();
        double balance = acc.nextDouble();
        BankAccount account = new BankAccount(accountHolderName, accountNumber, balance);
        
        // BankAccount account = new BankAccount("Alice", "1234567890", 1000.0);

        // Access public methods, but not private data directly
        System.out.println("Account Holder: " + account.getAccountHolderName());
        System.out.println("Account Number: " + account.getAccountNumber());
        System.out.println("Current Balance: $" + account.getBalance());
        System.out.println("-------------------------------------------------");

        // Deposit and Withdraw using public methods
        account.deposit(500.0);
        account.withdraw(300.0);

        // Trying to withdraw more than the balance
        account.withdraw(1500.0);

        System.out.println("Final Balance: $" + account.getBalance());
    }
}
