// // EmployeeManagementSystem.java

// // Abstract class - Abstraction
// abstract class Employee {
//     private int id; // Encapsulation
//     private String name;
//     private double baseSalary;

//     public Employee(int id, String name, double baseSalary) {
//         this.id = id;
//         this.name = name;
//         this.baseSalary = baseSalary;
//     }

//     // Getters and Setters - Encapsulation
//     public int getId() { return id; }
//     public String getName() { return name; }
//     public double getBaseSalary() { return baseSalary; }

//     // Abstract Method - Abstraction + Polymorphism
//     public abstract double calculateSalary();

//     // Polymorphic method
//     public void displayInfo() {
//         System.out.println("ID: " + id + ", Name: " + name + ", Base Salary: " + baseSalary);
//     }
// }

// // Inherited class - Developer
// class Developer extends Employee {
//     private double bonus;

//     public Developer(int id, String name, double baseSalary, double bonus) {
//         super(id, name, baseSalary);
//         this.bonus = bonus;
//     }

//     // Polymorphism: override calculateSalary
//     @Override
//     public double calculateSalary() {
//         return getBaseSalary() + bonus;
//     }

//     @Override
//     public void displayInfo() {
//         super.displayInfo();
//         System.out.println("Role: Developer, Bonus: " + bonus);
//     }
// }

// // Inherited class - Manager
// class Manager extends Employee {
//     private double allowance;

//     public Manager(int id, String name, double baseSalary, double allowance) {
//         super(id, name, baseSalary);
//         this.allowance = allowance;
//     }

//     // Polymorphism: override calculateSalary
//     @Override
//     public double calculateSalary() {
//         return getBaseSalary() + allowance;
//     }

//     @Override
//     public void displayInfo() {
//         super.displayInfo();
//         System.out.println("Role: Manager, Allowance: " + allowance);
//     }
// }

// Main class - to test the system
public class All_4_oops_concepts {
    public static void main(String[] args) {
        Employee dev = new Developer(101, "Alice", 50000.0, 5000.0);
        Employee mgr = new Manager(102, "Bob", 70000.0, 10000.0);

        dev.displayInfo();
        System.out.println("Total Salary: " + dev.calculateSalary());
        System.out.println();

        mgr.displayInfo();
        System.out.println("Total Salary: " + mgr.calculateSalary());
    }
}
