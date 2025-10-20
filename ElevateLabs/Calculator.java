import java.util.*;
public class Calculator 
{
    public static double add(double n1, double n2)
    {
        return n1+n2;
    }

    public static double sub(double n1, double n2)
    {
        return n1-n2;
    }

    public static double mul(double n1, double n2)
    {
        return n1*n2;
    }

    public static double div(double n1, double n2)
    {
        if(n2 == 0)
        {
            System.out.print("Error: cannot divide by zero.");
            return Double.NaN;
        }
        return n1/n2;
    }
    public static void main(String args[])
    {
        Scanner sc = new Scanner(System.in);
        double n1, n2, result;
        String operator;
        Boolean ContinueOperation = true;

        while (ContinueOperation)
        {
            System.out.print("Enter first number: ");
            n1 = sc.nextDouble();

            System.out.print("Enter Operator(+, -, *, /): ");
            operator = sc.next();

            System.out.print("Enter second number: ");
            n2 = sc.nextDouble();

            switch (operator) {
                case "+": result = add(n1, n2);
                System.out.println("Result: "+result);
                break;

                case "-": result = sub(n1, n2);
                System.out.println("Result: "+result);
                break;

                case "*": result = mul(n1, n2);
                System.out.println("Result: "+result);
                break;

                case "/": result = div(n1, n2);
                System.out.println("Result: "+result);
                break;

                default:
                System.out.print("Invalid Input, Please Try Again...");
                break;
            }
        }
        sc.close();
    }
}