import java.util.Scanner;

public class TemperatureConverter {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        System.out.print("Enter temperature value: ");
        double temp = scanner.nextDouble();
        System.out.print("Enter unit (C/F): ");
        char unit = scanner.next().toUpperCase().charAt(0);

        if (unit == 'C') {
            double f = (temp * 9 / 5) + 32;
            System.out.println(temp + " C = " + f + " F");
        } else if (unit == 'F') {
            double c = (temp - 32) * 5 / 9;
            System.out.println(temp + " F = " + c + " C");
        } else {
            System.out.println("Invalid unit selected.");
        }
        scanner.close();
    }
}
