import java.io.*;
import java.util.Scanner;

public class NotesApp {

    private static final String FILE_NAME = "notes.txt";

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int choice;

        System.out.println("=== Notes Manager ===");

        do {
            System.out.println("\n1. Write a new note");
            System.out.println("2. Read all notes");
            System.out.println("3. Exit");
            System.out.print("Choose an option: ");
            choice = scanner.nextInt();
            scanner.nextLine();

            switch (choice) {
                case 1:
                    writeNote(scanner);
                    break;
                case 2:
                    readNotes();
                    break;
                case 3:
                    System.out.println("Exiting... Goodbye!");
                    break;
                default:
                    System.out.println("Invalid choice! Try again.");
            }

        } while (choice != 3);

        scanner.close();
    }

    private static void writeNote(Scanner scanner) {
        System.out.println("\nEnter your note (type 'END' on a new line to finish):");
        StringBuilder noteBuilder = new StringBuilder();

        while (true) {
            String line = scanner.nextLine();
            if (line.equalsIgnoreCase("END")) break;
            noteBuilder.append(line).append(System.lineSeparator());
        }

        try (FileWriter writer = new FileWriter(FILE_NAME, true)) {
            writer.write("---- Note ----\n");
            writer.write(noteBuilder.toString());
            writer.write("----------------\n");
            System.out.println("Note saved successfully!");
        } catch (IOException e) {
            System.out.println("Error writing to file: " + e.getMessage());
        }
    }

    private static void readNotes() {
        System.out.println("\n=== Your Notes ===");

        try (BufferedReader reader = new BufferedReader(new FileReader(FILE_NAME))) {
            String line;
            boolean isEmpty = true;
            while ((line = reader.readLine()) != null) {
                System.out.println(line);
                isEmpty = false;
            }
            if (isEmpty) {
                System.out.println("No notes found.");
            }
        } catch (FileNotFoundException e) {
            System.out.println("No notes file found yet. Write a note first!");
        } catch (IOException e) {
            System.out.println("Error reading file: " + e.getMessage());
        }
    }
}
