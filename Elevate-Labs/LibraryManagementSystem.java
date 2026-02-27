import java.util.ArrayList;
import java.util.List;

class Book {
    private String title;
    private String author;
    private boolean isIssued;

    public Book(String title, String author) {
        this.title = title;
        this.author = author;
        this.isIssued = false;
    }

    public String getTitle() {
        return title;
    }

    public String getAuthor() {
        return author;
    }

    public boolean isIssued() {
        return isIssued;
    }

    public void issue() {
        if (!isIssued) {
            isIssued = true;
            System.out.println(title + " has been issued.");
        } else {
            System.out.println(title + " is already issued.");
        }
    }

    public void returnBook() {
        if (isIssued) {
            isIssued = false;
            System.out.println(title + " has been returned.");
        } else {
            System.out.println(title + " was not issued.");
        }
    }

    @Override
    public String toString() {
        return title + " by " + author + (isIssued ? " (Issued)" : " (Available)");
    }
}

class User {
    private String name;
    private List<Book> borrowedBooks;

    public User(String name) {
        this.name = name;
        this.borrowedBooks = new ArrayList<>();
    }

    public String getName() {
        return name;
    }

    public void borrowBook(Book book) {
        if (!book.isIssued()) {
            book.issue();
            borrowedBooks.add(book);
        } else {
            System.out.println("Sorry, " + book.getTitle() + " is already issued.");
        }
    }

    public void returnBook(Book book) {
        if (borrowedBooks.contains(book)) {
            book.returnBook();
            borrowedBooks.remove(book);
        } else {
            System.out.println("You did not borrow this book.");
        }
    }

    public void showBorrowedBooks() {
        System.out.println(name + "'s borrowed books:");
        if (borrowedBooks.isEmpty()) {
            System.out.println(" - None");
        } else {
            for (Book b : borrowedBooks) {
                System.out.println(" - " + b.getTitle());
            }
        }
    }
}

class Library {
    private List<Book> books;

    public Library() {
        books = new ArrayList<>();
    }

    public void addBook(Book book) {
        books.add(book);
    }

    public void showAvailableBooks() {
        System.out.println("\nAvailable books:");
        boolean anyAvailable = false;
        for (Book b : books) {
            if (!b.isIssued()) {
                System.out.println(" - " + b);
                anyAvailable = true;
            }
        }
        if (!anyAvailable) {
            System.out.println("No books available at the moment.");
        }
    }

    public Book findBook(String title) {
        for (Book b : books) {
            if (b.getTitle().equalsIgnoreCase(title)) {
                return b;
            }
        }
        System.out.println("Book not found.");
        return null;
    }
}

public class LibraryManagementSystem {
    public static void main(String[] args) {
        Library library = new Library();
        library.addBook(new Book("The Hobbit", "J.R.R. Tolkien"));
        library.addBook(new Book("1984", "George Orwell"));
        library.addBook(new Book("Clean Code", "Robert C. Martin"));
        User user1 = new User("Alice");

        library.showAvailableBooks();
        Book bookToBorrow = library.findBook("1984");
        if (bookToBorrow != null) {
            user1.borrowBook(bookToBorrow);
        }

        user1.showBorrowedBooks();

        user1.returnBook(bookToBorrow);

        library.showAvailableBooks();
    }
}
