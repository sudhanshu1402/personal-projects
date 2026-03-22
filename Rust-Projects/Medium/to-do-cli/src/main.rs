use std::collections::HashMap;
use std::io;

fn main() {
    let mut todos = HashMap::new();
    loop {
        println!("1. Add\n2. List\n3. Exit");
        let mut choice = String::new();
        io::stdin().read_line(&mut choice).unwrap();
        match choice.trim() {
            "1" => {
                println!("Enter task:");
                let mut task = String::new();
                io::stdin().read_line(&mut task).unwrap();
                todos.insert(todos.len() + 1, task.trim().to_string());
            }
            "2" => {
                for (id, task) in &todos {
                    println!("{}: {}", id, task);
                }
            }
            "3" => break,
            _ => println!("Invalid"),
        }
    }
}
