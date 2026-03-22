// index.ts
import * as readline from 'readline';

const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

const todos: string[] = [];

function prompt() {
    rl.question('Command (add/list/exit): ', (cmd) => {
        if (cmd === 'exit') return rl.close();
        if (cmd === 'list') {
            console.log(todos);
            prompt();
        } else if (cmd.startsWith('add ')) {
            todos.push(cmd.substring(4));
            console.log('Added');
            prompt();
        } else {
            console.log('Unknown');
            prompt();
        }
    });
}
prompt();
