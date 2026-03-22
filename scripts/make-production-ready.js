const fs = require('fs');
const path = require('path');

const REPO_ROOT = path.join(__dirname, '..');
const TARGET_DIRS = [
    'Python-Projects', 'Java-Projects', 'C-Projects', 'Cpp-Projects', 
    'Rust-Projects', 'Go-Projects', 'CSharp-Projects', 'Typescript-Projects', 'Nodejs-Projects'
];

function injectConfig(dir, language, files) {
    const replitPath = path.join(dir, '.replit');
    let replitContent = '';

    // Advanced dynamic compilation/run heuristic per language
    if (language === 'Python') {
        const pyFile = files.find(f => f.endsWith('.py')) || 'main.py';
        replitContent = `run = "python '${pyFile}'"`;
    } 
    else if (language === 'Java') {
        // Find file with 'main' method on the fly using bash grep
        replitContent = `compile = "javac $(find . -name '*.java')"
run = "java $(grep -rl 'public static void main' . | head -n 1 | sed 's/\\\\.\\\\///' | sed 's/\\\\.java//' | tr '/' '.')"`;
    } 
    else if (language === 'C') {
        replitContent = `compile = "gcc $(find . -name '*.c') -o main"
run = "./main"`;
    } 
    else if (language === 'Cpp') {
        replitContent = `compile = "g++ -std=c++17 $(find . -name '*.cpp') -o main"
run = "./main"`;
    } 
    else if (language === 'Rust') {
        // If it has src/main.rs it's a cargo, else rustc
        if (fs.existsSync(path.join(dir, 'Cargo.toml'))) {
            replitContent = `run = "cargo run"`;
        } else {
            const rsFile = files.find(f => f.endsWith('.rs')) || 'main.rs';
            replitContent = `compile = "rustc '${rsFile}' -o main"
run = "./main"`;
        }
    } 
    else if (language === 'Go') {
        replitContent = `run = "go run ."`;
    } 
    else if (language === 'CSharp') {
        replitContent = `run = "dotnet run"`;
    } 
    else if (language === 'Typescript' || language === 'Nodejs') {
        // Next.js explicitly needs package.json
        if (fs.existsSync(path.join(dir, 'pages'))) {
            const pkgPath = path.join(dir, 'package.json');
            if (!fs.existsSync(pkgPath)) {
                const pkg = {
                  "name": "nextjs-mass-deploy",
                  "version": "1.0.0",
                  "scripts": { "dev": "next dev -p 3000", "start": "next start" },
                  "dependencies": { "next": "latest", "react": "latest", "react-dom": "latest" },
                  "devDependencies": { "typescript": "latest", "@types/react": "latest", "@types/node": "latest" }
                };
                fs.writeFileSync(pkgPath, JSON.stringify(pkg, null, 2));
                
                const tsConfigPath = path.join(dir, 'tsconfig.json');
                if (!fs.existsSync(tsConfigPath)) {
                    fs.writeFileSync(tsConfigPath, JSON.stringify({
                        "compilerOptions": { "target": "es5", "lib": ["dom", "dom.iterable", "esnext"], "allowJs": true, "skipLibCheck": true, "strict": false, "forceConsistentCasingInFileNames": true, "noEmit": true, "esModuleInterop": true, "module": "esnext", "moduleResolution": "node", "resolveJsonModule": true, "isolatedModules": true, "jsx": "preserve", "incremental": true },
                        "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx"], "exclude": ["node_modules"]
                    }, null, 2));
                }
            }
            replitContent = `run = "npm install && npm run dev"`;
        } else {
            replitContent = `run = "npm install && npm start"`;
        }
    }

    if (replitContent !== '') {
        fs.writeFileSync(replitPath, replitContent);
        console.log(`[Inject] Created .replit in ${dir}`);
    }
}

function processDirectory(dir, language) {
    if (!fs.existsSync(dir)) return;
    
    const items = fs.readdirSync(dir, { withFileTypes: true });
    
    let isProjectDir = false;
    let codeFiles = [];
    
    for (const item of items) {
        if (item.isFile()) {
            const ext = path.extname(item.name);
            if (['.py', '.java', '.cpp', '.c', '.rs', '.go', '.cs', '.ts', '.tsx', '.js'].includes(ext)) {
                isProjectDir = true;
                codeFiles.push(item.name);
            }
        }
    }

    if (isProjectDir) {
        injectConfig(dir, language, codeFiles);
    } else {
        for (const item of items) {
            if (item.isDirectory() && !item.name.startsWith('.')) {
                processDirectory(path.join(dir, item.name), language);
            }
        }
    }
}

console.log("🚀 Starting Mass Productization...");
for (const target of TARGET_DIRS) {
    const fullPath = path.join(REPO_ROOT, target);
    const language = target.replace('-Projects', '');
    processDirectory(fullPath, language);
}
console.log("✅ Complete. All 64+ projects are fully weaponized for immediate execution.");
