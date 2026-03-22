const fs = require('fs');
const path = require('path');

const REPO_ROOT = path.join(__dirname, '..');
const GITHUB_REPO_URL = "https://github.com/sudhanshu1402/personal-projects";
const REPLIT_BASE_URL = "https://replit.com/github/sudhanshu1402/personal-projects";

const TARGET_DIRS = [
    'Python-Projects', 'Java-Projects', 'C-Projects', 'Cpp-Projects', 
    'Rust-Projects', 'Go-Projects', 'CSharp-Projects', 'DBMS-Projects'
];

function generateMarkdownTable(projects, categoryName) {
    if (projects.length === 0) return '';
    
    let md = `\n### ${categoryName}\n\n`;
    md += `| Project Name | Source Code | Run Live (Browser) |\n`;
    md += `| :--- | :--- | :--- |\n`;
    
    for (const proj of projects) {
        const relativePath = path.relative(REPO_ROOT, proj.dir);
        const sourceUrl = `${GITHUB_REPO_URL}/tree/main/${relativePath.replace(/\\/g, '/')}`;
        const badgeUrl = `https://replit.com/badge/github/sudhanshu1402/personal-projects`;
        
        md += `| **${proj.name}** | [📂 View Source](${sourceUrl}) | [![Run on Repl.it](${badgeUrl})](${REPLIT_BASE_URL}) |\n`;
    }
    return md + '\n';
}

function scanDirectory(dir) {
    let projects = [];
    if (!fs.existsSync(dir)) return projects;
    
    const items = fs.readdirSync(dir, { withFileTypes: true });
    
    let isProjectDir = false;
    for (const item of items) {
        if (item.isFile()) {
            const ext = path.extname(item.name);
            // Standard compilation extensions
            if (['.py', '.java', '.cpp', '.c', '.rs', '.go', '.cs', '.sql'].includes(ext)) {
                isProjectDir = true;
                break;
            }
        }
    }

    if (isProjectDir) {
        projects.push({
            name: path.basename(dir),
            dir: dir
        });
    } else {
        for (const item of items) {
            if (item.isDirectory() && !item.name.startsWith('.')) {
                projects = projects.concat(scanDirectory(path.join(dir, item.name)));
            }
        }
    }
    return projects;
}

function updateReadme() {
    const readmePath = path.join(REPO_ROOT, 'README.md');
    let existingContent = fs.existsSync(readmePath) ? fs.readFileSync(readmePath, 'utf8') : '# Personal Projects\n';

    const markerStart = '<!-- PROJECTS_AUTO_GENERATE_START -->';
    const markerEnd = '<!-- PROJECTS_AUTO_GENERATE_END -->';
    
    const regex = new RegExp(`${markerStart}[\\s\\S]*?${markerEnd}`, 'g');
    existingContent = existingContent.replace(regex, '');
    existingContent = existingContent.replace(markerStart, '');

    let newSection = `${markerStart}\n## 🚀 Live Interactive Projects (Zero-Config)\n\n`;
    newSection += `> **Recruiters & Developers:** Click the **Run on Repl.it** badge next to any project below to spin up a free, isolated cloud container. Replit will automatically provision the correct compiler (Java, C++, Rust, Python, etc.) and execute the script directly in your browser!\n\n`;

    let totalEmbedded = 0;
    for (const target of TARGET_DIRS) {
        const fullPath = path.join(REPO_ROOT, target);
        const language = target.replace('-Projects', '');
        const foundProjects = scanDirectory(fullPath);
        
        if (foundProjects.length > 0) {
            newSection += generateMarkdownTable(foundProjects, language);
            totalEmbedded += foundProjects.length;
        }
    }

    newSection += `${markerEnd}\n`;

    fs.writeFileSync(readmePath, existingContent.trim() + '\n\n' + newSection);
    console.log(`✅ successfully indexed ${totalEmbedded} CLI projects and injected Replit badges into README.md!`);
}

updateReadme();
