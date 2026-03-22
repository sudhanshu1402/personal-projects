const fs = require('fs');
const path = require('path');

const REPO_ROOT = path.join(__dirname, '..');
const TARGET_DIRS = [
    'Python-Projects', 'Java-Projects', 'C-Projects', 'Cpp-Projects', 
    'Rust-Projects', 'Go-Projects', 'CSharp-Projects', 'Typescript-Projects', 
    'Nodejs-Projects', 'Javascript-Projects'
];

let projectsData = [];

function scanDirectory(dir, category) {
    if (!fs.existsSync(dir)) return;
    const items = fs.readdirSync(dir, { withFileTypes: true });
    
    // Determine if this is a leaf project directory (contains code or index.html)
    let isProject = false;
    let hasHtml = false;
    for (const item of items) {
        if (item.isFile()) {
            const ext = path.extname(item.name);
            if (['.py','.java','.cpp','.c','.rs','.go','.cs','.ts','.js','.html'].includes(ext)) {
                isProject = true;
                if (ext === '.html') hasHtml = true;
            }
        }
    }

    if (isProject) {
        // Build project metadata
        const relativePath = path.relative(REPO_ROOT, dir);
        const parts = relativePath.split(path.sep);
        // Ex: ['Python-Projects', 'Hard', 'Fruit Ninja Game']
        const difficulty = parts.length > 1 ? parts[1] : 'Uncategorized';
        const name = parts[parts.length - 1].replace(/-/g, ' ');

        let type = category === 'Javascript' ? 'Frontend UI' : 'Backend / CLI';
        let actionUrl = '';
        let actionText = '';

        if (category === 'Javascript' && hasHtml) {
            actionUrl = relativePath + '/index.html';
            actionText = 'Launch Interface';
        } else {
            actionUrl = `https://replit.com/github/sudhanshu1402/personal-projects?folder=${encodeURIComponent(relativePath)}`;
            actionText = 'Execute in Replit';
        }

        projectsData.push({
            name,
            category,
            difficulty,
            type,
            path: relativePath,
            actionUrl,
            actionText
        });
    } else {
        // Recurse deeper
        for (const item of items) {
            if (item.isDirectory() && !item.name.startsWith('.')) {
                scanDirectory(path.join(dir, item.name), category);
            }
        }
    }
}

for (const target of TARGET_DIRS) {
    const fullPath = path.join(REPO_ROOT, target);
    const categoryName = target.replace('-Projects', '');
    scanDirectory(fullPath, categoryName);
}

const jsOutput = `const PROJECT_DATA = ${JSON.stringify(projectsData, null, 4)};`;
fs.writeFileSync(path.join(REPO_ROOT, 'projects_data.js'), jsOutput);
console.log(`Successfully generated projects_data.js with ${projectsData.length} projects.`);
