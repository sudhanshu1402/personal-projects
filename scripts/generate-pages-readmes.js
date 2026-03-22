const fs = require('fs');
const path = require('path');

const REPO_ROOT = path.join(__dirname, '..');
const GITHUB_PAGES_BASE = "https://sudhanshu1402.github.io/personal-projects";

function generatePagesTable(projects) {
    if (projects.length === 0) return '';
    
    let md = `\n### JavaScript (Static UIs)\n\n`;
    md += `| Project Name | Source Code | Play Live (Web) |\n`;
    md += `| :--- | :--- | :--- |\n`;
    
    for (const proj of projects) {
        const relativeDir = path.relative(REPO_ROOT, path.dirname(proj.file));
        const sourceUrl = `https://github.com/sudhanshu1402/personal-projects/tree/main/${relativeDir.replace(/\\/g, '/')}`;
        const liveUrl = `${GITHUB_PAGES_BASE}/${relativeDir.replace(/\\/g, '/')}/index.html`;
        
        md += `| **${proj.name}** | [📂 View Source](${sourceUrl}) | [🌐 View Live on GitHub Pages](${liveUrl}) |\n`;
    }
    return md + '\n';
}

function scanForStaticProjects(dir) {
    let projects = [];
    if (!fs.existsSync(dir)) return projects;
    
    const items = fs.readdirSync(dir, { withFileTypes: true });
    
    // Check if index.html exists in this directory
    const hasIndex = items.some(item => item.isFile() && item.name === 'index.html');
    if (hasIndex) {
        projects.push({
            name: path.basename(dir),
            file: path.join(dir, 'index.html')
        });
    } else {
        // Recurse into subdirectories
        for (const item of items) {
            if (item.isDirectory() && !item.name.startsWith('.')) {
                projects = projects.concat(scanForStaticProjects(path.join(dir, item.name)));
            }
        }
    }
    return projects;
}

function updateReadme() {
    const readmePath = path.join(REPO_ROOT, 'README.md');
    let existingContent = fs.existsSync(readmePath) ? fs.readFileSync(readmePath, 'utf8') : '';

    const markerStart = '<!-- PAGES_AUTO_GENERATE_START -->';
    const markerEnd = '<!-- PAGES_AUTO_GENERATE_END -->';
    
    const regex = new RegExp(`${markerStart}[\\s\\S]*?${markerEnd}`, 'g');
    existingContent = existingContent.replace(regex, '');

    const foundProjects = scanForStaticProjects(path.join(REPO_ROOT, 'Javascript-Projects'));
    
    if (foundProjects.length > 0) {
        let newSection = `${markerStart}\n${generatePagesTable(foundProjects)}${markerEnd}\n`;
        // Inject right before the CLI section we added previously
        const splitString = '<!-- PROJECTS_AUTO_GENERATE_START -->';
        if (existingContent.includes(splitString)) {
            const parts = existingContent.split(splitString);
            existingContent = parts[0] + newSection + splitString + parts[1];
        } else {
            existingContent += '\n' + newSection;
        }
        
        fs.writeFileSync(readmePath, existingContent);
        console.log(`✅ successfully indexed ${foundProjects.length} HTML projects and injected GitHub Pages links into README.md!`);
    } else {
        console.log('No static HTML projects found.');
    }
}

updateReadme();
