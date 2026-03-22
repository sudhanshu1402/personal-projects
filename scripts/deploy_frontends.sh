#!/bin/bash
# Universal Vercel Mass-Deployment Script
# Scans the Personal-Projects archive for static web apps and deploys them instantly

echo "🚀 Booting up the Universal Vercel Deployer..."
echo "Ensure you are logged in to Vercel (run 'npx vercel login' first)."

PROJECTS_DIR="../Javascript-Projects"

# Find all index.html files (representing static projects)
find "$PROJECTS_DIR" -name "index.html" | while read filepath; do
    project_dir=$(dirname "$filepath")
    project_name=$(basename "$project_dir")
    
    echo "============================================="
    echo "📦 Packaging and Deploying: $project_name"
    echo "============================================="
    
    cd "$project_dir"
    
    # Deploy to Vercel production automatically, skipping prompts
    npx vercel --prod --yes
    
    # Go back to the scripts folder
    cd - > /dev/null
done

echo ""
echo "✅ SUCCESS! All static projects successfully blasted to the Vercel Edge Network."
echo "Your entire Javascript learning history is now live on the internet."
