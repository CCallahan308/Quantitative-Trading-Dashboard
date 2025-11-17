#!/bin/bash
# Script to update repository name references after renaming
# Usage: ./update_repo_references.sh [new-repo-name]
#
# This script should be run AFTER renaming the repository on GitHub

set -e

OLD_NAME="test"
NEW_NAME="${1:-quantitative-trading-dashboard}"

echo "🔄 Updating repository references from 'test' to '$NEW_NAME'"
echo "=================================================="

# Update README.md
if [ -f "README.md" ]; then
    echo "📝 Updating README.md..."
    sed -i.bak "s|CCallahan308/test|CCallahan308/$NEW_NAME|g" README.md
    echo "✅ README.md updated"
fi

# Update SETUP.md
if [ -f "SETUP.md" ]; then
    echo "📝 Updating SETUP.md..."
    sed -i.bak "s|CCallahan308/test|CCallahan308/$NEW_NAME|g" SETUP.md
    echo "✅ SETUP.md updated"
fi

# Update CONTRIBUTING.md
if [ -f "CONTRIBUTING.md" ]; then
    echo "📝 Updating CONTRIBUTING.md..."
    sed -i.bak "s|CCallahan308/test|CCallahan308/$NEW_NAME|g" CONTRIBUTING.md
    echo "✅ CONTRIBUTING.md updated"
fi

# Update any other markdown files
echo "📝 Checking other documentation files..."
for file in *.md; do
    if [ "$file" != "README.md" ] && [ "$file" != "SETUP.md" ] && [ "$file" != "CONTRIBUTING.md" ]; then
        if grep -q "CCallahan308/test" "$file" 2>/dev/null; then
            echo "   Updating $file..."
            sed -i.bak "s|CCallahan308/test|CCallahan308/$NEW_NAME|g" "$file"
        fi
    fi
done

# Update git remote
echo "🔗 Updating git remote..."
git remote set-url origin "https://github.com/CCallahan308/$NEW_NAME.git"
echo "✅ Git remote updated"

# Verify changes
echo ""
echo "📊 Verification"
echo "=================================================="
echo "Current git remote:"
git remote -v

echo ""
echo "✅ All updates complete!"
echo ""
echo "📝 Next steps:"
echo "1. Review the changes with: git diff"
echo "2. Commit the changes: git add . && git commit -m 'Update repository name references'"
echo "3. Push the changes: git push"
echo "4. Remove backup files: rm *.bak"
