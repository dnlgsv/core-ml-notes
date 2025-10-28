---
layout: page
title: Setup and Customization Guide
permalink: /setup-guide/
---

# Setup and Customization Guide

## 🎯 Initial Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/dnlgsv/core-ml-notes.git
cd core-ml-notes
```

### Step 2: Set Up Ruby Environment

#### On Windows:
```bash
# Using Ruby installer or RVM
ruby --version
```

#### On macOS/Linux:
```bash
ruby --version
```

### Step 3: Install Dependencies

```bash
bundle install
```

### Step 4: Start Development Server

```bash
bundle exec jekyll serve
```

Visit `http://localhost:4000` to see your site!

## 🚀 Deploying to GitHub Pages

Your site will be automatically deployed when you push to the `main` branch. GitHub Actions will:
1. Build your Jekyll site
2. Generate static files in `_site/`
3. Deploy to GitHub Pages

No additional setup required! Your site will be live at: `https://dnlgsv.github.io/core-ml-notes/`

## 🎨 Customization Guide

### Change Site Settings

Edit `_config.yml`:

```yaml
title: Your Site Title
description: Your site description
author: Your Name
url: https://your-domain.com
baseurl: /your-repo-name
```

### Change Fonts and Colors

Edit `_sass/variables.scss` or create custom CSS in `_sass/custom.scss`:

```scss
$primary-color: #1f77b4;
$secondary-color: #2ca02c;
$accent-color: #d62728;
```

### Add Social Media Links

Edit `_config.yml`:

```yaml
social:
  - platform: github
    url: https://github.com/your-username
  - platform: twitter
    url: https://twitter.com/your-handle
```

## 📝 Adding Content

### Create a New Page

1. Create a markdown file in the appropriate directory:
   ```bash
   docs/your-section/your-page.md
   ```

2. Add Jekyll front matter at the top:
   ```markdown
   ---
   layout: page
   title: Your Page Title
   permalink: /your-section/your-page/
   ---
   ```

3. Write your content in markdown below the front matter

### Supported Markdown Features

#### Code Blocks with Syntax Highlighting
```python
def example_function():
    return "Hello, World!"
```

#### Tables
| Column 1 | Column 2 |
|----------|----------|
| Data 1   | Data 2   |
| Data 3   | Data 4   |

#### Lists
- Unordered item 1
- Unordered item 2

1. Ordered item 1
2. Ordered item 2

#### Links
[Link text](https://example.com)

#### Images
![Alt text](/assets/image.png)

## 🐛 Troubleshooting

### Bundle install fails
Try updating Ruby:
```bash
ruby --version
gem update bundler
```

### Port 4000 already in use
```bash
bundle exec jekyll serve --port 4001
```

### Build fails
Clear the build cache:
```bash
rm -rf _site/
bundle exec jekyll build
```

## 📚 Additional Resources

- [Jekyll Documentation](https://jekyllrb.com/)
- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [Markdown Guide](https://www.markdownguide.org/)

## ✅ Deployment Checklist

Before deploying to GitHub Pages:

- [ ] Update `title` and `description` in `_config.yml`
- [ ] Update `url` and `baseurl` in `_config.yml`
- [ ] Test locally with `bundle exec jekyll serve`
- [ ] Check that all links work
- [ ] Test on mobile
- [ ] Commit all changes
- [ ] Push to `main` branch
