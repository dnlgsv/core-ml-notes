/**
 * Custom JavaScript for Core ML Notes
 * Inspired by Lena Voita's NLP Course Design
 */

(function() {
  'use strict';

  // Initialize on document ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  /**
   * Main initialization function
   */
  function init() {
    enhanceCodeBlocks();
    initializeKaTeX();
    setupSmoothScrolling();
    enhanceAdmonitions();
    setupTableOfContents();
  }

  /**
   * Enhance code blocks with additional features
   */
  function enhanceCodeBlocks() {
    const codeBlocks = document.querySelectorAll('pre');
    
    codeBlocks.forEach(block => {
      // Add language indicator if available
      const code = block.querySelector('code');
      if (code && code.className) {
        const language = code.className.replace('language-', '').toUpperCase();
        const label = document.createElement('div');
        label.className = 'code-label';
        label.textContent = language;
        block.insertBefore(label, code);
      }
      
      // Add copy button
      addCopyButton(block);
    });
  }

  /**
   * Add copy-to-clipboard button to code blocks
   */
  function addCopyButton(block) {
    const button = document.createElement('button');
    button.className = 'copy-button';
    button.innerHTML = '📋 Copy';
    button.title = 'Copy to clipboard';
    
    button.addEventListener('click', function() {
      const code = block.querySelector('code').textContent;
      navigator.clipboard.writeText(code).then(() => {
        button.innerHTML = '✓ Copied';
        setTimeout(() => {
          button.innerHTML = '📋 Copy';
        }, 2000);
      });
    });
    
    block.appendChild(button);
  }

  /**
   * Initialize KaTeX for mathematical equations
   */
  function initializeKaTeX() {
    if (typeof window.renderMathInElement !== 'undefined') {
      window.renderMathInElement(document.body, {
        delimiters: [
          {left: '$$', right: '$$', display: true},
          {left: '$', right: '$', display: false},
          {left: '\\(', right: '\\)', display: false},
          {left: '\\[', right: '\\]', display: true}
        ],
        throwOnError: false
      });
    }
  }

  /**
   * Setup smooth scrolling for internal links
   */
  function setupSmoothScrolling() {
    document.addEventListener('click', function(e) {
      const link = e.target.closest('a[href^="#"]');
      if (!link) return;
      
      const href = link.getAttribute('href');
      const target = document.querySelector(href);
      
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth' });
      }
    });
  }

  /**
   * Enhance admonitions with icons and better styling
   */
  function enhanceAdmonitions() {
    const admonitions = document.querySelectorAll('.admonition');
    
    admonitions.forEach(admon => {
      const type = admon.classList[1] || 'note';
      const icons = {
        'note': '📝',
        'warning': '⚠️',
        'danger': '🚨',
        'tip': '💡',
        'info': 'ℹ️',
        'example': '📌'
      };
      
      const icon = icons[type] || '📝';
      const title = admon.querySelector('.admonition-title');
      if (title) {
        title.textContent = icon + ' ' + title.textContent;
      }
    });
  }

  /**
   * Setup table of contents enhancements
   */
  function setupTableOfContents() {
    const toc = document.querySelector('.toc');
    if (!toc) return;
    
    // Add smooth scroll behavior
    const tocLinks = toc.querySelectorAll('a');
    tocLinks.forEach(link => {
      link.addEventListener('click', function(e) {
        e.preventDefault();
        const href = this.getAttribute('href');
        const target = document.querySelector(href);
        if (target) {
          target.scrollIntoView({ behavior: 'smooth' });
          history.pushState(null, null, href);
        }
      });
    });
    
    // Highlight current section in TOC
    highlightCurrentSection();
    window.addEventListener('scroll', highlightCurrentSection);
  }

  /**
   * Highlight the current section in the table of contents
   */
  function highlightCurrentSection() {
    const headings = document.querySelectorAll('h2, h3');
    let currentHeading = null;
    
    headings.forEach(heading => {
      if (heading.getBoundingClientRect().top < window.innerHeight / 2) {
        currentHeading = heading;
      }
    });
    
    if (currentHeading) {
      const id = currentHeading.id;
      const tocLinks = document.querySelectorAll('.toc a');
      
      tocLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === '#' + id) {
          link.classList.add('active');
        }
      });
    }
  }

  /**
   * Add some utility functions to the global scope if needed
   */
  window.CoreML = {
    // Scroll to section smoothly
    scrollToSection: function(sectionId) {
      const target = document.querySelector('#' + sectionId);
      if (target) {
        target.scrollIntoView({ behavior: 'smooth' });
      }
    },
    
    // Toggle theme
    toggleTheme: function() {
      const html = document.documentElement;
      const currentScheme = html.getAttribute('data-md-color-scheme');
      const newScheme = currentScheme === 'dark' ? 'light' : 'dark';
      html.setAttribute('data-md-color-scheme', newScheme);
      localStorage.setItem('theme', newScheme);
    },
    
    // Get theme preference
    getTheme: function() {
      return localStorage.getItem('theme') || 'light';
    }
  };

})();
