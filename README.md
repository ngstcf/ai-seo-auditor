# AI SEO Auditor - GEO/AEO Optimization Tool

A Python-based SEO auditing tool that analyzes websites for **Generative Engine Optimization (GEO)** and **Answer Engine Optimization (AEO)** — optimizing for AI search visibility in 2025/2026.

## Features

The auditor evaluates your site across **11 scored categories** with programmatic checks feeding into an LLM-powered analysis:

### Domain-Level Checks (run once per audit)
- **Robots.txt AI Crawler Access**: Checks whether 12 AI bots (GPTBot, ClaudeBot, PerplexityBot, ChatGPT-User, OAI-SearchBot, Google-Extended, Amazonbot, etc.) are blocked, partially blocked, or allowed
- **Sitemap.xml Analysis**: Validates presence, URL count, and `lastmod` date coverage
- **AI Discovery Files**: Checks for `/llms.txt`, `/llms-full.txt`, and `/.well-known/ai.json`

### Page-Level Checks (run per crawled page)
- **Schema Markup Analysis**: Extracts JSON-LD structured data including `@type`, `datePublished`, `dateModified`, `author`, and FAQPage detection (with `@graph` support)
- **OpenGraph & Meta Tags**: Validates `og:title`, `og:description`, `og:type`, `og:image`, canonical URL, meta description, and meta author completeness
- **Content Freshness**: Parses dates from schema, meta tags, and HTML `<time>` elements; flags content as Fresh (<90 days), Aging (90-365), Stale (365-730), or Very Stale (>730 days)
- **Heading Hierarchy**: Validates H1-H6 nesting (single H1, no skipped levels), counts question-format headings that boost AI citation likelihood
- **Link Profile**: Counts internal/external links, identifies authoritative external citations (.gov, .edu), flags weak anchor text
- **E-E-A-T Signals**: Detects author bylines, bio sections, Person schema, credential indicators (PhD, MD, Dr., certifications), and fact-check/review attribution
- **Content Quality Metrics**: Measures word count, paragraph distribution, statistic density, definition patterns, and quotable statements
- **Semantic Clarity**: Evaluates topic focus, context richness, and entity mentions

### Additional Features
- **Anti-Bot Bypass**: Stealth mode, magic mode, user simulation, navigator override, and auto-retry (up to 3 attempts) for sites with bot protection
- **Deep Crawling**: Multi-page analysis with configurable depth via BFS strategy
- **CSV Export**: Detailed reports with all 11 metric scores for historical tracking

## Why This Matters

AI search engines (ChatGPT, Perplexity, Claude, Google AI Overviews, Gemini, Copilot) are reshaping content discovery. Key facts driving this tool:
- 60% of AI Overview citations come from pages NOT in the top 20 organic results
- 40%+ of sites accidentally block AI crawlers in robots.txt
- Pages not updated within 90 days are 3x more likely to lose AI citations
- 85% of AI-cited content is from the last 2 years
- Question-format headings increase citation likelihood by 40%
- Statistics with cited sources improve AI visibility by 40%

> **Reference:** For more on SEO in the AI era, see article [SEO for the AI Era: A 2025 Quick Guide](https://c3.unu.edu/blog/seo-for-the-ai-era-a-2025-quick-guide)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ngstcf/ai-seo-auditor.git
   cd ai-seo-auditor
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install crawl4ai browser setup:
   ```bash
   crawl4ai-setup
   ```

## Configuration

Create a `.env` file in the project root:

```env
# Required: OpenAI-compatible API key
LLM_API_KEY=your_api_key_here

# Optional: Custom API endpoint (defaults to OpenAI)
LLM_BASE_URL=https://api.openai.com/v1

# Optional: Model to use (defaults to gpt-4o-mini)
LLM_MODEL=gpt-4o-mini

# Optional: Crawl depth and page limits
MAX_DEPTH=1
MAX_PAGES=5
```

A `.env.example` file is provided for reference.

## Usage

Run the auditor:
```bash
python seo_audit.py
```

Follow the prompts to enter a URL. The tool will:
1. Run domain-level checks (robots.txt, sitemap.xml, llms.txt, AI discovery files)
2. Crawl the specified page(s) with anti-bot bypass enabled
3. Run programmatic extraction on each page (schema, meta tags, freshness, headings, links, E-E-A-T, content quality)
4. Send all pre-analyzed data to the LLM for scoring across 11 categories
5. Print a detailed report and save results to `ai_seo_report_2025.csv`

## Output

The tool generates:
- **Console Report**: Domain-level summary + per-page analysis with 11 scored categories, strengths, and prioritized recommendations
- **CSV Report**: All metrics in columns for historical tracking and comparison

### Score Categories

| Category | What It Measures |
|----------|------------------|
| Robots.txt AI Access | Whether AI crawlers (GPTBot, ClaudeBot, etc.) are blocked |
| Schema Markup | JSON-LD presence, types, completeness (dates, authors) |
| AI Discovery Files | llms.txt, llms-full.txt, ai.json presence |
| OpenGraph & Meta | OG tags completeness, canonical URL, meta description |
| Content Freshness | Publication/modification dates, staleness assessment |
| Content Structure | Lists, tables, FAQs, accordions, definition lists |
| Heading Structure | H1-H6 hierarchy validity, question-format headings |
| Link Profile | Internal/external links, authoritative citations, anchor text |
| E-E-A-T Signals | Author bylines, bios, credentials, fact-check attribution |
| Content Quality | Depth, statistic density, definitions, quotable statements |
| Semantic Clarity | Topic focus, entity mentions, context richness |

## Requirements

- Python 3.10+
- OpenAI API key (or compatible endpoint)

## Current Limitations

- **Schema Validation**: Extracts and identifies JSON-LD types and key fields but does not validate against the full Schema.org specification
- **llms.txt Content**: Checks for file existence; does not validate syntax or completeness of the file contents
- **E-E-A-T Heuristics**: Detects structural signals (bylines, bios, credentials, Person schema) but cannot verify real-world authority or credentials
- **Freshness Parsing**: Handles ISO 8601 and common date formats; unusual or locale-specific formats may not parse
- **Single-Language**: Optimized for English content analysis; other languages may have reduced accuracy
- **LLM Dependency**: Final scoring depends on the model used; different LLMs may produce varying scores for the same pre-analyzed data
- **Anti-Bot Limits**: Stealth mode and retries handle most protections, but sites with advanced anti-bot (Cloudflare Turnstile, Datadome) may still block crawling

## Future Directions

- [ ] **Multi-language Support**: Extend LLM prompts and content analysis for non-English sites
- [ ] **Historical Tracking**: Built-in comparison of previous audits to show progress over time
- [ ] **Competitor Analysis**: Compare multiple sites side-by-side
- [ ] **Export Formats**: Add JSON and HTML report options
- [ ] **CI/CD Integration**: GitHub Action for automated SEO checks on PRs
- [ ] **Batch Mode**: Analyze multiple URLs from a file without interactive prompts
- [ ] **Core Web Vitals**: Integrate page speed and UX metrics from Lighthouse/PageSpeed Insights
- [ ] **Markdown Alternate Routes**: Check for `.md` versions of pages and `<link rel="alternate" type="text/markdown">` tags
- [ ] **AI Citation Tracking**: Monitor how AI platforms actually cite/reference the audited content over time

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## Author

Created by Ng Chong [@ngstcf](https://github.com/ngstcf)
