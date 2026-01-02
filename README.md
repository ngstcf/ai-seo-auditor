# AI SEO Auditor - GEO/AEO Optimization Tool

A Python-based SEO auditing tool that analyzes websites for **Generative Engine Optimization (GEO)** and **Answer Engine Optimization (AEO)** — the 2025 standards for AI search visibility.

## Features

- **Schema Markup Analysis**: Detects and validates JSON-LD structured data
- **llms.txt Detection**: Checks for the new AI discovery standard at `/llms.txt`
- **Content Structure Analysis**: Evaluates AI-friendly formatting (lists, tables, FAQs, headings)
- **E-E-A-T Assessment**: Analyzes experience, expertise, authoritativeness, and trustworthiness signals
- **Semantic Clarity Scoring**: Measures topic focus and context richness
- **AI Citation Potential**: Estimates likelihood of being cited by AI search engines
- **Deep Crawling**: Supports multi-page website analysis with configurable depth
- **CSV Export**: Generates detailed reports for tracking progress over time

## Why This Matters in 2025

AI search engines (ChatGPT, Perplexity, Claude, Google AI Overviews, Gemini) are changing how content is discovered. This tool helps optimize for:
- Direct answer citations in AI responses
- Structured data that AI models can parse
- Clear, scannable content formats
- Authority signals that build trust

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
1. Check for `/llms.txt` at the domain root
2. Crawl the specified page(s)
3. Analyze schema markup, content structure, and E-E-A-T signals
4. Generate an AI-powered SEO report
5. Save results to `ai_seo_report_2025.csv`

## Output

The tool generates:
- **Console Report**: Real-time analysis with scores and recommendations
- **CSV Report**: Detailed metrics for historical tracking

### Score Categories

| Category | What It Measures |
|----------|------------------|
| Schema Markup | JSON-LD presence and quality |
| llms.txt | AI discovery file presence |
| Content Structure | Lists, tables, FAQs, headings |
| E-E-A-T Signals | Trust and authority indicators |
| Semantic Clarity | Topic focus and context |

## Requirements

- Python 3.10+
- OpenAI API key (or compatible endpoint)

## Current Limitations

- **Schema Validation**: Detects presence of JSON-LD but does not perform deep validation against Schema.org specifications
- **llms.txt Content**: Checks for file existence only; does not validate syntax or completeness of the file contents
- **E-E-A-T Heuristics**: Trust signals are inferred from content patterns; cannot verify real-world credentials or authority
- **Single-Language**: Optimized for English content analysis; other languages may have reduced accuracy
- **LLM Dependency**: Analysis quality depends on the model used; different LLMs may produce varying scores

## Future Directions

- [ ] **Schema Validation**: Integrate Schema.org validator to check markup correctness
- [ ] **llms.txt Parser**: Full parsing and validation of llms.txt file structure and content
- [ ] **Multi-language Support**: Extend LLM prompts and content analysis for non-English sites
- [ ] **Historical Tracking**: Built-in comparison of previous audits to show progress over time
- [ ] **Competitor Analysis**: Compare multiple sites side-by-side
- [ ] **Export Formats**: Add JSON and HTML report options
- [ ] **CI/CD Integration**: GitHub Action for automated SEO checks on PRs
- [ ] **Batch Mode**: Analyze multiple URLs from a file without interactive prompts
- [ ] **Core Web Vitals**: Integrate page speed and UX metrics from Lighthouse/PageSpeed Insights

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## Author

Created by Ng Chong [@ngstcf](https://github.com/ngstcf)
