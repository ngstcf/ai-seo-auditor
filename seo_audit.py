import os
import asyncio
import json
import re
import csv
import time
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Crawl4AI Imports
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy

# LLM Client
from openai import AsyncOpenAI

# Load environment variables
load_dotenv()

# --- 1. Helper Functions ---
def extract_schema_tags(html: str) -> Dict[str, Any]:
    """
    Extract JSON-LD schema blocks from HTML.
    Returns both raw schemas and parsed data for analysis.
    """
    if not html:
        return {"found": False, "schemas": [], "types": []}
    
    pattern = r'<script\s+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>'
    matches = re.findall(pattern, html, re.DOTALL | re.IGNORECASE)
    
    if not matches:
        return {"found": False, "schemas": [], "types": []}
    
    schemas = []
    schema_types = set()
    
    for content in matches:
        try:
            clean_content = re.sub(r'\s+', ' ', content).strip()
            parsed = json.loads(clean_content)
            
            # Extract @type
            if isinstance(parsed, dict):
                schema_type = parsed.get('@type', 'Unknown')
                schema_types.add(schema_type if isinstance(schema_type, str) else str(schema_type))
            elif isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        schema_type = item.get('@type', 'Unknown')
                        schema_types.add(schema_type if isinstance(schema_type, str) else str(schema_type))
            
            schemas.append(clean_content)
        except json.JSONDecodeError:
            schemas.append(clean_content)
    
    return {
        "found": True,
        "schemas": schemas,
        "types": list(schema_types)
    }

def check_llms_txt(base_url: str) -> Optional[str]:
    """
    Check if llms.txt exists at the root domain.
    """
    import requests
    
    try:
        llms_url = f"{base_url.rstrip('/')}/llms.txt"
        response = requests.get(llms_url, timeout=5, headers={
            'User-Agent': 'Mozilla/5.0 (compatible; SEO-Auditor/1.0)'
        })
        
        if response.status_code == 200:
            return "Present"
        else:
            return "Not Found"
    except Exception:
        return "Not Found"

def extract_structured_content(html: str) -> Dict[str, Any]:
    """
    Analyze content structure for GEO/AEO optimization.
    """
    # Check for FAQ sections
    faq_pattern = r'<(div|section)[^>]*?(faq|question|answer)[^>]*?>'
    has_faq = bool(re.search(faq_pattern, html, re.IGNORECASE))
    
    # Check for lists
    list_count = len(re.findall(r'<(ul|ol)', html, re.IGNORECASE))
    
    # Check for tables
    table_count = len(re.findall(r'<table', html, re.IGNORECASE))
    
    # Check for headings structure
    h2_count = len(re.findall(r'<h2', html, re.IGNORECASE))
    h3_count = len(re.findall(r'<h3', html, re.IGNORECASE))
    
    return {
        "has_faq": has_faq,
        "list_count": list_count,
        "table_count": table_count,
        "heading_count": h2_count + h3_count,
        "has_structured_content": list_count > 0 or table_count > 0 or has_faq
    }

# --- 2. LLM Analysis Function ---
async def analyze_content_with_llm(
    content: str, 
    schema_data: Dict[str, Any],
    structured_content: Dict[str, Any],
    llms_txt_status: str,
    url: str
):
    """Enhanced LLM analysis with 2025 GEO/AEO best practices"""
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1") 
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    if not api_key:
        print("❌ Error: LLM_API_KEY missing in .env")
        return None

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    system_prompt = """
    You are an expert AI SEO Auditor specializing in GEO (Generative Engine Optimization) and AEO (Answer Engine Optimization) for 2025.
    
    Analyze content for visibility in AI search engines (ChatGPT, Perplexity, Claude, Google AI Overviews, Gemini).
    
    CRITICAL 2025 FACTORS TO EVALUATE:
    
    1. SCHEMA MARKUP (High Priority)
       - Presence of JSON-LD structured data
       - Critical types: Article, FAQPage, Organization, LocalBusiness, Product, Person, Review/AggregateRating
       - Schema quality and completeness
    
    2. LLMS.TXT (New Standard)
       - Check if /llms.txt exists at root
       - Critical for AI discoverability
    
    3. CONTENT STRUCTURE FOR AI (Essential)
       - Clear Q&A format sections
       - Lists and tables (AI-friendly formats)
       - Concise, scannable paragraphs (75-300 words per section)
       - Question-style H2 headings
       - Direct answers in first sentences
    
    4. E-E-A-T SIGNALS (Trust Factor)
       - Author credentials and bylines
       - Expert quotes and citations
       - Real experience indicators
       - Authority signals
    
    5. SEMANTIC CLARITY
       - Topic focus and clarity
       - Context-rich language
       - Entity mentions
       - Internal linking
    
    6. AI CITATION OPTIMIZATION
       - Quotable statistics
       - Clear definitions
       - Comparison tables
       - Step-by-step instructions
    
    OUTPUT ONLY VALID JSON (no markdown):
    {
      "title": "Page Title",
      "url": "URL",
      "overall_score": 0-100,
      "ai_readiness": "Poor/Fair/Good/Excellent",
      "metrics": [
        {
          "category": "Schema Markup",
          "score": 0-100,
          "status": "Pass/Warning/Fail",
          "details": "Specific findings",
          "types_found": ["Article", "Organization"]
        },
        {
          "category": "llms.txt",
          "score": 0-100,
          "status": "Pass/Warning/Fail",
          "details": "Present or missing"
        },
        {
          "category": "Content Structure",
          "score": 0-100,
          "status": "Pass/Warning/Fail",
          "details": "AI-friendly formatting assessment"
        },
        {
          "category": "E-E-A-T Signals",
          "score": 0-100,
          "status": "Pass/Warning/Fail",
          "details": "Trust and authority indicators"
        },
        {
          "category": "Semantic Clarity",
          "score": 0-100,
          "status": "Pass/Warning/Fail",
          "details": "Topic focus and context"
        }
      ],
      "recommendations": [
        {
          "priority": "Critical/High/Medium/Low",
          "category": "Schema/Content/E-E-A-T/llms.txt",
          "action": "Specific actionable recommendation",
          "impact": "Expected improvement"
        }
      ],
      "strengths": ["List of what's working well"],
      "ai_citation_potential": "Low/Medium/High"
    }
    """

    user_prompt = f"""
    Analyze this page for AI search optimization (GEO/AEO):
    
    URL: {url}
    
    === LLMS.TXT STATUS ===
    {llms_txt_status}
    
    === SCHEMA.ORG DATA ===
    Found: {schema_data['found']}
    Types: {', '.join(schema_data['types']) if schema_data['types'] else 'None'}
    Raw Data: {schema_data['schemas'][:2000] if schema_data['schemas'] else 'No schema found'}
    
    === CONTENT STRUCTURE ANALYSIS ===
    Has FAQ: {structured_content['has_faq']}
    Lists: {structured_content['list_count']}
    Tables: {structured_content['table_count']}
    Headings: {structured_content['heading_count']}
    
    === PAGE CONTENT (First 12000 chars) ===
    {content[:12000]}
    
    Provide comprehensive analysis focusing on 2025 AI search optimization best practices.
    """

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"⚠️ LLM Call Failed: {e}")
        return None

# --- 3. Enhanced CSV Output ---
def save_to_csv(reports: List[Dict], filename="ai_seo_report_2025.csv"):
    """Save detailed reports with 2025 metrics"""
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        if not file_exists:
            writer.writerow([
                "Date", "URL", "Title", "Overall Score", "AI Readiness",
                "Schema Score", "Schema Types", "llms.txt", 
                "Content Structure Score", "E-E-A-T Score", "Semantic Score",
                "AI Citation Potential", "Top Priority Action", "Strengths"
            ])
        
        for r in reports:
            metrics = {m['category']: m for m in r.get('metrics', [])}
            rec = r.get('recommendations', [])
            top_rec = next((r['action'] for r in rec if r.get('priority') == 'Critical'), 
                          rec[0]['action'] if rec else "None")
            
            schema_metric = metrics.get('Schema Markup', {})
            schema_types = ', '.join(schema_metric.get('types_found', []))
            
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M"),
                r.get('url'),
                r.get('title'),
                r.get('overall_score'),
                r.get('ai_readiness'),
                metrics.get('Schema Markup', {}).get('score', 'N/A'),
                schema_types or 'None',
                metrics.get('llms.txt', {}).get('details', 'N/A'),
                metrics.get('Content Structure', {}).get('score', 'N/A'),
                metrics.get('E-E-A-T Signals', {}).get('score', 'N/A'),
                metrics.get('Semantic Clarity', {}).get('score', 'N/A'),
                r.get('ai_citation_potential', 'Unknown'),
                top_rec,
                '; '.join(r.get('strengths', [])[:2])
            ])
    
    print(f"\n💾 Saved {len(reports)} detailed reports to '{filename}'")

# --- 4. Main Crawler Logic ---
async def analyze_site(start_url: str):
    """Enhanced crawler with 2025 AI SEO checks"""
    print(f"🕵️  Initializing Enhanced AI SEO Crawler...")
    print(f"🎯 Target: {start_url}")
    
    # Extract base URL for llms.txt check
    from urllib.parse import urlparse
    parsed = urlparse(start_url)
    base_url = f"{parsed.scheme}://{parsed.netloc}"

    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        java_script_enabled=True,
    )

    deep_strategy = None
    max_depth = int(os.getenv("MAX_DEPTH", 1))
    max_pages = int(os.getenv("MAX_PAGES", 5))

    if max_depth > 0:
        deep_strategy = BFSDeepCrawlStrategy(max_depth=max_depth, max_pages=max_pages)

    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        process_iframes=False,
        remove_overlay_elements=True,
        delay_before_return_html=2.5,
        deep_crawl_strategy=deep_strategy
    )

    # Check llms.txt once for the domain
    print(f"\n🔍 Checking for /llms.txt...")
    llms_txt_status = check_llms_txt(base_url)
    print(f"   {'✅' if llms_txt_status == 'Present' else '⚠️'} llms.txt: {llms_txt_status}")

    async with AsyncWebCrawler(config=browser_config) as crawler:
        results = await crawler.arun(url=start_url, config=run_config)

        if not isinstance(results, list):
            results = [results]

        print(f"\n✅ Crawl Complete: {len(results)} pages fetched")
        aggregated_report = []

        for idx, res in enumerate(results, 1):
            if not res.success:
                print(f"⚠️  [{idx}] Error: {res.url} - {res.error_message}")
                continue

            print(f"\n🔎 [{idx}/{len(results)}] Analyzing: {res.url}")
            
            # Extract schema
            schema_data = extract_schema_tags(res.html)
            schema_icon = "✅" if schema_data['found'] else "❌"
            print(f"   {schema_icon} Schema: {', '.join(schema_data['types'][:3]) if schema_data['types'] else 'None found'}")
            
            # Extract structured content
            structured_content = extract_structured_content(res.html)
            print(f"   📊 Structure: Lists={structured_content['list_count']}, Tables={structured_content['table_count']}, FAQ={structured_content['has_faq']}")

            # Send to LLM
            print(f"   🧠 Analyzing with AI...")
            json_str = await analyze_content_with_llm(
                res.markdown,
                schema_data,
                structured_content,
                llms_txt_status,
                res.url
            )
            
            if not json_str:
                continue

            try:
                data = json.loads(json_str)
                aggregated_report.append(data)
                score = data.get('overall_score', 0)
                readiness = data.get('ai_readiness', 'Unknown')
                print(f"   ✨ Score: {score}/100 | AI Readiness: {readiness}")
            except json.JSONDecodeError as e:
                print(f"   ❌ JSON Parse Error: {e}")

        return aggregated_report

# --- 5. Enhanced Reporting ---
def print_detailed_report(reports: List[Dict]):
    """Print comprehensive AI SEO report"""
    if not reports:
        print("\n❌ No valid reports generated.")
        return

    scores = [r.get('overall_score', 0) for r in reports if isinstance(r.get('overall_score'), (int, float))]
    avg_score = sum(scores) / len(scores) if scores else 0

    print("\n" + "="*80)
    print(f"🤖 AI SEO AUDIT REPORT - 2025 STANDARDS")
    print(f"📊 Average Score: {avg_score:.1f}/100")
    print(f"📄 Pages Analyzed: {len(reports)}")
    print("="*80)

    for i, page in enumerate(reports, 1):
        print(f"\n{'='*80}")
        print(f"[{i}] {page.get('title', 'Untitled')}")
        print(f"🔗 {page.get('url')}")
        print(f"📈 Overall Score: {page.get('overall_score')}/100")
        print(f"🎯 AI Readiness: {page.get('ai_readiness', 'Unknown')}")
        print(f"🎖️  Citation Potential: {page.get('ai_citation_potential', 'Unknown')}")
        
        print(f"\n💪 Strengths:")
        for strength in page.get('strengths', [])[:3]:
            print(f"   ✅ {strength}")
        
        print(f"\n📊 Detailed Metrics:")
        for m in page.get('metrics', []):
            status_icon = {"Pass": "✅", "Warning": "⚠️", "Fail": "❌"}.get(m.get('status', ''), "❓")
            print(f"   {status_icon} {m.get('category')}: {m.get('score', 'N/A')}/100")
            print(f"      → {m.get('details')}")
        
        print(f"\n🎯 Priority Recommendations:")
        critical_recs = [r for r in page.get('recommendations', []) if r.get('priority') == 'Critical']
        high_recs = [r for r in page.get('recommendations', []) if r.get('priority') == 'High']
        
        for rec in (critical_recs + high_recs)[:5]:
            priority_icon = "🔴" if rec.get('priority') == 'Critical' else "🟡"
            print(f"   {priority_icon} [{rec.get('priority')}] {rec.get('action')}")
            if rec.get('impact'):
                print(f"      Impact: {rec.get('impact')}")

# --- 6. Main Execution ---
async def main():
    print("="*80)
    print("🚀 AI SEO AUDITOR 2025 - GEO/AEO Optimization Tool")
    print("="*80)
    print("\n📋 This tool analyzes your site for:")
    print("   • Schema.org markup (JSON-LD)")
    print("   • llms.txt presence")
    print("   • AI-friendly content structure")
    print("   • E-E-A-T signals")
    print("   • Semantic clarity")
    print("   • AI citation potential\n")
    
    target_url = input("🔗 Enter URL to analyze: ").strip()
    if not target_url.startswith("http"):
        target_url = "https://" + target_url

    reports = await analyze_site(target_url)
    
    if reports:
        print_detailed_report(reports)
        save_to_csv(reports)
        
        # Summary stats
        critical_issues = sum(
            1 for r in reports 
            for rec in r.get('recommendations', []) 
            if rec.get('priority') == 'Critical'
        )
        
        print(f"\n{'='*80}")
        print(f"📈 SUMMARY")
        print(f"{'='*80}")
        print(f"🔴 Critical Issues: {critical_issues}")
        print(f"💾 Full report saved to: ai_seo_report_2025.csv")
        print(f"\n🎯 Next Steps:")
        print(f"   1. Address Critical priority items first")
        print(f"   2. Implement missing schema markup")
        print(f"   3. Add llms.txt if not present")
        print(f"   4. Restructure content for AI readability")
        print("="*80)
    else:
        print("\n❌ No valid reports generated. Check your configuration.")

if __name__ == "__main__":
    asyncio.run(main())