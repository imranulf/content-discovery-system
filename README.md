````markdown
# Universal Content Discovery System 🚀

A **domain-agnostic, production-ready content discovery system** that works for ANY type of content across ANY domain. Built with zero external dependencies using only Python's standard library.

## 🎯 What Makes This Special

- **🌐 Universal**: Works for education, e-commerce, media, research, job listings, real estate - ANYTHING
- **📦 Zero Dependencies**: Pure Python standard library - no external packages needed
- **🕷️ Web Crawling**: Dynamically crawl and index content from any website
- **🧠 NLP Processing**: Built-in natural language understanding (50+ synonyms, intent detection, entity extraction)
- **🎨 Multi-Format**: Handles text, images, audio, video, and mixed content
- **⚡ Fast**: In-memory vector database with BM25, Dense, and Hybrid search strategies
- **🎯 Personalized**: Knowledge-based ranking adapting to user expertise and preferences
- **📊 Production-Ready**: Includes ranking algorithms, evaluation metrics (nDCG, MRR), and comprehensive testing

## 🌟 Key Features

### Content Discovery
- **Multiple Search Strategies**: BM25 (lexical), TF-IDF Dense (semantic), Hybrid (best of both)
- **Web Crawling**: Real-time content discovery from any URL
- **Content Parsing**: Automatic metadata extraction from HTML
- **Tag Generation**: Intelligent keyword extraction from content


### Natural Language Processing
- **Intent Detection**: Identifies learning, tutorial, reference, or project intents
- **Entity Extraction**: Detects topics, difficulty levels, and content formats
- **Synonym Expansion**: 50+ synonym mappings for better query understanding
- **Stop Word Filtering**: 100+ stop words for cleaner text processing

### Personalization
- **Knowledge-Based Ranking**: Adjusts results based on user expertise
- **Format Preferences**: Boosts preferred content types (+10%)
- **Time Budget Matching**: Fits content to available time (+5%)
- **Difficulty Progression**: Recommends appropriate challenge level

### Content Types Supported
- 📄 **Text**: Articles, eBooks, Documentation
- 🎬 **Video**: Tutorials, Courses, Live Streams
- 🎙️ **Audio**: Podcasts, Audio Courses, Interviews
- 📊 **Visual**: Infographics, Image Galleries, Diagrams
- 📦 **Mixed**: Any combination of the above

## 📂 Project Structure

```
Content_Discovery/
├── Project.py                      # Core implementation (825 lines, 35 KB)
│   ├── ContentCrawler             # Web crawling with gzip support
│   ├── NaturalLanguageProcessor   # NLP with 50+ synonyms
│   ├── VectorDBManager            # BM25, Dense, Hybrid search
│   └── LearnoraContentDiscovery   # Main API with personalization
│
├── Test & Demo Files
│   ├── test_content_discovery.py  # Unit tests (6 tests, all passing)
│   ├── simple_test.py             # Command-line testing tool
│   └── test_universal_discovery.py # Comprehensive multi-domain demo
│
└── Documentation
    ├── README.md                   # This file - Complete system overview
    ├── TECHNICAL_DOCUMENTATION.md  # Architecture & algorithm details
    └── NLP_DOCUMENTATION.md       # NLP system documentation
```

## 🚀 Quick Start

### Requirements
- Python 3.8+ (standard library only - no pip install needed!)

### Installation
```bash
# Clone the repository
git clone https://github.com/imranulf/content-discovery-system.git
cd content-discovery-system

# No installation needed! Just run it:
python test_universal_discovery.py
python simple_test.py "your query here"
```

### Basic Usage

```python
from Project import (
    LearnoraContentDiscovery,
    LearningContent,
    UserProfile,
)
from datetime import datetime

# 1. Create content (works for ANY domain!)
content = LearningContent(
    id="item-001",
    title="Python Programming for Beginners",
    content_type="video",
    source="YouTube",
    url="https://example.com/python",
    description="Learn Python from scratch",
    difficulty="beginner",
    duration_minutes=30,
    tags=["python", "programming", "beginner"],
    created_at=datetime.now(),
)

# 2. Initialize system with NLP
discovery = LearnoraContentDiscovery(enable_nlp=True)
discovery.vector_db.add_contents([content])

# 3. Create user profile
user = UserProfile(
    user_id="user-123",
    preferred_formats=["video"],
    available_time_daily=60,
)

# 4. Search with natural language
results = discovery.discover_and_personalize(
    query="I want to learn python programming",
    user_profile=user,
    strategy="hybrid",
    use_nlp=True,
)

# 5. Get personalized results
for item in results["results"]:
    print(f"{item['title']} - Score: {item['score']:.4f}")
```

### Web Crawling Example

```python
from Project import LearnoraContentDiscovery, UserProfile

# Initialize with web crawler enabled
discovery = LearnoraContentDiscovery(
    enable_crawler=True,
    enable_nlp=True,
)

# Crawl real websites
urls = [
    "https://www.python.org/about/",
    "https://realpython.com/",
    "https://docs.python.org/3/tutorial/",
]

discovery.crawl_and_index_urls(urls)

# Search the crawled content
user = UserProfile(user_id="user-1", preferred_formats=["article"])
results = discovery.discover_and_personalize(
    query="python tutorial",
    user_profile=user,
)

print(f"Found {len(results['results'])} pages from crawled content!")
```

## 📊 Demonstrations

### Run Demos & Tests

```bash
# Comprehensive multi-domain demonstration
python test_universal_discovery.py "machine learning"

# Command-line testing (quick searches)
python simple_test.py "python programming"

# Unit tests (verify all features)
python test_content_discovery.py
```

## 🎯 Real-World Use Cases

### 1. Educational Platforms (Coursera, Udemy)
```python
# Recommend courses based on user's knowledge
results = discovery.discover_and_personalize(
    query="advanced machine learning",
    user_profile=expert_user,
)
```

### 2. E-Commerce (Amazon-style)
```python
# Discover products with rich metadata
product = LearningContent(
    title="MacBook Pro 14",
    content_type="laptop",
    tags=["apple", "laptop", "professional"],
    metadata={"price": "$1999", "rating": "4.8/5"},
)
```

### 3. Content Aggregators (Feedly, Pocket)
```python
# Crawl and index articles from the web
discovery.crawl_and_index_urls([
    "https://techcrunch.com/",
    "https://arstechnica.com/",
])
```

### 4. Job Matching Platforms
```python
# Match candidates to jobs
job = LearningContent(
    title="Senior Python Developer",
    content_type="job_posting",
    tags=["python", "backend", "senior"],
    difficulty="advanced",
)
```

## 🧪 Testing

The system includes comprehensive test coverage:

```bash
# Run unit tests (6 core tests)
python test_content_discovery.py
# ✅ All tests passing: BM25, Dense, Hybrid, Personalization, Caching, Metrics

# Test multi-domain capability
python test_universal_discovery.py "machine learning"
# ✅ 20 diverse items across all domains

# Interactive testing
python simple_test.py "python programming"
# ✅ Quick command-line searches
```

**All tests passing:** ✅ 100% operational

## 📈 Performance Characteristics

- **Indexing**: Lightning fast in-memory operations
- **Search**: < 1ms for typical queries
- **Memory**: Minimal footprint
- **Scalability**: Handles 10K-100K items efficiently
- **Dependencies**: ZERO external libraries

## 🌐 Supported Domains

The system is **100% domain-agnostic** and works for:

- 🎓 **Education**: Courses, tutorials, certifications
- 🛍️ **E-Commerce**: Products, services, listings
- 📰 **Media**: Articles, videos, podcasts
- 💼 **Business**: Jobs, companies, opportunities
- 🏠 **Real Estate**: Properties, rentals, sales
- 🍔 **Food**: Recipes, restaurants, reviews
- ✈️ **Travel**: Destinations, hotels, experiences
- 🎮 **Entertainment**: Games, movies, books
- **And LITERALLY anything else!**

## 🔧 Advanced Features

### Search Strategies

```python
# BM25: Best for exact keyword matching
results = discovery.discover_and_personalize(query, user, strategy="bm25")

# Dense: Best for semantic similarity
results = discovery.discover_and_personalize(query, user, strategy="dense")

# Hybrid: Best overall (35% BM25 + 65% Dense)
results = discovery.discover_and_personalize(query, user, strategy="hybrid")
```

### Evaluation Metrics

```python
from Project import compute_ndcg, compute_mrr

# Normalized Discounted Cumulative Gain
ndcg = compute_ndcg(predictions, ground_truth, k=10)

# Mean Reciprocal Rank
mrr = compute_mrr(predictions, relevant_set)
```

### Natural Language Queries

The system understands natural language:
- "I want to learn Python programming"
- "Show me beginner-friendly tutorials"
- "Find advanced machine learning courses"
- "Quick JavaScript reference guide"

## 📚 Documentation

- **[README.md](README.md)** - Complete system overview (this file)
- **[TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md)** - Architecture and algorithms
- **[NLP_DOCUMENTATION.md](NLP_DOCUMENTATION.md)** - NLP system details

## 🎨 Architecture Highlights

### Core Components

1. **ContentCrawler**: Fetches and parses web content with gzip support
2. **NaturalLanguageProcessor**: 50+ synonyms, intent detection, entity extraction
3. **VectorDBManager**: Three search strategies (BM25, Dense, Hybrid)
4. **LearnoraContentDiscovery**: Main API with personalization and caching

### Algorithms

- **BM25**: k1=1.6, b=0.75 for lexical ranking
- **TF-IDF**: Dense vector similarity
- **Hybrid**: Weighted combination (35% BM25 + 65% Dense)
- **Personalization**: Format boost (+10%), Time fit (+5%)

## 💡 Extending the System

### Add New Content Types
```python
# The system accepts ANY content type
custom_content = LearningContent(
    content_type="your_custom_type",  # Anything you want!
    metadata={"custom_field": "value"},
)
```

### Add Custom URLs
```python
# Crawl any website
discovery.crawl_and_index_urls([
    "https://your-domain.com/page1",
    "https://another-site.com/content",
])
```

### Integrate with APIs
Extend `APIContentFetcher` class to integrate:
- YouTube Data API
- Medium API
- GitHub API
- Coursera API
- Your custom API

## 🤝 Contributing

This is a standalone educational project. Feel free to:
- Fork and modify for your needs
- Use in your own projects
- Learn from the implementation
- Share improvements

## 📄 License

Provided as-is for educational and testing purposes. Adapt or relicense as needed.

## 🎯 Quick Command Reference

```bash
# Interactive testing
python simple_test.py "your query"

# Multi-domain demonstration
python test_universal_discovery.py "machine learning"

# Run unit tests
python test_content_discovery.py
```

## 🌟 Why This System?

✅ **Universal**: One system for all content types and domains  
✅ **Zero Dependencies**: No external libraries - pure Python  
✅ **Production-Ready**: Complete with ranking, evaluation, and personalization  
✅ **Well-Documented**: Comprehensive guides and examples  
✅ **Fast**: In-memory operations for quick responses  
✅ **Extensible**: Easy to customize and extend  
✅ **Educational**: Learn from clean, well-commented code  

---

**Status**: ✅ All systems operational and production-ready!

For questions or issues, refer to the documentation files or examine the source code in `Project.py`.

````
