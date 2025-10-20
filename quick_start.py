"""Quick Start Example - Dynamic Content Discovery for Different Content Types

This script shows how easy it is to use the content discovery system
for ANY type of content.
"""

from Project import (
    LearningContent,
    UserProfile,
    LearnoraContentDiscovery,
)
from datetime import datetime


def example_1_courses():
    """Example: Discovering online courses"""
    print("\n" + "=" * 70)
    print("Example 1: Online Course Discovery")
    print("=" * 70)
    
    # Create course content
    courses = [
        LearningContent(
            id="course-1",
            title="Complete Python Bootcamp",
            content_type="course",
            source="Udemy",
            url="https://example.com/python-bootcamp",
            description="Master Python programming from beginner to advanced",
            difficulty="beginner",
            duration_minutes=120,
            tags=["python", "programming", "bootcamp", "beginner"],
            created_at=datetime.now(),
        ),
        LearningContent(
            id="course-2",
            title="Data Science with Python",
            content_type="course",
            source="Coursera",
            url="https://example.com/data-science",
            description="Learn data science, machine learning, and statistics",
            difficulty="intermediate",
            duration_minutes=180,
            tags=["python", "data science", "machine learning"],
            created_at=datetime.now(),
        ),
    ]
    
    # Setup discovery
    discovery = LearnoraContentDiscovery()
    discovery.vector_db.add_contents(courses)
    
    # Create user
    user = UserProfile(
        user_id="student-1",
        preferred_formats=["course"],
        available_time_daily=120,
    )
    
    # Search
    results = discovery.discover_and_personalize(
        query="learn python programming",
        user_profile=user,
        strategy="hybrid",
    )
    
    print(f"\n✨ Found {len(results['results'])} courses:")
    for item in results["results"]:
        print(f"  • {item['title']} (Score: {item['score']:.4f})")
        print(f"    {item['description'][:60]}...")


def example_2_products():
    """Example: E-commerce product discovery"""
    print("\n" + "=" * 70)
    print("Example 2: Product Discovery (E-commerce)")
    print("=" * 70)
    
    # Create product content
    products = [
        LearningContent(
            id="prod-1",
            title="Wireless Bluetooth Headphones",
            content_type="electronics",
            source="Amazon",
            url="https://shop.com/headphones",
            description="Premium noise-cancelling wireless headphones with 30-hour battery",
            difficulty="consumer",
            duration_minutes=0,
            tags=["headphones", "wireless", "bluetooth", "audio", "music"],
            metadata={"price": "$199", "rating": "4.5/5", "brand": "Sony"},
            created_at=datetime.now(),
        ),
        LearningContent(
            id="prod-2",
            title="Ergonomic Office Chair",
            content_type="furniture",
            source="Amazon",
            url="https://shop.com/chair",
            description="Comfortable ergonomic chair with lumbar support for long work hours",
            difficulty="consumer",
            duration_minutes=0,
            tags=["chair", "office", "ergonomic", "furniture", "comfort"],
            metadata={"price": "$299", "rating": "4.7/5", "brand": "Herman Miller"},
            created_at=datetime.now(),
        ),
    ]
    
    discovery = LearnoraContentDiscovery()
    discovery.vector_db.add_contents(products)
    
    user = UserProfile(
        user_id="shopper-1",
        preferred_formats=["electronics", "furniture"],
    )
    
    results = discovery.discover_and_personalize(
        query="comfortable office setup",
        user_profile=user,
        strategy="hybrid",
    )
    
    print(f"\n🛍️  Found {len(results['results'])} products:")
    for item in results["results"]:
        print(f"  • {item['title']} (Score: {item['score']:.4f})")
        print(f"    {item['description'][:60]}...")


def example_3_articles():
    """Example: Blog article/content discovery"""
    print("\n" + "=" * 70)
    print("Example 3: Article & Blog Post Discovery")
    print("=" * 70)
    
    # Create article content
    articles = [
        LearningContent(
            id="article-1",
            title="10 Tips for Better Code Reviews",
            content_type="blog-post",
            source="Dev.to",
            url="https://blog.com/code-reviews",
            description="Improve your code review process with these practical tips",
            difficulty="intermediate",
            duration_minutes=10,
            tags=["code review", "software engineering", "best practices", "team"],
            metadata={"author": "John Developer", "views": "10K"},
            created_at=datetime.now(),
        ),
        LearningContent(
            id="article-2",
            title="Understanding Async Programming in Python",
            content_type="tutorial",
            source="Real Python",
            url="https://blog.com/async-python",
            description="Learn how to write asynchronous Python code with asyncio",
            difficulty="advanced",
            duration_minutes=20,
            tags=["python", "async", "programming", "asyncio", "tutorial"],
            metadata={"author": "Python Expert", "views": "50K"},
            created_at=datetime.now(),
        ),
    ]
    
    discovery = LearnoraContentDiscovery()
    discovery.vector_db.add_contents(articles)
    
    user = UserProfile(
        user_id="reader-1",
        preferred_formats=["blog-post", "tutorial"],
        available_time_daily=30,
    )
    
    results = discovery.discover_and_personalize(
        query="improve python programming skills",
        user_profile=user,
        strategy="hybrid",
    )
    
    print(f"\n📰 Found {len(results['results'])} articles:")
    for item in results["results"]:
        print(f"  • {item['title']} (Score: {item['score']:.4f})")
        print(f"    {item['description'][:60]}...")


def example_4_mixed_content():
    """Example: Mixed content types in one system"""
    print("\n" + "=" * 70)
    print("Example 4: Mixed Content Discovery (Any Type!)")
    print("=" * 70)
    
    # Mix different content types!
    mixed_content = [
        LearningContent(
            id="mix-1",
            title="Python Programming Course",
            content_type="course",
            source="Udemy",
            url="https://example.com/python",
            description="Learn Python programming from scratch",
            difficulty="beginner",
            duration_minutes=90,
            tags=["python", "programming", "course"],
            created_at=datetime.now(),
        ),
        LearningContent(
            id="mix-2",
            title="Python Programming Book",
            content_type="book",
            source="O'Reilly",
            url="https://example.com/book",
            description="Comprehensive Python programming reference book",
            difficulty="intermediate",
            duration_minutes=0,
            tags=["python", "programming", "book", "reference"],
            metadata={"pages": "500", "author": "Expert Author"},
            created_at=datetime.now(),
        ),
        LearningContent(
            id="mix-3",
            title="Python IDE Software",
            content_type="software",
            source="JetBrains",
            url="https://example.com/pycharm",
            description="Professional Python IDE for developers",
            difficulty="professional",
            duration_minutes=0,
            tags=["python", "IDE", "software", "development"],
            metadata={"price": "$199/year", "platform": "Windows/Mac/Linux"},
            created_at=datetime.now(),
        ),
    ]
    
    discovery = LearnoraContentDiscovery()
    discovery.vector_db.add_contents(mixed_content)
    
    user = UserProfile(
        user_id="learner-1",
        preferred_formats=["course", "book"],
    )
    
    results = discovery.discover_and_personalize(
        query="python programming resources",
        user_profile=user,
        strategy="hybrid",
    )
    
    print(f"\n🎯 Found {len(results['results'])} items of different types:")
    for item in results["results"]:
        print(f"  • [{item['content_type'].upper()}] {item['title']}")
        print(f"    Score: {item['score']:.4f} | Tags: {', '.join(item['tags'][:3])}")


def main():
    """Run all examples"""
    print("\n" + "🚀" * 35)
    print("      QUICK START: Content Discovery for ANY Content Type")
    print("🚀" * 35)
    
    example_1_courses()
    example_2_products()
    example_3_articles()
    example_4_mixed_content()
    
    print("\n" + "=" * 70)
    print("✅ All examples completed!")
    print("=" * 70)
    print("\n💡 Key Takeaway:")
    print("   The same system works for courses, products, articles, books,")
    print("   software, or ANY content type you can imagine!")
    print("\n📚 Next Steps:")
    print("   1. Run the full demo: python demo.py")
    print("   2. Read the guide: USAGE_GUIDE.md")
    print("   3. Check the tests: python test_content_discovery.py")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
