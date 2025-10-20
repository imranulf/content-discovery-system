"""Demo script showcasing dynamic content discovery for any type of content.

This demonstrates how the content discovery system can be used for:
- Learning content (courses, articles, videos)
- Any other type of content (products, documents, media, etc.)
"""

from Project import (
    LearningContent,
    UserProfile,
    VectorDBManager,
    LearnoraContentDiscovery,
)
from datetime import datetime


def demo_learning_content():
    """Demonstrate content discovery with learning materials."""
    print("=" * 70)
    print("DEMO 1: Learning Content Discovery")
    print("=" * 70)
    
    # Create diverse learning content
    contents = [
        LearningContent(
            id="py-basics",
            title="Python Programming Basics",
            content_type="article",
            source="CodeAcademy",
            url="https://example.com/python-basics",
            description="Learn Python fundamentals including variables, loops, and functions",
            difficulty="beginner",
            duration_minutes=30,
            tags=["python", "programming", "beginner", "fundamentals"],
            created_at=datetime.now(),
        ),
        LearningContent(
            id="ml-intro",
            title="Introduction to Machine Learning",
            content_type="video",
            source="DataCamp",
            url="https://example.com/ml-intro",
            description="Understand machine learning concepts, algorithms, and applications",
            difficulty="intermediate",
            duration_minutes=60,
            tags=["machine learning", "AI", "algorithms", "data science"],
            created_at=datetime.now(),
        ),
        LearningContent(
            id="web-dev",
            title="Full Stack Web Development",
            content_type="course",
            source="Udemy",
            url="https://example.com/fullstack",
            description="Build modern web applications using React, Node.js, and databases",
            difficulty="intermediate",
            duration_minutes=120,
            tags=["web development", "react", "nodejs", "javascript"],
            created_at=datetime.now(),
        ),
        LearningContent(
            id="data-viz",
            title="Data Visualization with Python",
            content_type="tutorial",
            source="Kaggle",
            url="https://example.com/data-viz",
            description="Create stunning visualizations using matplotlib, seaborn, and plotly",
            difficulty="intermediate",
            duration_minutes=45,
            tags=["python", "visualization", "data science", "matplotlib"],
            created_at=datetime.now(),
        ),
    ]
    
    # Initialize discovery system
    discovery = LearnoraContentDiscovery()
    discovery.vector_db.add_contents(contents)
    
    # Create user profile
    user = UserProfile(
        user_id="user-001",
        knowledge_areas={"python": "intermediate", "web": "beginner"},
        learning_goals=["improve python skills", "learn data science"],
        preferred_formats=["video", "tutorial"],
        available_time_daily=60,
        learning_style="hands-on",
    )
    
    # Search with different queries
    queries = [
        "python programming",
        "machine learning and AI",
        "web development",
    ]
    
    for query in queries:
        print(f"\n📚 Query: '{query}'")
        print("-" * 70)
        
        result = discovery.discover_and_personalize(
            query=query,
            user_profile=user,
            strategy="hybrid",
            top_k=3,
        )
        
        print(f"Found {result['stats']['returned']} results:")
        for i, item in enumerate(result["results"], 1):
            print(f"\n  {i}. {item['title']}")
            print(f"     Type: {item['content_type']} | Difficulty: {item['difficulty']}")
            print(f"     Duration: {item['duration_minutes']} min | Score: {item['score']:.4f}")
            print(f"     Tags: {', '.join(item['tags'][:3])}")


def demo_general_content():
    """Demonstrate that the system works for ANY type of content."""
    print("\n\n" + "=" * 70)
    print("DEMO 2: General Content Discovery (Products, Articles, Media, etc.)")
    print("=" * 70)
    
    # Create diverse content types - could be products, documents, media, etc.
    contents = [
        LearningContent(
            id="laptop-001",
            title="Dell XPS 13 Laptop",
            content_type="product",
            source="TechStore",
            url="https://store.com/laptop-xps",
            description="High-performance ultrabook with 13-inch display, Intel i7, 16GB RAM, perfect for programming and data science",
            difficulty="professional",
            duration_minutes=0,  # N/A for products
            tags=["laptop", "computer", "programming", "portable", "high-performance"],
            metadata={"price": "$1299", "brand": "Dell", "category": "Electronics"},
            created_at=datetime.now(),
        ),
        LearningContent(
            id="book-001",
            title="Clean Code by Robert Martin",
            content_type="book",
            source="BookStore",
            url="https://books.com/clean-code",
            description="A handbook of agile software craftsmanship with best practices for writing maintainable code",
            difficulty="intermediate",
            duration_minutes=0,
            tags=["programming", "software engineering", "best practices", "coding"],
            metadata={"author": "Robert C. Martin", "pages": "464", "category": "Technical Books"},
            created_at=datetime.now(),
        ),
        LearningContent(
            id="keyboard-001",
            title="Mechanical Keyboard RGB",
            content_type="product",
            source="TechStore",
            url="https://store.com/keyboard",
            description="Professional mechanical keyboard with RGB lighting, perfect for programming and gaming",
            difficulty="beginner",
            duration_minutes=0,
            tags=["keyboard", "mechanical", "programming", "accessories", "RGB"],
            metadata={"price": "$149", "brand": "Keychron", "category": "Accessories"},
            created_at=datetime.now(),
        ),
        LearningContent(
            id="article-001",
            title="The Future of AI in Software Development",
            content_type="article",
            source="TechBlog",
            url="https://blog.com/ai-future",
            description="Exploring how artificial intelligence is transforming software development and programming practices",
            difficulty="intermediate",
            duration_minutes=15,
            tags=["AI", "software development", "future", "programming", "technology"],
            metadata={"author": "Jane Smith", "publication": "Tech Insights", "category": "Technology"},
            created_at=datetime.now(),
        ),
    ]
    
    # Initialize discovery system
    discovery = LearnoraContentDiscovery()
    discovery.vector_db.add_contents(contents)
    
    # Create user profile
    user = UserProfile(
        user_id="user-002",
        knowledge_areas={"programming": "advanced"},
        learning_goals=["improve coding setup", "stay updated on tech trends"],
        preferred_formats=["article", "book"],
        available_time_daily=30,
        learning_style="practical",
    )
    
    # Search for different types of content
    queries = [
        "programming equipment and tools",
        "software engineering best practices",
        "artificial intelligence trends",
    ]
    
    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 70)
        
        result = discovery.discover_and_personalize(
            query=query,
            user_profile=user,
            strategy="hybrid",
            top_k=2,
        )
        
        print(f"Found {result['stats']['returned']} results:")
        for i, item in enumerate(result["results"], 1):
            print(f"\n  {i}. {item['title']}")
            print(f"     Type: {item['content_type']} | Tags: {', '.join(item['tags'][:3])}")
            print(f"     Score: {item['score']:.4f}")
            print(f"     Description: {item['description'][:80]}...")


def demo_search_strategies():
    """Demonstrate different search strategies."""
    print("\n\n" + "=" * 70)
    print("DEMO 3: Comparing Different Search Strategies")
    print("=" * 70)
    
    # Create some content
    contents = [
        LearningContent(
            id="advanced-ml",
            title="Advanced Machine Learning Techniques",
            content_type="course",
            source="AI Academy",
            url="https://example.com/advanced-ml",
            description="Deep dive into neural networks, deep learning, and advanced ML algorithms",
            difficulty="advanced",
            duration_minutes=180,
            tags=["machine learning", "deep learning", "neural networks", "AI"],
            created_at=datetime.now(),
        ),
        LearningContent(
            id="ml-basics",
            title="Machine Learning Basics",
            content_type="article",
            source="Tech Blog",
            url="https://example.com/ml-basics",
            description="Introduction to machine learning concepts and algorithms for beginners",
            difficulty="beginner",
            duration_minutes=20,
            tags=["machine learning", "basics", "introduction", "algorithms"],
            created_at=datetime.now(),
        ),
    ]
    
    discovery = LearnoraContentDiscovery()
    discovery.vector_db.add_contents(contents)
    
    user = UserProfile(
        user_id="user-003",
        preferred_formats=["course"],
        available_time_daily=90,
    )
    
    query = "machine learning"
    strategies = ["bm25", "dense", "hybrid"]
    
    print(f"\nQuery: '{query}'")
    print("=" * 70)
    
    for strategy in strategies:
        print(f"\n🔧 Strategy: {strategy.upper()}")
        print("-" * 70)
        
        result = discovery.discover_and_personalize(
            query=query,
            user_profile=user,
            strategy=strategy,
            top_k=2,
        )
        
        for i, item in enumerate(result["results"], 1):
            print(f"  {i}. {item['title']} (Score: {item['score']:.4f})")


if __name__ == "__main__":
    print("\n" + "🎯" * 35)
    print("   DYNAMIC CONTENT DISCOVERY SYSTEM DEMONSTRATION")
    print("🎯" * 35)
    
    # Run all demos
    demo_learning_content()
    demo_general_content()
    demo_search_strategies()
    
    print("\n\n" + "=" * 70)
    print("✅ All demonstrations completed successfully!")
    print("=" * 70)
    print("\nKey Features Demonstrated:")
    print("  • Flexible content modeling (learning, products, articles, etc.)")
    print("  • Multiple search strategies (BM25, Dense, Hybrid)")
    print("  • Personalized recommendations based on user preferences")
    print("  • Tag-based content organization")
    print("  • Efficient in-memory vector database")
    print("  • No external dependencies required")
    print("=" * 70)
