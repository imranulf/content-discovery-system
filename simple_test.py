"""Simple Command-Line Test - Search and See Results

Usage:
    python simple_test.py "your search query here"
    
Examples:
    python simple_test.py "python tutorial"
    python simple_test.py "beginner machine learning"
    python simple_test.py "advanced web development"
"""

from Project import (
    LearnoraContentDiscovery,
    LearningContent,
    UserProfile,
)
from datetime import datetime
import sys


def create_sample_content():
    """Create some sample content to search through."""
    return [
        LearningContent(
            id="py-beginner",
            title="Python Programming for Beginners",
            content_type="video",
            source="YouTube",
            url="https://example.com/py-beginner",
            description="Learn Python programming from scratch with easy tutorials for beginners",
            difficulty="beginner",
            duration_minutes=30,
            tags=["python", "programming", "beginner", "tutorial"],
            created_at=datetime.now(),
        ),
        LearningContent(
            id="py-advanced",
            title="Advanced Python Techniques",
            content_type="course",
            source="Udemy",
            url="https://example.com/py-advanced",
            description="Master advanced Python concepts including decorators, generators, and metaclasses",
            difficulty="advanced",
            duration_minutes=120,
            tags=["python", "advanced", "programming", "decorators"],
            created_at=datetime.now(),
        ),
        LearningContent(
            id="ml-intro",
            title="Introduction to Machine Learning",
            content_type="article",
            source="Medium",
            url="https://example.com/ml-intro",
            description="Understanding machine learning fundamentals, algorithms, and practical applications",
            difficulty="intermediate",
            duration_minutes=45,
            tags=["machine learning", "ML", "data science", "python", "AI"],
            created_at=datetime.now(),
        ),
        LearningContent(
            id="web-dev",
            title="Full Stack Web Development",
            content_type="course",
            source="Coursera",
            url="https://example.com/web-dev",
            description="Build modern web applications with JavaScript, React, Node.js and databases",
            difficulty="intermediate",
            duration_minutes=90,
            tags=["web development", "javascript", "react", "nodejs", "fullstack"],
            created_at=datetime.now(),
        ),
        LearningContent(
            id="data-viz",
            title="Data Visualization with Python",
            content_type="tutorial",
            source="DataCamp",
            url="https://example.com/data-viz",
            description="Create beautiful visualizations using matplotlib, seaborn, and plotly libraries",
            difficulty="intermediate",
            duration_minutes=60,
            tags=["python", "data visualization", "matplotlib", "data science"],
            created_at=datetime.now(),
        ),
        LearningContent(
            id="js-basics",
            title="JavaScript Fundamentals",
            content_type="video",
            source="freeCodeCamp",
            url="https://example.com/js-basics",
            description="Learn JavaScript basics including variables, functions, and DOM manipulation",
            difficulty="beginner",
            duration_minutes=40,
            tags=["javascript", "programming", "beginner", "web development"],
            created_at=datetime.now(),
        ),
        LearningContent(
            id="ai-deeplearning",
            title="Deep Learning and Neural Networks",
            content_type="course",
            source="Coursera",
            url="https://example.com/ai-deep",
            description="Advanced AI course covering neural networks, CNNs, RNNs, and deep learning",
            difficulty="advanced",
            duration_minutes=180,
            tags=["AI", "deep learning", "neural networks", "machine learning", "advanced"],
            created_at=datetime.now(),
        ),
        LearningContent(
            id="git-tutorial",
            title="Git and GitHub Tutorial",
            content_type="tutorial",
            source="GitHub",
            url="https://example.com/git-tutorial",
            description="Learn version control with Git and collaborate using GitHub",
            difficulty="beginner",
            duration_minutes=25,
            tags=["git", "github", "version control", "beginner", "tutorial"],
            created_at=datetime.now(),
        ),
    ]


def main():
    """Main function."""
    print("=" * 80)
    print("SIMPLE CONTENT DISCOVERY TEST")
    print("=" * 80)
    
    # Check if query provided
    if len(sys.argv) < 2:
        print("\nUsage: python simple_test.py \"your search query\"")
        print("\nExamples:")
        print('  python simple_test.py "python tutorial"')
        print('  python simple_test.py "beginner machine learning"')
        print('  python simple_test.py "I want to learn web development"')
        print("\n" + "=" * 80)
        return
    
    # Get query from command line
    query = " ".join(sys.argv[1:])
    
    # Setup system
    print("\nSetting up system...")
    discovery = LearnoraContentDiscovery(enable_nlp=True)
    sample_content = create_sample_content()
    discovery.vector_db.add_contents(sample_content)
    print(f"Loaded {len(sample_content)} content items")
    
    # Create user profile
    user_profile = UserProfile(
        user_id="test-user",
        preferred_formats=["video", "course"],
        available_time_daily=60,
    )
    
    print(f"\nUser Profile:")
    print(f"  - Preferred formats: {', '.join(user_profile.preferred_formats)}")
    print(f"  - Available time: {user_profile.available_time_daily} min/day")
    
    # Search
    print("\n" + "=" * 80)
    print(f"SEARCH QUERY: '{query}'")
    print("=" * 80)
    
    results = discovery.discover_and_personalize(
        query=query,
        user_profile=user_profile,
        strategy="hybrid",
        top_k=5,
        use_nlp=True,
    )
    
    # Display NLP analysis if available
    if 'nlp_analysis' in results:
        nlp = results['nlp_analysis']
        print(f"\nNLP Analysis:")
        print(f"  Intent: {nlp['intent']['primary']}")
        if nlp['entities']['topics']:
            print(f"  Topics: {', '.join(nlp['entities']['topics'][:3])}")
        if nlp['entities']['difficulty']:
            print(f"  Difficulty: {', '.join(nlp['entities']['difficulty'])}")
        if nlp['entities']['formats']:
            print(f"  Formats: {', '.join(nlp['entities']['formats'])}")
    
    # Display results
    print(f"\nFound {len(results['results'])} results:")
    print(f"Strategy: {results['strategy']}")
    print(f"Total indexed: {results['stats']['total_indexed']}")
    
    print("\n" + "-" * 80)
    print("RESULTS:")
    print("-" * 80)
    
    for i, item in enumerate(results['results'], 1):
        print(f"\n{i}. {item['title']}")
        print(f"   Score: {item['score']:.4f}")
        print(f"   Type: {item['content_type']:<10} | Difficulty: {item['difficulty']:<12}")
        print(f"   Duration: {item['duration_minutes']} minutes")
        print(f"   Description: {item['description'][:100]}...")
        print(f"   Tags: {', '.join(item['tags'][:5])}")
        print(f"   URL: {item['url']}")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    
    # Show some example queries
    print("\nTry these queries:")
    print('  python simple_test.py "advanced python programming"')
    print('  python simple_test.py "beginner tutorial"')
    print('  python simple_test.py "machine learning course"')
    print('  python simple_test.py "I want to learn JavaScript"')


if __name__ == "__main__":
    main()
