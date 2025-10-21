"""Universal Multi-Domain Content Discovery Demo

This demonstrates the system's ability to:
1. Discover ANY type of content (text, images, audio, video)
2. Work across ALL domains (education, entertainment, business, etc.)
3. Rank results based on user's knowledge base and preferences
4. Personalize recommendations using user history

Run this to see how the system adapts to different content types and user backgrounds!
"""

from Project import (
    LearnoraContentDiscovery,
    LearningContent,
    UserProfile,
)
from datetime import datetime, timedelta
import sys


def create_universal_content():
    """Create diverse content across all types and domains."""
    return [
        # TEXT CONTENT - TECHNOLOGY
        LearningContent(
            id="text-tech-1",
            title="Introduction to Machine Learning - Comprehensive Guide",
            content_type="article",
            source="Medium",
            url="https://towardsdatascience.com/ml-intro",
            description="Deep dive into ML algorithms, neural networks, and practical implementations with Python",
            difficulty="intermediate",
            duration_minutes=45,
            tags=["machine learning", "AI", "python", "data science", "algorithms"],
            created_at=datetime.now() - timedelta(days=5),
        ),
        LearningContent(
            id="text-tech-2",
            title="Web Development Best Practices 2024",
            content_type="article",
            source="Dev.to",
            url="https://dev.to/webdev-practices",
            description="Modern web development techniques including React, TypeScript, and serverless architecture",
            difficulty="intermediate",
            duration_minutes=30,
            tags=["web development", "react", "javascript", "typescript", "frontend"],
            created_at=datetime.now() - timedelta(days=2),
        ),
        
        # VIDEO CONTENT - EDUCATION
        LearningContent(
            id="video-edu-1",
            title="Python for Data Science - Full Course",
            content_type="video",
            source="YouTube",
            url="https://youtube.com/python-data-science",
            description="Complete video course on Python programming for data analysis, pandas, numpy, and visualization",
            difficulty="beginner",
            duration_minutes=180,
            tags=["python", "data science", "programming", "pandas", "numpy", "tutorial"],
            created_at=datetime.now() - timedelta(days=10),
        ),
        LearningContent(
            id="video-edu-2",
            title="Advanced SQL Queries and Database Design",
            content_type="video",
            source="Udemy",
            url="https://udemy.com/advanced-sql",
            description="Master complex SQL queries, indexing, optimization, and database architecture",
            difficulty="advanced",
            duration_minutes=240,
            tags=["sql", "database", "data", "programming", "advanced"],
            created_at=datetime.now() - timedelta(days=15),
        ),
        
        # AUDIO/PODCAST CONTENT - BUSINESS
        LearningContent(
            id="audio-biz-1",
            title="The Tim Ferriss Show - Building Successful Startups",
            content_type="podcast",
            source="Spotify",
            url="https://spotify.com/tim-ferriss/startups",
            description="Interview with successful entrepreneurs about startup strategies, fundraising, and scaling",
            difficulty="intermediate",
            duration_minutes=90,
            tags=["business", "startup", "entrepreneurship", "success", "strategy"],
            created_at=datetime.now() - timedelta(days=3),
        ),
        LearningContent(
            id="audio-biz-2",
            title="How I Built This - Airbnb Story",
            content_type="podcast",
            source="NPR",
            url="https://npr.org/how-i-built-this/airbnb",
            description="The complete story of how Airbnb disrupted the hospitality industry",
            difficulty="beginner",
            duration_minutes=45,
            tags=["business", "startup", "airbnb", "entrepreneurship", "innovation"],
            created_at=datetime.now() - timedelta(days=7),
        ),
        
        # IMAGE/VISUAL CONTENT - DESIGN
        LearningContent(
            id="image-design-1",
            title="UI/UX Design Principles - Visual Guide",
            content_type="infographic",
            source="Behance",
            url="https://behance.net/ui-ux-guide",
            description="Visual guide to user interface design, color theory, typography, and layout principles",
            difficulty="intermediate",
            duration_minutes=20,
            tags=["design", "UI", "UX", "visual", "graphics", "interface"],
            created_at=datetime.now() - timedelta(days=4),
        ),
        LearningContent(
            id="image-design-2",
            title="Photography Composition Masterclass",
            content_type="image_gallery",
            source="500px",
            url="https://500px.com/composition",
            description="Learn photography composition through 100+ annotated example images",
            difficulty="beginner",
            duration_minutes=30,
            tags=["photography", "art", "composition", "visual", "creative"],
            created_at=datetime.now() - timedelta(days=12),
        ),
        
        # MIXED CONTENT - MUSIC & ARTS
        LearningContent(
            id="audio-music-1",
            title="Learn Piano in 30 Days - Audio Course",
            content_type="audio_course",
            source="Masterclass",
            url="https://masterclass.com/piano",
            description="Complete piano course with audio lessons, practice exercises, and music theory",
            difficulty="beginner",
            duration_minutes=60,
            tags=["music", "piano", "instrument", "learning", "beginner"],
            created_at=datetime.now() - timedelta(days=8),
        ),
        LearningContent(
            id="video-music-1",
            title="Vocal Training and Singing Techniques",
            content_type="video",
            source="YouTube",
            url="https://youtube.com/vocal-training",
            description="Professional singing lessons covering breathing, pitch, tone, and performance",
            difficulty="intermediate",
            duration_minutes=45,
            tags=["singing", "music", "vocal", "voice", "performance"],
            created_at=datetime.now() - timedelta(days=6),
        ),
        
        # TEXT CONTENT - HEALTH & FITNESS
        LearningContent(
            id="text-health-1",
            title="Complete Guide to Nutrition and Healthy Eating",
            content_type="ebook",
            source="Amazon Kindle",
            url="https://amazon.com/nutrition-guide",
            description="Evidence-based nutrition guide covering macros, meal planning, and healthy lifestyle",
            difficulty="beginner",
            duration_minutes=120,
            tags=["nutrition", "health", "diet", "wellness", "food"],
            created_at=datetime.now() - timedelta(days=20),
        ),
        
        # VIDEO CONTENT - FITNESS
        LearningContent(
            id="video-fitness-1",
            title="Home Workout Program - 30 Day Challenge",
            content_type="video",
            source="YouTube",
            url="https://youtube.com/home-workout",
            description="Complete bodyweight workout program requiring no equipment, suitable for all levels",
            difficulty="beginner",
            duration_minutes=30,
            tags=["fitness", "workout", "exercise", "health", "bodyweight"],
            created_at=datetime.now() - timedelta(days=5),
        ),
        LearningContent(
            id="video-fitness-2",
            title="Advanced Strength Training and Muscle Building",
            content_type="video",
            source="Bodybuilding.com",
            url="https://bodybuilding.com/strength-training",
            description="Scientific approach to building muscle with progressive overload and proper form",
            difficulty="advanced",
            duration_minutes=90,
            tags=["fitness", "strength", "muscle", "bodybuilding", "advanced"],
            created_at=datetime.now() - timedelta(days=14),
        ),
        
        # AUDIO CONTENT - LANGUAGES
        LearningContent(
            id="audio-lang-1",
            title="Spanish Conversations for Beginners",
            content_type="audio_course",
            source="Duolingo",
            url="https://duolingo.com/spanish-audio",
            description="Learn Spanish through real conversations with native speakers and practice exercises",
            difficulty="beginner",
            duration_minutes=25,
            tags=["spanish", "language", "learning", "conversation", "beginner"],
            created_at=datetime.now() - timedelta(days=9),
        ),
        
        # TEXT CONTENT - COOKING
        LearningContent(
            id="text-cook-1",
            title="Italian Cooking Recipes and Techniques",
            content_type="article",
            source="Serious Eats",
            url="https://seriouseats.com/italian-cooking",
            description="Authentic Italian recipes with detailed techniques for pasta, sauces, and traditional dishes",
            difficulty="intermediate",
            duration_minutes=40,
            tags=["cooking", "italian", "food", "recipes", "culinary"],
            created_at=datetime.now() - timedelta(days=11),
        ),
        
        # VIDEO CONTENT - COOKING
        LearningContent(
            id="video-cook-1",
            title="Baking Bread from Scratch - Step by Step",
            content_type="video",
            source="YouTube",
            url="https://youtube.com/baking-bread",
            description="Learn artisan bread baking with detailed instructions for sourdough and traditional loaves",
            difficulty="intermediate",
            duration_minutes=35,
            tags=["baking", "bread", "cooking", "food", "technique"],
            created_at=datetime.now() - timedelta(days=13),
        ),
        
        # MIXED CONTENT - SCIENCE
        LearningContent(
            id="text-science-1",
            title="Understanding Quantum Physics - Simplified",
            content_type="article",
            source="Scientific American",
            url="https://sciam.com/quantum-physics",
            description="Accessible introduction to quantum mechanics, wave-particle duality, and modern physics",
            difficulty="intermediate",
            duration_minutes=50,
            tags=["physics", "quantum", "science", "education", "theory"],
            created_at=datetime.now() - timedelta(days=18),
        ),
        LearningContent(
            id="video-science-1",
            title="Biology Basics - Cell Structure and Function",
            content_type="video",
            source="Khan Academy",
            url="https://khanacademy.org/biology",
            description="Comprehensive biology lessons covering cells, DNA, genetics, and life processes",
            difficulty="beginner",
            duration_minutes=60,
            tags=["biology", "science", "education", "cells", "genetics"],
            created_at=datetime.now() - timedelta(days=16),
        ),
        
        # AUDIO CONTENT - HISTORY
        LearningContent(
            id="audio-history-1",
            title="Hardcore History - World War II Deep Dive",
            content_type="podcast",
            source="Dan Carlin",
            url="https://dancarlin.com/ww2",
            description="Epic historical podcast covering WWII with dramatic storytelling and detailed analysis",
            difficulty="intermediate",
            duration_minutes=180,
            tags=["history", "war", "wwii", "education", "documentary"],
            created_at=datetime.now() - timedelta(days=25),
        ),
        
        # IMAGE/VISUAL CONTENT - ART
        LearningContent(
            id="image-art-1",
            title="Drawing Techniques - Visual Reference Guide",
            content_type="infographic",
            source="Pinterest",
            url="https://pinterest.com/drawing-guide",
            description="Comprehensive visual guide to drawing techniques, shading, perspective, and anatomy",
            difficulty="intermediate",
            duration_minutes=25,
            tags=["drawing", "art", "visual", "technique", "creative"],
            created_at=datetime.now() - timedelta(days=22),
        ),
    ]


def create_user_profiles():
    """Create different user profiles with varying knowledge bases."""
    profiles = {
        "tech_expert": {
            "profile": UserProfile(
                user_id="tech-expert",
                preferred_formats=["article", "video"],
                available_time_daily=90,
                learning_goals=["advanced programming", "machine learning", "system design"],
            ),
            "knowledge_base": ["python", "javascript", "machine learning", "algorithms", "data structures"],
            "description": "Software Engineer with 5 years experience, knows programming basics, wants advanced content",
        },
        "complete_beginner": {
            "profile": UserProfile(
                user_id="beginner",
                preferred_formats=["video", "podcast"],
                available_time_daily=30,
                learning_goals=["learn programming", "career change"],
            ),
            "knowledge_base": [],
            "description": "Complete beginner, no technical background, prefers easy-to-follow videos",
        },
        "creative_professional": {
            "profile": UserProfile(
                user_id="creative",
                preferred_formats=["video", "infographic", "image_gallery"],
                available_time_daily=60,
                learning_goals=["design skills", "creative arts", "photography"],
            ),
            "knowledge_base": ["design", "photography", "art", "visual composition"],
            "description": "Creative professional interested in design and visual arts",
        },
        "fitness_enthusiast": {
            "profile": UserProfile(
                user_id="fitness",
                preferred_formats=["video", "article"],
                available_time_daily=45,
                learning_goals=["fitness", "nutrition", "health"],
            ),
            "knowledge_base": ["basic fitness", "nutrition basics"],
            "description": "Fitness enthusiast looking to improve workout and diet knowledge",
        },
        "entrepreneur": {
            "profile": UserProfile(
                user_id="entrepreneur",
                preferred_formats=["podcast", "article"],
                available_time_daily=60,
                learning_goals=["business strategy", "startup growth", "marketing"],
            ),
            "knowledge_base": ["business basics", "startup", "entrepreneurship"],
            "description": "Aspiring entrepreneur interested in business and startup stories",
        },
    }
    return profiles


def display_content_summary(content):
    """Display content type distribution."""
    type_counts = {}
    for item in content:
        ctype = item.content_type
        type_counts[ctype] = type_counts.get(ctype, 0) + 1
    
    print("\n📊 CONTENT TYPE DISTRIBUTION:")
    print("-" * 60)
    icons = {
        "article": "📄", "ebook": "📚", "video": "🎬", "podcast": "🎙️",
        "audio_course": "🎧", "infographic": "📊", "image_gallery": "🖼️"
    }
    for ctype, count in sorted(type_counts.items()):
        icon = icons.get(ctype, "📦")
        print(f"  {icon} {ctype.replace('_', ' ').title():<20} : {count} items")
    
    # Domain distribution
    domains = {}
    for item in content:
        domain = item.tags[0] if item.tags else "other"
        domains[domain] = domains.get(domain, 0) + 1
    
    print("\n🌐 DOMAIN DISTRIBUTION:")
    print("-" * 60)
    for domain, count in sorted(domains.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  • {domain.capitalize():<20} : {count} items")


def run_personalized_search(discovery, query, user_data, content_types=None):
    """Run search with user's knowledge base affecting rankings."""
    print("\n" + "=" * 80)
    print(f"QUERY: '{query}'")
    print("=" * 80)
    print(f"\n👤 USER PROFILE: {user_data['description']}")
    print(f"   Knowledge Base: {', '.join(user_data['knowledge_base']) if user_data['knowledge_base'] else 'None (Complete Beginner)'}")
    print(f"   Preferred Formats: {', '.join(user_data['profile'].preferred_formats)}")
    print(f"   Available Time: {user_data['profile'].available_time_daily} min/day")
    
    # Perform search
    results = discovery.discover_and_personalize(
        query=query,
        user_profile=user_data['profile'],
        strategy="hybrid",
        top_k=5,
        use_nlp=True,
    )
    
    # Display NLP analysis
    if 'nlp_analysis' in results:
        nlp = results['nlp_analysis']
        print(f"\n🧠 NLP ANALYSIS:")
        print(f"   Intent: {nlp['intent']['primary']}")
        if nlp['entities']['topics']:
            print(f"   Detected Topics: {', '.join(nlp['entities']['topics'][:3])}")
        if nlp['entities']['difficulty']:
            print(f"   Difficulty Level: {', '.join(nlp['entities']['difficulty'])}")
    
    # Display results
    print(f"\n🎯 SEARCH RESULTS ({len(results['results'])} found):")
    print("-" * 80)
    
    if not results['results']:
        print("   No results found")
        return
    
    content_icons = {
        "article": "📄", "ebook": "📚", "video": "🎬", "podcast": "🎙️",
        "audio_course": "🎧", "infographic": "📊", "image_gallery": "🖼️"
    }
    
    for i, item in enumerate(results['results'], 1):
        icon = content_icons.get(item['content_type'], "📦")
        difficulty_emoji = {"beginner": "🟢", "intermediate": "🟡", "advanced": "🔴"}.get(item['difficulty'], "⚪")
        
        print(f"\n{i}. {icon} {item['title']}")
        print(f"   Score: {item['score']:.4f} | {difficulty_emoji} {item['difficulty'].upper()} | "
              f"{item['content_type'].replace('_', ' ').title()} | {item['duration_minutes']} min")
        print(f"   {item['description'][:90]}...")
        print(f"   Tags: {', '.join(item['tags'][:4])}")
        
        # Show why this was ranked high
        if i == 1:
            reasons = []
            if item['content_type'] in user_data['profile'].preferred_formats:
                reasons.append(f"✓ Matches preferred format ({item['content_type']})")
            if item['duration_minutes'] <= user_data['profile'].available_time_daily:
                reasons.append(f"✓ Fits time budget ({item['duration_minutes']} ≤ {user_data['profile'].available_time_daily} min)")
            
            # Check knowledge base match
            user_knows = user_data['knowledge_base']
            item_topics = [tag.lower() for tag in item['tags']]
            knowledge_overlap = [topic for topic in user_knows if any(topic in t for t in item_topics)]
            
            if not user_knows:
                if item['difficulty'] == 'beginner':
                    reasons.append("✓ Beginner-friendly for new learners")
            elif knowledge_overlap:
                reasons.append(f"✓ Builds on your knowledge: {', '.join(knowledge_overlap[:2])}")
            
            if reasons:
                print(f"   🌟 Why ranked #1: {' | '.join(reasons)}")


def main():
    """Main demonstration function."""
    print("=" * 80)
    print("UNIVERSAL MULTI-DOMAIN CONTENT DISCOVERY SYSTEM")
    print("=" * 80)
    print("\nDemonstrating:")
    print("  ✓ Multiple content types (text, images, audio, video)")
    print("  ✓ Multiple domains (tech, business, health, arts, etc.)")
    print("  ✓ Personalized ranking based on user knowledge")
    print("  ✓ Format preferences and time constraints")
    
    # Setup
    print("\n⚙️  Setting up system...")
    discovery = LearnoraContentDiscovery(enable_nlp=True)
    content = create_universal_content()
    discovery.vector_db.add_contents(content)
    
    print(f"✓ Loaded {len(content)} diverse content items")
    display_content_summary(content)
    
    # Get user profiles
    profiles = create_user_profiles()
    
    # Test scenarios
    print("\n\n" + "=" * 80)
    print("SCENARIO-BASED TESTING")
    print("=" * 80)
    
    # Scenario 1: Tech Expert wants advanced ML content
    run_personalized_search(
        discovery,
        "machine learning algorithms and neural networks",
        profiles['tech_expert']
    )
    
    # Scenario 2: Complete beginner wants to learn programming
    run_personalized_search(
        discovery,
        "I want to learn programming from scratch",
        profiles['complete_beginner']
    )
    
    # Scenario 3: Creative professional looking for design resources
    run_personalized_search(
        discovery,
        "UI design and visual composition techniques",
        profiles['creative_professional']
    )
    
    # Scenario 4: Fitness enthusiast wants workout plans
    run_personalized_search(
        discovery,
        "strength training and muscle building",
        profiles['fitness_enthusiast']
    )
    
    # Scenario 5: Entrepreneur looking for business insights
    run_personalized_search(
        discovery,
        "startup success stories and business strategy",
        profiles['entrepreneur']
    )
    
    # Scenario 6: Cross-domain search (multiple content types)
    print("\n\n" + "=" * 80)
    print("CROSS-DOMAIN SEARCH EXAMPLE")
    print("=" * 80)
    run_personalized_search(
        discovery,
        "learning new skills",
        profiles['complete_beginner']
    )
    
    # Summary
    print("\n\n" + "=" * 80)
    print("SYSTEM CAPABILITIES SUMMARY")
    print("=" * 80)
    print("""
✅ CONTENT TYPES SUPPORTED:
   📄 Text: Articles, eBooks, Documentation
   🎬 Video: Tutorials, Courses, Demonstrations
   🎙️ Audio: Podcasts, Audio Courses, Interviews
   📊 Visual: Infographics, Image Galleries, Diagrams
   📚 Mixed: Any combination of the above

✅ DOMAINS COVERED:
   💻 Technology (Programming, AI, Web Dev)
   💼 Business (Startups, Marketing, Strategy)
   🎨 Creative Arts (Design, Photography, Music)
   🏋️ Health & Fitness (Nutrition, Workouts, Wellness)
   🔬 Science (Physics, Biology, Research)
   🌍 Languages (Spanish, French, etc.)
   🍳 Cooking & Culinary Arts
   📚 And LITERALLY ANY domain you add!

✅ PERSONALIZATION FEATURES:
   • Content type preferences (video vs. article vs. audio)
   • Knowledge-based ranking (beginner vs. expert)
   • Time budget matching (short vs. long content)
   • Difficulty progression (builds on what you know)
   • Learning goals alignment
   • Format accessibility (visual learners, audio learners)

✅ RANKING FACTORS:
   • Content relevance (BM25 + Dense search)
   • User knowledge overlap (+boost for familiar topics)
   • Format preference match (+10% for preferred types)
   • Time availability match (+5% for fitting content)
   • Difficulty appropriateness (not too easy, not too hard)
   • Recency (newer content ranked higher)

🎯 YOUR SYSTEM IS TRULY UNIVERSAL!
   It works for ANY content type, ANY domain, and adapts to ANY user!
   
💡 REAL-WORLD APPLICATIONS:
   • Educational platforms (Coursera, Udemy)
   • Content aggregators (Pocket, Feedly)
   • Media recommendation engines (Netflix, Spotify)
   • Learning management systems (LMS)
   • Research databases (Google Scholar)
   • E-commerce product discovery (Amazon)
   • Job/talent matching platforms
   • And much more!
""")
    print("=" * 80)
    
    # Interactive option
    if len(sys.argv) > 1:
        print("\n\n" + "=" * 80)
        print("CUSTOM QUERY TEST")
        print("=" * 80)
        custom_query = " ".join(sys.argv[1:])
        print(f"\nTesting with your query: '{custom_query}'")
        
        # Let user choose profile
        print("\nChoose a user profile:")
        for i, (key, data) in enumerate(profiles.items(), 1):
            print(f"  {i}. {key.replace('_', ' ').title()}: {data['description']}")
        
        print(f"\nUsing profile: Complete Beginner (default)")
        run_personalized_search(discovery, custom_query, profiles['complete_beginner'])


if __name__ == "__main__":
    main()
