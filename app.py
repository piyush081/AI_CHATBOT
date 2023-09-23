import requests
from bs4 import BeautifulSoup
from transformers import pipeline


def scrape_courses(url):
    
    sample_data = [
        {'title': 'Tell me about Python courses.', 'description': 'Learn Python programming basics.', 'url': 'https://brainlox.com/courses/fc29b015-962f-41fc-bc93-181d3ed87842'},
        {'title': 'Tell me about Java scripts Learning Fundamentals', 'description': 'Introduction to java scripts learning concepts.', 'url': 'https://brainlox.com/courses/fc9e2faf-dbe1-47bf-994c-f566a9ad3b42'},
        {'title': 'Tell me about Application Development', 'description': 'Build mobile application development.', 'url': 'https://brainlox.com/courses/2cf11f62-6452-41f1-9b42-303fb371b873'},
    ]
    
    return sample_data


def preprocess_data(courses):
    
    unique_courses = []
    seen_titles = set()
    
    for course in courses:
        title = course['title']
        if title not in seen_titles:
            unique_courses.append(course)
            seen_titles.add(title)
    
    return unique_courses


nlp = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")


def extract_recommendations(query, courses):
    
    recommendations = []
    for course in courses:
        if query.lower() in course['title'].lower() or query.lower() in course['description'].lower():
            recommendations.append(course)
    
    return recommendations


def chatbot(courses):
    print("Chatbot: Hi! I'm your course recommendation chatbot.")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye Piyush!")
            break
        
        recommendations = extract_recommendations(user_input, courses)
        
        if recommendations:
            print("Chatbot: Here are some relevant courses:")
            for i, course in enumerate(recommendations, start=1):
                print(f"{i}. {course['title']}: {course['description']}")
                print(f"   URL: {course['url']}")
        else:
            print("Chatbot: I couldn't find any relevant courses. Please try a different query.")

if __name__ == "__main__":
    
    sample_url = "https://brainlox.com/courses/category/technical"  
    courses = preprocess_data(scrape_courses(sample_url))
    
    
    chatbot(courses)
