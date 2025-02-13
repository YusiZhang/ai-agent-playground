from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import faiss
import numpy as np
import pickle
import os
import random
from sentence_transformers import SentenceTransformer
from enum import Enum

class Position(BaseModel):
    """Represents a job position"""
    title: str
    company: str
    industry: str
    start_year: int = Field(..., ge=1900, le=datetime.now().year)
    end_year: Optional[int] = Field(None, ge=1900, le=datetime.now().year)
    is_current: bool = False

    def __str__(self) -> str:
        return f"{self.title} at {self.company} ({self.industry})"

class Education(BaseModel):
    """Represents an educational qualification"""
    institution: str
    degree: str
    field_of_study: str
    graduation_year: int = Field(..., ge=1900, le=datetime.now().year)

    def __str__(self) -> str:
        return f"{self.degree} in {self.field_of_study} from {self.institution}"

class Certification(BaseModel):
    """Represents a professional certification"""
    name: str
    issuing_organization: str
    year_obtained: int = Field(..., ge=1900, le=datetime.now().year)
    expiry_year: Optional[int] = None

    def __str__(self) -> str:
        return f"{self.name} from {self.issuing_organization}"

class Profile(BaseModel):
    """Represents a complete member profile"""
    name: str
    current_position: Position
    previous_positions: List[Position] = []
    skills: List[str] = []
    certifications: List[Certification] = []
    education: List[Education] = []

    def to_embedding_text(self) -> str:
        """Convert profile to text for embedding generation"""
        components = [
            self.name,
            str(self.current_position),
            " ".join(self.skills),
            " ".join(str(edu) for edu in self.education),
            " ".join(str(cert) for cert in self.certifications)
        ]
        return " ".join(components)
    
class ProfileGenerator:
    """Utility class to generate random realistic profiles"""
    
    def __init__(self):
        self.first_names = [
            "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", 
            "Linda", "William", "Elizabeth", "David", "Barbara", "Richard", "Susan",
            "Wei", "Li", "Juan", "Maria", "Mohammed", "Fatima", "Yuki", "Haruka",
            "Alex", "Sam", "Jordan", "Taylor", "Morgan", "Casey", "Aiden", "Riley"
        ]
        
        self.last_names = [
            "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
            "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
            "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
            "Chen", "Wang", "Kim", "Lee", "Patel", "Singh", "Kumar", "Sato",
            "Thompson", "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis"
        ]
        
        self.companies = [
            "Tech Solutions Inc", "Global Systems", "Data Dynamics", "Cloud Corp",
            "Innovation Labs", "Digital Ventures", "Smart Technologies", "Future Systems",
            "Next Generation Tech", "Quantum Computing", "AI Research Ltd", "Data Science Co",
            "Web Solutions", "Mobile Innovations", "Security Systems Inc",
            "Cyber Defense", "Network Solutions", "Cloud Services Pro", "DevOps Masters",
            "Enterprise Solutions"
        ]
        
        self.industries = [
            "Software Development", "Cloud Computing", "Artificial Intelligence",
            "Cybersecurity", "E-commerce", "FinTech", "HealthTech", "EdTech",
            "Telecommunications", "Internet of Things", "Blockchain", "Big Data",
            "Digital Marketing", "Gaming", "Enterprise Software"
        ]
        
        self.skills = [
            # Programming Languages
            "Python", "Java", "JavaScript", "C++", "Go", "Rust", "TypeScript",
            "Ruby", "PHP", "Swift", "Kotlin", "Scala",
            
            # Web Technologies
            "React", "Angular", "Vue.js", "Node.js", "Django", "Flask",
            "Spring Boot", "GraphQL", "REST APIs", "HTML5", "CSS3",
            
            # Cloud & DevOps
            "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform",
            "Jenkins", "GitLab CI", "CircleCI",
            
            # Data & AI
            "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch",
            "Scikit-learn", "Data Science", "Natural Language Processing",
            "Computer Vision", "Big Data", "Spark", "Hadoop",
            
            # Databases
            "PostgreSQL", "MongoDB", "MySQL", "Redis", "Elasticsearch",
            "Cassandra", "DynamoDB",
            
            # Tools & Others
            "Git", "Linux", "Agile", "Scrum", "Microservices", "System Design",
            "CI/CD", "Test Driven Development"
        ]
        
        self.job_titles = [
            # Engineering
            "Software Engineer", "Senior Software Engineer", "Principal Engineer",
            "Full Stack Developer", "Backend Engineer", "Frontend Developer",
            "DevOps Engineer", "Site Reliability Engineer", "Cloud Architect",
            
            # Data & AI
            "Data Scientist", "Machine Learning Engineer", "AI Researcher",
            "Data Engineer", "Analytics Engineer", "Research Scientist",
            
            # Management & Leadership
            "Engineering Manager", "Technical Lead", "Project Manager",
            "Product Manager", "Scrum Master", "Technical Project Manager",
            
            # Specialized
            "Security Engineer", "Blockchain Developer", "iOS Developer",
            "Android Developer", "Database Administrator", "UX Engineer"
        ]
        
        self.universities = [
            "Tech University", "State College", "Global Institute", "City University",
            "National College", "International University", "Metropolitan University",
            "Pacific College", "Atlantic University", "Central State University",
            "Institute of Technology", "College of Engineering", "Digital University",
            "Science & Tech Institute", "Engineering Academy"
        ]
        
        self.degrees = [
            "BS", "BA", "MS", "PhD", "MBA", "BSc", "MSc", 
            "BEng", "MEng", "BTech", "MTech"
        ]
        
        self.fields = [
            "Computer Science", "Software Engineering", "Information Technology",
            "Data Science", "Artificial Intelligence", "Computer Engineering",
            "Information Systems", "Cybersecurity", "Robotics",
            "Electrical Engineering", "Mathematics", "Physics"
        ]
        
        self.certifications = [
            ("AWS Certified Solutions Architect", "Amazon Web Services"),
            ("Certified Kubernetes Administrator", "Cloud Native Computing Foundation"),
            ("Google Cloud Professional", "Google"),
            ("Azure Solutions Architect", "Microsoft"),
            ("Certified Information Systems Security Professional", "ISC2"),
            ("Professional Scrum Master", "Scrum.org"),
            ("Project Management Professional", "PMI"),
            ("Certified Data Scientist", "IBM"),
            ("TensorFlow Developer Certificate", "Google"),
            ("Certified Ethical Hacker", "EC-Council"),
            ("Oracle Certified Professional", "Oracle"),
            ("Red Hat Certified Engineer", "Red Hat")
        ]

    def generate_name(self) -> str:
        """Generate a random full name"""
        return f"{random.choice(self.first_names)} {random.choice(self.last_names)}"

    def generate_skills(self, min_skills: int = 3, max_skills: int = 8) -> List[str]:
        """Generate a random list of skills"""
        num_skills = random.randint(min_skills, max_skills)
        return random.sample(self.skills, num_skills)

    def generate_position(self, start_year: Optional[int] = None, is_current: bool = False) -> Position:
        """Generate a random position"""
        if not start_year:
            start_year = random.randint(2015, 2023)
            
        end_year = None if is_current else min(start_year + random.randint(1, 3), 2024)
        
        return Position(
            title=random.choice(self.job_titles),
            company=random.choice(self.companies),
            industry=random.choice(self.industries),
            start_year=start_year,
            end_year=end_year,
            is_current=is_current
        )

    def generate_education(self) -> Education:
        """Generate a random education entry"""
        return Education(
            institution=random.choice(self.universities),
            degree=random.choice(self.degrees),
            field_of_study=random.choice(self.fields),
            graduation_year=random.randint(2010, 2023)
        )

    def generate_certification(self) -> Certification:
        """Generate a random certification"""
        cert_name, org = random.choice(self.certifications)
        year_obtained = random.randint(2018, 2023)
        
        return Certification(
            name=cert_name,
            issuing_organization=org,
            year_obtained=year_obtained,
            expiry_year=year_obtained + random.randint(2, 4)
        )

    def generate_profile(self) -> Profile:
        """Generate a complete random profile"""
        # Generate current position first
        current_position = self.generate_position(is_current=True)
        
        # Generate 0-3 previous positions
        num_prev_positions = random.randint(0, 3)
        previous_positions = []
        last_year = current_position.start_year
        
        for _ in range(num_prev_positions):
            start_year = last_year - random.randint(2, 4)
            pos = self.generate_position(start_year=start_year)
            previous_positions.append(pos)
            last_year = start_year
        
        # Generate 1-2 education entries
        num_education = random.randint(1, 2)
        education = [self.generate_education() for _ in range(num_education)]
        
        # Generate 0-3 certifications
        num_certs = random.randint(0, 3)
        certifications = [self.generate_certification() for _ in range(num_certs)]
        
        return Profile(
            name=self.generate_name(),
            current_position=current_position,
            previous_positions=previous_positions,
            skills=self.generate_skills(),
            certifications=certifications,
            education=education
        )

class SearchType(Enum):
    """Types of semantic search available"""
    GENERAL = "general"
    SKILLS = "skills"
    POSITION = "position"
    EDUCATION = "education"

class ProfileStore:
    """
    A profile storage and search system using Faiss for vector search and pickle for persistence.
    """
    
    def __init__(self, save_dir: str = "profile_store"):
        """
        Initialize the profile store.
        
        Args:
            save_dir: Directory to save/load persistence files
        """
        self.save_dir = save_dir
        self.dimension = 384  # embedding dimension for 'all-MiniLM-L6-v2'
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Try to load existing data or initialize new
        if self._check_existing_data():
            self.load_state()
        else:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.profiles: Dict[int, Profile] = {}
            self.id_counter = 0
    
    def _check_existing_data(self) -> bool:
        """Check if saved data exists"""
        index_path = os.path.join(self.save_dir, "vectors.index")
        profiles_path = os.path.join(self.save_dir, "profiles.pkl")
        return os.path.exists(index_path) and os.path.exists(profiles_path)
    
    def save_state(self) -> None:
        """Save both the Faiss index and profile data"""
        # Save Faiss index
        index_path = os.path.join(self.save_dir, "vectors.index")
        faiss.write_index(self.index, index_path)
        
        # Save profile data and metadata
        profiles_path = os.path.join(self.save_dir, "profiles.pkl")
        metadata = {
            'profiles': self.profiles,
            'id_counter': self.id_counter,
            'last_saved': datetime.now(),
            'total_profiles': len(self.profiles)
        }
        with open(profiles_path, 'wb') as f:
            pickle.dump(metadata, f)
    
    def load_state(self) -> None:
        """Load both the Faiss index and profile data"""
        try:
            # Load Faiss index
            index_path = os.path.join(self.save_dir, "vectors.index")
            self.index = faiss.read_index(index_path)
            
            # Load profile data and metadata
            profiles_path = os.path.join(self.save_dir, "profiles.pkl")
            with open(profiles_path, 'rb') as f:
                metadata = pickle.load(f)
            
            self.profiles = metadata['profiles']
            self.id_counter = metadata['id_counter']
            
            print(f"Loaded {len(self.profiles)} profiles")
            print(f"Last saved: {metadata['last_saved']}")
            
        except Exception as e:
            print(f"Error loading saved state: {e}")
            self.index = faiss.IndexFlatIP(self.dimension)
            self.profiles = {}
            self.id_counter = 0
    
    def add_profile(self, profile: Profile) -> int:
        """
        Add a new profile to the store.
        
        Args:
            profile: Profile object to add
            
        Returns:
            int: ID of the added profile
        """
        # Generate embedding
        embedding = self.encoder.encode([profile.to_embedding_text()])[0]
        
        # Add to Faiss index
        self.index.add(np.array([embedding]).astype('float32'))
        
        # Store profile
        profile_id = self.id_counter
        self.profiles[profile_id] = profile
        self.id_counter += 1
        
        # Autosave after every 100 additions
        if self.id_counter % 100 == 0:
            self.save_state()
        
        return profile_id
    
    def search(self, 
              query: str, 
              search_type: SearchType = SearchType.GENERAL,
              k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for profiles using semantic search.
        
        Args:
            query: Search query
            search_type: Type of search to perform
            k: Number of results to return
            
        Returns:
            List of dictionaries containing profile and similarity score
        """
        # Generate query embedding
        query_embedding = self.encoder.encode([query])[0]
        
        # Search in Faiss
        distances, indices = self.index.search(
            np.array([query_embedding]).astype('float32'), k
        )
        
        # Process results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx != -1:
                profile = self.profiles[idx]
                result = {
                    'profile': profile,
                    'similarity_score': float(distance)
                }
                
                # Add search-type specific information
                if search_type == SearchType.SKILLS:
                    result['matching_skills'] = [
                        skill for skill in profile.skills 
                        if skill.lower() in query.lower()
                    ]
                elif search_type == SearchType.POSITION:
                    result['position_match'] = query.lower() in profile.current_position.title.lower()
                
                results.append(result)
        
        return results

def main():
    """Demo the profile store functionality with 1000 random profiles"""
    # Initialize store
    store = ProfileStore("demo_store")
    
    # Initialize profile generator
    generator = ProfileGenerator()
    
    # Check if the profiles and vectors files already exist
    profiles_path = os.path.join(store.save_dir, "profiles.pkl")
    vectors_path = os.path.join(store.save_dir, "vectors.index")
    
    if os.path.exists(profiles_path) and os.path.exists(vectors_path):
        print("Profiles and vectors already exist. Skipping generation of 1000 random profiles.")
    else:
        # Generate and add 1000 profiles
        print("Generating 1000 random profiles...")
        for i in range(1000):
            profile = generator.generate_profile()
            profile_id = store.add_profile(profile)
            
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1} profiles")
        
        # Save final state
        store.save_state()
        print("All profiles generated and saved.")
    
    # Perform some example searches
    print("\nPerforming example searches...")
    
    # Search for machine learning experts
    print("\n1. Search for 'machine learning experts':")
    results = store.search("machine learning expert", SearchType.GENERAL, k=5)
    for result in results:
        profile = result['profile']
        print(f"- {profile.name} (Score: {result['similarity_score']:.2f})")
        print(f"  Current: {profile.current_position}")
        print(f"  Skills: {', '.join(profile.skills)}")
        print()
    
    # Search for cloud architects
    print("\n2. Search for 'cloud architect with AWS experience':")
    results = store.search("cloud architect AWS", SearchType.SKILLS, k=5)
    for result in results:
        profile = result['profile']
        print(f"- {profile.name}")
        print(f"  Current: {profile.current_position}")
        print(f"  Matching skills: {', '.join(result.get('matching_skills', []))}")
        print()
    
    # Search for senior engineers
    print("\n3. Search for 'senior software engineer':")
    results = store.search("senior software engineer", SearchType.POSITION, k=5)
    for result in results:
        profile = result['profile']
        print(f"- {profile.name}")
        print(f"  Position: {profile.current_position}")
        print(f"  Skills: {', '.join(profile.skills)}")
        print()

if __name__ == "__main__":
    main()