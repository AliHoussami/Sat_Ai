from flask import Flask, request, jsonify
import mysql.connector
import requests
from flask_cors import CORS
import json
import re
import os
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from PIL import Image
import pytesseract
import matplotlib.pyplot as plt
import io
import base64
import fitz  # PyMuPDF
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import logging
import pickle
from datetime import datetime
import time
import hashlib



app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Alinx123@',  # Your actual MySQL password
    'database': 'sat_prep'
}


def get_db_connection():
    conn = mysql.connector.connect(**db_config)
    return conn, conn.cursor(dictionary=True)


SAT_MATERIALS = {


    'practice_test': 'sat-practice-test-7-digital.pdf',
    'answers': 'sat-practice-test-7-answers-digital.pdf',
    'scoring': 'scoring-sat-practice-test-7-digital.pdf'
}


@app.route('/register', methods=['POST'])
def register():
    """Register a new user"""
    data = request.json
    username = data.get('username', '')
    password = data.get('password', '')
    
    
    if not username or not password:
        return jsonify({'error': 'Username and password are required'}), 400
    
    if len(username) < 3:
        return jsonify({'error': 'Username must be at least 3 characters long'}), 400
    
    if len(password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters long'}), 400
    
    
    if not re.match(r'^[a-zA-Z0-9_]+$', username):
        return jsonify({'error': 'Username can only contain letters, numbers, and underscores'}), 400
    
    try:
        conn, cursor = get_db_connection()
        
        
        cursor.execute("SELECT user_id FROM users WHERE username = %s", (username,))
        existing_user = cursor.fetchone()
        
        if existing_user:
            cursor.close()
            conn.close()
            return jsonify({'error': 'Username already exists'}), 400
        
        
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        
        
        cursor.execute(
            "INSERT INTO users (username, password) VALUES (%s, %s)",
            (username, hashed_password)
        )
        
        
        user_id = cursor.lastrowid
        
        
        cursor.execute(
            "UPDATE users SET weak_areas = '' WHERE user_id = %s",
            (user_id,)
        )
        
        conn.commit()
        cursor.close()
        conn.close()
        
        
        return jsonify({
            'success': True,
            'user_id': user_id,
            'username': username
        })
        
    except Exception as e:
        logger.error(f"Error in user registration: {str(e)}")
        return jsonify({'error': 'Registration failed'}), 500

@app.route('/login', methods=['POST'])
def login():
    """Log in an existing user"""
    data = request.json
    username = data.get('username', '')
    password = data.get('password', '')
    
    
    if not username or not password:
        return jsonify({'error': 'Username and password are required'}), 400
    
    try:
        conn, cursor = get_db_connection()
        
        
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        
        
        cursor.execute(
            "SELECT user_id, username, weak_areas FROM users WHERE username = %s AND password = %s",
            (username, hashed_password)
        )
        
        user = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if not user:
            return jsonify({'error': 'Invalid username or password'}), 401
        
        
        return jsonify({
            'success': True,
            'user_id': user['user_id'],
            'username': user['username'],
            'weak_areas': user['weak_areas'].split(',') if user['weak_areas'] else []
        })
        
    except Exception as e:
        logger.error(f"Error in user login: {str(e)}")
        return jsonify({'error': 'Login failed'}), 500


class VectorDatabase:
    def __init__(self):
        self.documents = []
        self.embeddings = None
        self.document_type = []
        self.section_info = []
        
        self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
        logger.info("Initialized VectorDatabase with all-mpnet-base-v2 model")
        
    def add_document(self, text, doc_type="general", section=""):
        """Add a document to the vector database with proper chunking"""
        if not text or len(text.strip()) == 0:
            logger.warning(f"Skipping empty document with type={doc_type}, section={section}")
            return
            
        
        chunks = self.chunk_sat_content(text, doc_type, section)
        
        original_doc_count = len(self.documents)
        for chunk in chunks:
            self.documents.append(chunk["text"])
            self.document_type.append(chunk["metadata"]["doc_type"])
            self.section_info.append(chunk["metadata"]["section"])
        
        new_doc_count = len(self.documents)
        logger.info(f"Added {new_doc_count - original_doc_count} chunks from document")
        
        
        self._update_embeddings()
    
    def chunk_sat_content(self, text, doc_type, section, max_chunk_size=256, overlap=50):
        """
        Specialized chunking function for SAT PDFs that respects document structure
        and creates semantically meaningful chunks with rich metadata
        """
        chunks = []
        
        
        test_num = "unknown"
        section_type = "general"
        
        
        test_match = re.search(r"Test\s+(\d+)", section)
        if test_match:
            test_num = test_match.group(1)
        
        
        if "Math" in section:
            section_type = "Math"
        elif "Reading" in section or "Writing" in section:
            section_type = "Reading and Writing"
        
        
        if doc_type == "practice_test" and "QUESTION" in text:
            
            q_chunks = re.split(r"(QUESTION\s+\d+)", text)
            
            
            if q_chunks and not q_chunks[0].strip():
                q_chunks = q_chunks[1:]
                
            
            for i in range(0, len(q_chunks)-1, 2):
                if i+1 < len(q_chunks):
                    q_header = q_chunks[i]
                    q_content = q_chunks[i+1]
                    
                    
                    q_num_match = re.search(r"QUESTION\s+(\d+)", q_header)
                    q_num = q_num_match.group(1) if q_num_match else "unknown"
                    
                    
                    enhanced_chunk = f"""
SAT Practice Test {test_num} - {section}
Test Number: {test_num}
Question Number: {q_num}
Document Type: {doc_type}
Section Type: {section_type}

{q_header}{q_content}
"""
                    
                    chunks.append({
                        "text": enhanced_chunk.strip(),
                        "metadata": {
                            "test_num": test_num,
                            "section": section,
                            "question": q_num,
                            "doc_type": doc_type,
                            "section_type": section_type
                        }
                    })
                    
            
            if chunks:
                return chunks
        
        
        if doc_type == "answers" and "QUESTION" in text:
            
            explanation_chunks = re.split(r"(QUESTION\s+\d+)", text)
            
            
            if explanation_chunks and not explanation_chunks[0].strip():
                explanation_chunks = explanation_chunks[1:]
            
            
            for i in range(0, len(explanation_chunks)-1, 2):
                if i+1 < len(explanation_chunks):
                    explanation_header = explanation_chunks[i]
                    explanation_content = explanation_chunks[i+1]
                    
                    
                    if len(explanation_content.strip()) > 50:
                        
                        q_num_match = re.search(r"QUESTION\s+(\d+)", explanation_header)
                        q_num = q_num_match.group(1) if q_num_match else "unknown"
                        
                        
                        enhanced_chunk = f"""
SAT Practice Test {test_num} - Answer Explanation
Test Number: {test_num}
Question Number: {q_num}
Document Type: Answer Explanation
Section Type: {section_type}

{explanation_header}{explanation_content}
"""
                        
                        chunks.append({
                            "text": enhanced_chunk.strip(),
                            "metadata": {
                                "test_num": test_num,
                                "section": section,
                                "question": q_num,
                                "doc_type": "answers",
                                "section_type": section_type
                            }
                        })
            
            
            if chunks:
                return chunks
        
        
        paragraphs = re.split(r"\n\s*\n", text)
        
        current_chunk = []
        current_chunk_size = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            paragraph_size = len(paragraph)
            
            
            if current_chunk_size + paragraph_size > max_chunk_size and current_chunk:
                
                enhanced_chunk = f"""
SAT Practice Test {test_num} - {section}
Test Number: {test_num}
Section: {section}
Document Type: {doc_type}
Section Type: {section_type}

{"".join(current_chunk)}
"""
                
                chunks.append({
                    "text": enhanced_chunk.strip(),
                    "metadata": {
                        "test_num": test_num,
                        "section": section,
                        "doc_type": doc_type,
                        "section_type": section_type
                    }
                })
                
                
                if len(current_chunk) > 2 and overlap > 0:
                    current_chunk = current_chunk[-2:]
                    current_chunk_size = sum(len(p) for p in current_chunk)
                else:
                    current_chunk = []
                    current_chunk_size = 0
            
            
            current_chunk.append(paragraph + "\n\n")
            current_chunk_size += paragraph_size
        
        
        if current_chunk:
            
            enhanced_chunk = f"""
SAT Practice Test {test_num} - {section}
Test Number: {test_num}
Section: {section}
Document Type: {doc_type}
Section Type: {section_type}

{"".join(current_chunk)}
"""
            
            chunks.append({
                "text": enhanced_chunk.strip(),
                "metadata": {
                    "test_num": test_num,
                    "section": section,
                    "doc_type": doc_type,
                    "section_type": section_type
                }
            })
        
        return chunks
    
    def _update_embeddings(self):
        """Create or update embeddings for all documents"""
        if not self.documents:
            logger.warning("No documents to embed")
            return
            
        logger.info(f"Updating embeddings for {len(self.documents)} documents")
        try:
            self.embeddings = self.embedding_model.encode(self.documents)
            logger.info(f"Embeddings updated successfully with shape: {self.embeddings.shape if self.embeddings is not None else 'None'}")
        except Exception as e:
            logger.error(f"Error updating embeddings: {str(e)}")
    
    def search(self, query, top_k=3, doc_type=None, section=None):
        """Combined vector and keyword-based search (hybrid approach)"""
        if not self.documents or self.embeddings is None:
            logger.warning("Vector database is empty, no documents to search")
            return []
            
        logger.info(f"Searching for query: '{query}', type={doc_type}, section={section}")
        
        
        exact_matches = []
        query_lower = query.lower().strip()
        query_keywords = query_lower.split()
        
        
        for i, doc in enumerate(self.documents):
            
            if doc_type and not (doc_type.lower() in self.document_type[i].lower()):
                continue
            
            
            if section and section not in self.section_info[i]:
                continue
            
            doc_lower = doc.lower()
            
            
            if query_lower == doc_lower:
                exact_matches.append({
                    "text": doc,
                    "similarity": 1.0,
                    "type": self.document_type[i],
                    "section": self.section_info[i],
                    "match_type": "exact_equality"
                })
                continue
            
            
            if query_lower in doc_lower:
                exact_matches.append({
                    "text": doc,
                    "similarity": 0.98,
                    "type": self.document_type[i],
                    "section": self.section_info[i],
                    "match_type": "substring_match"
                })
                continue
                
            
            if all(keyword in doc_lower for keyword in query_keywords):
                exact_matches.append({
                    "text": doc,
                    "similarity": 0.95,
                    "type": self.document_type[i],
                    "section": self.section_info[i],
                    "match_type": "keywords_match"
                })
                continue
        
        
        query_embedding = self.embedding_model.encode([query])[0]
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        
        vector_results = []
        for i, sim in enumerate(similarities):
            # Skip if doc_type filter doesn't match
            if doc_type and not (doc_type.lower() in self.document_type[i].lower()):
                continue
            
            # Skip if section filter doesn't match
            if section and section not in self.section_info[i]:
                continue
                
            vector_results.append({
                "text": self.documents[i],
                "similarity": float(sim),
                "type": self.document_type[i],
                "section": self.section_info[i],
                "match_type": "vector"
            })
        
        
        vector_results.sort(key=lambda x: x["similarity"], reverse=True)
        
        
        all_results = exact_matches + vector_results
        
        
        seen_texts = set()
        unique_results = []
        for result in all_results:
            
            text_sig = result["text"][:100]
            if text_sig not in seen_texts:
                seen_texts.add(text_sig)
                unique_results.append(result)
                if len(unique_results) >= top_k:
                    break
        
        
        if unique_results:
            logger.info(f"Top search result similarity: {unique_results[0]['similarity']:.4f}")
            logger.info(f"Top result match type: {unique_results[0].get('match_type', 'unknown')}")
            logger.info(f"Top result preview: {unique_results[0]['text'][:50]}...")
        else:
            logger.warning(f"No results found for query: {query}")
            
        return unique_results

    def add_exact_match_documents(self):
        """Add exact match documents for key search terms"""
        logger.info("Adding exact match documents for key search terms")
        
        
        test_numbers = ['4', '5', '6', '7', '8', '9', '10']
        
        
        self.add_document(
            f"""SAT MATH SECTION SAT MATH SECTION SAT MATH SECTION
                This document contains the exact search term: SAT MATH SECTION
                The SAT Math section includes algebra, data analysis, problem-solving, and advanced math concepts.
                EXACT SEARCH TERM: SAT MATH SECTION""", 
            doc_type="practice_test", 
            section="Math"
        )
        
        self.add_document(
            f"""READING PASSAGES SAT READING PASSAGES SAT READING PASSAGES SAT
                This document contains the exact search term: READING PASSAGES SAT
                The SAT Reading section includes passages from literature, science, history, and social studies.
                EXACT SEARCH TERM: READING PASSAGES SAT""", 
            doc_type="practice_test", 
            section="Reading and Writing"
        )
        
        
        for test_num in test_numbers:
            
            self.add_document(
                f"""Test {test_num} math Test {test_num} math Test {test_num} math
                    This document contains the exact search term: Test {test_num} math
                    This refers to the math section of SAT Practice Test {test_num}.
                    EXACT SEARCH TERM: Test {test_num} math""", 
                doc_type="practice_test", 
                section=f"Test {test_num} - Math"
            )
            
            
            self.add_document(
                f"""Test {test_num} reading Test {test_num} reading Test {test_num} reading
                    This document contains the exact search term: Test {test_num} reading
                    This refers to the reading and writing section of SAT Practice Test {test_num}.
                    EXACT SEARCH TERM: Test {test_num} reading""", 
                doc_type="practice_test", 
                section=f"Test {test_num} - Reading and Writing"
            )
        
        logger.info(f"Added {2 + 2*len(test_numbers)} exact match documents")
        
    def save(self, directory="rag_data"):
        """Save the vector database to disk"""
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")
        
        # Save documents and metadata
        with open(os.path.join(directory, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)
        
        with open(os.path.join(directory, "document_types.pkl"), "wb") as f:
            pickle.dump(self.document_type, f)
        
        with open(os.path.join(directory, "section_info.pkl"), "wb") as f:
            pickle.dump(self.section_info, f)
        
        # Save embeddings
        with open(os.path.join(directory, "embeddings.pkl"), "wb") as f:
            pickle.dump(self.embeddings, f)
        
        logger.info(f"Vector database saved to {directory} ({len(self.documents)} documents)")

def load_vector_database(directory="rag_data"):
    """Load the vector database from disk"""
    if not os.path.exists(directory):
        logger.warning(f"No saved vector database found in {directory}")
        return None
    
    try:
        # Create an empty database
        db = VectorDatabase()
        
        # Load documents and metadata
        with open(os.path.join(directory, "documents.pkl"), "rb") as f:
            db.documents = pickle.load(f)
        
        with open(os.path.join(directory, "document_types.pkl"), "rb") as f:
            db.document_type = pickle.load(f)
        
        with open(os.path.join(directory, "section_info.pkl"), "rb") as f:
            db.section_info = pickle.load(f)
        
        # Load embeddings
        with open(os.path.join(directory, "embeddings.pkl"), "rb") as f:
            db.embeddings = pickle.load(f)
        
        logger.info(f"Vector database loaded from {directory} ({len(db.documents)} documents)")
        return db
    except Exception as e:
        logger.error(f"Error loading vector database: {str(e)}")
        return None

# Extract text from PDF with improved preprocessing
def extract_text_from_pdf(pdf_path):
    """Extract text and identify sections from PDF with improved preprocessing"""
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        return []
    
    sections = []
    try:
        doc = fitz.open(pdf_path)
        logger.info(f"Opened PDF: {pdf_path} with {len(doc)} pages")
        
        for page_num, page in enumerate(doc):
            text = page.get_text()
            
            # Clean the text
            text = re.sub(r'SAT Practice Test #\d+', '', text)
            text = re.sub(r'©\s*\d+\s*College Board', '', text)
            text = re.sub(r'Page \d+ of \d+', '', text)
            
            # Try to identify section headers
            lines = text.split('\n')
            current_section = f"Page {page_num+1}"
            
            # Look for section markers
            for line in lines:
                if re.match(r'^Module\s+\d+', line):
                    current_section = line.strip()
                    logger.debug(f"Found section marker: {current_section}")
                elif "QUESTION" in line and re.search(r'\d+', line):
                    current_section = line.strip()
                elif "Reading and Writing" in line:
                    current_section = "Reading and Writing"
                elif "Math" in line and not "Mathematics" in current_section:
                    current_section = "Math"
            
            sections.append({
                "text": text,
                "section": current_section,
                "page": page_num + 1
            })
            
            logger.debug(f"Extracted page {page_num+1}, section: {current_section}, text length: {len(text)}")
        
        logger.info(f"Extracted {len(sections)} sections from PDF {pdf_path}")
        return sections
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return []

# Extract tables from PDF
def extract_tables_from_pdf(pdf_path):
    """Extract tables from PDF using visual detection"""
    if not os.path.exists(pdf_path):
        return []
    
    tables = []
    try:
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc):
            # Get page text
            text = page.get_text()
            
            # Look for table indicators
            if any(indicator in text for indicator in ["Table", "table", "Column", "column", "Row", "row"]):
                # Extract a screenshot of this page
                pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                img_data = pix.tobytes("png")
                
                tables.append({
                    "page": page_num + 1,
                    "image_data": base64.b64encode(img_data).decode('utf-8'),
                    "text": text
                })
        
        return tables
    except Exception as e:
        logger.error(f"Error extracting tables from PDF: {e}")
        return []

# Extract charts and images from PDF
def extract_charts_from_pdf(pdf_path):
    """Extract charts and images from PDF"""
    if not os.path.exists(pdf_path):
        return []
    
    charts = []
    try:
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc):
            # Get images from page
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Store the image
                charts.append({
                    "page": page_num + 1,
                    "image_index": img_index,
                    "image_data": base64.b64encode(image_bytes).decode('utf-8')
                })
        
        return charts
    except Exception as e:
        logger.error(f"Error extracting charts from PDF: {e}")
        return []

# Function to generate context for queries
def generate_context_for_query(query, question_type=None):
    """Generate context for the query based on its content using hybrid search"""
    context = ""
    
    # Add SAT-specific terms to the query based on question type
    enhanced_query = query
    if "sat" not in query.lower():
        if question_type == "math":
            enhanced_query = f"SAT Math section {query}"
        elif question_type == "reading_writing":
            enhanced_query = f"SAT Reading and Writing section {query}"
        elif question_type == "scoring":
            enhanced_query = f"SAT score {query}"
        else:
            enhanced_query = f"SAT {query}"
    
    logger.info(f"Enhanced query: {enhanced_query}")
    
    # Try general search first without filtering (increased top_k for more options)
    general_results = vector_db.search(enhanced_query, top_k=5)
    
    # Filter for only HIGHLY relevant documents (threshold of 0.7)
    filtered_results = [r for r in general_results if r["similarity"] > 0.7]
    
    # If we have at least one good result, use only those
    if filtered_results:
        # Get diverse results from different tests if possible
        test_results = {}
        for result in filtered_results:
            test_identifier = result["section"].split('-')[0].strip() if '-' in result["section"] else 'Test 7'
            if test_identifier not in test_results:
                test_results[test_identifier] = result
            
            # Limit to 3 different tests
            if len(test_results) >= 3:
                break
        
        results = list(test_results.values())
        logger.info(f"Using {len(results)} highly relevant documents from different practice tests")
    # If general search found good results, use them    
    elif general_results and len(general_results) > 0 and general_results[0]["similarity"] > 0.7:
        logger.info(f"Using best search result. Found {len(general_results)} documents")
        results = [general_results[0]]  # Just use the single best result
    else:
        # Otherwise, try type-specific search
        logger.info(f"General search insufficient, trying type-specific search")
        if question_type == "math":
            math_results = vector_db.search(enhanced_query, top_k=3, doc_type="practice_test")
            answer_results = vector_db.search(enhanced_query, top_k=2, doc_type="answers")
            # Filter to most relevant results only
            filtered_math = [r for r in math_results if r["similarity"] > 0.7][:2]
            filtered_answers = [r for r in answer_results if r["similarity"] > 0.7][:1]
            results = filtered_math + filtered_answers
        elif question_type == "reading_writing":
            reading_results = vector_db.search(enhanced_query, top_k=3, doc_type="practice_test")
            answer_results = vector_db.search(enhanced_query, top_k=2, doc_type="answers")
            # Filter to most relevant results only
            filtered_reading = [r for r in reading_results if r["similarity"] > 0.7][:2]
            filtered_answers = [r for r in answer_results if r["similarity"] > 0.7][:1]
            results = filtered_reading + filtered_answers
        elif question_type == "scoring":
            results = vector_db.search(enhanced_query, top_k=3, doc_type="scoring")
            # Filter to most relevant results only
            results = [r for r in results if r["similarity"] > 0.7][:2]
        else:
            # If no specific type, try all document types with different queries
            practice_results = vector_db.search(f"SAT practice {query}", top_k=2)
            answer_results = vector_db.search(f"SAT answer {query}", top_k=2)
            scoring_results = vector_db.search(f"SAT scoring {query}", top_k=1)
            
            # Combine and sort all results
            all_results = practice_results + answer_results + scoring_results
            all_results.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Filter to most relevant results only
            results = [r for r in all_results if r["similarity"] > 0.7][:3]
    
    # If we still don't have good results, add some general SAT information
    if not results or len(results) == 0:
        logger.warning(f"No search results found for query: {enhanced_query}")
        # Add some general SAT info as fallback
        general_info = [
            {"text": "The SAT is a standardized test widely used for college admissions in the United States. It has sections on Reading, Writing, and Mathematics."},
            {"text": "The SAT includes multiple-choice questions and student-produced responses in the Math section."},
            {"text": "The Reading and Writing section tests comprehension, analysis, and grammar skills."},
            {"text": "Practice tests 4 through 10 are available to help students prepare for the actual SAT exam."}
        ]
        for info in general_info:
            context += info["text"] + "\n\n"
    else:
        # Add results to context, but limit to just the most relevant ones
        for result in results:
            context += result["text"] + "\n\n"
        
        # Log what we found
        logger.info(f"Found {len(results)} relevant documents with similarities: {[round(r['similarity'], 2) for r in results]}")
    
    return context

# DeepSeek R1 API function with RAG capabilities
def ask_deepseek_with_rag(question, question_type=None, additional_context=""):
    import requests
    import re
    start_time = time.time()
    logger.info(f"Processing question: {question}, type: {question_type}")
    
    # Get context from RAG system
    context = generate_context_for_query(question, question_type)
    
    # Add any additional context provided
    if additional_context:
        context += "\n" + additional_context
    
    # Ollama API endpoint
    url = "http://localhost:11434/api/generate"
    
    # Check if this is a math question
    is_math = '=' in question or 'f(' in question or re.search(r'[x-zX-Z]²', question)
    
    # Create a prompt with context
    if is_math:
        # Format prompt to enable proper LaTeX formatting
        prompt = f"""You are an SAT math tutor. For the question: {question}

Here is additional context to help provide a more accurate answer:
{context}

Solve using proper LaTeX formatting with these STRICT rules:
1. ALL LaTeX expressions MUST be wrapped in $ $ for inline math or $$ $$ for displayed equations.
2. Never use raw LaTeX commands outside of math delimiters.
3. Always place \\textbf, \\boxed, \\frac, and all other LaTeX commands inside $ $ delimiters.

Format your answer as follows:

Solve the equation:
$$[Equation with LaTeX notation]$$

Step 1: [Describe first step]
$$[Step 1 calculation with LaTeX notation]$$

Step 2: [Describe second step]
$$[Step 2 calculation with LaTeX notation]$$

Final Answer:
$$\\boxed{{[Answer with LaTeX notation]}}$$"""
    else:
        prompt = f"""You are an SAT preparation tutor. Answer the following question:

Question: {question}

Here is additional context from official SAT materials to help provide a more accurate answer:
{context}

Provide a clear, structured answer explaining the reasoning behind your response. 
Use specific details from the official SAT materials when possible.
If the question involves charts or tables, describe them and explain their relevance to your answer.
Focus on being accurate and helpful for students preparing for the SAT."""
    
    # Speed-optimized configuration
    data = {
        "model": "deepseek-r1:1.5b",
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_gpu": 1,
            "temperature": 0.3,
            "top_p": 0.9,
            "top_k": 40,
            "gpu_layers": -1,
            "f16_kv": True,
            "num_ctx": 4096,  # Increased context window
            "batch_size": 16
        }
    }
    
    try:
        logger.info("Sending request to DeepSeek model")
        t = time.time()
        logger.info("Loading model")
        logger.info(t)
        response = requests.post(url, json=data)      
        logger.info("finished loading model")
        logger.info(time.time() - t)
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get("response", "Error connecting to AI model.")
            
            # Clean up any "think" sections if they appear
            if "<think>" in answer or "think" in answer.lower():
                answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)
                answer = re.sub(r'think.*?/think', '', answer, flags=re.DOTALL)
            
            answer = re.sub(r'\*\*(Step\s+\d+:\s*)\*\*', r'\1', answer)
            answer = re.sub(r'\*\*(Final\s+Answer:\s*)\*\*', r'\1', answer, flags=re.IGNORECASE)
            answer = re.sub(r'\*\*(Answer:)\*\*', r'\1', answer)

            if is_math:
                answer = re.sub(r'\$\$?\s*\\boxed\{\s*\}\s*\$\$?', '', answer)
                answer = answer.replace('\n\n', '\n')
            
            
            logger.info(f"Generated response of length {len(answer)}")
            end_time = time.time()
            print(f"TIMING: Function ask_deepseek_with_rag took {end_time - start_time:.2f} seconds")
            return answer
        else:
            logger.error(f"Error from DeepSeek API: {response.status_code}")
            return f"Error connecting to AI model. Status code: {response.status_code}"
    except Exception as e:
        logger.error(f"Exception in ask_deepseek_with_rag: {str(e)}")
        return f"Error connecting to AI model: {str(e)}"

# Handle chart questions
def handle_chart_question(question, chart_type=None):
    """Special handling for questions involving charts or graphs"""
    logger.info(f"Handling chart question: {question}")
    
    # Extract and analyze charts if needed
    charts = []
    
    # Find relevant charts based on the question
    try:
        if "graph" in question.lower() or "chart" in question.lower() or "table" in question.lower():
            # Check all practice tests for charts
            for key, path in SAT_MATERIALS.items():
                if 'practice_test' in key:
                    test_charts = extract_charts_from_pdf(path)
                    charts.extend(test_charts)
                    logger.info(f"Extracted {len(test_charts)} potential charts from {key}")
    except Exception as e:
        logger.error(f"Error extracting charts: {str(e)}")
    
    # Generate analysis of charts
    chart_context = ""
    if charts:
        chart_context = "The question involves chart(s)/graph(s). Here's what I can identify:\n"
        for i, chart in enumerate(charts[:3]):  # Limit to first 3 charts
            page_num = chart.get('page', 'unknown')
            
            # Basic chart detection - cannot do proper analysis without image processing
            chart_context += f"- Chart/Figure on page {page_num}. "
            chart_context += "Without advanced image analysis, I cannot describe the specific contents, "
            chart_context += "but SAT charts typically present data for interpretation and analysis.\n"
    else:
        chart_context = "No specific charts were automatically detected in the materials, but I can still provide general guidance on SAT charts and tables."
    
    # Enhanced query for chart questions
    enhanced_question = question
    if chart_context:
        enhanced_question = f"{question}\n\n{chart_context}"
        
    logger.info(f"Enhanced chart question: {enhanced_question}")
    
    # Add special context for chart questions based on known SAT content
    additional_context = """
SAT charts and graphs test data interpretation skills. Common types include:
1. Bar charts - comparing quantities across categories
2. Line graphs - showing trends over time
3. Scatterplots - showing relationships between variables
4. Tables - organized data in rows and columns
5. Circle/pie charts - showing proportions of a whole

Questions typically ask about trends, comparisons, extremes (max/min), calculations from data,
or drawing conclusions from the presented information.
"""
    
    # Process with DeepSeek using RAG
    return ask_deepseek_with_rag(enhanced_question, question_type=chart_type, additional_context=additional_context)

# Initialize vector database with SAT materials
def initialize_vector_database():
    """Initialize the vector database with content from SAT materials"""
    logger.info("Initializing vector database with SAT materials...")
    
    # Process all practice tests dynamically
    for key, path in SAT_MATERIALS.items():
        try:
            # Extract test number and type
            test_type = None
            test_num = None
            
            if key.startswith('practice_test'):
                test_type = 'practice_test'
                if key == 'practice_test':
                    test_num = '7'  # Default for the original practice test
                else:
                    test_num = key.split('_')[-1]
            elif key.startswith('answers'):
                test_type = 'answers'
                if key == 'answers':
                    test_num = '7'  # Default for the original answers
                else:
                    test_num = key.split('_')[-1]
            elif key.startswith('scoring'):
                test_type = 'scoring'
                if key == 'scoring':
                    test_num = '7'  # Default for the original scoring
                else:
                    test_num = key.split('_')[-1]
            
            if test_type and path and os.path.exists(path):
                logger.info(f"Processing {test_type} {test_num} from {path}")
                sections = extract_text_from_pdf(path)
                logger.info(f"Extracted {len(sections)} sections from {test_type} {test_num}")
                
                for section in sections:
                    # Properly format the section label to include test number
                    section_label = f"Test {test_num} - {section['section']}"
                    logger.info(f"Adding section: {section_label}")
                    
                    # Enhancement: Add test number to the beginning of the text
                    enhanced_text = f"SAT Practice Test {test_num}: {section['text']}"
                    
                    vector_db.add_document(
                        enhanced_text, 
                        doc_type=test_type, 
                        section=section_label
                    )
                    
        except Exception as e:
            logger.error(f"Error processing {key}: {str(e)}")
    
    # Add exact match documents for better search
    vector_db.add_exact_match_documents()
    
    logger.info(f"Vector database initialized with {len(vector_db.documents)} documents")

@app.route('/chart_diagnostic', methods=['GET'])
def chart_diagnostic():
    """Run a diagnostic on chart extraction capabilities"""
    results = {
        "status": "running",
        "charts_found": [],
        "extraction_details": {}
    }
    
    try:
        # Extract charts from all practice tests
        all_charts = []
        chart_counts = {}
        
        for key, path in SAT_MATERIALS.items():
            if 'practice_test' in key:
                test_charts = extract_charts_from_pdf(path)
                all_charts.extend([(key, chart) for chart in test_charts])
                chart_counts[key] = len(test_charts)
        
        # Basic stats
        results["extraction_details"] = chart_counts
        
        # Sample chart info (without the actual image data which would be too large)
        for source_key, chart in all_charts[:10]:  # Show first 10 charts across all tests
            chart_info = {
                "source": source_key,
                "page": chart.get('page', 'unknown'),
                "image_index": chart.get('image_index', 0),
                "has_image_data": "image_data" in chart and bool(chart["image_data"]),
                "image_size_bytes": len(base64.b64decode(chart["image_data"])) if "image_data" in chart and chart["image_data"] else 0
            }
            results["charts_found"].append(chart_info)
        
        results["status"] = "complete"
        return jsonify(results)
    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)
        return jsonify(results), 500


@app.route('/user_questions/<int:user_id>', methods=['GET'])
def get_user_questions(user_id):
    """Get all questions and RAG responses for a specific user"""
    try:
        conn, cursor = get_db_connection()
        
        # Get all questions for this user, ordered by timestamp
        cursor.execute(
            "SELECT id, user_id, question_text, rag_response, question_type, timestamp FROM user_questions_rag WHERE user_id = %s ORDER BY timestamp DESC",
            (user_id,)
        )
        questions = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return jsonify({'questions': questions})
    except Exception as e:
        logger.error(f"Error getting user questions: {str(e)}")
        return jsonify({'error': str(e)}), 500


def init_rag_tables():
    """Initialize the user_questions_rag table if it doesn't exist"""
    try:
        conn, cursor = get_db_connection()
        
        # Create user_questions_rag table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_questions_rag (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT,
            question_text TEXT NOT NULL,
            rag_response TEXT NOT NULL,
            question_type VARCHAR(50),
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
        ''')
        
        conn.commit()
        cursor.close()
        conn.close()
        logger.info("RAG database tables initialized")
    except Exception as e:
        logger.error(f"Error initializing RAG database tables: {str(e)}")



def format_math_with_latex(text):
    """Simple LaTeX formatting that works reliably across platforms"""
    if not text:
        return text
    
    # Don't attempt to process text that already contains LaTeX delimiters
    if '$' in text:
        return text
    
    # Don't overprocess - just focus on basic formatting
    return text


# Add these helper functions to your existing functions
def generate_sat_practice_question(topic="Math", difficulty="medium"):
    """Generate a SAT practice question from database with fallback"""
    print(f"Attempting to get {topic} question from database...")
    
    try:
        # Try to get a question from the database first
        conn, cursor = get_db_connection()
        print("Database connection successful")
        
        # Query the database for a random question of the specified topic
        query = "SELECT question_id, question_text, answer_text, explanation, topic FROM questions WHERE topic = %s ORDER BY RAND() LIMIT 1"
        print(f"Executing query: {query} with params: ({topic},)")
        
        cursor.execute(query, (topic,))
        print("Query executed successfully")
        
        db_question = cursor.fetchone()
        print(f"Query result: {db_question}")
        
        cursor.close()
        conn.close()
        
        # If we found a question in the database, format it and return it
        if db_question:
            print(f"Using database question ID: {db_question['question_id']}")
            
            # Format explanation with clear steps for math questions
            explanation = db_question['explanation']
            if topic == "Math" and not explanation.startswith("Step"):
                # Simple step-by-step formatting
                explanation_lines = explanation.split('\n')
                formatted_lines = []
                step_num = 1
                
                for line in explanation_lines:
                    line = line.strip()
                    if not line:
                        continue
                        
                    # Basic step formatting
                    if "=" in line or any(keyword in line.lower() for keyword in ["solve", "substitute", "find"]):
                        formatted_lines.append(f"Step {step_num}: {line}")
                        step_num += 1
                    elif line.lower().startswith(("therefore", "thus", "finally", "so")):
                        formatted_lines.append(f"**Final Answer:** {line}")
                    else:
                        formatted_lines.append(line)
                
                # If no steps were identified, just number all lines
                if step_num == 1 and formatted_lines:
                    formatted_lines = []
                    for i, line in enumerate(explanation_lines):
                        if line.strip():
                            if i == len(explanation_lines) - 1:
                                formatted_lines.append(f"**Final Answer:** {line.strip()}")
                            else:
                                formatted_lines.append(f"Step {i+1}: {line.strip()}")
                
                # Join with double line breaks for readability
                explanation = "\n\n".join(formatted_lines)
            
            return {
                "question_id": db_question['question_id'],
                "question_text": db_question['question_text'],
                "answer_text": db_question['answer_text'],
                "explanation": explanation,
                "topic": db_question['topic']
            }
        else:
            print("No questions found in database, using fallback")
            
    except Exception as e:
        print(f"Error querying database: {str(e)}")
        print("Using fallback questions due to database error")
    
    # If database query failed or returned no results, use the fallback
    print("Returning fallback question")
    return generate_fallback_sat_question(topic, difficulty)

def generate_fallback_sat_question(topic="Math", difficulty="medium"):
    """Generate a fallback SAT question when database retrieval fails"""
    if topic == "Math":
        questions = [
            {
                "question_text": "If f(x) = 2x² - 3x + 5, what is f(2)?",
                "answer_text": "7",
                "explanation": "Step 1: Substitute x = 2 into the function.\n\nStep 2: Calculate the value.\nf(2) = 2(2)² - 3(2) + 5\n\nStep 3: Simplify the expression.\nf(2) = 2(4) - 6 + 5\nf(2) = 8 - 6 + 5\n\n**Final Answer:** f(2) = 7",
                "topic": "Math"
            },
            {
                "question_text": "In a right triangle, if one leg has length 5 and the hypotenuse has length 13, what is the length of the other leg?",
                "answer_text": "12",
                "explanation": "Step 1: Use the Pythagorean theorem.\nFor a right triangle with legs a and b and hypotenuse c: a² + b² = c²\n\nStep 2: Substitute the known values.\nWe know one leg is 5 and the hypotenuse is 13, so: 5² + b² = 13²\n\nStep 3: Solve for b.\n25 + b² = 169\nb² = 169 - 25\nb² = 144\nb = 12\n\n**Final Answer:** The length of the other leg is 12 units.",
                "topic": "Math"
            },
            {
                "question_text": "If 3x - 5y = 15 and 2x + 3y = 6, what is the value of x?",
                "answer_text": "5",
                "explanation": "Step 1: Solve for y in the second equation.\n2x + 3y = 6\n3y = 6 - 2x\ny = (6 - 2x)/3\n\nStep 2: Substitute this expression for y into the first equation.\n3x - 5((6 - 2x)/3) = 15\n\nStep 3: Simplify and solve for x.\n3x - 5(6 - 2x)/3 = 15\n3x - (30 - 10x)/3 = 15\n9x - (30 - 10x) = 45\n9x - 30 + 10x = 45\n19x = 75\nx = 75/19\nx = 5\n\n**Final Answer:** x = 5",
                "topic": "Math"
            },
            {
                "question_text": "To solve the equation x⁸ = 64, we first express both sides with a common base. Since 64 = 2⁶, the equation becomes x⁸ = 2⁶. What is the value of x²⁴?",
                "answer_text": "8",
                "explanation": "Step 1: Rewrite the equation with a common base.\nx⁸ = 64\nx⁸ = 2⁶\n\nStep 2: Solve for x by taking the 8th root of both sides.\nx = (2⁶)^(1/8)\nx = 2^(6/8)\nx = 2^(3/4)\n\nStep 3: Calculate x²⁴.\nx²⁴ = (2^(3/4))²⁴\nx²⁴ = 2^((3/4)·24)\nx²⁴ = 2^18\nx²⁴ = 2^18 = 2^18 = 262,144\n\n**Final Answer:** x²⁴ = 262,144",
                "topic": "Math"
            }
        ]
        
        # Adjust difficulty by selecting appropriate questions
        if difficulty == "easy":
            easy_indices = [0, 1]  # Indices of easier questions
            filtered_questions = [questions[i] for i in easy_indices if i < len(questions)]
            questions = filtered_questions if filtered_questions else questions
        elif difficulty == "hard":
            hard_indices = [2, 3]  # Indices of harder questions
            filtered_questions = [questions[i] for i in hard_indices if i < len(questions)]
            questions = filtered_questions if filtered_questions else questions
            
    elif topic == "Reading":
        questions = [
            {
                "question_text": "Based on the passage, the author's attitude toward traditional farming methods can best be described as:\n\nA) critical\nB) nostalgic\nC) ambivalent\nD) enthusiastic",
                "answer_text": "B",
                "explanation": "The author describes traditional farming methods with positive language and fond remembrance, indicating a nostalgic attitude. There is no criticism, ambivalence, or over-enthusiasm - just an appreciation for traditional approaches.",
                "topic": "Reading"
            },
            {
                "question_text": "Which statement best expresses the main idea of the third paragraph?\n\nA) Climate change poses significant threats to global ecosystems.\nB) Human activity has accelerated the natural cycle of climate change.\nC) The relationship between climate change and biodiversity loss is complex.\nD) Scientists disagree about the primary causes of climate change.",
                "answer_text": "C",
                "explanation": "The third paragraph focuses on explaining the various ways climate change and biodiversity loss interact and influence each other, highlighting the complexity of their relationship rather than simply stating that climate change threatens ecosystems or discussing human involvement or scientific disagreement.",
                "topic": "Reading"
            }
        ]
        
    else:  # Writing
        questions = [
            {
                "question_text": "Which choice best maintains the style and tone of the paragraph?\n\nA) It was really awesome how she did that\nB) Her accomplishment was noteworthy\nC) She totally killed it with her performance\nD) One can certainly observe her achievement",
                "answer_text": "B",
                "explanation": "Option B maintains a formal, academic tone consistent with SAT writing expectations. The other options are either too casual (A, C) or too stilted (D).",
                "topic": "Writing"
            },
            {
                "question_text": "Choose the most effective transition to begin the underlined sentence.\n\nRenewable energy sources have become increasingly popular. _____, fossil fuels still provide the majority of the world's energy needs.\n\nA) Therefore\nB) For instance\nC) Similarly\nD) However",
                "answer_text": "D",
                "explanation": "The sentence presents a contrast to the previous statement about renewable energy becoming popular. 'However' is the appropriate transition to show this contrast. 'Therefore' would indicate a result, 'For instance' would introduce an example, and 'Similarly' would suggest the ideas are alike rather than contrasting.",
                "topic": "Writing"
            }
        ]
    
    import random
    question = random.choice(questions)
    
    # Make sure the explanation has proper step formatting but no LaTeX
    if topic == "Math" and not question["explanation"].startswith("Step"):
        explanation_lines = question["explanation"].split('\n')
        formatted_lines = []
        step_num = 1
        
        for line in explanation_lines:
            line = line.strip()
            if not line:
                continue
                
            # Basic step formatting
            if "=" in line or any(keyword in line.lower() for keyword in ["solve", "substitute", "find"]):
                formatted_lines.append(f"Step {step_num}: {line}")
                step_num += 1
            elif line.lower().startswith(("therefore", "thus", "finally", "so")):
                formatted_lines.append(f"**Final Answer:** {line}")
            else:
                formatted_lines.append(line)
        
        # If no steps were identified, just number all lines
        if step_num == 1 and explanation_lines:
            formatted_lines = []
            for i, line in enumerate(explanation_lines):
                if line.strip():
                    if i == len(explanation_lines) - 1:
                        formatted_lines.append(f"**Final Answer:** {line.strip()}")
                    else:
                        formatted_lines.append(f"Step {i+1}: {line.strip()}")
        
        # Join with double line breaks for readability
        question["explanation"] = "\n\n".join(formatted_lines)
    
    return question


def check_sat_answer(question_text, correct_answer, user_answer, explanation):
    """Check if a user's answer to an SAT question is correct"""
    # Simple exact match check
    is_correct = False
    
    # Clean up answers for comparison
    clean_user_answer = user_answer.lower().strip().replace(" ", "")
    clean_correct_answer = correct_answer.lower().strip().replace(" ", "")
    
    # Check for exact match first
    if clean_user_answer == clean_correct_answer:
        is_correct = True
    # For math questions, allow for equivalent forms
    elif any(c.isdigit() for c in clean_correct_answer):
        # Try to be more lenient with numeric answers
        try:
            user_nums = [float(s) for s in re.findall(r'-?\d+\.?\d*', clean_user_answer)]
            correct_nums = [float(s) for s in re.findall(r'-?\d+\.?\d*', clean_correct_answer)]
            if user_nums and correct_nums and abs(user_nums[0] - correct_nums[0]) < 0.001:
                is_correct = True
        except:
            pass
    
    # Generate personalized feedback
    feedback_prompt = f"""The user answered "{user_answer}" to this SAT question: 
    
{question_text}

The correct answer is "{correct_answer}". 
The user's answer is {"CORRECT" if is_correct else "INCORRECT"}.

Provide a supportive and educational response that:
1. Confirms if their answer is correct or not
2. Explains why the answer is {correct_answer}
3. If they're incorrect, identify what concept they might be misunderstanding
4. Provide a brief tip to help them with similar questions in the future

Explanation details:
{explanation}
"""
    
    personalized_feedback = ask_deepseek_with_rag(feedback_prompt)
    
    return {
        'is_correct': is_correct,
        'feedback': personalized_feedback
    }



def format_math_for_display(text):
    """
    Simple function to properly format math expressions with LaTeX
    Specifically designed for SAT practice questions
    """
    import re
    
    # Skip if already has LaTeX formatting
    if '$' in text:
        return text
        
    # Format numbers in survey/word problems to improve readability
    # This uses a simple rule: numbers in the context of a problem should have $ $ around them
    number_pattern = r'(\b\d+\b)'
    text = re.sub(number_pattern, r'$\1$', text)
    
    # Handle special case for fractions
    fraction_pattern = r'(\$\d+\$)/(\$\d+\$)'
    text = re.sub(fraction_pattern, r'$\\frac{\1}{\2}$', text)
    
    # Remove unnecessary nested $ signs from the fractions
    text = text.replace('$$\\frac{$', '$\\frac{')
    text = text.replace('$}$', '}')
    
    return text


def format_sat_question_display(question_text):
    """
    Format SAT questions to display properly with correct parentheses and layout
    """
    import re
    
    # Fix coordinate pairs - ensure they have proper parentheses formatting
    # Look for patterns like (3, -2) and ensure they're formatted correctly
    coordinate_pattern = r'\((\d+),\s*(-?\d+)\)'
    question_text = re.sub(coordinate_pattern, r'(\1, \2)', question_text)
    
    # Fix equation formatting - make sure equal signs and operators have proper spacing
    # Look for patterns like |2x-3|=7 and format as |2x - 3| = 7
    equation_pattern = r'(\|?)(\d*[a-z])([+\-])(\d+)(\|?)([=])(\d+)'
    question_text = re.sub(equation_pattern, r'\1\2 \3 \4\5 \6 \7', question_text)
    
    # Fix any variable expressions like 2x-3 to 2x - 3
    variable_expr_pattern = r'(\d)([a-z])([+\-])(\d+)'
    question_text = re.sub(variable_expr_pattern, r'\1\2 \3 \4', question_text)
    
    return question_text

def format_sat_practice_question(question_text):
    """Format SAT practice questions to display properly"""
    import re
    
    # Fix coordinate pairs
    question_text = re.sub(r'\((\d+),\s*(-?\d+)\)', r'(\1, \2)', question_text)
    
    # Add spaces around math operators in equations
    question_text = re.sub(r'([0-9a-zA-Z])\s*([+\-=])\s*([0-9a-zA-Z])', r'\1 \2 \3', question_text)
    
    # Fix spacing in perpendicular/parallel line equations
    question_text = re.sub(r'(perpendicular|parallel)\s+to\s+([a-z])\s*=\s*(\d+)\s*([a-z])', 
                          r'\1 to \2 = \3\4', question_text)
    
    # Fix spacing for variable coefficients (e.g., 3x should not have a space)
    question_text = re.sub(r'(\d+)\s+([a-zA-Z])', r'\1\2', question_text)
    
    # Format "y = 3x + 2" type equations properly
    question_text = re.sub(r'([a-z])\s*=\s*(\d+)([a-z])\s*([+\-])\s*(\d+)', 
                          r'\1 = \2\3 \4 \5', question_text)
    
    return question_text

def format_explanation_with_steps(explanation):
    """Format an explanation into clear steps with proper spacing"""
    # Skip if already correctly formatted
    if "Step 1:" in explanation and "\n\n" in explanation:
        return explanation
    
    # Split by natural breaks
    lines = re.split(r'\n+|(?<=[.!?])\s+', explanation)
    formatted_lines = []
    step_count = 0
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check if this line starts a new step
        if any(keyword in line.lower() for keyword in ["step", "solve", "find", "calculate", "therefore"]):
            # If it doesn't already have a step number, add one
            if not re.match(r'Step \d+:', line):
                step_count += 1
                line = f"Step {step_count}: {line}"
            formatted_lines.append(line)
        elif "=" in line and not line.startswith("Step"):
            # This is an equation - make it a new step
            step_count += 1
            formatted_lines.append(f"Step {step_count}: {line}")
        elif line.lower().startswith(("therefore", "thus", "hence", "so the")):
            # This is a conclusion
            formatted_lines.append(f"**Final Answer:** {line}")
        else:
            formatted_lines.append(line)
    
    # If no steps were added, divide it into logical steps
    if step_count == 0:
        parts = re.split(r'(?<=[.!?])\s+', explanation)
        formatted_lines = []
        for i, part in enumerate(parts):
            if part.strip():
                if i == len(parts) - 2:  # Second to last meaningful part
                    formatted_lines.append(f"**Final Answer:** {part.strip()}")
                elif i < len(parts) - 2:  # Not the last parts
                    formatted_lines.append(f"Step {i+1}: {part.strip()}")
                else:
                    formatted_lines.append(part.strip())
    
    # Join with double line breaks for better readability
    return "\n\n".join(formatted_lines)

# Enhanced endpoint to answer questions
@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data.get('question', '')
    user_id = data.get('user_id', None)
    question_type = data.get('question_type', None)
    print(f"\n==== REQUEST RECEIVED: {datetime.now()} ====")
    logger.info(f"Received question: '{question}' with type: '{question_type}'")

    try:
        # FIRST check if this is a SAT practice question request
        practice_phrases = [
            "sat practice question",
            "sat math practice question",
            "sat reading practice question", 
            "sat writing practice question",
            "give me a sat question",
            "give me a practice question",
            "give me an sat problem",
            "sat practice problem",
            "sat practice test question"
        ]
        
        # Check if ANY of these exact phrases are in the question
        is_practice_request = any(phrase.lower() in question.lower() for phrase in practice_phrases)
        
        # Also look for patterns like "Give me a SAT math question"
        if not is_practice_request:
            practice_pattern = re.search(r"(give me|i want|i need).+(sat|practice).+(question|problem)", question.lower())
            is_practice_request = practice_pattern is not None
            
        if is_practice_request:
            logger.info("SAT PRACTICE QUESTION REQUESTED")
            
            # Determine question type
            sat_topic = "Math"  # Default
            if "math" in question.lower():
                sat_topic = "Math"
            elif "reading" in question.lower():
                sat_topic = "Reading"
            elif "writing" in question.lower():
                sat_topic = "Writing"
                
            # Determine difficulty
            difficulty = "medium"  # Default
            if "easy" in question.lower():
                difficulty = "easy"
            elif "hard" in question.lower() or "difficult" in question.lower():
                difficulty = "hard"
                
            # Generate a practice question
            question_data = generate_sat_practice_question(topic=sat_topic, difficulty=difficulty)
            
            # Format math questions if needed
            if sat_topic == "Math":
                question_data['question_text'] = format_math_for_display(question_data['question_text'])
                
            # Store in database
            conn, cursor = get_db_connection()
            cursor.execute(
                "INSERT INTO questions (question_text, answer_text, explanation, topic) VALUES (%s, %s, %s, %s)",
                (question_data['question_text'], question_data['answer_text'], question_data['explanation'], sat_topic)
            )
            question_id = cursor.lastrowid
            
            # Store the last question for this user
            if user_id:
                try:
                    # First check if the users table has a last_question_id column
                    cursor.execute("SHOW COLUMNS FROM users LIKE 'last_question_id'")
                    column_exists = cursor.fetchone()
                    
                    if not column_exists:
                        # Add the column if it doesn't exist
                        cursor.execute("ALTER TABLE users ADD COLUMN last_question_id INT NULL")
                    
                    # Update the user's last question ID
                    cursor.execute(
                        "UPDATE users SET last_question_id = %s WHERE user_id = %s",
                        (question_id, user_id)
                    )
                except Exception as e:
                    logger.error(f"Error updating user's last question: {str(e)}")
                    # Continue anyway, not critical
            
            conn.commit()
            cursor.close()
            conn.close()
            
            # Format response for chat - WITHOUT QUESTION ID DISPLAYED
            answer = f"""📝 SAT {sat_topic} Practice Question ({difficulty}): {format_sat_practice_question(question_data['question_text'])}

To submit your answer, reply with:

My answer is: your answer here"""
            
            logger.info(f"Generated SAT practice question with ID {question_id}")
            
            # Record this interaction
            if user_id:
                conn, cursor = get_db_connection()
                cursor.execute(
                    "INSERT INTO user_questions_rag (user_id, question_text, rag_response, question_type) VALUES (%s, %s, %s, %s)",
                    (user_id, question, answer, "SAT_Practice_Request")
                )
                conn.commit()
                cursor.close()
                conn.close()
                
            return jsonify({'answer': answer})
            
        # NEXT check if this is an answer submission
        # More flexible answer patterns with better capturing groups
        answer_patterns = [
            r"my answer is\s*:?\s*([a-zA-Z0-9./-]+)",
            r"my answer is\s*([a-zA-Z0-9./-]+)",
            r"answer\s*:?\s*([a-zA-Z0-9./-]+)",
            r"answer is\s*:?\s*([a-zA-Z0-9./-]+)",
            r"i think the answer is\s*:?\s*([a-zA-Z0-9./-]+)"
        ]
        
        user_answer = None
        for pattern in answer_patterns:
            match = re.search(pattern, question.lower())
            if match:
                user_answer = match.group(1).strip()
                logger.info(f"Found answer: '{user_answer}' using pattern '{pattern}'")
                break
        
        if user_answer:
            logger.info(f"SAT PRACTICE ANSWER SUBMITTED: {user_answer}")
            
            # Get the question ID from user's last question instead of requiring it in the response
            question_id = None
            
            if user_id:
                conn, cursor = get_db_connection()
                try:
                    cursor.execute("SELECT last_question_id FROM users WHERE user_id = %s", (user_id,))
                    user_data = cursor.fetchone()
                    
                    if user_data and user_data.get('last_question_id'):
                        question_id = user_data['last_question_id']
                        logger.info(f"Found user's last question ID: {question_id}")
                except Exception as e:
                    logger.error(f"Error retrieving last question ID: {str(e)}")
                    # Will continue and try to extract from the message as fallback
            
            # If we still don't have a question ID, try to extract it from the message (backward compatibility)
            if not question_id:
                question_id_patterns = [
                    r"question id\s*:?\s*(\d+)",
                    r"question id\s*(\d+)",
                    r"id\s*:?\s*(\d+)",
                    r"id\s*(\d+)"
                ]
                
                for pattern in question_id_patterns:
                    match = re.search(pattern, question.lower())
                    if match:
                        question_id = match.group(1)
                        logger.info(f"Found Question ID from message: {question_id} using pattern '{pattern}'")
                        break
                    
            if not question_id:
                logger.info("No Question ID found")
                answer = "I couldn't determine which question you're answering. Please ask for a new practice question."
                return jsonify({'answer': answer})
            
            # Get question from database
            conn, cursor = get_db_connection()
            cursor.execute(
                "SELECT question_text, answer_text, explanation, topic FROM questions WHERE question_id = %s",
                (question_id,)
            )
            question_data = cursor.fetchone()
            
            if not question_data:
                cursor.close()
                conn.close()
                answer = "I don't have a record of your last question. Please ask for a new practice question."
                return jsonify({'answer': answer})
                
            # Improved answer checking with better cleaning and matching
            logger.info(f"User answer: '{user_answer}', Correct answer: '{question_data['answer_text']}'")
            
            # Clean answers for comparison - be more aggressive with cleaning
            clean_user_answer = re.sub(r'[^a-zA-Z0-9.]', '', user_answer.lower())
            clean_correct_answer = re.sub(r'[^a-zA-Z0-9.]', '', question_data['answer_text'].lower())
            
            logger.info(f"Clean user answer: '{clean_user_answer}', Clean correct answer: '{clean_correct_answer}'")
            
            # Check for exact match first
            is_correct = False
            if clean_user_answer == clean_correct_answer:
                is_correct = True
                logger.info("Exact match found")
            # For math questions, allow for equivalent forms
            elif any(c.isdigit() for c in clean_correct_answer):
                # Try to be more lenient with numeric answers
                try:
                    user_nums = [float(s) for s in re.findall(r'-?\d+\.?\d*', clean_user_answer)]
                    correct_nums = [float(s) for s in re.findall(r'-?\d+\.?\d*', clean_correct_answer)]
                    logger.info(f"Numeric comparison: user_nums={user_nums}, correct_nums={correct_nums}")
                    if user_nums and correct_nums and abs(user_nums[0] - correct_nums[0]) < 0.001:
                        is_correct = True
                        logger.info("Numeric match found")
                except Exception as e:
                    logger.error(f"Error in numeric comparison: {str(e)}")
                    pass
                    
            # For multiple choice, be very lenient
            if len(clean_user_answer) == 1 and len(clean_correct_answer) == 1:
                # Single letter answers like A, B, C, D
                if clean_user_answer == clean_correct_answer:
                    is_correct = True
                    logger.info("Multiple choice match found")
                    
            # Special case for A/B/C/D answers
            if clean_user_answer in ["a", "b", "c", "d"] and clean_correct_answer in ["a", "b", "c", "d"]:
                is_correct = (clean_user_answer == clean_correct_answer)
                logger.info(f"A/B/C/D comparison: {clean_user_answer} == {clean_correct_answer} = {is_correct}")
                    
            # Record response
            try:
                cursor.execute(
                    "INSERT INTO user_responses (user_id, question_id, user_answer, is_correct) VALUES (%s, %s, %s, %s)",
                    (user_id, question_id, user_answer, is_correct)
                )
                
                # Update weak areas if incorrect
                if not is_correct:
                    cursor.execute("SELECT weak_areas FROM users WHERE user_id = %s", (user_id,))
                    user_data = cursor.fetchone()
                    
                    if user_data and user_data.get('weak_areas') is not None:
                        weak_areas = user_data['weak_areas'].split(',') if user_data['weak_areas'] else []
                        topic = question_data['topic']
                        
                        if topic not in weak_areas:
                            weak_areas.append(topic)
                            new_weak_areas = ','.join(weak_areas)
                            cursor.execute("UPDATE users SET weak_areas = %s WHERE user_id = %s", (new_weak_areas, user_id))
            except Exception as e:
                logger.error(f"Error updating database: {str(e)}")
                # Continue anyway to give the user feedback
                        
            conn.commit()
            cursor.close()
            conn.close()
            
            # Format explanation with simple step approach (no LaTeX)
            explanation = question_data['explanation']
            
            # For math questions, ensure we have clear steps
            if question_data['topic'] == "Math":
                # Only format if not already formatted
                if not explanation.strip().startswith("Step"):
                    explanation_lines = explanation.split('\n')
                    formatted_lines = []
                    step_num = 1
                    
                    for line in explanation_lines:
                        line = line.strip()
                        if not line:
                            continue
                            
                        # Basic step formatting
                        if "=" in line or any(keyword in line.lower() for keyword in ["solve", "substitute", "find"]):
                            formatted_lines.append(f"Step {step_num}: {line}")
                            step_num += 1
                        elif line.lower().startswith(("therefore", "thus", "finally", "so")):
                            formatted_lines.append(f"**Final Answer:** {line}")
                        else:
                            formatted_lines.append(line)
                    
                    # If no steps were identified, just number all lines
                    if step_num == 1:
                        formatted_lines = []
                        for i, line in enumerate(explanation_lines):
                            if line.strip():
                                if i == len(explanation_lines) - 1:
                                    formatted_lines.append(f"**Final Answer:** {line.strip()}")
                                else:
                                    formatted_lines.append(f"Step {i+1}: {line.strip()}")
                    
                    # Join with double line breaks for readability
                    explanation = "\n\n".join(formatted_lines)
            
            # Format feedback
            if is_correct:
                explanation = format_explanation_with_steps(question_data['explanation'])
                answer = f"""✅ Correct!

Your answer: {user_answer}
Correct answer: {question_data['answer_text']}

Explanation:
{explanation}

Would you like another practice question? Just ask!
"""
            else:
                explanation = format_explanation_with_steps(question_data['explanation'])
                answer = f"""❌ Incorrect

Your answer: {user_answer}
Correct answer: {question_data['answer_text']}

Explanation:
{explanation}

Would you like to try another practice question? Just ask!
"""
            logger.info(f"SAT practice answer feedback (correct: {is_correct})")
            
            # Record this interaction
            if user_id:
                try:
                    conn, cursor = get_db_connection()
                    cursor.execute(
                        "INSERT INTO user_questions_rag (user_id, question_text, rag_response, question_type) VALUES (%s, %s, %s, %s)",
                        (user_id, question, answer, "SAT_Practice_Answer")
                    )
                    conn.commit()
                    cursor.close()
                    conn.close()
                except Exception as e:
                    logger.error(f"Error recording interaction: {str(e)}")
                    # Continue anyway
                
            return jsonify({'answer': answer})
        
        # Continue with your existing chart/math question detection
        chart_related_terms = ['chart', 'graph', 'bar graph', 'line graph', 'pie chart', 'plot', 'histogram']
        table_related_terms = ['table', 'grid', 'matrix']
        
        # Exclude math/geometry problems
        math_terms = [
            'rectangle', 'area', 'square inches', 'equation', 'solve',
            'triangle', 'square', 'polygon', 'trapezoid', 'rhombus', 'parallelogram',
            'circle', 'radius', 'diameter', 'circumference', 'perimeter', 'volume',
            'surface area', 'angle', 'degrees', 'obtuse', 'acute', 'right angle',
            'congruent', 'similar', 'geometry', 'prism', 'cube', 'cylinder',
            'cone', 'sphere', 'vertices', 'edges', 'faces',
            'variable', 'expression', 'simplify', 'factor', 'multiple', 'divide',
            'multiply', 'addition', 'subtraction', 'sum', 'difference', 'product',
            'quotient', 'average', 'mean', 'median', 'mode', 'range',
            'probability', 'percentage', 'decimal', 'fraction', 'integer',
            'whole number', 'inequality', 'proportion', 'ratio', 'percent',
            'square root', 'exponent', 'power', 'logarithm', 'algebra',
            'polynomial', 'linear', 'quadratic', 'system of equations',
            'inequality', 'function', 'domain', 'range', 'derivative',
            'integral', 'calculus', 'matrix', 'vector', 'coordinate',
            'slope', 'intercept', 'tangent', 'secant', 'yards','linear','=' , '+'
        ]

        # A question is only chart-related if it EXPLICITLY mentions charts/graphs AND isn't just a math problem
        is_chart_question = (
            any(term.lower() in question.lower() for term in chart_related_terms + table_related_terms) and
            not any(term.lower() in question.lower() for term in math_terms)
        )

        is_math_question = (
            any(term.lower() in question.lower() for term in math_terms)     
        )
        
        if is_chart_question:
            logger.info("Chart-related question detected")
            answer = handle_chart_question(question, chart_type=question_type)

        elif is_math_question:     
            logger.info("Math question detected")
            answer = ask_deepseek_with_rag(question, question_type=question_type)  
            
        else:
            # Regular question handling
            logger.info("Regular question detected")
            answer = ask_deepseek_with_rag(question, question_type=question_type)

        # Save the interaction to database
        try:
            if user_id:  # Only save if we have a user_id
                conn, cursor = get_db_connection()
                cursor.execute(
                    "INSERT INTO user_questions_rag (user_id, question_text, rag_response, question_type) VALUES (%s, %s, %s, %s)",
                    (user_id, question, answer, question_type)
                )
                conn.commit()
                cursor.close()
                conn.close()
                logger.info(f"Saved question and response for user {user_id}")
            else:
                logger.info("No user_id provided, skipping database save")
        except Exception as db_error:
            logger.error(f"Error saving to database: {str(db_error)}")
            # Continue even if saving to DB fails

        logger.info(f"Generated answer of length {len(answer)}")
        print(f"\n==== SENDING RESPONSE: {datetime.now()} ====")
        return jsonify({'answer': answer})
    except Exception as e:
        logger.error(f"Error in ask_question: {str(e)}")
        return jsonify({'error': str(e), 'answer': "I encountered an error while processing your question. Please try again."}), 500

def transform_math_solution(answer):
    """Transform math solutions from model format to the expected frontend format"""
    import re
    
    # Create a copy to work with
    transformed = answer
    
    # Handle Step-by-Step Explanation section
    transformed = transformed.replace("**Step-by-Step Explanation:**", "")
    
    # Handle Answer section
    transformed = transformed.replace("**Answer:**", "Final Answer:")
    transformed = re.sub(r'The value of (\w+) is \*\*(\d+)\*\*', r'Final Answer: \1 = \2', transformed)
    
    # Handle step patterns
    step_pattern = r'(\d+)\. \*\*([^*]+)\*\*'
    step_replacement = r'Step \1: \2'
    transformed = re.sub(step_pattern, step_replacement, transformed)
    
    # Format for steps with numbers and asterisks like "**Step 2:**"
    transformed = re.sub(r'\*\*Step (\d+):\*\*', r'Step \1:', transformed)
    
    # Handle "Final Answer:" formatting
    transformed = transformed.replace("Final Answer:", "\n\nFinal Answer:")
    
    # Ensure proper equation formatting with line breaks
    equation_pattern = r'([\w\s]+)=([\w\s]+)'
    equation_replacement = r'\n\1 = \2\n'
    transformed = re.sub(equation_pattern, equation_replacement, transformed)
    
    # Clean up excessive newlines
    transformed = re.sub(r'\n{3,}', '\n\n', transformed)
    
    # Ensure steps are separated by double newlines for proper parsing
    transformed = re.sub(r'(Step \d+:.*?)(\n\s*Step \d+:)', r'\1\n\n\2', transformed)
    
    return transformed


# Endpoint to get questions by topic
@app.route('/questions/topic/<topic>', methods=['GET'])
def get_questions_by_topic(topic):
   try:
       conn, cursor = get_db_connection()
       cursor.execute("SELECT * FROM questions WHERE topic = %s", (topic,))
       questions = cursor.fetchall()
       cursor.close()
       conn.close()
       
       if questions:
           return jsonify({'questions': questions})
       else:
           return jsonify({'error': 'No questions found for this topic'}), 404
   except Exception as e:
       return jsonify({'error': str(e)}), 500

# Endpoint to submit an answer to a question
@app.route('/submit_answer', methods=['POST'])
def submit_answer():
   data = request.json
   user_id = data.get('user_id')
   question_id = data.get('question_id')
   user_answer = data.get('user_answer')
   
   if not all([user_id, question_id, user_answer]):
       return jsonify({'error': 'Missing required fields'}), 400
   
   try:
       conn, cursor = get_db_connection()
       
       # Get correct answer from the database
       cursor.execute("SELECT answer_text, explanation, topic, question_text FROM questions WHERE question_id = %s", (question_id,))
       question_data = cursor.fetchone()
       
       if not question_data:
           cursor.close()
           conn.close()
           return jsonify({'error': 'Question not found'}), 404
       
       # Check if the answer is correct (simple string comparison for now)
       is_correct = user_answer.lower().strip() == question_data['answer_text'].lower().strip()
       
       # Record the user's response
       cursor.execute(
           "INSERT INTO user_responses (user_id, question_id, user_answer, is_correct) VALUES (%s, %s, %s, %s)",
           (user_id, question_id, user_answer, is_correct)
       )
       
       # Update user's weak areas if the answer was incorrect
       if not is_correct:
           cursor.execute("SELECT weak_areas FROM users WHERE user_id = %s", (user_id,))
           user_data = cursor.fetchone()
           
           if user_data:
               weak_areas = user_data['weak_areas'].split(',') if user_data['weak_areas'] else []
               topic = question_data['topic']
               
               if topic not in weak_areas:
                   weak_areas.append(topic)
                   new_weak_areas = ','.join(weak_areas)
                   cursor.execute("UPDATE users SET weak_areas = %s WHERE user_id = %s", (new_weak_areas, user_id))
       
       conn.commit()
       cursor.close()
       conn.close()
       
       # Get enhanced explanation if incorrect
       explanation = question_data['explanation']
       if not is_correct:
           question_text = question_data['question_text']
           
           enhanced_explanation = ask_deepseek_with_rag(
               f"Explain why the answer to this SAT question is '{question_data['answer_text']}' and not '{user_answer}'? Question: {question_text}",
               question_type=question_data['topic'].lower()
           )
           
           if enhanced_explanation and len(enhanced_explanation) > 50:
               explanation = enhanced_explanation
       
       return jsonify({
           'is_correct': is_correct,
           'correct_answer': question_data['answer_text'],
           'explanation': explanation if not is_correct else None
       })
   except Exception as e:
       logger.error(f"Error in submit_answer: {str(e)}")
       return jsonify({'error': str(e)}), 500

# Endpoint to get user performance statistics
@app.route('/user_stats/<int:user_id>', methods=['GET'])
def get_user_stats(user_id):
   try:
       conn, cursor = get_db_connection()
       
       # Get total questions answered
       cursor.execute("SELECT COUNT(*) as total FROM user_responses WHERE user_id = %s", (user_id,))
       total_questions = cursor.fetchone()['total']
       
       # Get correct answers
       cursor.execute("SELECT COUNT(*) as correct FROM user_responses WHERE user_id = %s AND is_correct = TRUE", (user_id,))
       correct_answers = cursor.fetchone()['correct']
       
       # Get performance by topic
       cursor.execute("""
           SELECT q.topic, COUNT(*) as total, SUM(CASE WHEN ur.is_correct THEN 1 ELSE 0 END) as correct
           FROM user_responses ur
           JOIN questions q ON ur.question_id = q.question_id
           WHERE ur.user_id = %s
           GROUP BY q.topic
       """, (user_id,))
       
       topic_performance = cursor.fetchall()
       
       # Get recommended study topics based on performance
       cursor.execute("SELECT weak_areas FROM users WHERE user_id = %s", (user_id,))
       user_data = cursor.fetchone()
       weak_areas = user_data['weak_areas'].split(',') if user_data and user_data['weak_areas'] else []
       
       cursor.close()
       conn.close()
       
       # Generate personalized study plan
       study_plan = []
       if weak_areas:
           for area in weak_areas:
               if area:
                   study_recommendation = ask_deepseek_with_rag(
                       f"Create a brief study plan for the SAT {area} section. The student is struggling with this area.",
                       question_type=area.lower()
                   )
                   study_plan.append({
                       "topic": area,
                       "recommendation": study_recommendation
                   })
       
       return jsonify({
           'total_questions': total_questions,
           'correct_answers': correct_answers,
           'accuracy': (correct_answers / total_questions * 100) if total_questions > 0 else 0,
           'topic_performance': topic_performance,
           'weak_areas': weak_areas,
           'study_plan': study_plan
       })
   except Exception as e:
       return jsonify({'error': str(e)}), 500

# Endpoint to seed additional SAT questions
@app.route('/seed_questions', methods=['GET'])
def seed_questions():
   try:
       conn, cursor = get_db_connection()
       
       # Additional SAT sample questions
       questions = [
           {
               'question_text': 'If f(x) = 2x² + 3x - 5, what is f(3)?',
               'answer_text': '22',
               'explanation': 'Substitute x = 3 into the function: f(3) = 2(3)² + 3(3) - 5 = 2(9) + 9 - 5 = 18 + 9 - 5 = 22',
               'topic': 'Math'
           },
           {
               'question_text': 'What is the slope of a line perpendicular to y = 3x + 2?',
               'answer_text': '-1/3',
               'explanation': 'The slope of y = 3x + 2 is 3. Perpendicular lines have slopes that are negative reciprocals of each other. The negative reciprocal of 3 is -1/3.',
               'topic': 'Math'
           },
           {
               'question_text': 'The main purpose of the passage is to...',
               'answer_text': 'describe a natural phenomenon and explain its scientific basis',
               'explanation': 'The passage primarily focuses on describing a natural occurrence and then providing the scientific explanation behind it.',
               'topic': 'Reading'
           },
           {
               'question_text': 'The author uses the phrase "delicate balance" (line 45) primarily to emphasize...',
               'answer_text': 'the fragility of ecosystems',
               'explanation': 'In context, the phrase "delicate balance" is used to highlight how easily ecosystems can be disrupted, emphasizing their fragility.',
               'topic': 'Reading'
           },
           {
               'question_text': 'Choose the word or phrase that best maintains the tone of the passage.',
               'answer_text': 'significant',
               'explanation': 'The word "significant" maintains the formal, academic tone of the passage, while the other options are either too casual or too technical.',
               'topic': 'Writing'
           },
           {
               'question_text': 'Correct the sentence: "The committee, including the chairman and treasurer were in attendance."',
               'answer_text': 'The committee, including the chairman and treasurer, was in attendance.',
               'explanation': 'The subject of the sentence is "committee," which is singular. Therefore, the verb should be "was" not "were." Also, the phrase "including the chairman and treasurer" should be set off by commas on both sides.',
               'topic': 'Writing'
           }
       ]
       
       # Insert questions
       for q in questions:
           cursor.execute(
               "INSERT INTO questions (question_text, answer_text, explanation, topic) VALUES (%s, %s, %s, %s)",
               (q['question_text'], q['answer_text'], q['explanation'], q['topic'])
           )
       
       conn.commit()
       cursor.close()
       conn.close()
       
       return jsonify({'message': f'Successfully added {len(questions)} new questions'})
   except Exception as e:
       return jsonify({'error': str(e)}), 500

# Endpoint to get study materials
@app.route('/study_materials/<topic>', methods=['GET'])
def get_study_materials(topic):
   """Get personalized study materials for a specific topic"""
   try:
       # Generate study materials using RAG
       study_content = ask_deepseek_with_rag(
           f"Create comprehensive study notes for the SAT {topic} section. Include key concepts, common question types, and strategies.",
           question_type=topic.lower()
       )
       
       # Extract practice questions from the database
       conn, cursor = get_db_connection()
       cursor.execute("SELECT question_id, question_text FROM questions WHERE topic = %s LIMIT 5", (topic,))
       practice_questions = cursor.fetchall()
       cursor.close()
       conn.close()
       
       return jsonify({
           'topic': topic,
           'study_content': study_content,
           'practice_questions': practice_questions
       })
   except Exception as e:
       return jsonify({'error': str(e)}), 500

@app.route('/rag_status', methods=['GET'])
def rag_status():
   """Check the status of the RAG system"""
   return jsonify({
       'documents_count': len(vector_db.documents),
       'document_types': list(set(vector_db.document_type)),
       'sections': list(set(vector_db.section_info)),
       'embeddings_shape': vector_db.embeddings.shape if vector_db.embeddings is not None else None,
       'sat_materials': {
           name: {
               'exists': os.path.exists(path),
               'size_mb': round(os.path.getsize(path) / (1024 * 1024), 2) if os.path.exists(path) else None
           } for name, path in SAT_MATERIALS.items()
       }
   })

# Endpoint to get explanations for SAT concepts
@app.route('/explain_concept', methods=['POST'])
def explain_concept():
   """Get explanations for SAT concepts using RAG"""
   data = request.json
   concept = data.get('concept', '')
   topic = data.get('topic', 'general')
   
   if not concept:
       return jsonify({'error': 'Missing concept to explain'}), 400
   
   try:
       # Generate explanation using RAG
       explanation = ask_deepseek_with_rag(
           f"Explain the SAT concept: {concept}. Provide a detailed explanation with examples.",
           question_type=topic.lower()
       )
       
       return jsonify({
           'concept': concept,
           'explanation': explanation
       })
   except Exception as e:
       return jsonify({'error': str(e)}), 500

# New endpoint to get available practice tests
@app.route('/available_practice_tests', methods=['GET'])
def available_practice_tests():
   """List all available practice tests"""
   try:
       # Extract test numbers from SAT_MATERIALS
       tests = {}
       for key, path in SAT_MATERIALS.items():
           if '_' in key:
               key_parts = key.split('_')
               if key_parts[-1].isdigit():
                   test_type = '_'.join(key_parts[:-1])
                   test_num = key_parts[-1]
               else:
                   test_type = key
                   test_num = '7'  # Default for original files
           else:
               test_type = key
               test_num = '7'  # Default for original files
           
           if test_num not in tests:
               tests[test_num] = {}
           
           tests[test_num][test_type] = {
               "path": path,
               "exists": os.path.exists(path),
               "size_mb": round(os.path.getsize(path) / (1024 * 1024), 2) if os.path.exists(path) else None
           }
       
       return jsonify({
           "available_tests": tests,
           "total_tests": len(tests)
       })
   except Exception as e:
       return jsonify({'error': str(e)}), 500

# New endpoints for RAG database management
@app.route('/rebuild_rag', methods=['GET'])
def rebuild_rag():
    """Rebuild the entire RAG database with improved chunking"""
    try:
        global vector_db
        
        # Check if an existing database directory exists and rename it for backup
        if os.path.exists("rag_data"):
            backup_dir = f"rag_data_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            import shutil
            shutil.move("rag_data", backup_dir)
            logger.info(f"Backed up existing RAG data to {backup_dir}")
            
        # Create a new vector database
        vector_db = VectorDatabase()
        
        # Initialize the database
        initialize_vector_database()
        
        # Save the database
        vector_db.save()
        
        return jsonify({
            "status": "success",
            "message": f"RAG database rebuilt with {len(vector_db.documents)} documents",
            "document_types": list(set(vector_db.document_type)),
            "sections": list(set(vector_db.section_info))
        })
    except Exception as e:
        logger.error(f"Error rebuilding RAG database: {str(e)}")
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/add_exact_match_documents', methods=['GET'])
def add_exact_match_documents_endpoint():
    """Add exact match documents for key search terms"""
    try:
        # Add exact match documents
        vector_db.add_exact_match_documents()
        
        # Save the database
        vector_db.save()
        
        return jsonify({
            "status": "success",
            "message": "Added exact match documents to the RAG database",
            "documents_count": len(vector_db.documents)
        })
    except Exception as e:
        logger.error(f"Error adding exact match documents: {str(e)}")
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/test_search', methods=['POST'])
def test_search():
    """Test the search functionality with a specific query"""
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({'error': 'Missing query'}), 400
    
    try:
        # Search with the query
        results = vector_db.search(query, top_k=5)
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "text": result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"],
                "similarity": result["similarity"],
                "type": result["type"],
                "section": result["section"],
                "match_type": result.get("match_type", "unknown")
            })
        
        return jsonify({
            "query": query,
            "results": formatted_results,
            "total_results": len(formatted_results)
        })
    except Exception as e:
        logger.error(f"Error testing search: {str(e)}")
        return jsonify({"status": "error", "error": str(e)}), 500

# Initialize vector database on startup
vector_db = VectorDatabase()
vector_db_initialized = False

@app.before_request
def initialize_before_request():
   global vector_db_initialized, vector_db
   if not vector_db_initialized:
       logger.info("Checking for saved vector database...")
       
       # Try to load from disk first
       loaded_db = load_vector_database()
       
       if loaded_db is not None:
           # Use the loaded database
           vector_db = loaded_db
           vector_db_initialized = True
           logger.info(f"Using saved vector database with {len(vector_db.documents)} documents")
       else:
           # Initialize from scratch
           logger.info("No saved database found. Initializing from PDFs...")
           initialize_vector_database()
           
           # Save for next time
           vector_db.save()
           
           vector_db_initialized = True
       
       logger.info(f"Vector database initialized with {len(vector_db.documents)} documents")

@app.before_request
def initialize_tables():
    global rag_tables_initialized
    if not 'rag_tables_initialized' in globals() or not rag_tables_initialized:
        init_rag_tables()
        rag_tables_initialized = True


@app.route('/initialize_database', methods=['GET'])
def initialize_database_endpoint():
    """Manually initialize the RAG database"""
    try:
        global vector_db
        
        # Create a new vector database
        vector_db = VectorDatabase()
        
        # Initialize it
        logger.info("Manually initializing vector database...")
        initialize_vector_database()
        
        # Save the database
        vector_db.save()
        
        return jsonify({
            "status": "success",
            "message": f"Vector database initialized with {len(vector_db.documents)} documents",
            "document_count": len(vector_db.documents)
        })
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/add_test7_complete', methods=['GET'])
def add_test7_complete():
    """Add Practice Test 7 Questions, Answers and Scoring files to the database"""
    try:
        global vector_db
        
        # Check if the database exists
        if not hasattr(vector_db, 'documents') or vector_db.documents is None:
            vector_db = VectorDatabase()
        
        # Files to process
        target_files = [
            ('practice_test', SAT_MATERIALS['practice_test']),
            ('answers', SAT_MATERIALS['answers']),
            ('scoring', SAT_MATERIALS['scoring'])
        ]
        
        total_docs_added = 0
        
        # Process each file
        for key, path in target_files:
            if not os.path.exists(path):
                logger.warning(f"File {path} not found, skipping")
                continue
                
            # Extract test type and number
            test_type = key  # 'practice_test', 'answers', or 'scoring'
            test_num = '7'
            
            logger.info(f"Processing {key}: {path}")
            
            # Extract text with smaller chunk size for faster processing
            try:
                doc = fitz.open(path)
                
                # Process each page
                for page_num, page in enumerate(doc):
                    text = page.get_text()
                    
                    # Clean the text
                    text = re.sub(r'SAT Practice Test #\d+', '', text)
                    text = re.sub(r'©\s*\d+\s*College Board', '', text)
                    text = re.sub(r'Page \d+ of \d+', '', text)
                    
                    # Identify section
                    section = f"Page {page_num+1}"
                    
                    # Look for module markers
                    module_match = re.search(r"Module\s+(\d+)", text)
                    if module_match:
                        section = f"Module {module_match.group(1)}"
                    
                    # For practice test, look for question markers
                    if test_type == 'practice_test' and "QUESTION" in text:
                        # Extract question numbers
                        question_matches = re.finditer(r"QUESTION\s+(\d+)", text)
                        
                        for match in question_matches:
                            q_num = match.group(1)
                            q_pos = match.start()
                            
                            # Get text around this question (up to 500 chars)
                            q_text = text[max(0, q_pos-50):min(len(text), q_pos+450)]
                            
                            # Determine section type (math or reading)
                            section_type = "Math" if "Math" in section else "Reading and Writing"
                            
                            # Create enhanced text with metadata
                            enhanced_text = f"""
SAT Practice Test 7 - Question
Test Number: 7
Question Number: {q_num}
Section: {section}
Section Type: {section_type}
Document Type: Practice Test

{q_text}
"""
                            
                            # Add to database
                            vector_db.add_document(
                                enhanced_text.strip(),
                                doc_type="practice_test",
                                section=f"Test 7 - {section} - Question {q_num}"
                            )
                            
                            total_docs_added += 1
                    
                    # For answer documents, look for question markers
                    elif test_type == 'answers' and "QUESTION" in text:
                        # Extract question numbers
                        question_matches = re.finditer(r"QUESTION\s+(\d+)", text)
                        
                        for match in question_matches:
                            q_num = match.group(1)
                            q_pos = match.start()
                            
                            # Get text around this question (up to 500 chars)
                            q_text = text[max(0, q_pos-50):min(len(text), q_pos+450)]
                            
                            # Create enhanced text with metadata
                            enhanced_text = f"""
SAT Practice Test 7 - Answer Explanation
Test Number: 7
Question Number: {q_num}
Document Type: Answer Explanation

{q_text}
"""
                            
                            # Add to database
                            vector_db.add_document(
                                enhanced_text.strip(),
                                doc_type="answers",
                                section=f"Test 7 - Question {q_num}"
                            )
                            
                            total_docs_added += 1
                    
                    # For scoring documents, chunk by paragraphs
                    elif test_type == 'scoring':
                        # Split by paragraphs
                        paragraphs = re.split(r"\n\s*\n", text)
                        
                        current_chunk = []
                        current_size = 0
                        max_size = 256  # Smaller chunks for faster processing
                        
                        for para in paragraphs:
                            para = para.strip()
                            if not para:
                                continue
                                
                            if current_size + len(para) > max_size and current_chunk:
                                # Create enhanced chunk
                                enhanced_text = f"""
SAT Practice Test 7 - Scoring Information
Test Number: 7
Section: Scoring Guide
Page: {page_num + 1}

{"".join(current_chunk)}
"""
                                
                                # Add to database
                                vector_db.add_document(
                                    enhanced_text.strip(),
                                    doc_type="scoring",
                                    section=f"Test 7 - Scoring"
                                )
                                
                                total_docs_added += 1
                                current_chunk = []
                                current_size = 0
                            
                            # Add paragraph
                            current_chunk.append(para + "\n\n")
                            current_size += len(para)
                        
                        # Add the last chunk
                        if current_chunk:
                            enhanced_text = f"""
SAT Practice Test 7 - Scoring Information
Test Number: 7
Section: Scoring Guide
Page: {page_num + 1}

{"".join(current_chunk)}
"""
                            
                            vector_db.add_document(
                                enhanced_text.strip(),
                                doc_type="scoring",
                                section=f"Test 7 - Scoring"
                            )
                            
                            total_docs_added += 1
                    
                    # For general content not covered by specific handlers
                    else:
                        # Create enhanced text with metadata
                        enhanced_text = f"""
SAT Practice Test 7 - {section}
Test Number: 7
Section: {section}
Document Type: {test_type}
Page: {page_num + 1}

{text}
"""
                        
                        # Add to database
                        vector_db.add_document(
                            enhanced_text.strip(),
                            doc_type=test_type,
                            section=f"Test 7 - {section}"
                        )
                        
                        total_docs_added += 1
                        
            except Exception as e:
                logger.error(f"Error processing {path}: {str(e)}")
        
        # Add exact match documents for Test 7
        exact_match_docs = [
            {
                "text": "Test 7 math Test 7 math Test 7 math - This is the math section of SAT Practice Test 7",
                "doc_type": "practice_test",
                "section": "Test 7 - Math"
            },
            {
                "text": "Test 7 reading Test 7 reading Test 7 reading - This is the reading section of SAT Practice Test 7",
                "doc_type": "practice_test", 
                "section": "Test 7 - Reading and Writing"
            },
            {
                "text": "Test 7 math answers Test 7 math answers - This is the math section answers for SAT Practice Test 7",
                "doc_type": "answers",
                "section": "Test 7 - Math"
            },
            {
                "text": "Test 7 reading answers Test 7 reading answers - This is the reading section answers for SAT Practice Test 7",
                "doc_type": "answers",
                "section": "Test 7 - Reading and Writing"
            },
            {
                "text": "Scoring SAT Test 7 - This contains information about scoring the SAT Practice Test 7",
                "doc_type": "scoring",
                "section": "Test 7 - Scoring"
            }
        ]
        
        for doc in exact_match_docs:
            vector_db.add_document(doc["text"], doc_type=doc["doc_type"], section=doc["section"])
            total_docs_added += 1
        
        # Save the database
        vector_db.save()
        
        return jsonify({
            "status": "success",
            "message": f"Added Test 7 complete (questions, answers, scoring) with {total_docs_added} documents",
            "document_count": len(vector_db.documents)
        })
    except Exception as e:
        logger.error(f"Error adding Test 7 complete: {str(e)}")
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/add_test4_complete', methods=['GET'])
def add_test4_complete():
    """Add Practice Test 4 Questions, Answers and Scoring files to the database"""
    try:
        global vector_db
        
        # Check if the database exists
        if not hasattr(vector_db, 'documents') or vector_db.documents is None:
            vector_db = VectorDatabase()
        
        # Files to process - specifically for Test 4
        target_files = [
            ('practice_test_4', 'sat-practice-test-4-digital.pdf'),
            ('answers_4', 'sat-practice-test-4-answers-digital.pdf'),
            ('scoring_4', 'scoring-sat-practice-test-4-digital.pdf')
        ]
        
        total_docs_added = 0
        
        # Process each file
        for key, path in target_files:
            if not os.path.exists(path):
                logger.warning(f"File {path} not found, skipping")
                continue
                
            # Extract test type and number
            if key.startswith('practice_test'):
                test_type = 'practice_test'
            elif key.startswith('answers'):
                test_type = 'answers'
            elif key.startswith('scoring'):
                test_type = 'scoring'
            else:
                test_type = key
                
            test_num = '4'  # Specifically for Test 4
            
            logger.info(f"Processing {key}: {path}")
            
            # Extract text with smaller chunk size for faster processing
            try:
                doc = fitz.open(path)
                
                # Process each page
                for page_num, page in enumerate(doc):
                    text = page.get_text()
                    
                    # Clean the text
                    text = re.sub(r'SAT Practice Test #\d+', '', text)
                    text = re.sub(r'©\s*\d+\s*College Board', '', text)
                    text = re.sub(r'Page \d+ of \d+', '', text)
                    
                    # Identify section
                    section = f"Page {page_num+1}"
                    
                    # Look for module markers
                    module_match = re.search(r"Module\s+(\d+)", text)
                    if module_match:
                        section = f"Module {module_match.group(1)}"
                    
                    # For practice test, look for question markers
                    if test_type == 'practice_test' and "QUESTION" in text:
                        # Extract question numbers
                        question_matches = re.finditer(r"QUESTION\s+(\d+)", text)
                        
                        for match in question_matches:
                            q_num = match.group(1)
                            q_pos = match.start()
                            
                            # Get text around this question (up to 400 chars - smaller for speed)
                            q_text = text[max(0, q_pos-50):min(len(text), q_pos+350)]
                            
                            # Determine section type (math or reading)
                            section_type = "Math" if "Math" in section else "Reading and Writing"
                            
                            # Create enhanced text with metadata
                            enhanced_text = f"""
SAT Practice Test 4 - Question
Test Number: 4
Question Number: {q_num}
Section: {section}
Section Type: {section_type}
Document Type: Practice Test

{q_text}
"""
                            
                            # Add to database
                            vector_db.add_document(
                                enhanced_text.strip(),
                                doc_type="practice_test",
                                section=f"Test 4 - {section} - Question {q_num}"
                            )
                            
                            total_docs_added += 1
                    
                    # For answer documents, look for question markers
                    elif test_type == 'answers' and "QUESTION" in text:
                        # Extract question numbers
                        question_matches = re.finditer(r"QUESTION\s+(\d+)", text)
                        
                        for match in question_matches:
                            q_num = match.group(1)
                            q_pos = match.start()
                            
                            # Get text around this question (up to 400 chars - smaller for speed)
                            q_text = text[max(0, q_pos-50):min(len(text), q_pos+350)]
                            
                            # Create enhanced text with metadata
                            enhanced_text = f"""
SAT Practice Test 4 - Answer Explanation
Test Number: 4
Question Number: {q_num}
Document Type: Answer Explanation

{q_text}
"""
                            
                            # Add to database
                            vector_db.add_document(
                                enhanced_text.strip(),
                                doc_type="answers",
                                section=f"Test 4 - Question {q_num}"
                            )
                            
                            total_docs_added += 1
                    
                    # For scoring documents, chunk by paragraphs
                    elif test_type == 'scoring':
                        # Split by paragraphs
                        paragraphs = re.split(r"\n\s*\n", text)
                        
                        current_chunk = []
                        current_size = 0
                        max_size = 400  # Smaller chunks for faster processing
                        
                        for para in paragraphs:
                            para = para.strip()
                            if not para:
                                continue
                                
                            if current_size + len(para) > max_size and current_chunk:
                                # Create enhanced chunk
                                enhanced_text = f"""
SAT Practice Test 4 - Scoring Information
Test Number: 4
Section: Scoring Guide
Page: {page_num + 1}

{"".join(current_chunk)}
"""
                                
                                # Add to database
                                vector_db.add_document(
                                    enhanced_text.strip(),
                                    doc_type="scoring",
                                    section=f"Test 4 - Scoring"
                                )
                                
                                total_docs_added += 1
                                current_chunk = []
                                current_size = 0
                            
                            # Add paragraph
                            current_chunk.append(para + "\n\n")
                            current_size += len(para)
                        
                        # Add the last chunk
                        if current_chunk:
                            enhanced_text = f"""
SAT Practice Test 4 - Scoring Information
Test Number: 4
Section: Scoring Guide
Page: {page_num + 1}

{"".join(current_chunk)}
"""
                            
                            vector_db.add_document(
                                enhanced_text.strip(),
                                doc_type="scoring",
                                section=f"Test 4 - Scoring"
                            )
                            
                            total_docs_added += 1
                    
                    # For general content not covered by specific handlers
                    else:
                        # Create enhanced text with metadata
                        enhanced_text = f"""
SAT Practice Test 4 - {section}
Test Number: 4
Section: {section}
Document Type: {test_type}
Page: {page_num + 1}

{text}
"""
                        
                        # Add to database
                        vector_db.add_document(
                            enhanced_text.strip(),
                            doc_type=test_type,
                            section=f"Test 4 - {section}"
                        )
                        
                        total_docs_added += 1
                        
            except Exception as e:
                logger.error(f"Error processing {path}: {str(e)}")
        
        # Add exact match documents for Test 4
        exact_match_docs = [
            {
                "text": "Test 4 math Test 4 math Test 4 math - This is the math section of SAT Practice Test 4",
                "doc_type": "practice_test",
                "section": "Test 4 - Math"
            },
            {
                "text": "Test 4 reading Test 4 reading Test 4 reading - This is the reading section of SAT Practice Test 4",
                "doc_type": "practice_test", 
                "section": "Test 4 - Reading and Writing"
            },
            {
                "text": "Test 4 math answers Test 4 math answers - This is the math section answers for SAT Practice Test 4",
                "doc_type": "answers",
                "section": "Test 4 - Math"
            },
            {
                "text": "Test 4 reading answers Test 4 reading answers - This is the reading section answers for SAT Practice Test 4",
                "doc_type": "answers",
                "section": "Test 4 - Reading and Writing"
            },
            {
                "text": "Scoring SAT Test 4 - This contains information about scoring the SAT Practice Test 4",
                "doc_type": "scoring",
                "section": "Test 4 - Scoring"
            }
        ]
        
        for doc in exact_match_docs:
            vector_db.add_document(doc["text"], doc_type=doc["doc_type"], section=doc["section"])
            total_docs_added += 1
        
        # Save the database
        vector_db.save()
        
        return jsonify({
            "status": "success",
            "message": f"Added Test 4 complete (questions, answers, scoring) with {total_docs_added} documents",
            "document_count": len(vector_db.documents)
        })
    except Exception as e:
        logger.error(f"Error adding Test 4 complete: {str(e)}")
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route('/add_test10_complete', methods=['GET'])
def add_test10_complete():
    """Add Practice Test 10 Questions, Answers and Scoring files to the database"""
    try:
        global vector_db
        
        # Check if the database exists
        if not hasattr(vector_db, 'documents') or vector_db.documents is None:
            vector_db = VectorDatabase()
        
        # Files to process - specifically for Test 10
        target_files = [
            ('practice_test_10', 'sat-practice-test-10-digital.pdf'),
            ('answers_10', 'sat-practice-test-10-answers-digital.pdf'),
            ('scoring_10', 'scoring-sat-practice-test-10-digital.pdf')
        ]
        
        total_docs_added = 0
        
        # Process each file
        for key, path in target_files:
            if not os.path.exists(path):
                logger.warning(f"File {path} not found, skipping")
                continue
                
            # Extract test type and number
            if key.startswith('practice_test'):
                test_type = 'practice_test'
            elif key.startswith('answers'):
                test_type = 'answers'
            elif key.startswith('scoring'):
                test_type = 'scoring'
            else:
                test_type = key
                
            test_num = '10'  # Specifically for Test 10
            
            logger.info(f"Processing {key}: {path}")
            
            # Extract text with smaller chunk size for faster processing
            try:
                doc = fitz.open(path)
                
                # Process each page
                for page_num, page in enumerate(doc):
                    text = page.get_text()
                    
                    # Clean the text
                    text = re.sub(r'SAT Practice Test #\d+', '', text)
                    text = re.sub(r'©\s*\d+\s*College Board', '', text)
                    text = re.sub(r'Page \d+ of \d+', '', text)
                    
                    # Identify section
                    section = f"Page {page_num+1}"
                    
                    # Look for module markers
                    module_match = re.search(r"Module\s+(\d+)", text)
                    if module_match:
                        section = f"Module {module_match.group(1)}"
                    
                    # For practice test, look for question markers
                    if test_type == 'practice_test' and "QUESTION" in text:
                        # Extract question numbers
                        question_matches = re.finditer(r"QUESTION\s+(\d+)", text)
                        
                        for match in question_matches:
                            q_num = match.group(1)
                            q_pos = match.start()
                            
                            # Get text around this question (up to 400 chars - smaller for speed)
                            q_text = text[max(0, q_pos-50):min(len(text), q_pos+350)]
                            
                            # Determine section type (math or reading)
                            section_type = "Math" if "Math" in section else "Reading and Writing"
                            
                            # Create enhanced text with metadata
                            enhanced_text = f"""
SAT Practice Test 10 - Question
Test Number: 10
Question Number: {q_num}
Section: {section}
Section Type: {section_type}
Document Type: Practice Test

{q_text}
"""
                            
                            # Add to database
                            vector_db.add_document(
                                enhanced_text.strip(),
                                doc_type="practice_test",
                                section=f"Test 10 - {section} - Question {q_num}"
                            )
                            
                            total_docs_added += 1
                    
                    # For answer documents, look for question markers
                    elif test_type == 'answers' and "QUESTION" in text:
                        # Extract question numbers
                        question_matches = re.finditer(r"QUESTION\s+(\d+)", text)
                        
                        for match in question_matches:
                            q_num = match.group(1)
                            q_pos = match.start()
                            
                            # Get text around this question (up to 400 chars - smaller for speed)
                            q_text = text[max(0, q_pos-50):min(len(text), q_pos+350)]
                            
                            # Create enhanced text with metadata
                            enhanced_text = f"""
SAT Practice Test 10 - Answer Explanation
Test Number: 10
Question Number: {q_num}
Document Type: Answer Explanation

{q_text}
"""
                            
                            # Add to database
                            vector_db.add_document(
                                enhanced_text.strip(),
                                doc_type="answers",
                                section=f"Test 10 - Question {q_num}"
                            )
                            
                            total_docs_added += 1
                    
                    # For scoring documents, chunk by paragraphs
                    elif test_type == 'scoring':
                        # Split by paragraphs
                        paragraphs = re.split(r"\n\s*\n", text)
                        
                        current_chunk = []
                        current_size = 0
                        max_size = 400  # Smaller chunks for faster processing
                        
                        for para in paragraphs:
                            para = para.strip()
                            if not para:
                                continue
                                
                            if current_size + len(para) > max_size and current_chunk:
                                # Create enhanced chunk
                                enhanced_text = f"""
SAT Practice Test 10 - Scoring Information
Test Number: 10
Section: Scoring Guide
Page: {page_num + 1}

{"".join(current_chunk)}
"""
                                
                                # Add to database
                                vector_db.add_document(
                                    enhanced_text.strip(),
                                    doc_type="scoring",
                                    section=f"Test 10 - Scoring"
                                )
                                
                                total_docs_added += 1
                                current_chunk = []
                                current_size = 0
                            
                            # Add paragraph
                            current_chunk.append(para + "\n\n")
                            current_size += len(para)
                        
                        # Add the last chunk
                        if current_chunk:
                            enhanced_text = f"""
SAT Practice Test 10 - Scoring Information
Test Number: 10
Section: Scoring Guide
Page: {page_num + 1}

{"".join(current_chunk)}
"""
                            
                            vector_db.add_document(
                                enhanced_text.strip(),
                                doc_type="scoring",
                                section=f"Test 10 - Scoring"
                            )
                            
                            total_docs_added += 1
                    
                    # For general content not covered by specific handlers
                    else:
                        # Create enhanced text with metadata
                        enhanced_text = f"""
SAT Practice Test 10 - {section}
Test Number: 10
Section: {section}
Document Type: {test_type}
Page: {page_num + 1}

{text}
"""
                        
                        # Add to database
                        vector_db.add_document(
                            enhanced_text.strip(),
                            doc_type=test_type,
                            section=f"Test 10 - {section}"
                        )
                        
                        total_docs_added += 1
                        
            except Exception as e:
                logger.error(f"Error processing {path}: {str(e)}")
        
        # Add exact match documents for Test 10
        exact_match_docs = [
            {
                "text": "Test 10 math Test 10 math Test 10 math - This is the math section of SAT Practice Test 10",
                "doc_type": "practice_test",
                "section": "Test 10 - Math"
            },
            {
                "text": "Test 10 reading Test 10 reading Test 10 reading - This is the reading section of SAT Practice Test 10",
                "doc_type": "practice_test", 
                "section": "Test 10 - Reading and Writing"
            },
            {
                "text": "Test 10 math answers Test 10 math answers - This is the math section answers for SAT Practice Test 10",
                "doc_type": "answers",
                "section": "Test 10 - Math"
            },
            {
                "text": "Test 10 reading answers Test 10 reading answers - This is the reading section answers for SAT Practice Test 10",
                "doc_type": "answers",
                "section": "Test 10 - Reading and Writing"
            },
            {
                "text": "Scoring SAT Test 10 - This contains information about scoring the SAT Practice Test 10",
                "doc_type": "scoring",
                "section": "Test 10 - Scoring"
            }
        ]
        
        for doc in exact_match_docs:
            vector_db.add_document(doc["text"], doc_type=doc["doc_type"], section=doc["section"])
            total_docs_added += 1
        
        # Save the database
        vector_db.save()
        
        return jsonify({
            "status": "success",
            "message": f"Added Test 10 complete (questions, answers, scoring) with {total_docs_added} documents",
            "document_count": len(vector_db.documents)
        })
    except Exception as e:
        logger.error(f"Error adding Test 10 complete: {str(e)}")
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route('/add_test9_complete', methods=['GET'])
def add_test9_complete():
    """Add Practice Test 9 Questions, Answers and Scoring files to the database"""
    try:
        global vector_db
        
        # Check if the database exists
        if not hasattr(vector_db, 'documents') or vector_db.documents is None:
            vector_db = VectorDatabase()
        
        # Files to process - specifically for Test 9
        target_files = [
            ('practice_test_9', 'sat-practice-test-9-digital.pdf'),
            ('answers_9', 'sat-practice-test-9-answers-digital.pdf'),
            ('scoring_9', 'scoring-sat-practice-test-9-digital.pdf')
        ]
        
        total_docs_added = 0
        
        # Process each file
        for key, path in target_files:
            if not os.path.exists(path):
                logger.warning(f"File {path} not found, skipping")
                continue
                
            # Extract test type and number
            if key.startswith('practice_test'):
                test_type = 'practice_test'
            elif key.startswith('answers'):
                test_type = 'answers'
            elif key.startswith('scoring'):
                test_type = 'scoring'
            else:
                test_type = key
                
            test_num = '9'  # Specifically for Test 9
            
            logger.info(f"Processing {key}: {path}")
            
            # Extract text with smaller chunk size for faster processing
            try:
                doc = fitz.open(path)
                
                # Process each page
                for page_num, page in enumerate(doc):
                    text = page.get_text()
                    
                    # Clean the text
                    text = re.sub(r'SAT Practice Test #\d+', '', text)
                    text = re.sub(r'©\s*\d+\s*College Board', '', text)
                    text = re.sub(r'Page \d+ of \d+', '', text)
                    
                    # Identify section
                    section = f"Page {page_num+1}"
                    
                    # Look for module markers
                    module_match = re.search(r"Module\s+(\d+)", text)
                    if module_match:
                        section = f"Module {module_match.group(1)}"
                    
                    # For practice test, look for question markers
                    if test_type == 'practice_test' and "QUESTION" in text:
                        # Extract question numbers
                        question_matches = re.finditer(r"QUESTION\s+(\d+)", text)
                        
                        for match in question_matches:
                            q_num = match.group(1)
                            q_pos = match.start()
                            
                            # Get text around this question (up to 400 chars - smaller for speed)
                            q_text = text[max(0, q_pos-50):min(len(text), q_pos+350)]
                            
                            # Determine section type (math or reading)
                            section_type = "Math" if "Math" in section else "Reading and Writing"
                            
                            # Create enhanced text with metadata
                            enhanced_text = f"""
SAT Practice Test 9 - Question
Test Number: 9
Question Number: {q_num}
Section: {section}
Section Type: {section_type}
Document Type: Practice Test

{q_text}
"""
                            
                            # Add to database
                            vector_db.add_document(
                                enhanced_text.strip(),
                                doc_type="practice_test",
                                section=f"Test 9 - {section} - Question {q_num}"
                            )
                            
                            total_docs_added += 1
                    
                    # For answer documents, look for question markers
                    elif test_type == 'answers' and "QUESTION" in text:
                        # Extract question numbers
                        question_matches = re.finditer(r"QUESTION\s+(\d+)", text)
                        
                        for match in question_matches:
                            q_num = match.group(1)
                            q_pos = match.start()
                            
                            # Get text around this question (up to 400 chars - smaller for speed)
                            q_text = text[max(0, q_pos-50):min(len(text), q_pos+350)]
                            
                            # Create enhanced text with metadata
                            enhanced_text = f"""
SAT Practice Test 9 - Answer Explanation
Test Number: 9
Question Number: {q_num}
Document Type: Answer Explanation

{q_text}
"""
                            
                            # Add to database
                            vector_db.add_document(
                                enhanced_text.strip(),
                                doc_type="answers",
                                section=f"Test 9 - Question {q_num}"
                            )
                            
                            total_docs_added += 1
                    
                    # For scoring documents, chunk by paragraphs
                    elif test_type == 'scoring':
                        # Split by paragraphs
                        paragraphs = re.split(r"\n\s*\n", text)
                        
                        current_chunk = []
                        current_size = 0
                        max_size = 400  # Smaller chunks for faster processing
                        
                        for para in paragraphs:
                            para = para.strip()
                            if not para:
                                continue
                                
                            if current_size + len(para) > max_size and current_chunk:
                                # Create enhanced chunk
                                enhanced_text = f"""
SAT Practice Test 9 - Scoring Information
Test Number: 9
Section: Scoring Guide
Page: {page_num + 1}

{"".join(current_chunk)}
"""
                                
                                # Add to database
                                vector_db.add_document(
                                    enhanced_text.strip(),
                                    doc_type="scoring",
                                    section=f"Test 9 - Scoring"
                                )
                                
                                total_docs_added += 1
                                current_chunk = []
                                current_size = 0
                            
                            # Add paragraph
                            current_chunk.append(para + "\n\n")
                            current_size += len(para)
                        
                        # Add the last chunk
                        if current_chunk:
                            enhanced_text = f"""
SAT Practice Test 9 - Scoring Information
Test Number: 9
Section: Scoring Guide
Page: {page_num + 1}

{"".join(current_chunk)}
"""
                            
                            vector_db.add_document(
                                enhanced_text.strip(),
                                doc_type="scoring",
                                section=f"Test 9 - Scoring"
                            )
                            
                            total_docs_added += 1
                    
                    # For general content not covered by specific handlers
                    else:
                        # Create enhanced text with metadata
                        enhanced_text = f"""
SAT Practice Test 9 - {section}
Test Number: 9
Section: {section}
Document Type: {test_type}
Page: {page_num + 1}

{text}
"""
                        
                        # Add to database
                        vector_db.add_document(
                            enhanced_text.strip(),
                            doc_type=test_type,
                            section=f"Test 9 - {section}"
                        )
                        
                        total_docs_added += 1
                        
            except Exception as e:
                logger.error(f"Error processing {path}: {str(e)}")
        
        # Add exact match documents for Test 9
        exact_match_docs = [
            {
                "text": "Test 9 math Test 9 math Test 9 math - This is the math section of SAT Practice Test 9",
                "doc_type": "practice_test",
                "section": "Test 9 - Math"
            },
            {
                "text": "Test 9 reading Test 9 reading Test 9 reading - This is the reading section of SAT Practice Test 9",
                "doc_type": "practice_test", 
                "section": "Test 9 - Reading and Writing"
            },
            {
                "text": "Test 9 math answers Test 9 math answers - This is the math section answers for SAT Practice Test 9",
                "doc_type": "answers",
                "section": "Test 9 - Math"
            },
            {
                "text": "Test 9 reading answers Test 9 reading answers - This is the reading section answers for SAT Practice Test 9",
                "doc_type": "answers",
                "section": "Test 9 - Reading and Writing"
            },
            {
                "text": "Scoring SAT Test 9 - This contains information about scoring the SAT Practice Test 9",
                "doc_type": "scoring",
                "section": "Test 9 - Scoring"
            }
        ]
        
        for doc in exact_match_docs:
            vector_db.add_document(doc["text"], doc_type=doc["doc_type"], section=doc["section"])
            total_docs_added += 1
        
        # Save the database
        vector_db.save()
        
        return jsonify({
            "status": "success",
            "message": f"Added Test 9 complete (questions, answers, scoring) with {total_docs_added} documents",
            "document_count": len(vector_db.documents)
        })
    except Exception as e:
        logger.error(f"Error adding Test 9 complete: {str(e)}")
        return jsonify({"status": "error", "error": str(e)}), 500


if __name__ == '__main__':
   app.run(debug=True)
