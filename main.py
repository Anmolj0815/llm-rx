from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Union, Optional, Dict, Any
import time
import os
import requests
import tempfile
import json
import re
from urllib.parse import urlparse
from dotenv import load_dotenv
from datetime import datetime
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Load environment variables from .env file
load_dotenv()

# Only suppress tokenizer parallelism warnings (still useful for HuggingFace)
os.environ['TOKENIZERS_PARALLELISM'] = os.getenv('TOKENIZERS_PARALLELISM', 'false')

try:
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document
    from langchain_groq import ChatGroq
    from langchain.chains import RetrievalQA
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_huggingface import HuggingFaceEndpointEmbeddings
    from langchain.prompts import PromptTemplate
    from langchain.retrievers import EnsembleRetriever
    from langchain_community.retrievers import BM25Retriever
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install missing dependencies:")
    print("pip install langchain langchain-community langchain-groq langchain-huggingface faiss-cpu pypdf requests rank-bm25")
    exit(1)

# Enhanced request and response models for insurance claim processing
class DebugRequest(BaseModel):
    question: str

class ClaimRequest(BaseModel):
    documents: Union[List[str], str]
    claim_details: Dict[str, Any] = Field(default_factory=dict)
    questions: List[str]

class CoordinationOfBenefits(BaseModel):
    has_other_insurance: bool = False
    primary_insurance: Optional[str] = None
    secondary_insurance: Optional[str] = None
    primary_payment: Optional[float] = None
    remaining_amount: Optional[float] = None

class ClaimDecision(BaseModel):
    question: str
    decision: str  # "APPROVED", "DENIED", "PENDING_REVIEW"
    confidence_score: float = Field(ge=0.0, le=1.0)
    payout_amount: Optional[float] = None
    reasoning: str
    policy_sections_referenced: List[str] = Field(default_factory=list)
    exclusions_applied: List[str] = Field(default_factory=list)
    coordination_of_benefits: Optional[CoordinationOfBenefits] = None
    processing_notes: List[str] = Field(default_factory=list)

class ProcessingMetadata(BaseModel):
    request_id: str
    processing_time: float
    chunks_analyzed: int
    model_used: str
    timestamp: str

class EnhancedAnswerResponse(BaseModel):
    decisions: List[ClaimDecision]
    processing_metadata: ProcessingMetadata
    audit_trail: List[str] = Field(default_factory=list)

# Initialize FastAPI application with enhanced documentation
app = FastAPI(
    title="HackRx 6.0 Insurance RAG Backend", 
    version="2.0.0",
    description="Advanced RAG Backend with Insurance Claim Decision Engine and Structured Analysis"
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

security = HTTPBearer()

# Configuration
HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 5000))

# Enhanced Insurance-Specific Prompt Template
INSURANCE_CLAIM_PROMPT = """
You are an expert insurance claim processor with deep knowledge of policy terms, coverage rules, and claim evaluation. You must analyze claims systematically and provide structured decisions.

ANALYSIS FRAMEWORK:
1. **Eligibility Assessment**: Determine if the claim is covered under the policy
2. **Coverage Limits**: Identify applicable limits, deductibles, and caps
3. **Coordination of Benefits**: Check for multiple insurance policies and calculate remaining amounts
4. **Exclusion Review**: Identify any policy exclusions that apply
5. **Decision Logic**: Apply business rules to determine approval/denial
6. **Payout Calculation**: Calculate exact amounts considering all factors

RESPONSE FORMAT (Must be valid JSON):
{{
    "decision": "[APPROVED/DENIED/PENDING_REVIEW]",
    "confidence_score": [0.0-1.0],
    "payout_amount": [amount or null],
    "reasoning": "Detailed explanation with specific policy references",
    "policy_sections_referenced": ["section1", "section2"],
    "exclusions_applied": ["exclusion1", "exclusion2"],
    "coordination_of_benefits": {{
        "has_other_insurance": [true/false],
        "primary_insurance": "name or null",
        "secondary_insurance": "name or null", 
        "primary_payment": [amount or null],
        "remaining_amount": [amount or null]
    }},
    "processing_notes": ["note1", "note2"]
}}

IMPORTANT RULES:
- Base decisions ONLY on information in the policy context
- For coordination of benefits, calculate remaining amounts after primary insurance
- Include confidence scores based on clarity of policy language
- Reference specific policy sections in your reasoning
- If information is unclear, use "PENDING_REVIEW" decision

Policy Context:
{context}

Claim Question: {question}

Insurance Analysis (JSON format only):
"""

# Create the enhanced prompt template
ENHANCED_PROMPT = PromptTemplate(
    template=INSURANCE_CLAIM_PROMPT,
    input_variables=["context", "question"]
)

class InsuranceDecisionEngine:
    """Core decision engine for insurance claim processing"""
    
    def __init__(self):
        self.decision_rules = {
            'min_confidence_for_approval': 0.7,
            'max_payout_without_review': 10000,
            'coordination_keywords': [
                'coordination of benefits', 'other insurance', 'secondary claim',
                'primary insurance', 'remaining amount', 'balance claim'
            ],
            'exclusion_keywords': [
                'excluded', 'not covered', 'limitation', 'restriction'
            ]
        }
    
    def extract_financial_amounts(self, text: str) -> List[float]:
        """Extract dollar amounts from text"""
        amounts = re.findall(r'\$?[\d,]+\.?\d*', text)
        return [float(amt.replace('$', '').replace(',', '')) for amt in amounts if amt]
    
    def detect_coordination_of_benefits(self, context: str, question: str) -> bool:
        """Detect if coordination of benefits applies"""
        combined_text = (context + " " + question).lower()
        return any(keyword in combined_text for keyword in self.decision_rules['coordination_keywords'])
    
    def calculate_confidence_score(self, context: str, decision_factors: Dict) -> float:
        """Calculate confidence score based on various factors"""
        score = 0.5  # Base score
        
        # Boost confidence if specific policy sections are mentioned
        if decision_factors.get('policy_sections_referenced'):
            score += 0.2
        
        # Reduce confidence if coordination of benefits is involved
        if decision_factors.get('has_coordination'):
            score -= 0.1
        
        # Boost confidence if clear dollar amounts are present
        if decision_factors.get('has_amounts'):
            score += 0.1
        
        # Ensure score is within bounds
        return max(0.0, min(1.0, score))

class HybridRetriever:
    """Enhanced retrieval system combining semantic and keyword search"""
    
    def __init__(self, vector_store, documents):
        self.vector_store = vector_store
        self.documents = documents
        self.setup_hybrid_retrieval()
    
    def setup_hybrid_retrieval(self):
        """Setup hybrid retrieval combining vector and BM25 search"""
        try:
            # Create BM25 retriever for keyword matching
            doc_texts = [doc.page_content for doc in self.documents]
            self.bm25_retriever = BM25Retriever.from_texts(doc_texts)
            self.bm25_retriever.k = 6
            print("✅ BM25 retriever initialized")
        except Exception as e:
            print(f"⚠️ BM25 retriever failed, using vector-only: {e}")
            self.bm25_retriever = None
    
    def retrieve_relevant_docs(self, query: str, k: int = 6) -> List[Document]:
        """Retrieve documents using hybrid approach"""
        all_docs = []
        
        # Vector-based retrieval
        vector_retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k, "fetch_k": k * 2}
        )
        vector_docs = vector_retriever.get_relevant_documents(query)
        all_docs.extend(vector_docs)
        
        # BM25 keyword retrieval (if available)
        if self.bm25_retriever:
            try:
                bm25_docs = self.bm25_retriever.get_relevant_documents(query)
                all_docs.extend(bm25_docs)
            except Exception as e:
                print(f"⚠️ BM25 retrieval failed: {e}")
        
        # Remove duplicates and return top k
        unique_docs = []
        seen_content = set()
        for doc in all_docs:
            if doc.page_content not in seen_content:
                unique_docs.append(doc)
                seen_content.add(doc.page_content)
        
        return unique_docs[:k]

# Initialize components with enhanced error handling
try:
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-mpnet-base-v2",
        huggingfacehub_api_token=HF_TOKEN
    )
    print("✅ HuggingFace Endpoint Embeddings initialized successfully")
except Exception as e:
    print(f"❌ Error initializing endpoint embeddings: {e}")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        print("✅ Fallback local embeddings initialized successfully")
    except Exception as e2:
        print(f"❌ All embedding methods failed: {e2}")
        embeddings = None

try:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=GROQ_API_KEY,
        temperature=0
    )
    print("✅ LLM initialized successfully")
except Exception as e:
    print(f"❌ Error initializing LLM: {e}")
    llm = None

# Enhanced text splitter for insurance documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=[
        "\n\n### ",  # Policy sections
        "\n\nSection ",  # Section breaks
        "\n\nClause ",   # Clause breaks
        "\n\n",          # Paragraphs
        "\n",            # Lines
        ". ",            # Sentences
        " ",             # Words
    ],
    length_function=len,
    keep_separator=True
)

# Initialize decision engine and global variables
decision_engine = InsuranceDecisionEngine()
vector_store = None
hybrid_retriever = None
processed_documents = []

# Helper functions (keeping existing ones and adding new)
def is_url(string: str) -> bool:
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except:
        return False

def extract_pdf_content(pdf_url: str) -> str:
    try:
        print(f"📥 Downloading PDF from: {pdf_url}")
        
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(response.content)
            temp_path = temp_file.name
        
        loader = PyPDFLoader(temp_path)
        pages = loader.load()
        
        content = ""
        for i, page in enumerate(pages):
            page_text = page.page_content.strip()
            if page_text:
                content += f"\n--- Page {i+1} ---\n{page_text}\n"
        
        content = content.replace('\n\n\n', '\n\n')
        content = content.replace('\t', ' ')
        
        os.unlink(temp_path)
        
        print(f"✅ PDF extracted successfully. Content length: {len(content)} characters")
        return content
        
    except Exception as e:
        print(f"❌ Error extracting PDF content: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to extract PDF content: {str(e)}")

def process_documents(documents: Union[List[str], str]) -> str:
    if isinstance(documents, str):
        documents = [documents]
    
    processed_content = []
    
    for doc in documents:
        if is_url(doc):
            if doc.lower().endswith('.pdf') or 'pdf' in doc.lower():
                pdf_content = extract_pdf_content(doc)
                processed_content.append(pdf_content)
            else:
                raise HTTPException(status_code=400, detail=f"URL format not supported: {doc}")
        else:
            processed_content.append(doc)
    
    return "\n".join(processed_content)

def parse_llm_response(response_text: str) -> Dict:
    """Parse structured JSON response from LLM"""
    try:
        # Try to extract JSON from response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
        else:
            # Fallback parsing for non-JSON responses
            return {
                "decision": "PENDING_REVIEW",
                "confidence_score": 0.5,
                "payout_amount": None,
                "reasoning": response_text,
                "policy_sections_referenced": [],
                "exclusions_applied": [],
                "coordination_of_benefits": {
                    "has_other_insurance": False,
                    "primary_insurance": None,
                    "secondary_insurance": None,
                    "primary_payment": None,
                    "remaining_amount": None
                },
                "processing_notes": ["Response parsing required fallback method"]
            }
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON parsing failed: {e}")
        return {
            "decision": "PENDING_REVIEW",
            "confidence_score": 0.3,
            "payout_amount": None,
            "reasoning": f"Error parsing response: {response_text}",
            "policy_sections_referenced": [],
            "exclusions_applied": [],
            "coordination_of_benefits": {
                "has_other_insurance": False,
                "primary_insurance": None,
                "secondary_insurance": None,
                "primary_payment": None,
                "remaining_amount": None
            },
            "processing_notes": [f"JSON parsing error: {str(e)}"]
        }

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    received_token = credentials.credentials
    expected_token = os.getenv("AUTH_TOKEN")
    
    if not expected_token:
        raise HTTPException(status_code=500, detail="Server configuration error: AUTH_TOKEN not set.")

    if received_token == expected_token or received_token.strip() == expected_token.strip():
        return received_token
    
    raise HTTPException(status_code=403, detail="Invalid or expired token.")

# Enhanced API Endpoints
@app.get("/")
def root():
    return {
        "message": "HackRx 6.0 Insurance RAG Backend with Decision Engine",
        "version": "2.0.0",
        "status": "running",
        "features": [
            "Insurance claim decision engine",
            "Coordination of benefits analysis",
            "Structured JSON responses",
            "Hybrid retrieval (Vector + BM25)",
            "Confidence scoring",
            "Audit trail support"
        ],
        "supported_formats": ["text", "pdf_urls"],
        "endpoints": {
            "health": "/health",
            "rag_status": "/rag-status", 
            "run_query": "/hackrx/run",
            "debug_search": "/debug-search",
            "vector_stats": "/vector-stats",
            "decision_engine_status": "/decision-engine-status"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "embeddings_ready": embeddings is not None,
        "llm_ready": llm is not None,
        "vector_store_ready": vector_store is not None,
        "decision_engine_ready": decision_engine is not None,
        "hybrid_retriever_ready": hybrid_retriever is not None
    }

@app.get("/decision-engine-status")
def decision_engine_status():
    """Check decision engine configuration and rules"""
    return {
        "engine_active": True,
        "decision_rules": decision_engine.decision_rules,
        "supported_decisions": ["APPROVED", "DENIED", "PENDING_REVIEW"],
        "coordination_benefits_supported": True,
        "confidence_scoring_enabled": True
    }

@app.post("/debug-search")
async def debug_search(request: DebugRequest):
    """Enhanced debug endpoint with hybrid retrieval information"""
    global hybrid_retriever
    
    if vector_store is None:
        raise HTTPException(status_code=400, detail="No vector store available")
    
    try:
        if hybrid_retriever:
            docs = hybrid_retriever.retrieve_relevant_docs(request.question, k=6)
        else:
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 6, "fetch_k": 12}
            )
            docs = retriever.get_relevant_documents(request.question)
        
        retrieved_chunks = []
        for i, doc in enumerate(docs):
            retrieved_chunks.append({
                "chunk_id": i,
                "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                "full_length": len(doc.page_content)
            })
        
        # Add decision engine analysis
        has_cob = decision_engine.detect_coordination_of_benefits(
            " ".join([doc.page_content for doc in docs]), 
            request.question
        )
        
        return {
            "question": request.question,
            "total_chunks_retrieved": len(docs),
            "chunks": retrieved_chunks,
            "decision_engine_analysis": {
                "coordination_of_benefits_detected": has_cob,
                "retrieval_method": "hybrid" if hybrid_retriever else "vector_only"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debug search error: {str(e)}")

@app.post("/hackrx/run", response_model=EnhancedAnswerResponse)
async def run_enhanced_query(request: ClaimRequest, token: str = Depends(verify_token)):
    global vector_store, hybrid_retriever, processed_documents
    
    if embeddings is None or llm is None:
        raise HTTPException(status_code=500, detail="Core components not initialized")
    
    request_id = str(uuid.uuid4())
    start_time = time.time()
    audit_trail = [f"Request {request_id} started at {datetime.now().isoformat()}"]
    
    try:
        # Step 1: Process documents
        document_content = process_documents(request.documents)
        audit_trail.append("Documents processed successfully")
        
        if document_content.strip():
            docs = [Document(page_content=document_content)]
            chunks = text_splitter.split_documents(docs)
            print(f"Created {len(chunks)} chunks from documents")
            audit_trail.append(f"Created {len(chunks)} document chunks")
            
            # Create or update vector store
            if vector_store is None:
                print("Creating new vector store...")
                vector_store = FAISS.from_documents(chunks, embeddings)
                processed_documents = chunks
                print("✅ Vector store created successfully")
                audit_trail.append("New vector store created")
            else:
                print("Adding documents to existing vector store...")
                vector_store.add_documents(chunks)
                processed_documents.extend(chunks)
                audit_trail.append("Documents added to existing vector store")
            
            # Initialize hybrid retriever
            hybrid_retriever = HybridRetriever(vector_store, processed_documents)
            audit_trail.append("Hybrid retriever initialized")
        
        # Step 2: Process each question with enhanced decision logic
        decisions = []
        
        if vector_store is None:
            for question in request.questions:
                decisions.append(ClaimDecision(
                    question=question,
                    decision="PENDING_REVIEW",
                    confidence_score=0.0,
                    reasoning="No documents available for analysis",
                    processing_notes=["No vector store available"]
                ))
        else:
            for question in request.questions:
                try:
                    print(f"Processing question: {question}")
                    audit_trail.append(f"Processing question: {question[:50]}...")
                    
                    # Enhanced retrieval with multiple strategies
                    if hybrid_retriever:
                        relevant_docs = hybrid_retriever.retrieve_relevant_docs(question, k=8)
                    else:
                        retriever = vector_store.as_retriever(
                            search_type="similarity",
                            search_kwargs={"k": 8, "fetch_k": 16}
                        )
                        relevant_docs = retriever.get_relevant_documents(question)
                    
                    if relevant_docs:
                        context = "\n\n".join([doc.page_content for doc in relevant_docs])
                        
                        # Generate enhanced prompt with decision structure
                        formatted_prompt = ENHANCED_PROMPT.format(context=context, question=question)
                        
                        # Get LLM response
                        llm_result = llm.invoke(formatted_prompt)
                        response_text = llm_result.content if hasattr(llm_result, 'content') else str(llm_result)
                        
                        # Parse structured response
                        parsed_response = parse_llm_response(response_text)
                        
                        # Create decision object with enhanced data
                        decision = ClaimDecision(
                            question=question,
                            decision=parsed_response.get("decision", "PENDING_REVIEW"),
                            confidence_score=parsed_response.get("confidence_score", 0.5),
                            payout_amount=parsed_response.get("payout_amount"),
                            reasoning=parsed_response.get("reasoning", "Analysis completed"),
                            policy_sections_referenced=parsed_response.get("policy_sections_referenced", []),
                            exclusions_applied=parsed_response.get("exclusions_applied", []),
                            processing_notes=parsed_response.get("processing_notes", [])
                        )
                        
                        # Add coordination of benefits if detected
                        cob_data = parsed_response.get("coordination_of_benefits", {})
                        if cob_data and cob_data.get("has_other_insurance"):
                            decision.coordination_of_benefits = CoordinationOfBenefits(**cob_data)
                        
                        decisions.append(decision)
                        audit_trail.append(f"Decision generated: {decision.decision} (confidence: {decision.confidence_score})")
                    else:
                        decisions.append(ClaimDecision(
                            question=question,
                            decision="PENDING_REVIEW",
                            confidence_score=0.1,
                            reasoning="No relevant policy information found for this question",
                            processing_notes=["No relevant documents retrieved"]
                        ))
                        audit_trail.append("No relevant documents found")
                        
                except Exception as e:
                    print(f"❌ Error processing question '{question}': {str(e)}")
                    decisions.append(ClaimDecision(
                        question=question,
                        decision="PENDING_REVIEW",
                        confidence_score=0.0,
                        reasoning=f"Error during processing: {str(e)}",
                        processing_notes=[f"Processing error: {str(e)}"]
                    ))
                    audit_trail.append(f"Error processing question: {str(e)}")
        
        processing_time = time.time() - start_time
        audit_trail.append(f"Processing completed in {processing_time:.2f} seconds")
        
        # Create processing metadata
        metadata = ProcessingMetadata(
            request_id=request_id,
            processing_time=processing_time,
            chunks_analyzed=len(processed_documents) if processed_documents else 0,
            model_used="llama-3.3-70b-versatile",
            timestamp=datetime.now().isoformat()
        )
        
        return EnhancedAnswerResponse(
            decisions=decisions,
            processing_metadata=metadata,
            audit_trail=audit_trail
        )
        
    except Exception as e:
        print(f"❌ Error in enhanced RAG processing: {str(e)}")
        audit_trail.append(f"Fatal error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Enhanced RAG Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Enhanced HackRx 6.0 RAG Backend Server...")
    print("📍 Server will be available at:")
    print("   - http://localhost:5000")
    print("   - http://127.0.0.1:5000") 
    print("🎯 HackRx 6.0 Features:")
    print("   - Insurance claim decision engine")
    print("   - Coordination of benefits analysis")
    print("   - Structured JSON responses with confidence scores")
    print("   - Hybrid retrieval (Vector + BM25)")
    print("   - Audit trail and processing metadata")
    print("   - Policy section referencing")
    
    uvicorn.run(app, host=HOST, port=PORT)
