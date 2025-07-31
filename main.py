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
    from langchain_community.document_loaders import PyPDFLoader # Keep only PyPDFLoader
    from langchain_huggingface import HuggingFaceEndpointEmbeddings
    from langchain.prompts import PromptTemplate
    from langchain.retrievers import EnsembleRetriever
    from langchain_community.retrievers import BM25Retriever
    from langchain_core.exceptions import OutputParserException

    print("✅ All necessary LangChain imports successful")
except ImportError as e:
    print(f"❌ LangChain import error: {e}")
    print("Please install missing dependencies:")
    # Removed unstructured and its specific dependencies from this list
    print("pip install langchain langchain-community langchain-groq langchain-huggingface faiss-cpu pypdf requests rank-bm25")
    exit(1)
except Exception as e:
    print(f"❌ General import error: {e}")
    exit(1)

# Enhanced request and response models for insurance claim processing
class DebugRequest(BaseModel):
    question: str

class ClaimRequest(BaseModel):
    documents: Union[List[str], str]
    claim_details: Dict[str, Any] = Field(default_factory=dict)
    questions: List[str]

class ParsedClaimDetails(BaseModel):
    patient_age: Optional[int] = None
    procedure: Optional[str] = None
    location: Optional[str] = None
    policy_start_date: Optional[str] = None # YYYY-MM-DD
    claim_date: Optional[str] = None # YYYY-MM-DD
    requested_amount: Optional[float] = None
    has_other_insurance: Optional[bool] = None
    primary_insurance_payment: Optional[float] = None
    policy_duration_months: Optional[int] = None

class CoordinationOfBenefits(BaseModel):
    has_other_insurance: bool = False
    primary_insurance: Optional[str] = None
    secondary_insurance: Optional[str] = None
    primary_payment: Optional[float] = None
    remaining_amount: Optional[float] = None

class ClaimDecision(BaseModel):
    question: str
    decision: str
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
    llm_parser_used: bool
    llm_parser_output: Optional[Dict[str, Any]]

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

# Enhanced Insurance-Specific Prompt Template for Claim Decision
INSURANCE_CLAIM_PROMPT = """
You are an expert insurance claim processor with deep knowledge of policy terms, coverage rules, and claim evaluation. You must analyze claims systematically and provide structured decisions.

ANALYSIS FRAMEWORK:
1. **Eligibility Assessment**: Determine if the claim is covered under the policy based on provided claim details.
2. **Coverage Limits**: Identify applicable limits, deductibles, and caps.
3. **Coordination of Benefits**: Check for multiple insurance policies and calculate remaining amounts if applicable.
4. **Exclusion Review**: Identify any policy exclusions that apply based on the claim.
5. **Decision Logic**: Apply business rules to determine approval/denial based on policy context and claim details.
6. **Payout Calculation**: Calculate exact amounts considering all factors (limits, deductibles, primary payments).

RESPONSE FORMAT (Must be valid JSON):
{{
    "decision": "[APPROVED/DENIED/PENDING_REVIEW]",
    "confidence_score": [0.0-1.0],
    "payout_amount": [amount or null],
    "reasoning": "Detailed explanation with specific policy references and how claim details interact with policy rules.",
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
- Base decisions ONLY on information in the policy context and provided claim details.
- For coordination of benefits, if 'has_other_insurance' is true, assume primary_payment is the amount already paid by primary insurer, and calculate remaining_amount from the requested amount and policy limits.
- Include confidence scores based on clarity of policy language and completeness of claim details.
- Reference specific policy sections (e.g., 'Section A', 'Clause 3.1') in your reasoning and 'policy_sections_referenced' list.
- If information is unclear, ambiguous, or missing for a conclusive decision, use "PENDING_REVIEW" decision and explain why.

Policy Context:
{context}

Claim Details (Structured, if available):
{claim_details_json}

Claim Question: {question}

Insurance Analysis (JSON format only):
"""

# Prompt for LLM Parser to extract structured details from natural language query
LLM_PARSER_PROMPT = """
You are an AI assistant designed to extract key claim details from natural language queries.
Your goal is to parse the user's question and identify structured information relevant to an insurance claim.

Extract the following details if mentioned or inferable:
- 'patient_age' (integer)
- 'procedure' (string)
- 'location' (string)
- 'policy_duration_months' (integer, infer from phrases like '3-month-old policy')
- 'requested_amount' (float)
- 'has_other_insurance' (boolean: true if phrases like 'other insurance', 'secondary claim' are present, else false)
- 'primary_insurance_payment' (float, if a specific amount paid by primary insurer is mentioned)

Return the extracted details in JSON format. If a detail is not found or cannot be inferred, omit it from the JSON.

Example Input: "46-year-old male, knee surgery in Pune, 3-month-old insurance policy, seeking $25,000, with $10,000 paid by primary insurance"
Example Output:
{{
  "patient_age": 46,
  "procedure": "knee surgery",
  "location": "Pune",
  "policy_duration_months": 3,
  "requested_amount": 25000.0,
  "has_other_insurance": true,
  "primary_insurance_payment": 10000.0
}}

Query: {query}

Extracted Claim Details (JSON format only):
"""

# Create the enhanced prompt templates
ENHANCED_CLAIM_PROMPT = PromptTemplate(
    template=INSURANCE_CLAIM_PROMPT,
    input_variables=["context", "question", "claim_details_json"]
)

LLM_PARSER_PROMPT_TEMPLATE = PromptTemplate(
    template=LLM_PARSER_PROMPT,
    input_variables=["query"]
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
        """Calculate confidence score based on various factors - now mostly handled by LLM, but can be refined"""
        score = decision_factors.get('confidence_score', 0.5)
        if score < 0.1:
            if decision_factors.get('policy_sections_referenced'):
                score += 0.2
            if decision_factors.get('has_coordination'):
                score -= 0.1
            if decision_factors.get('has_amounts'):
                score += 0.1
        return max(0.0, min(1.0, score))


class HybridRetriever:
    """Enhanced retrieval system combining semantic and keyword search"""

    def __init__(self, vector_store, documents):
        self.vector_store = vector_store
        self.documents = documents
        self.bm25_retriever = None
        self.setup_bm25_retriever()

    def setup_bm25_retriever(self):
        """Setup BM25 retriever for keyword matching"""
        if not self.documents:
            print("⚠️ No documents to initialize BM25 retriever.")
            self.bm25_retriever = None
            return

        try:
            doc_texts = [doc.page_content for doc in self.documents if doc.page_content]
            if not doc_texts:
                print("⚠️ No text content found in documents to initialize BM25 retriever.")
                self.bm25_retriever = None
                return

            self.bm25_retriever = BM25Retriever.from_texts(doc_texts)
            self.bm25_retriever.k = 6
            print("✅ BM25 retriever initialized")
        except Exception as e:
            print(f"⚠️ BM25 retriever failed, using vector-only: {e}")
            self.bm25_retriever = None

    def retrieve_relevant_docs(self, query: str, k: int = 6) -> List[Document]:
        """Retrieve documents using hybrid approach"""
        all_docs = []

        vector_retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k, "fetch_k": k * 2}
        )
        vector_docs = vector_retriever.get_relevant_documents(query)
        all_docs.extend(vector_docs)

        if self.bm25_retriever:
            try:
                bm25_docs = self.bm25_retriever.get_relevant_documents(query)
                all_docs.extend(bm25_docs)
            except Exception as e:
                print(f"⚠️ BM25 retrieval failed: {e}")

        unique_docs = []
        seen_content = set()
        for doc in all_docs:
            if doc.page_content not in seen_content:
                unique_docs.append(doc)
                seen_content.add(doc.page_content)

        return unique_docs[:k]

# Initialize components with enhanced error handling
embeddings = None
llm = None
try:
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-mpnet-base-v2",
        huggingfacehub_api_token=HF_TOKEN
    )
    print("✅ HuggingFace Endpoint Embeddings initialized successfully")
except Exception as e:
    print(f"❌ Error initializing HuggingFaceEndpointEmbeddings: {e}")
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

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=[
        "\n\n### ",
        "\n\nSection ",
        "\n\nClause ",
        "\n\n",
        "\n",
        ". ",
        " ",
    ],
    length_function=len,
    keep_separator=True
)

decision_engine = InsuranceDecisionEngine()
vector_store = None
hybrid_retriever = None
processed_documents_global = []

# Helper functions (Modified to remove DOCX/EML specific parts)
def is_url(string: str) -> bool:
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except:
        return False

def extract_pdf_content(pdf_url: str) -> List[Document]:
    try:
        print(f"📥 Downloading PDF from: {pdf_url}")
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(response.content)
            temp_path = temp_file.name

        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        os.unlink(temp_path)
        print(f"✅ PDF extracted successfully. Loaded {len(docs)} pages.")
        return docs
    except Exception as e:
        print(f"❌ Error extracting PDF content from {pdf_url}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to extract PDF content: {str(e)}")

# This function is now simplified to only handle PDF URLs and plain text
def process_input_documents(documents_input: Union[List[str], str]) -> List[Document]:
    """Processes PDF URLs and plain text into LangChain Document objects."""
    if isinstance(documents_input, str):
        documents_input = [documents_input]

    all_loaded_docs = []

    for doc_item in documents_input:
        if is_url(doc_item):
            doc_item_lower = doc_item.lower()
            if doc_item_lower.endswith('.pdf'):
                all_loaded_docs.extend(extract_pdf_content(doc_item))
            else:
                # Only PDF URLs supported now
                raise HTTPException(status_code=400, detail=f"URL format not supported: {doc_item}. Only .pdf URLs are supported.")
        else:
            # Assume it's plain text content
            all_loaded_docs.append(Document(page_content=doc_item))

    return all_loaded_docs

def parse_llm_response(response_text: str, default_confidence: float = 0.5) -> Dict:
    """Parse structured JSON response from LLM with more robust error handling."""
    try:
        json_match = re.search(r'\{.*\}', response_text.strip(), re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            parsed_data = json.loads(json_str)
            return {
                "decision": parsed_data.get("decision", "PENDING_REVIEW"),
                "confidence_score": parsed_data.get("confidence_score", default_confidence),
                "payout_amount": parsed_data.get("payout_amount"),
                "reasoning": parsed_data.get("reasoning", "No specific reasoning provided by LLM."),
                "policy_sections_referenced": parsed_data.get("policy_sections_referenced", []),
                "exclusions_applied": parsed_data.get("exclusions_applied", []),
                "coordination_of_benefits": parsed_data.get("coordination_of_benefits"),
                "processing_notes": parsed_data.get("processing_notes", [])
            }
        else:
            raise ValueError("No JSON object found in LLM response.")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"⚠️ JSON parsing failed or no JSON found: {e}. Raw response: {response_text[:200]}...")
        return {
            "decision": "PENDING_REVIEW",
            "confidence_score": 0.1,
            "payout_amount": None,
            "reasoning": f"LLM response parsing failed or unstructured response: {response_text}. Error: {str(e)}",
            "policy_sections_referenced": [],
            "exclusions_applied": [],
            "coordination_of_benefits": None,
            "processing_notes": [f"LLM response parsing error: {str(e)}"]
        }


def parse_claim_details_from_llm(query: str, llm_model: ChatGroq) -> Optional[Dict[str, Any]]:
    """Uses LLM to extract structured claim details from a natural language query."""
    try:
        formatted_prompt = LLM_PARSER_PROMPT_TEMPLATE.format(query=query)
        llm_result = llm_model.invoke(formatted_prompt)
        response_text = llm_result.content if hasattr(llm_result, 'content') else str(llm_result)

        json_match = re.search(r'\{.*\}', response_text.strip(), re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            parsed_data = json.loads(json_str)
            return parsed_data
        else:
            print(f"⚠️ LLM Parser: No JSON object found for query: '{query}'. Raw LLM response: {response_text[:200]}")
            return None
    except Exception as e:
        print(f"❌ Error during LLM parsing of claim details: {e}")
        return None


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
            "Audit trail support",
            "PDF document processing", # Updated
            "LLM-powered query parsing for claim details"
        ],
        "supported_formats": ["text", "pdf_urls"], # Updated
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
        "hybrid_retriever_ready": hybrid_retriever is not None,
        "processed_documents_count": len(processed_documents_global)
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
        raise HTTPException(status_code=400, detail="No vector store available. Please run /hackrx/run with documents first.")

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
                "full_length": len(doc.page_content),
                "source": doc.metadata.get('source', 'unknown'),
                "page": doc.metadata.get('page', 'N/A')
            })

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
    global vector_store, hybrid_retriever, processed_documents_global

    if embeddings is None or llm is None:
        raise HTTPException(status_code=500, detail="Core components (Embeddings/LLM) not initialized. Check server logs.")

    request_id = str(uuid.uuid4())
    start_time = time.time()
    audit_trail = [f"Request {request_id} started at {datetime.now().isoformat()}"]
    llm_parser_used = False
    parsed_claim_details_output = {}
    effective_claim_details = request.claim_details.copy()

    try:
        if not effective_claim_details and request.questions:
            audit_trail.append("Attempting LLM-powered claim details extraction.")
            parsed_details = parse_claim_details_from_llm(request.questions[0], llm)
            if parsed_details:
                effective_claim_details.update(parsed_details)
                llm_parser_used = True
                parsed_claim_details_output = parsed_details
                audit_trail.append(f"LLM parser extracted: {parsed_details}")
            else:
                audit_trail.append("LLM parser found no additional claim details or failed.")
        elif effective_claim_details:
            audit_trail.append("Claim details provided, skipping LLM parser for details.")
        else:
            audit_trail.append("No claim details provided and no questions to parse from. Proceeding without specific claim details.")


        loaded_docs_from_input = process_input_documents(request.documents)
        audit_trail.append(f"Input documents processed. Total loaded documents: {len(loaded_docs_from_input)}")

        if not loaded_docs_from_input:
            raise HTTPException(status_code=400, detail="No valid document content could be extracted or provided.")

        chunks = text_splitter.split_documents(loaded_docs_from_input)
        print(f"Created {len(chunks)} chunks from documents")
        audit_trail.append(f"Created {len(chunks)} document chunks from all inputs")

        if vector_store is None:
            print("Creating new vector store...")
            vector_store = FAISS.from_documents(chunks, embeddings)
            processed_documents_global = chunks
            print("✅ Vector store created successfully")
            audit_trail.append("New vector store created from all document chunks.")
        else:
            print(f"Adding {len(chunks)} documents to existing vector store...")
            vector_store.add_documents(chunks)
            processed_documents_global.extend(chunks)
            print("✅ Documents added to existing vector store.")
            audit_trail.append(f"{len(chunks)} new document chunks added to existing vector store.")

        hybrid_retriever = HybridRetriever(vector_store, processed_documents_global)
        audit_trail.append("Hybrid retriever re-initialized with current document set.")


        decisions = []
        if not request.questions:
            raise HTTPException(status_code=400, detail="No questions provided for processing.")

        for question in request.questions:
            try:
                print(f"Processing question: {question}")
                audit_trail.append(f"Processing question: '{question[:70]}...'")

                relevant_docs = hybrid_retriever.retrieve_relevant_docs(question, k=8)

                if relevant_docs:
                    context = "\n\n".join([doc.page_content for doc in relevant_docs])
                    claim_details_for_llm = json.dumps(effective_claim_details, indent=2) if effective_claim_details else "{}"
                    formatted_prompt = ENHANCED_CLAIM_PROMPT.format(
                        context=context,
                        question=question,
                        claim_details_json=claim_details_for_llm
                    )

                    llm_result = llm.invoke(formatted_prompt)
                    response_text = llm_result.content if hasattr(llm_result, 'content') else str(llm_result)

                    parsed_response = parse_llm_response(response_text)

                    decision = ClaimDecision(
                        question=question,
                        decision=parsed_response.get("decision", "PENDING_REVIEW"),
                        confidence_score=parsed_response.get("confidence_score", 0.5),
                        payout_amount=parsed_response.get("payout_amount"),
                        reasoning=parsed_response.get("reasoning", "Analysis completed."),
                        policy_sections_referenced=parsed_response.get("policy_sections_referenced", []),
                        exclusions_applied=parsed_response.get("exclusions_applied", []),
                        processing_notes=parsed_response.get("processing_notes", [])
                    )

                    cob_data = parsed_response.get("coordination_of_benefits")
                    if cob_data and isinstance(cob_data, dict) and cob_data.get("has_other_insurance") is not None:
                        if 'primary_payment' in cob_data and not isinstance(cob_data['primary_payment'], (int, float)):
                            try: cob_data['primary_payment'] = float(cob_data['primary_payment'])
                            except ValueError: cob_data['primary_payment'] = None
                        if 'remaining_amount' in cob_data and not isinstance(cob_data['remaining_amount'], (int, float)):
                            try: cob_data['remaining_amount'] = float(cob_data['remaining_amount'])
                            except ValueError: cob_data['remaining_amount'] = None
                        if 'has_other_insurance' in cob_data and not isinstance(cob_data['has_other_insurance'], bool):
                            cob_data['has_other_insurance'] = str(cob_data['has_other_insurance']).lower() == 'true'

                        try:
                            decision.coordination_of_benefits = CoordinationOfBenefits(**cob_data)
                        except Exception as cob_e:
                            decision.processing_notes.append(f"Error parsing COB data: {cob_e}. Raw: {cob_data}")
                            decision.coordination_of_benefits = None
                    else:
                        decision.coordination_of_benefits = CoordinationOfBenefits(has_other_insurance=False)

                    decisions.append(decision)
                    audit_trail.append(f"Decision generated for '{question[:50]}...': {decision.decision} (confidence: {decision.confidence_score})")
                else:
                    decisions.append(ClaimDecision(
                        question=question,
                        decision="PENDING_REVIEW",
                        confidence_score=0.1,
                        payout_amount=None,
                        reasoning="No relevant policy information found for this question in the provided documents.",
                        processing_notes=["No relevant documents retrieved for this query."]
                    ))
                    audit_trail.append(f"No relevant documents found for '{question[:50]}...'")

            except Exception as e:
                print(f"❌ Error processing question '{question}': {str(e)}")
                decisions.append(ClaimDecision(
                    question=question,
                    decision="PENDING_REVIEW",
                    confidence_score=0.0,
                    payout_amount=None,
                    reasoning=f"An internal error occurred during processing: {str(e)}",
                    processing_notes=[f"Processing error: {str(e)}"]
                ))
                audit_trail.append(f"Error processing question '{question[:50]}...': {str(e)}")

        processing_time = time.time() - start_time
        audit_trail.append(f"Processing completed in {processing_time:.2f} seconds")

        metadata = ProcessingMetadata(
            request_id=request_id,
            processing_time=processing_time,
            chunks_analyzed=len(processed_documents_global) if processed_documents_global else 0,
            model_used=llm.model_name if llm else "N/A",
            timestamp=datetime.now().isoformat(),
            llm_parser_used=llm_parser_used,
            llm_parser_output=effective_claim_details # Changed to effective_claim_details for full output
        )

        return EnhancedAnswerResponse(
            decisions=decisions,
            processing_metadata=metadata,
            audit_trail=audit_trail
        )

    except HTTPException as http_e:
        audit_trail.append(f"Client-side error: {http_e.detail}")
        raise http_e
    except Exception as e:
        print(f"❌ Fatal error in /hackrx/run processing: {str(e)}")
        audit_trail.append(f"Fatal error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error in RAG processing: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Enhanced HackRx 6.0 RAG Backend Server...")
    print("📍 Server will be available at:")
    print(f"   - http://{HOST}:{PORT}")
    print("🎯 HackRx 6.0 Features:")
    print("   - Insurance claim decision engine")
    print("   - Coordination of benefits analysis")
    print("   - Structured JSON responses with confidence scores")
    print("   - Hybrid retrieval (Vector + BM25)")
    print("   - Audit trail and processing metadata")
    print("   - Policy section referencing")
    print("   - **PDF document processing via URL**") # Updated feature list
    print("   - **LLM-powered query parsing for structured claim details**")

    uvicorn.run(app, host=HOST, port=PORT)
