// API configuration and utility functions
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface DetectionResponse {
  disease: string;
  confidence: number;
  recommendations: string[];
  image_processed: boolean;
}

export interface RAGResponse {
  answer: string;
  sources: string[];
  query: string;
  confidence: number;
}

export interface CombinedAnalysisResponse {
  disease_detection: DetectionResponse;
  rag_response: RAGResponse;
  combined_insights: string;
  confidence_score: number;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  timestamp: string;
}

export interface InitialAnalysisResponse {
  response: string;
  session_id: string;
  chat_history: ChatMessage[];
}

export interface ChatResponse {
  response: string;
  chat_history: ChatMessage[];
  session_id: string;
}

export interface ApiError {
  detail: string;
}

/**
 * Combined analysis endpoint - sends both image and query
 */
export async function analyzeCrop(
  imageFile: File,
  query: string
): Promise<CombinedAnalysisResponse> {
  const formData = new FormData();
  formData.append("image", imageFile);
  formData.append("query", query);

  const response = await fetch(`${API_BASE_URL}/api/analyze`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const errorData: ApiError = await response.json();
    throw new Error(errorData.detail || "Analysis failed");
  }

  return response.json();
}

/**
 * RAG query endpoint - text-only queries
 */
export async function queryRAG(query: string): Promise<RAGResponse> {
  const response = await fetch(`${API_BASE_URL}/api/rag`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ query }),
  });

  if (!response.ok) {
    const errorData: ApiError = await response.json();
    throw new Error(errorData.detail || "Query failed");
  }

  return response.json();
}

/**
 * Disease detection endpoint - image-only analysis
 */
export async function detectDisease(imageFile: File): Promise<DetectionResponse> {
  const formData = new FormData();
  formData.append("image", imageFile);

  const response = await fetch(`${API_BASE_URL}/api/detect`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const errorData: ApiError = await response.json();
    throw new Error(errorData.detail || "Detection failed");
  }

  return response.json();
}

/**
 * Health check endpoint
 */
export async function checkHealth(): Promise<{ status: string }> {
  const response = await fetch(`${API_BASE_URL}/api/health`);

  if (!response.ok) {
    throw new Error("Health check failed");
  }

  return response.json();
}

/**
 * Convert data URL to File object
 */
export function dataURLtoFile(dataURL: string, filename: string): File {
  const arr = dataURL.split(",");
  const mime = arr[0].match(/:(.*?);/)![1];
  const bstr = atob(arr[1]);
  let n = bstr.length;
  const u8arr = new Uint8Array(n);

  while (n--) {
    u8arr[n] = bstr.charCodeAt(n);
  }

  return new File([u8arr], filename, { type: mime });
}

/**
 * Fast image analysis - only plant detection and health status (2-3 seconds)
 */
export async function fastImageAnalysis(
  imageFile: File,
  query?: string
): Promise<{
  plant: string;
  status: "healthy" | "diseased";
  disease?: string;
  confidence: number;
  session_id: string;
  user_query?: string;
}> {
  // Simple plant detection only
  const formData = new FormData();
  formData.append("image", imageFile);
  if (query) {
    formData.append("query", query); // Store query for later use
  }

  const response = await fetch(`${API_BASE_URL}/api/detect`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const errorData: ApiError = await response.json();
    throw new Error(errorData.detail || "Plant detection failed");
  }

  const result = await response.json();

  // Backend now returns the simplified format we need
  return {
    plant: result.plant || "Plant",
    status: result.status || "healthy",
    disease: result.disease,
    confidence: result.confidence || 0,
    session_id:
      result.session_id || `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
    user_query: result.user_query || query,
  };
}

/**
 * Initial analysis endpoint - starts a new chat session with image and query
 */
export async function initialAnalysis(
  imageFile: File,
  query: string
): Promise<InitialAnalysisResponse> {
  const formData = new FormData();
  formData.append("image", imageFile);
  formData.append("query", query);

  const response = await fetch(`${API_BASE_URL}/api/initial-analysis`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const errorData: ApiError = await response.json();
    throw new Error(errorData.detail || "Initial analysis failed");
  }

  return response.json();
}

/**
 * Get detailed RAG analysis for an existing session
 */
export async function getDetailedAnalysis(
  sessionId: string,
  originalQuery: string,
  detectionResult: any,
  chatHistory: ChatMessage[] = []
): Promise<ChatResponse> {
  // Create enhanced contextual query based on user's original question and detection results
  const crop = detectionResult?.crop || "crop";
  const disease = detectionResult?.disease || "condition";
  const confidence = detectionResult?.confidence || 0;

  // Build comprehensive detailed query based on user's original intent
  let detailedQuery = "";

  const queryLower = originalQuery.toLowerCase();

  if (
    queryLower.includes("what") &&
    (queryLower.includes("spots") || queryLower.includes("dots") || queryLower.includes("marks"))
  ) {
    // Identification query - provide comprehensive disease information
    detailedQuery = `Based on the ${disease} detected on ${crop} with ${(confidence * 100).toFixed(0)}% confidence, please provide:

1. **Complete Disease Information**: Detailed explanation of ${disease}, its causes, and how it spreads
2. **Symptoms & Identification**: How to identify ${disease} at different stages, what the spots/lesions look like, and how they develop
3. **Immediate Treatment**: Step-by-step treatment protocol including specific fungicides, application rates, and timing
4. **Prevention Strategies**: Long-term prevention methods, resistant varieties, cultural practices
5. **Management Timeline**: When to apply treatments, monitoring schedule, and follow-up actions
6. **Local Context**: Specific advice for Indian/West Bengal agricultural conditions, seasonal considerations
7. **Cost-Effective Solutions**: Both chemical and organic treatment options with expected costs

Please provide detailed, actionable guidance that a farmer can immediately implement.`;
  } else if (
    queryLower.includes("treat") ||
    queryLower.includes("cure") ||
    queryLower.includes("control")
  ) {
    // Treatment focused query
    detailedQuery = `The farmer is asking about treatment for ${disease} on ${crop}. Please provide comprehensive treatment guidance:

1. **Immediate Actions**: What to do right now to stop disease progression
2. **Treatment Protocol**: Detailed fungicide recommendations with specific products, concentrations, and application methods
3. **Application Schedule**: Exact timing, frequency, and conditions for treatment
4. **Monitoring & Assessment**: How to evaluate treatment effectiveness and when to reapply
5. **Integrated Management**: Combining chemical, biological, and cultural control methods
6. **Resistance Management**: How to prevent fungicide resistance development
7. **Cost Analysis**: Treatment costs vs potential yield losses
8. **Local Availability**: Products commonly available in Indian markets

Provide step-by-step instructions that are practical for field implementation.`;
  } else if (
    queryLower.includes("prevent") ||
    queryLower.includes("avoid") ||
    queryLower.includes("stop")
  ) {
    // Prevention focused query
    detailedQuery = `The farmer wants to prevent ${disease} on ${crop}. Please provide comprehensive prevention strategies:

1. **Preventive Fungicide Program**: Seasonal spray schedule and preventive treatments
2. **Cultural Practices**: Crop rotation, planting density, irrigation management
3. **Resistant Varieties**: Disease-resistant cultivars suitable for local conditions
4. **Field Sanitation**: Proper field hygiene and residue management
5. **Environmental Management**: Optimizing field conditions to reduce disease pressure
6. **Monitoring Systems**: Early detection methods and scouting protocols
7. **Seasonal Planning**: Timing of preventive measures based on local climate
8. **Economic Considerations**: Cost-benefit analysis of prevention vs treatment

Focus on practical, implementable prevention strategies for Indian farming conditions.`;
  } else {
    // General query - provide comprehensive overview
    detailedQuery = `Based on the user's question "${originalQuery}" and the detected ${disease} on ${crop}, please provide comprehensive agricultural guidance:

1. **Disease Analysis**: Complete overview of ${disease} including symptoms, causes, and impact
2. **Treatment Options**: Both immediate and long-term treatment strategies with specific product recommendations
3. **Prevention Methods**: Integrated disease management approaches including cultural, biological, and chemical controls
4. **Best Practices**: Optimal farming practices for ${crop} to minimize disease risks
5. **Seasonal Guidance**: Timing of interventions based on crop growth stages and local climate
6. **Economic Impact**: Cost analysis of treatment vs potential yield losses
7. **Local Context**: Specific recommendations for Indian agricultural conditions, market availability of inputs
8. **Follow-up Actions**: Monitoring protocols and long-term management strategies

Provide detailed, actionable advice that addresses the farmer's specific concern while offering comprehensive crop management guidance.`;
  }

  const response = await fetch(`${API_BASE_URL}/api/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      message: detailedQuery,
      chat_history: chatHistory,
    }),
  });

  if (!response.ok) {
    const errorData: ApiError = await response.json();
    throw new Error(errorData.detail || "Detailed analysis failed");
  }

  return response.json();
}

/**
 * Chat follow-up endpoint - continues conversation
 */
export async function chatFollowUp(
  message: string,
  chatHistory: ChatMessage[] = []
): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE_URL}/api/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      message,
      chat_history: chatHistory,
    }),
  });

  if (!response.ok) {
    const errorData: ApiError = await response.json();
    throw new Error(errorData.detail || "Chat failed");
  }

  return response.json();
}

/**
 * Process user query with detection context (only if user provided a query)
 */
export async function processQueryWithDetection(detectionResult: {
  plant: string;
  status: "healthy" | "diseased";
  disease?: string;
  confidence: number;
  session_id: string;
  user_query?: string;
}): Promise<ChatResponse> {
  if (!detectionResult.user_query) {
    throw new Error("No user query to process");
  }

  const { plant, status, disease, confidence, user_query } = detectionResult;

  // Build contextual query based on detection results + user question
  let contextualQuery = "";

  if (status === "healthy") {
    contextualQuery = `Based on the image analysis, this appears to be a healthy ${plant} (confidence: ${(confidence * 100).toFixed(0)}%). 
    
User question: "${user_query}"

Please provide helpful guidance about this ${plant} addressing their question, including general care tips, preventive measures, and best practices for maintaining plant health.`;
  } else {
    contextualQuery = `Based on the image analysis, this ${plant} appears to have ${disease} (confidence: ${(confidence * 100).toFixed(0)}%). 
    
User question: "${user_query}"

Please provide comprehensive guidance addressing their specific question about this ${disease} on ${plant}, including immediate actions, treatment options, and management strategies.`;
  }

  const response = await fetch(`${API_BASE_URL}/api/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      message: contextualQuery,
      chat_history: [],
    }),
  });

  if (!response.ok) {
    const errorData: ApiError = await response.json();
    throw new Error(errorData.detail || "Query processing failed");
  }

  return response.json();
}
