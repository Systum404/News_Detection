export interface AnalyzeResponse {
  result: string;
  is_fake: boolean;
  verdict: "FAKE" | "AUTHENTIC" | "UNCONFIRMED";
  claim_sha256?: string;
  is_polarized?: boolean;
  polarization_index?: number;
  confidence: string;
  agreement: string;
  soft_score: number;
  soft_prob_fake: number;
  soft_prob_auth: number;
  bar_width: string;
  bar_color: string;
  color: string;
  bg_glow: string;
  raw_color: string;
  snippet: string;
  is_short: boolean;
  vote1: string;
  score1: string;
  is_fake1: boolean;
  vote2: string;
  score2: string;
  is_fake2: boolean;
  vote3: string;
  score3: string;
  is_fake3: boolean;
  label1: string;
  label2: string;
  label3: string;
  vote4?: string;
  score4?: string;
  is_fake4?: boolean;
  label4?: string;
  has_xlmr?: boolean;
  total_models?: number;
  language_code: string;
  language_name: string;
  is_multilingual: boolean;
  routing_engine: string;
  acceleration_device?: string;
  hardware_accelerated?: boolean;
  precision?: string;
  fact_check?: FactCheckData;
  article_meta?: ScrapedArticleMeta;
  category?: CategoryPrediction;
  calibration?: {
    enabled: boolean;
    temperature_scaled: boolean;
    temperatures?: Record<string, number>;
  };
  cached?: boolean;
  cache_hit?: boolean;
  cache_key?: string;
  cache_tier?: "L1_RAM" | "L2_SQLITE" | string;
  execution_time_ms?: number;
}

export interface ScrapedArticleMeta {
  url: string;
  domain: string;
  publisher: string;
  title: string;
  author?: string | null;
  published_date?: string | null;
  image_url?: string | null;
  favicon?: string;
  word_count: number;
  char_count: number;
}

export type NewsCategoryName =
  | "Politics"
  | "Technology"
  | "Business"
  | "Sports"
  | "Science"
  | "Entertainment"
  | "International News";

export interface CategoryPrediction {
  category: NewsCategoryName | string;
  confidence: number;
  distribution: Record<string, number>;
  runner_up?: {
    category: string;
    confidence: number;
  };
  matched_signals: string[];
  is_ambiguous?: boolean;
}


export interface FactCheckSource {
  title: string;
  publisher: string;
  url: string;
  published_at?: string;
  is_authoritative: boolean;
  type?: "news_article" | "encyclopedia";
  snippet?: string;
}

export interface FactCheckData {
  corroboration_score: number;
  corroboration_status:
    | "HIGHLY_CORROBORATED"
    | "PARTIALLY_CORROBORATED"
    | "WEAK_CORROBORATION"
    | "UNVERIFIED_CLAIM"
    | "INSUFFICIENT_INPUT"
    | "SKIPPED"
    | "TIMEOUT"
    | "HISTORICAL_FACT_GROUNDED"
    | string;
  summary: string;
  sources: FactCheckSource[];
  authoritative_count: number;
  total_found: number;
  search_query_used: string;
  cached?: boolean;
  is_historical?: boolean;
  temporal_metadata?: {
    is_historical: boolean;
    historical_fact: boolean;
    verified_years?: number[];
    knowledge_source?: string;
    entity_title?: string;
  };
}


export interface AcceleratorStatus {
  active_device: string;
  device_name: string;
  half_precision_fp16: boolean;
  mps_available: boolean;
  cuda_available: boolean;
  onnx_available: boolean;
  onnx_version: string | null;
  onnx_providers: string[];
  coreml_available: boolean;
  quantization_modes: string[];
}

export interface BenchmarkResult {
  device_name: string;
  active_device: string;
  precision: string;
  cpu_latency_ms: number;
  accelerated_latency_ms: number;
  speedup_factor: string;
  workload: string;
}

export interface BackendHealth {
  status: string;
  models_loaded: boolean;
  models: string[];
  multilingual_support: boolean;
  concurrency_limit: number;
  timeout_seconds: number;
  runtime: string;
  hardware_acceleration?: AcceleratorStatus;
}

export interface AnalysisHistoryItem {
  id: string;
  timestamp: string;
  textSnippet: string;
  response: AnalyzeResponse;
}

export interface SamplePreset {
  id: string;
  title: string;
  langTag: string;
  category: "authentic" | "fake";
  text: string;
}
