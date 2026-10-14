import reflex as rx
from transformers import pipeline
from typing import Optional, Tuple, Dict, Any

# --- AI ENSEMBLE CONFIGURATION (LAZY-LOADED) ---
print("--- INITIALIZING ENSEMBLE FAKE NEWS DETECTION SYSTEM ---")

pipe1 = None  # RoBERTa Base
pipe2 = None  # Distilled TinyBERT
pipe3 = None  # BERT-Detect
fact_checker = None  # Zero-shot NLI fact verification
classifier = None  # Backward-compatibility alias


def get_pipes() -> Tuple[object, object, object]:
    global pipe1, pipe2, pipe3
    if pipe1 is None:
        print("1. Loading RoBERTa Base Model (hamzab/roberta-fake-news-classification)...")
        pipe1 = pipeline("text-classification", model="hamzab/roberta-fake-news-classification")
    if pipe2 is None:
        print("2. Loading Distilled TinyBERT Model (mrm8488/bert-tiny-finetuned-fake-news-detection)...")
        pipe2 = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news-detection")
    if pipe3 is None:
        print("3. Loading BERT-Detect Model (jy46604790/Fake-News-Bert-Detect)...")
        pipe3 = pipeline("text-classification", model="jy46604790/Fake-News-Bert-Detect")
    print("--- ENSEMBLE SYSTEM READY ---")
    return pipe1, pipe2, pipe3


def get_fact_checker():
    global fact_checker
    if fact_checker is None:
        print("Loading Smart Fact Verification Model (facebook/bart-large-mnli)...")
        fact_checker = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    return fact_checker


def are_models_loaded() -> bool:
    return all(p is not None for p in (pipe1, pipe2, pipe3))


def analyze_text(
    text: str,
    categories: Optional[str] = None,
    pipes: Optional[Tuple[object, object, object]] = None,
) -> Dict[str, Any]:
    global pipe1, pipe2, pipe3

    sanitized = text.strip() if text else ""
    if not sanitized:
        raise ValueError("Article text cannot be empty.")

    words = sanitized.split()
    is_short = len(words) < 15

    # Short query (< 15 words): Evaluate with Smart Fact Verification layer
    if is_short:
        fc = get_fact_checker()
        c_set = ["common knowledge and true", "conspiracy theory or falsehood"]
        f_res = fc(sanitized, candidate_labels=c_set)
        top_fact = f_res["labels"][0]
        fact_score = float(f_res["scores"][0])
        is_fact_true = (top_fact == c_set[0]) and (fact_score >= 0.55)

        score_str = f"{fact_score * 100:.1f}%"
        snippet = sanitized[:60] + "..." if len(sanitized) > 60 else sanitized

        if is_fact_true:
            final_res = "AUTHENTIC FACT"
            final_color = "text-emerald-400"
            bg_glow = "shadow-emerald-500/20"
            bar_color = "bg-emerald-500"
            raw_color = "#34d399"
            final_conf = f"Fact Check Score ({score_str})"
            bar_width = f"{fact_score * 100:.0f}%"

            return {
                "result": final_res,
                "is_fake": False,
                "verdict": "AUTHENTIC",
                "confidence": final_conf,
                "agreement": "Fact Check",
                "bar_width": bar_width,
                "bar_color": bar_color,
                "color": final_color,
                "bg_glow": bg_glow,
                "raw_color": raw_color,
                "snippet": snippet,
                "is_short": True,
                "vote1": "AUTHENTIC (Fact Verified)",
                "score1": score_str,
                "is_fake1": False,
                "vote2": "AUTHENTIC (Fact Verified)",
                "score2": score_str,
                "is_fake2": False,
                "vote3": "AUTHENTIC (Fact Verified)",
                "score3": score_str,
                "is_fake3": False,
                "label1": "AUTHENTIC",
                "label2": "AUTHENTIC",
                "label3": "AUTHENTIC",
            }
        else:
            final_res = "FALSEHOOD / FAKE CLAIM"
            final_color = "text-red-400"
            bg_glow = "shadow-red-500/20"
            bar_color = "bg-red-500"
            raw_color = "#f87171"
            final_conf = f"Fact Check Score ({score_str})"
            bar_width = f"{fact_score * 100:.0f}%"

            return {
                "result": final_res,
                "is_fake": True,
                "verdict": "FAKE",
                "confidence": final_conf,
                "agreement": "Fact Check",
                "bar_width": bar_width,
                "bar_color": bar_color,
                "color": final_color,
                "bg_glow": bg_glow,
                "raw_color": raw_color,
                "snippet": snippet,
                "is_short": True,
                "vote1": "FAKE (Fact Flagged)",
                "score1": score_str,
                "is_fake1": True,
                "vote2": "FAKE (Fact Flagged)",
                "score2": score_str,
                "is_fake2": True,
                "vote3": "FAKE (Fact Flagged)",
                "score3": score_str,
                "is_fake3": True,
                "label1": "FAKE",
                "label2": "FAKE",
                "label3": "FAKE",
            }

    # For full news text (>= 15 words): Run the 3-Model Transformer Ensemble
    if pipes is not None:
        p1, p2, p3 = pipes
    else:
        p1, p2, p3 = get_pipes()

    # Truncate text cleanly for context window
    truncated_text = sanitized[:2000]

    # Model 1: RoBERTa Base (hamzab/roberta-fake-news-classification)
    # Outputs: 'FAKE' or 'TRUE'
    res1 = p1(truncated_text, truncation=True, max_length=512)[0]
    label1_raw = str(res1.get("label", "")).upper()
    score1_val = float(res1.get("score", 0.0))
    is_fake1 = "FAKE" in label1_raw
    vote1 = "FAKE" if is_fake1 else "AUTHENTIC"
    score1_str = f"{score1_val * 100:.1f}%"

    # Model 2: Distilled TinyBERT (mrm8488/bert-tiny-finetuned-fake-news-detection)
    # Outputs: LABEL_1 (FAKE), LABEL_0 (REAL)
    res2 = p2(truncated_text, truncation=True, max_length=512)[0]
    label2_raw = str(res2.get("label", "")).upper()
    score2_val = float(res2.get("score", 0.0))
    is_fake2 = label2_raw == "LABEL_1"
    vote2 = "FAKE" if is_fake2 else "AUTHENTIC"
    score2_str = f"{score2_val * 100:.1f}%"

    # Model 3: BERT-Detect (jy46604790/Fake-News-Bert-Detect)
    # Outputs: LABEL_0 (FAKE), LABEL_1 (REAL)
    res3 = p3(truncated_text, truncation=True, max_length=512)[0]
    label3_raw = str(res3.get("label", "")).upper()
    score3_val = float(res3.get("score", 0.0))
    is_fake3 = label3_raw == "LABEL_0"
    vote3 = "FAKE" if is_fake3 else "AUTHENTIC"
    score3_str = f"{score3_val * 100:.1f}%"

    # Majority Voting Logic
    votes = [vote1, vote2, vote3]
    fake_count = votes.count("FAKE")
    authentic_count = votes.count("AUTHENTIC")
    is_fake = fake_count >= 2

    if is_fake:
        final_res = "FAKE NEWS DETECTED"
        final_color = "text-red-400"
        bg_glow = "shadow-red-500/20"
        bar_color = "bg-red-500"
        raw_color = "#f87171"
        agreement = fake_count
    else:
        final_res = "AUTHENTIC NEWS"
        final_color = "text-emerald-400"
        bg_glow = "shadow-emerald-500/20"
        bar_color = "bg-emerald-500"
        raw_color = "#34d399"
        agreement = authentic_count

    conf_pct = (agreement / 3.0) * 100.0
    final_conf = f"{agreement}/3 Models ({conf_pct:.1f}%)"
    bar_width = f"{conf_pct:.0f}%"

    snippet = sanitized[:60] + "..." if len(sanitized) > 60 else sanitized

    return {
        "result": final_res,
        "is_fake": is_fake,
        "verdict": "FAKE" if is_fake else "AUTHENTIC",
        "confidence": final_conf,
        "agreement": f"{agreement}/3",
        "bar_width": bar_width,
        "bar_color": bar_color,
        "color": final_color,
        "bg_glow": bg_glow,
        "raw_color": raw_color,
        "snippet": snippet,
        "is_short": False,
        # Model details
        "vote1": vote1,
        "score1": score1_str,
        "is_fake1": is_fake1,
        "vote2": vote2,
        "score2": score2_str,
        "is_fake2": is_fake2,
        "vote3": vote3,
        "score3": score3_str,
        "is_fake3": is_fake3,
        # Backward-compatible labels
        "label1": vote1,
        "label2": vote2,
        "label3": vote3,
    }


class State(rx.State):
    news_content: str = ""
    result: str = ""
    status_type: str = ""  # "fake", "authentic", "warning", "loading", ""
    confidence_score: str = ""
    bar_width: str = "0%"
    loading: bool = False

    # Model 1: RoBERTa
    vote1: str = ""
    score1: str = ""
    is_fake1: bool = False

    # Model 2: TinyBERT
    vote2: str = ""
    score2: str = ""
    is_fake2: bool = False

    # Model 3: BERT-Detect
    vote3: str = ""
    score3: str = ""
    is_fake3: bool = False
    is_short_text: bool = False

    # History entries: [snippet, result, is_fake_str ("true"/"false"), confidence_score]
    history: list[list[str]] = []

    def set_news_content(self, value: str):
        self.news_content = value

    def clear_form(self):
        self.news_content = ""
        self.result = ""
        self.status_type = ""
        self.confidence_score = ""
        self.bar_width = "0%"
        self.loading = False
        self.vote1 = ""
        self.score1 = ""
        self.is_fake1 = False
        self.vote2 = ""
        self.score2 = ""
        self.is_fake2 = False
        self.vote3 = ""
        self.score3 = ""
        self.is_fake3 = False
        self.is_short_text = False

    def load_sample_real(self):
        self.news_content = (
            "WASHINGTON (Reuters) - The U.S. Senate approved bipartisan legislation "
            "to fund federal agencies, following extensive negotiations between congressional leadership."
        )
        self.result = ""
        self.status_type = ""
        self.is_short_text = False

    def load_sample_fake(self):
        self.news_content = (
            "BREAKING EXCLUSIVE: Secret government underground base discovered where alien "
            "technology is being used to control international weather patterns and manipulate elections!"
        )
        self.result = ""
        self.status_type = ""
        self.is_short_text = False

    def analyze_news(self):
        content = self.news_content.strip() if self.news_content else ""
        if not content:
            self.result = "Please paste an article first."
            self.status_type = "warning"
            self.bar_width = "0%"
            self.confidence_score = ""
            self.vote1 = ""
            self.vote2 = ""
            self.vote3 = ""
            self.is_short_text = False
            return

        self.is_short_text = len(content.split()) < 15
        self.result = "Analyzing Article..."
        self.status_type = "loading"
        self.loading = True
        yield

        try:
            res = analyze_text(content)
            self.result = res["result"]
            is_fake = res["is_fake"]
            self.status_type = "fake" if is_fake else "authentic"
            self.confidence_score = res["confidence"]
            self.bar_width = res["bar_width"]

            self.vote1 = res["vote1"]
            self.score1 = res["score1"]
            self.is_fake1 = res["is_fake1"]

            self.vote2 = res["vote2"]
            self.score2 = res["score2"]
            self.is_fake2 = res["is_fake2"]

            self.vote3 = res["vote3"]
            self.score3 = res["score3"]
            self.is_fake3 = res["is_fake3"]

            snippet = res["snippet"]
            is_fake_str = "true" if is_fake else "false"
            self.history.insert(0, [snippet, res["result"], is_fake_str, res["confidence"]])
        except Exception as e:
            self.result = f"System Error: {e}"
            self.status_type = "warning"
            print(f"Error during analysis: {e}")
        finally:
            self.loading = False


def header_component():
    return rx.box(
        rx.box(
            rx.text(
                "ENSEMBLE AI FACT-CHECKER",
                class_name="text-[11px] font-extrabold tracking-widest text-indigo-400 uppercase bg-indigo-500/10 border border-indigo-500/30 px-3 py-1 rounded-full w-fit mb-3",
            ),
            rx.heading(
                "Veritas",
                class_name="text-5xl font-black tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-indigo-300 via-purple-300 to-pink-300",
            ),
            rx.text(
                "Multi-Model Deep Learning Fake News Detection System",
                class_name="text-gray-400 mt-2 text-base font-medium",
            ),
            class_name="w-full",
        ),
        class_name="mb-8 w-full",
    )


def model_vote_card(model_name: str, subtitle: str, vote: rx.Var[str], score: rx.Var[str], is_fake: rx.Var[bool]):
    return rx.box(
        rx.text(model_name, class_name="text-sm font-bold text-white tracking-wide"),
        rx.text(subtitle, class_name="text-xs text-gray-400 mb-3"),
        rx.cond(
            vote == "",
            rx.text("WAITING", class_name="text-gray-500 font-bold text-sm tracking-wider animate-pulse"),
            rx.box(
                rx.cond(
                    is_fake,
                    rx.box(
                        rx.text("FAKE", class_name="text-red-400 font-extrabold text-lg"),
                        class_name="bg-red-500/10 border border-red-500/30 rounded-lg px-4 py-1 mb-2",
                    ),
                    rx.box(
                        rx.text("AUTHENTIC", class_name="text-emerald-400 font-extrabold text-lg"),
                        class_name="bg-emerald-500/10 border border-emerald-500/30 rounded-lg px-4 py-1 mb-2",
                    ),
                ),
                rx.text("Confidence: " + score, class_name="text-gray-300 text-xs font-mono"),
                class_name="w-full flex flex-col items-center",
            ),
        ),
        class_name="bg-white/5 border border-white/10 rounded-xl p-4 flex flex-col items-center justify-center text-center backdrop-blur-md transition-all hover:border-white/20 w-full",
    )


def results_dashboard():
    return rx.box(
        rx.cond(
            State.status_type != "",
            rx.box(
                # Main Verdict Card
                rx.cond(
                    State.status_type == "fake",
                    rx.box(
                        rx.text("ENSEMBLE VERDICT", class_name="text-xs text-red-300 uppercase tracking-widest font-semibold mb-2"),
                        rx.text(State.result, class_name="text-3xl font-extrabold text-red-400 drop-shadow-md"),
                        class_name="bg-red-950/30 border border-red-500/30 rounded-2xl p-6 text-center shadow-2xl shadow-red-500/20 backdrop-blur-lg mb-8",
                    ),
                    rx.cond(
                        State.status_type == "authentic",
                        rx.box(
                            rx.text("ENSEMBLE VERDICT", class_name="text-xs text-emerald-300 uppercase tracking-widest font-semibold mb-2"),
                            rx.text(State.result, class_name="text-3xl font-extrabold text-emerald-400 drop-shadow-md"),
                            class_name="bg-emerald-950/30 border border-emerald-500/30 rounded-2xl p-6 text-center shadow-2xl shadow-emerald-500/20 backdrop-blur-lg mb-8",
                        ),
                        rx.cond(
                            State.status_type == "loading",
                            rx.box(
                                rx.text("ANALYSIS IN PROGRESS", class_name="text-xs text-indigo-300 uppercase tracking-widest font-semibold mb-2"),
                                rx.text("Analyzing Article Across 3 Neural Networks...", class_name="text-2xl font-bold text-indigo-400 animate-pulse"),
                                class_name="bg-indigo-950/30 border border-indigo-500/30 rounded-2xl p-6 text-center shadow-2xl shadow-indigo-500/20 backdrop-blur-lg mb-8",
                            ),
                            # Warning (empty input)
                            rx.box(
                                rx.text("INPUT REQUIRED", class_name="text-xs text-yellow-300 uppercase tracking-widest font-semibold mb-2"),
                                rx.text(State.result, class_name="text-2xl font-bold text-yellow-400"),
                                class_name="bg-yellow-950/30 border border-yellow-500/30 rounded-2xl p-6 text-center shadow-2xl shadow-yellow-500/20 backdrop-blur-lg mb-8",
                            ),
                        ),
                    ),
                ),
                # Smart Fact Verification Notice
                rx.cond(
                    State.is_short_text & ((State.status_type == "fake") | (State.status_type == "authentic")),
                    rx.box(
                        rx.hstack(
                            rx.icon(tag="sparkles", class_name="h-4 w-4 text-cyan-400 shrink-0 mt-0.5"),
                            rx.box(
                                rx.text(
                                    "Smart Fact-Checking Active (< 15 words):",
                                    class_name="text-xs font-bold text-cyan-300 mb-0.5",
                                ),
                                rx.text(
                                    "Short factual claims are verified using our Zero-Shot NLI Factuality Layer. Full news articles (>= 15 words) utilize the 3-model Transformer Ensemble.",
                                    class_name="text-xs text-cyan-200/90 leading-relaxed",
                                ),
                            ),
                            spacing="3",
                            align="start",
                        ),
                        class_name="bg-cyan-950/40 border border-cyan-500/30 rounded-xl p-3 mb-6 backdrop-blur-md",
                    ),
                ),
                # Confidence Progress Bar
                rx.cond(
                    (State.status_type == "fake") | (State.status_type == "authentic"),
                    rx.box(
                        rx.hstack(
                            rx.text("Ensemble Agreement", class_name="text-sm font-medium text-gray-400"),
                            rx.text(State.confidence_score, class_name="text-sm font-bold text-white font-mono"),
                            justify="between",
                            class_name="mb-2",
                        ),
                        rx.box(
                            rx.cond(
                                State.status_type == "fake",
                                rx.box(class_name="h-2 rounded-full bg-red-500 transition-all duration-1000 ease-out", width=State.bar_width),
                                rx.box(class_name="h-2 rounded-full bg-emerald-500 transition-all duration-1000 ease-out", width=State.bar_width),
                            ),
                            class_name="w-full bg-gray-800 rounded-full h-2 mb-8 overflow-hidden border border-white/5",
                        ),
                        width="100%",
                    ),
                ),
                # 3 Individual Model Cards
                rx.cond(
                    (State.status_type == "fake") | (State.status_type == "authentic"),
                    rx.box(
                        rx.text("Individual Neural Network Breakdown", class_name="text-xs text-gray-400 uppercase tracking-widest font-semibold mb-3"),
                        rx.grid(
                            model_vote_card("RoBERTa Base", "Contextual NLP", State.vote1, State.score1, State.is_fake1),
                            model_vote_card("Distilled TinyBERT", "Distilled Student", State.vote2, State.score2, State.is_fake2),
                            model_vote_card("BERT-Detect", "Fine-Tuned Classifier", State.vote3, State.score3, State.is_fake3),
                            columns="3",
                            spacing="4",
                            width="100%",
                        ),
                        width="100%",
                    ),
                ),
                class_name="animate-fade-in-up",
            ),
            # Empty / Awaiting State
            rx.box(
                rx.icon(tag="shield-alert", class_name="h-16 w-16 text-indigo-400/60 mb-4 mx-auto"),
                rx.text("Awaiting Article...", class_name="text-gray-300 font-semibold text-xl"),
                rx.text(
                    "Paste any news article, headline, or excerpt on the left to evaluate its authenticity using our 3-model deep learning ensemble.",
                    class_name="text-gray-500 text-sm mt-2 max-w-sm text-center leading-relaxed",
                ),
                class_name="bg-white/5 border border-white/10 border-dashed rounded-2xl h-full min-h-[340px] flex flex-col items-center justify-center p-8 backdrop-blur-md",
            ),
        ),
        class_name="w-full h-full flex flex-col justify-center",
    )


def history_sidebar():
    return rx.box(
        rx.hstack(
            rx.icon(tag="history", class_name="h-4 w-4 text-gray-400"),
            rx.text("Recent Scans", class_name="text-sm font-semibold text-gray-300 uppercase tracking-widest"),
            align="center",
            class_name="mb-4",
        ),
        rx.cond(
            State.history,
            rx.vstack(
                rx.foreach(
                    State.history,
                    lambda item: rx.box(
                        rx.text(item[0], class_name="text-gray-300 text-xs truncate mb-2 font-serif"),
                        rx.hstack(
                            rx.cond(
                                item[2] == "true",
                                rx.hstack(
                                    rx.box(class_name="w-2 h-2 rounded-full bg-red-400"),
                                    rx.text("FAKE NEWS", class_name="text-[11px] font-bold text-red-400"),
                                    align="center",
                                    spacing="1",
                                ),
                                rx.hstack(
                                    rx.box(class_name="w-2 h-2 rounded-full bg-emerald-400"),
                                    rx.text("AUTHENTIC", class_name="text-[11px] font-bold text-emerald-400"),
                                    align="center",
                                    spacing="1",
                                ),
                            ),
                            rx.text(item[3], class_name="text-[11px] text-gray-500 font-mono"),
                            justify="between",
                            align="center",
                            width="100%",
                        ),
                        class_name="bg-white/5 hover:bg-white/10 transition-colors border border-white/5 rounded-xl p-3 w-full cursor-default",
                    ),
                ),
                spacing="3",
                class_name="w-full overflow-y-auto max-h-[70vh] pr-1 custom-scrollbar",
            ),
            rx.box(
                rx.text("No scans recorded yet in this session.", class_name="text-gray-500 text-xs italic"),
                class_name="p-4 bg-white/5 rounded-xl border border-white/5",
            ),
        ),
        class_name="w-full lg:w-80 bg-black/30 p-6 rounded-3xl border border-white/10 backdrop-blur-xl h-fit",
    )


def main_content():
    return rx.box(
        header_component(),
        rx.box(
            rx.hstack(
                rx.text("Article Text Content", class_name="text-sm font-semibold text-gray-400 uppercase tracking-widest"),
                rx.hstack(
                    rx.text("Try sample:", class_name="text-xs text-gray-500 self-center"),
                    rx.button(
                        "Real News",
                        on_click=State.load_sample_real,
                        size="1",
                        variant="surface",
                        class_name="text-xs bg-emerald-500/10 text-emerald-300 hover:bg-emerald-500/20 border border-emerald-500/30 cursor-pointer rounded-md px-2 py-1",
                    ),
                    rx.button(
                        "Fake News",
                        on_click=State.load_sample_fake,
                        size="1",
                        variant="surface",
                        class_name="text-xs bg-red-500/10 text-red-300 hover:bg-red-500/20 border border-red-500/30 cursor-pointer rounded-md px-2 py-1",
                    ),
                    spacing="2",
                    align="center",
                ),
                justify="between",
                align="center",
                class_name="mb-2",
            ),
            rx.text_area(
                placeholder="Paste news article, press release, or headline here to evaluate with our ensemble AI...",
                value=State.news_content,
                on_change=State.set_news_content,
                class_name="w-full h-64 bg-gray-900/60 border border-gray-700/80 rounded-2xl p-6 text-gray-200 placeholder-gray-600 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-all resize-none shadow-inner text-base font-serif leading-relaxed",
            ),
            rx.hstack(
                rx.button(
                    rx.cond(
                        State.loading,
                        rx.hstack(
                            rx.icon(tag="loader", class_name="animate-spin mr-2 h-5 w-5"),
                            "Running Neural Ensemble...",
                        ),
                        rx.hstack(
                            rx.icon(tag="shield-check", class_name="mr-2 h-5 w-5"),
                            "Analyze News",
                        ),
                    ),
                    on_click=State.analyze_news,
                    disabled=State.loading,
                    class_name="bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white font-bold py-4 px-8 rounded-xl shadow-lg shadow-indigo-500/30 transition-all hover:-translate-y-0.5 w-full sm:w-auto cursor-pointer",
                ),
                rx.button(
                    "Clear",
                    on_click=State.clear_form,
                    class_name="bg-gray-800/80 hover:bg-gray-700 text-gray-300 font-semibold py-4 px-8 rounded-xl border border-gray-700 transition-all w-full sm:w-auto cursor-pointer",
                ),
                spacing="4",
                class_name="mt-6 flex-wrap",
            ),
            class_name="mb-10",
        ),
        results_dashboard(),
        class_name="w-full lg:flex-1 flex flex-col",
    )


def index() -> rx.Component:
    return rx.box(
        rx.box(
            class_name="absolute top-0 left-1/4 w-96 h-96 bg-purple-600/20 rounded-full blur-[140px] pointer-events-none"
        ),
        rx.box(
            class_name="absolute bottom-0 right-1/4 w-[30rem] h-[30rem] bg-indigo-600/15 rounded-full blur-[160px] pointer-events-none"
        ),
        rx.box(
            rx.hstack(
                main_content(),
                history_sidebar(),
                class_name="flex flex-col lg:flex-row gap-10 w-full max-w-7xl mx-auto items-start",
            ),
            class_name="relative z-10 w-full min-h-screen p-6 sm:p-10 lg:p-14 flex justify-center",
        ),
        class_name="bg-[#0B0F19] min-h-screen text-slate-200 font-sans relative overflow-x-hidden selection:bg-purple-500/30",
    )


app = rx.App(
    theme=rx.theme(
        appearance="dark",
        has_background=True,
    ),
    stylesheets=[
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap",
    ],
)
app.add_page(index, title="Veritas | Ensemble Fake News Detection")