import logging
import reflex as rx

from .ensemble import (
    pipe1,
    pipe2,
    pipe3,
    get_pipes,
    are_models_loaded,
    extract_model_probabilities,
    analyze_text,
)

logger = logging.getLogger("news_detection.reflex")


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

    # Multilingual State
    detected_lang: str = "English"
    lang_code: str = "en"
    is_multilingual: bool = False
    routing_engine: str = "3-Model Deep Learning Ensemble"

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
        self.detected_lang = "English"
        self.lang_code = "en"
        self.is_multilingual = False
        self.routing_engine = "3-Model Deep Learning Ensemble"

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

    def load_sample_hindi(self):
        self.news_content = (
            "नई दिल्ली: भारतीय अंतरिक्ष अनुसंधान संगठन (ISRO) ने आज अपने महत्वपूर्ण मौसम उपग्रह का "
            "सफल प्रक्षेपण श्रीहरिकोटा से किया, जिससे वैज्ञानिक समुदाय में नई उपलब्धि दर्ज हुई।"
        )
        self.result = ""
        self.status_type = ""
        self.is_short_text = False

    def load_sample_spanish(self):
        self.news_content = (
            "MADRID (EFE) - El Ministerio de Economía anunció hoy medidas fiscales para contener "
            "la inflación y respaldar la producción industrial durante el próximo trimestre."
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

            self.detected_lang = res.get("language_name", "English")
            self.lang_code = res.get("language_code", "en")
            self.is_multilingual = res.get("is_multilingual", False)
            self.routing_engine = res.get("routing_engine", "3-Model Deep Learning Ensemble")

            snippet = res["snippet"]
            is_fake_str = "true" if is_fake else "false"
            self.history.insert(0, [snippet, res["result"], is_fake_str, res["confidence"]])
            self.history = self.history[:50]
        except Exception as e:
            self.result = f"System Error: {e}"
            self.status_type = "warning"
            logger.exception(f"Error during analysis: {e}")
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
                "News Detection",
                class_name="text-5xl font-black tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-indigo-300 via-purple-300 to-pink-300",
            ),
            rx.text(
                "Multi-Model Deep Learning System",
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
                # Short Claim Mode Notice
                rx.cond(
                    State.is_short_text & ((State.status_type == "fake") | (State.status_type == "authentic")),
                    rx.box(
                        rx.hstack(
                            rx.icon(tag="info", class_name="h-4 w-4 text-cyan-400 shrink-0 mt-0.5"),
                            rx.box(
                                rx.text(
                                    "Headline / Short Claim Mode (< 15 words):",
                                    class_name="text-xs font-bold text-cyan-300 mb-0.5",
                                ),
                                rx.text(
                                    "All 3 neural networks evaluated this claim based on headline veracity and linguistic patterns. Note: Full articles provide richer contextual cues.",
                                    class_name="text-xs text-cyan-200/90 leading-relaxed",
                                ),
                            ),
                            spacing="3",
                            align="start",
                        ),
                        class_name="bg-cyan-950/40 border border-cyan-500/30 rounded-xl p-3 mb-6 backdrop-blur-md",
                    ),
                ),
                # Detected Language Badge
                rx.cond(
                    (State.status_type == "fake") | (State.status_type == "authentic"),
                    rx.box(
                        rx.hstack(
                            rx.hstack(
                                rx.icon(tag="globe", class_name="h-4 w-4 text-indigo-400 shrink-0"),
                                rx.text("Detected Language: ", class_name="text-xs text-gray-400 font-medium"),
                                rx.text(State.detected_lang, class_name="text-xs font-bold text-indigo-300"),
                                align="center",
                                spacing="2",
                            ),
                            rx.box(
                                rx.text(
                                    State.routing_engine,
                                    class_name="text-[10px] font-mono font-semibold px-2.5 py-1 rounded-full bg-white/5 border border-white/10 text-gray-300",
                                ),
                            ),
                            justify="between",
                            align="center",
                            width="100%",
                        ),
                        class_name="bg-indigo-950/30 border border-indigo-500/20 rounded-xl px-4 py-2.5 mb-6 backdrop-blur-md",
                    ),
                ),
                # Confidence Progress Bar
                rx.cond(
                    (State.status_type == "fake") | (State.status_type == "authentic"),
                    rx.box(
                        rx.hstack(
                            rx.text("Consensus Confidence", class_name="text-sm font-medium text-gray-400"),
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
                    rx.text("Try sample:", class_name="text-xs text-gray-500 self-center hidden sm:inline"),
                    rx.button(
                        "Real (EN)",
                        on_click=State.load_sample_real,
                        size="1",
                        variant="surface",
                        class_name="text-xs bg-emerald-500/10 text-emerald-300 hover:bg-emerald-500/20 border border-emerald-500/30 cursor-pointer rounded-md px-2 py-1",
                    ),
                    rx.button(
                        "Fake (EN)",
                        on_click=State.load_sample_fake,
                        size="1",
                        variant="surface",
                        class_name="text-xs bg-red-500/10 text-red-300 hover:bg-red-500/20 border border-red-500/30 cursor-pointer rounded-md px-2 py-1",
                    ),
                    rx.button(
                        "Hindi (हिंदी)",
                        on_click=State.load_sample_hindi,
                        size="1",
                        variant="surface",
                        class_name="text-xs bg-amber-500/10 text-amber-300 hover:bg-amber-500/20 border border-amber-500/30 cursor-pointer rounded-md px-2 py-1",
                    ),
                    rx.button(
                        "Spanish (ES)",
                        on_click=State.load_sample_spanish,
                        size="1",
                        variant="surface",
                        class_name="text-xs bg-purple-500/10 text-purple-300 hover:bg-purple-500/20 border border-purple-500/30 cursor-pointer rounded-md px-2 py-1",
                    ),
                    spacing="2",
                    align="center",
                    wrap="wrap",
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
app.add_page(index, title="News Detection | AI Ensemble")