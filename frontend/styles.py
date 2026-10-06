import streamlit as st
import plotly.graph_objects as go

CUSTOM_COLORWAY = ["#6366F1", "#EC4899", "#10B981", "#F59E0B", "#3B82F6", "#8B5CF6", "#14B8A6", "#F43F5E"]

# Monotonic counter used to generate unique keys for section_card() containers.
_CARD_COUNTER = {"n": 0}

def inject_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

        /* ============================================================
           RETENTIX AI - DESIGN TOKENS
           Single source of truth for spacing, radius, border, shadow.
           ============================================================ */
        :root {
            /* Surfaces */
            --rt-bg: #0A0F1C;
            --rt-surface: rgba(15, 23, 42, 0.72);

            /* Borders */
            --rt-border: rgba(255, 255, 255, 0.08);
            --rt-border-strong: rgba(255, 255, 255, 0.14);
            --rt-border-subtle: rgba(255, 255, 255, 0.05);

            /* Text */
            --rt-text: #F8FAFC;
            --rt-text-muted: #94A3B8;
            --rt-text-subtle: #64748B;

            /* Brand */
            --rt-brand: #6366F1;
            --rt-brand-soft: #A5B4FC;

            /* Radius scale */
            --rt-radius-sm: 8px;
            --rt-radius: 12px;
            --rt-radius-md: 14px;
            --rt-radius-lg: 18px;

            /* Elevation */
            --rt-shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.24);
            --rt-shadow: 0 8px 24px -8px rgba(0, 0, 0, 0.45);
            --rt-shadow-lg: 0 18px 40px -16px rgba(0, 0, 0, 0.6);
        }

        /* ============================================================
           BASE / TYPOGRAPHY
           ============================================================ */
        html, body, [class*="css"], .stApp, button, input, select, textarea {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }

        .stApp {
            background: var(--rt-bg);
        }

        /* Scrollbars - subtle, enterprise */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        ::-webkit-scrollbar-track {
            background: transparent;
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(148, 163, 184, 0.22);
            border-radius: 999px;
            border: 2px solid transparent;
            background-clip: content-box;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(148, 163, 184, 0.38);
            background-clip: content-box;
        }

        .stApp::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, #6366F1 0%, #8B5CF6 35%, #EC4899 70%, #10B981 100%);
            z-index: 999999;
        }

        /* Vertical rhythm for the main canvas */
        .stMainBlockContainer.block-container {
            padding-top: 2.25rem;
            padding-bottom: 4rem;
            padding-left: 2.5rem;
            padding-right: 2.5rem;
            max-width: 1680px;
        }
        @media (max-width: 1200px) {
            .stMainBlockContainer.block-container {
                padding-left: 1.5rem;
                padding-right: 1.5rem;
            }
        }
        @media (max-width: 768px) {
            .stMainBlockContainer.block-container {
                padding-top: 1.5rem;
                padding-left: 1rem;
                padding-right: 1rem;
            }
        }

        .hero-banner {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
            border: 1px solid var(--rt-border);
            box-shadow: var(--rt-shadow-lg), inset 0 1px 0 rgba(255, 255, 255, 0.1);
            border-radius: var(--rt-radius-lg);
            padding: 1.85rem 2.25rem;
            margin-bottom: 1.75rem;
            position: relative;
            overflow: hidden;
        }
        .hero-banner::after {
            content: '';
            position: absolute;
            top: -50%;
            right: -10%;
            width: 350px;
            height: 350px;
            background: radial-gradient(circle, rgba(99, 102, 241, 0.15) 0%, rgba(236, 72, 153, 0.05) 50%, transparent 70%);
            pointer-events: none;
        }
        .hero-title {
            font-size: 1.95rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            line-height: 1.2;
            background: linear-gradient(135deg, #FFFFFF 30%, #A5B4FC 100%);
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            flex-wrap: wrap;
        }
        .hero-subtitle {
            color: var(--rt-text-muted);
            font-size: 0.95rem;
            margin: 0.6rem 0 0 0;
            font-weight: 400;
            max-width: 900px;
            line-height: 1.6;
        }
        .hero-tag {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 12px;
            background: rgba(99, 102, 241, 0.15);
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-radius: 9999px;
            color: var(--rt-brand-soft);
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.85rem;
        }
        .pulse-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: #34D399;
            box-shadow: 0 0 0 3px rgba(52, 211, 153, 0.18);
            flex-shrink: 0;
        }
        .kpi-card {
            background: linear-gradient(145deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
            border: 1px solid var(--rt-border);
            border-radius: var(--rt-radius-md);
            padding: 1.15rem 1.25rem;
            box-shadow: var(--rt-shadow);
            transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease;
            position: relative;
            overflow: hidden;
            height: 100%;
        }
        .kpi-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, rgba(99, 102, 241, 0.7), rgba(139, 92, 246, 0.25));
            opacity: 0.85;
        }
        .kpi-card:hover {
            transform: translateY(-2px);
            border-color: rgba(99, 102, 241, 0.4);
            box-shadow: 0 14px 30px -12px rgba(99, 102, 241, 0.25);
        }
        .kpi-title {
            color: var(--rt-text-muted);
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            margin-bottom: 0.6rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.5rem;
        }
        .kpi-title span {
            font-size: 0.95rem;
            opacity: 0.85;
        }
        .kpi-value {
            font-family: 'JetBrains Mono', ui-monospace, monospace;
            font-size: 1.6rem;
            font-weight: 700;
            color: var(--rt-text);
            letter-spacing: -0.03em;
            line-height: 1.15;
            margin-bottom: 0.55rem;
            white-space: nowrap;
        }
        .kpi-badge {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            font-size: 0.7rem;
            font-weight: 600;
            padding: 3px 8px;
            border-radius: 6px;
            line-height: 1.4;
        }
        .badge-danger {
            background: rgba(244, 63, 94, 0.15);
            color: #FB7185;
            border: 1px solid rgba(244, 63, 94, 0.3);
        }
        .badge-success {
            background: rgba(16, 185, 129, 0.15);
            color: #34D399;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }
        .badge-info {
            background: rgba(59, 130, 246, 0.15);
            color: #60A5FA;
            border: 1px solid rgba(59, 130, 246, 0.3);
        }
        .badge-warning {
            background: rgba(245, 158, 11, 0.15);
            color: #FBBF24;
            border: 1px solid rgba(245, 158, 11, 0.3);
        }

        /* ============================================================
           CONTENT CARDS
           Widgets are placed inside a keyed container (st.container with
           key="rt_card"), which renders as .st-key-rt_card. That node is
           a real DOM ancestor of the charts/tables, so the card surface,
           padding and radius reliably wrap the content. .content-card
           remains for cards whose body is pure HTML.
           ============================================================ */
        .content-card {
            background: linear-gradient(145deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.7) 100%);
            border: 1px solid var(--rt-border);
            border-radius: var(--rt-radius-lg);
            padding: 1.35rem 1.5rem;
            margin-bottom: 0.5rem;
            box-shadow: var(--rt-shadow);
        }
        .content-card-header {
            font-size: 1.02rem;
            font-weight: 700;
            color: var(--rt-text);
            letter-spacing: -0.01em;
            line-height: 1.35;
            margin-bottom: 1.1rem;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid var(--rt-border-subtle);
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }
        .content-card-header span {
            font-size: 1.05rem;
            line-height: 1;
        }

        /* Keyed section container used by section_card() in the views */
        [class*="st-key-rt_card"] {
            background: linear-gradient(145deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.7) 100%);
            border: 1px solid var(--rt-border);
            border-radius: var(--rt-radius-lg);
            box-shadow: var(--rt-shadow);
            padding: 1.35rem 1.5rem;
        }
        /* Consistent breathing room between the header and the body */
        [class*="st-key-rt_card"] [data-testid="stVerticalBlock"] {
            gap: 0.5rem;
        }
        [class*="st-key-rt_card"] [data-testid="stPlotlyChart"],
        [class*="st-key-rt_card"] [data-testid="stDataFrame"],
        [class*="st-key-rt_card"] [data-testid="stImage"],
        [class*="st-key-rt_card"] [data-testid="stTabs"] {
            border-radius: var(--rt-radius);
            overflow: hidden;
        }

        .prediction-hero-card {
            border-radius: var(--rt-radius-lg);
            padding: 1.75rem;
            text-align: center;
            position: relative;
            overflow: hidden;
            margin: 0.25rem 0 0.75rem;
            border: 1px solid rgba(255, 255, 255, 0.12);
            box-shadow: var(--rt-shadow-lg);
        }
        .prediction-churn {
            background: linear-gradient(135deg, rgba(225, 29, 72, 0.22) 0%, rgba(159, 18, 57, 0.32) 100%);
            border-color: rgba(244, 63, 94, 0.45);
        }
        .prediction-stay {
            background: linear-gradient(135deg, rgba(5, 150, 105, 0.22) 0%, rgba(4, 120, 87, 0.32) 100%);
            border-color: rgba(16, 185, 129, 0.45);
        }
        .prediction-status-title {
            font-size: 1.15rem;
            font-weight: 800;
            letter-spacing: 0.02em;
            margin-bottom: 0.35rem;
        }
        .prediction-status-sub {
            font-size: 0.85rem;
            color: var(--rt-text-muted);
            max-width: 560px;
            margin: 0 auto;
            line-height: 1.55;
        }

        .reco-box {
            background: rgba(15, 23, 42, 0.55);
            border: 1px dashed rgba(99, 102, 241, 0.35);
            border-radius: var(--rt-radius);
            padding: 1.1rem 1.25rem;
            margin-top: 0.5rem;
        }
        .reco-item {
            display: flex;
            align-items: flex-start;
            gap: 0.6rem;
            margin-bottom: 0.7rem;
            color: #E2E8F0;
            font-size: 0.87rem;
            line-height: 1.55;
        }
        .reco-item:last-child {
            margin-bottom: 0;
        }

        /* ============================================================
           TABS
           The legacy BaseWeb attribute hooks no longer exist in this
           Streamlit release, so the tab strip is targeted through the
           ARIA tab pattern (tablist / tab / aria-selected).
           ============================================================ */
        [data-testid="stTabs"] [role="tablist"] {
            gap: 6px;
            background-color: rgba(15, 23, 42, 0.55);
            padding: 5px;
            border-radius: var(--rt-radius);
            border: 1px solid var(--rt-border-subtle);
        }
        [data-testid="stTabs"] [role="tab"] {
            min-height: 40px;
            border-radius: var(--rt-radius-sm);
            color: var(--rt-text-muted);
            font-weight: 600;
            font-size: 0.85rem;
            padding: 0 16px;
            border: none;
            transition: background-color 0.15s ease, color 0.15s ease;
        }
        [data-testid="stTabs"] [role="tab"]:hover {
            background: rgba(255, 255, 255, 0.05);
            color: var(--rt-text);
        }
        [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
            background: rgba(99, 102, 241, 0.9) !important;
            color: #FFFFFF !important;
            box-shadow: 0 2px 8px rgba(99, 102, 241, 0.3);
        }
        [data-testid="stTabPanel"] {
            padding-top: 0.75rem;
        }

        /* ============================================================
           SIDEBAR / NAVIGATION
           ============================================================ */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #080D18 0%, #0F172A 100%);
            border-right: 1px solid var(--rt-border);
        }
        [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
            padding-top: 1.25rem;
        }
        [data-testid="stSidebar"] hr {
            margin: 1rem 0;
            border-color: var(--rt-border-subtle);
        }
        /* Sidebar section labels */
        [data-testid="stSidebar"] .rt-side-label {
            font-size: 0.68rem;
            font-weight: 700;
            color: var(--rt-text-subtle);
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin: 0.35rem 0 0.55rem 0;
        }

        /* Navigation radio list */
        [data-testid="stSidebar"] [role="radiogroup"] {
            gap: 2px;
        }
        [data-testid="stSidebar"] [data-testid="stRadioOption"] {
            padding: 1px 0;
        }
        [data-testid="stSidebar"] [data-testid="stRadioOption"] > label {
            border-radius: var(--rt-radius-sm);
            padding: 8px 10px;
            border: 1px solid transparent;
            transition: background-color 0.15s ease, border-color 0.15s ease;
        }
        [data-testid="stSidebar"] [data-testid="stRadioOption"] > label:hover {
            background: rgba(255, 255, 255, 0.045);
        }
        [data-testid="stSidebar"] [data-testid="stRadioOption"] > label:has(input:checked) {
            background: rgba(99, 102, 241, 0.14);
            border-color: rgba(99, 102, 241, 0.35);
        }
        [data-testid="stSidebar"] [data-testid="stRadioOption"] p {
            font-size: 0.83rem;
            font-weight: 500;
            color: var(--rt-text-muted);
        }
        [data-testid="stSidebar"] [data-testid="stRadioOption"] label:has(input:checked) p {
            color: var(--rt-text);
            font-weight: 600;
        }

        /* ============================================================
           INPUT CONTROLS
           ============================================================ */
        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input,
        [data-testid="stSelectbox"] input,
        [data-testid="stMultiSelect"] input {
            background-color: rgba(15, 23, 42, 0.6) !important;
            border: 1px solid var(--rt-border) !important;
            border-radius: var(--rt-radius-sm) !important;
            color: var(--rt-text) !important;
            font-size: 0.85rem !important;
            transition: border-color 0.15s ease, box-shadow 0.15s ease;
        }
        [data-testid="stTextInput"] input:hover,
        [data-testid="stNumberInput"] input:hover,
        [data-testid="stSelectbox"] input:hover,
        [data-testid="stMultiSelect"] input:hover {
            border-color: var(--rt-border-strong) !important;
        }
        [data-testid="stTextInput"] input:focus,
        [data-testid="stNumberInput"] input:focus,
        [data-testid="stSelectbox"] input:focus,
        [data-testid="stMultiSelect"] input:focus {
            border-color: rgba(99, 102, 241, 0.7) !important;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.16) !important;
            outline: none !important;
        }
        [data-testid="stTextInput"] input::placeholder {
            color: var(--rt-text-subtle) !important;
        }

        [data-testid="stMultiSelect"] [data-testid="stMultiSelectTagsContainer"] {
            background-color: rgba(15, 23, 42, 0.6) !important;
            border: 1px solid var(--rt-border) !important;
            border-radius: var(--rt-radius-sm) !important;
        }
        [data-testid="stMultiSelect"] [data-tag] {
            background: rgba(99, 102, 241, 0.16) !important;
            border: 1px solid rgba(99, 102, 241, 0.3) !important;
            border-radius: 6px !important;
            font-size: 0.75rem !important;
        }

        /* Dropdown list surfaces */
        [data-testid="stSelectboxVirtualDropdown"],
        [role="listbox"] {
            background-color: #0F172A !important;
            border: 1px solid var(--rt-border-strong) !important;
            border-radius: var(--rt-radius) !important;
            box-shadow: var(--rt-shadow-lg) !important;
        }
        [role="option"] {
            font-size: 0.85rem !important;
            color: var(--rt-text-muted) !important;
        }
        [role="option"][aria-selected="true"] {
            background: rgba(99, 102, 241, 0.2) !important;
            color: var(--rt-text) !important;
        }

        /* Number input steppers */
        [data-testid="stNumberInput"] button {
            background: rgba(30, 41, 59, 0.8) !important;
            border: 1px solid var(--rt-border) !important;
            color: var(--rt-text-muted) !important;
            transition: background-color 0.15s ease, color 0.15s ease;
        }
        [data-testid="stNumberInput"] button:hover {
            background: rgba(99, 102, 241, 0.25) !important;
            color: var(--rt-text) !important;
        }

        /* ============================================================
           SLIDERS
           ============================================================ */
        [data-testid="stSlider"] [role="slider"] {
            background: rgba(148, 163, 184, 0.2) !important;
            border-radius: 999px !important;
        }
        [data-testid="stSlider"] [data-testid="stSliderThumbValue"] {
            background: #6366F1 !important;
            border: 2px solid #0F172A !important;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.4) !important;
        }
        [data-testid="stSliderTickBar"] {
            background: rgba(148, 163, 184, 0.18) !important;
            border-radius: 999px !important;
        }

        /* ============================================================
           BUTTONS
           ============================================================ */
        [data-testid="stBaseButton-primary"] {
            background: linear-gradient(180deg, #6D6FF5 0%, #5B5BD6 100%) !important;
            border: 1px solid rgba(129, 140, 248, 0.55) !important;
            border-radius: var(--rt-radius-sm) !important;
            box-shadow: 0 2px 10px -2px rgba(99, 102, 241, 0.45) !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
            transition: filter 0.15s ease, box-shadow 0.15s ease, transform 0.1s ease;
        }
        [data-testid="stBaseButton-primary"]:hover {
            filter: brightness(1.08) !important;
            box-shadow: 0 4px 16px -2px rgba(99, 102, 241, 0.6) !important;
        }
        [data-testid="stBaseButton-primary"]:active {
            transform: translateY(1px) !important;
        }
        [data-testid="stBaseButton-secondary"] {
            background: rgba(30, 41, 59, 0.75) !important;
            border: 1px solid var(--rt-border-strong) !important;
            border-radius: var(--rt-radius-sm) !important;
            color: var(--rt-text) !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
            box-shadow: var(--rt-shadow-sm) !important;
            transition: background-color 0.15s ease, border-color 0.15s ease;
        }
        [data-testid="stBaseButton-secondary"]:hover {
            background: rgba(51, 65, 85, 0.85) !important;
            border-color: rgba(99, 102, 241, 0.4) !important;
        }

        /* ============================================================
           DATA TABLES
           ============================================================ */
        [data-testid="stDataFrame"] {
            border: 1px solid var(--rt-border-subtle);
            border-radius: var(--rt-radius);
            overflow: hidden;
        }
        [data-testid="stDataFrame"] [role="grid"] {
            font-size: 0.82rem;
        }

        /* ============================================================
           ALERTS / EMPTY STATES
           ============================================================ */
        [data-testid="stAlertContainer"] {
            border-radius: var(--rt-radius) !important;
            border: 1px solid var(--rt-border) !important;
            font-size: 0.85rem !important;
        }
        [data-testid="stAlertContentInfo"] {
            background: rgba(59, 130, 246, 0.1) !important;
            border-color: rgba(59, 130, 246, 0.3) !important;
            color: #93C5FD !important;
        }
        [data-testid="stAlertContentSuccess"] {
            background: rgba(16, 185, 129, 0.1) !important;
            border-color: rgba(16, 185, 129, 0.3) !important;
            color: #6EE7B7 !important;
        }
        [data-testid="stAlertContentError"] {
            background: rgba(244, 63, 94, 0.1) !important;
            border-color: rgba(244, 63, 94, 0.3) !important;
            color: #FDA4AF !important;
        }
        [data-testid="stAlertContentWarning"] {
            background: rgba(245, 158, 11, 0.1) !important;
            border-color: rgba(245, 158, 11, 0.3) !important;
            color: #FCD34D !important;
        }

        /* ============================================================
           CAPTIONS / HELPER TEXT
           ============================================================ */
        [data-testid="stCaptionContainer"],
        [data-testid="stCaption"] {
            color: var(--rt-text-subtle) !important;
            font-size: 0.78rem !important;
            line-height: 1.5;
        }

        /* ============================================================
           METRICS
           ============================================================ */
        [data-testid="stMetric"] {
            background: rgba(15, 23, 42, 0.45);
            border: 1px solid var(--rt-border-subtle);
            border-radius: var(--rt-radius);
            padding: 0.85rem 1rem;
        }
        [data-testid="stMetricLabel"] {
            color: var(--rt-text-muted) !important;
            font-size: 0.75rem !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        [data-testid="stMetricValue"] {
            font-family: 'JetBrains Mono', ui-monospace, monospace;
            font-size: 1.35rem !important;
            font-weight: 700 !important;
            color: var(--rt-text) !important;
        }
        [data-testid="stMetricDelta"] {
            font-size: 0.76rem !important;
        }

        /* ============================================================
           FILE UPLOADER
           ============================================================ */
        [data-testid="stFileUploaderDropzone"] {
            background: rgba(15, 23, 42, 0.5) !important;
            border: 1px dashed var(--rt-border-strong) !important;
            border-radius: var(--rt-radius) !important;
            transition: border-color 0.15s ease, background-color 0.15s ease;
        }
        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: rgba(99, 102, 241, 0.5) !important;
            background: rgba(30, 41, 59, 0.5) !important;
        }
        [data-testid="stFileUploaderDropzoneInstructions"] {
            color: var(--rt-text-muted) !important;
            font-size: 0.8rem !important;
        }
        [data-testid="stFileUploaderDropzoneInstructions"] span {
            color: var(--rt-text-subtle) !important;
            font-size: 0.75rem !important;
        }

        /* ============================================================
           ACCESSIBILITY
           ============================================================ */
        :focus-visible {
            outline: 2px solid rgba(99, 102, 241, 0.8);
            outline-offset: 2px;
        }
    </style>
    """, unsafe_allow_html=True)

def render_hero_banner(title: str, subtitle: str, tag: str = "PULSE ENGINE ONLINE"):
    st.markdown(f"""
    <div class="hero-banner">
        <div class="hero-tag"><span class="pulse-dot"></span> {tag}</div>
        <h1 class="hero-title">{title}</h1>
        <p class="hero-subtitle">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def card_header(icon: str, title: str):
    """Renders a section card header (icon + title with a hairline rule)."""
    st.markdown(
        f'<div class="content-card-header"><span>{icon}</span> {title}</div>',
        unsafe_allow_html=True,
    )


def _next_card_key() -> str:
    """
    Returns a unique Streamlit element key for a section card.

    Streamlit requires element keys to be unique within a single script run,
    and every view renders more than one card. A per-run counter keeps the
    keys distinct while still producing the stable `st-key-rt_card*` CSS
    hook used to draw the card surface.
    """
    _CARD_COUNTER["n"] += 1
    return f"rt_card_{_CARD_COUNTER['n']}"


def section_card(icon: str, title: str):
    """
    Opens a real (keyed) Streamlit container styled as a section card.

    The keyed container renders as `.st-key-rt_card`, so the surface, padding
    and radius genuinely wrap every widget written inside it. This replaces
    the previous pattern of a standalone `<div class="content-card">` markdown
    call, whose element was auto-closed by the browser before the charts.

    Returns:
        DeltaGenerator: use as `card = section_card(...)` then `with card:`
    """
    card = st.container(key=_next_card_key())
    with card:
        card_header(icon, title)
    return card


def apply_plotly_theme(fig, height=380):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#CBD5E1", size=12),
        margin=dict(l=16, r=16, t=40, b=20),
        height=height,
        colorway=CUSTOM_COLORWAY,
        hoverlabel=dict(
            bgcolor="#0F172A",
            bordercolor="rgba(255, 255, 255, 0.15)",
            font=dict(size=12, color="#F8FAFC", family="Plus Jakarta Sans, sans-serif"),
        ),
        legend=dict(
            bgcolor="rgba(15, 23, 42, 0.75)",
            bordercolor="rgba(255, 255, 255, 0.1)",
            borderwidth=1,
            font=dict(size=11),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        title=dict(
            font=dict(size=13, color="#94A3B8", family="Plus Jakarta Sans, sans-serif"),
            x=0,
            xanchor="left",
        ),
        xaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.06)",
            zerolinecolor="rgba(255, 255, 255, 0.1)",
            linecolor="rgba(255, 255, 255, 0.1)",
            tickfont=dict(size=11, color="#94A3B8"),
            title=dict(font=dict(size=11, color="#64748B")),
            automargin=True,
        ),
        yaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.06)",
            zerolinecolor="rgba(255, 255, 255, 0.1)",
            linecolor="rgba(255, 255, 255, 0.1)",
            tickfont=dict(size=11, color="#94A3B8"),
            title=dict(font=dict(size=11, color="#64748B")),
            automargin=True,
        ),
    )
    return fig

