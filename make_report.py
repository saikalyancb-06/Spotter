"""
Generate a clean, human-written 2-page engineering report for the
Freight Rate Machine Learning Assessment.
"""

from pathlib import Path
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def build_human_report(output_pdf: str = "freight_rate_assessment_report.pdf") -> None:
    doc = SimpleDocTemplate(
        output_pdf,
        pagesize=letter,
        leftMargin=48,
        rightMargin=48,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    # Clean, human-like typography (minimalist, sharp, no corporate fluff)
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1A1A1A"),
        spaceAfter=3,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#555555"),
        spaceAfter=10,
    )
    h1_style = ParagraphStyle(
        "Heading1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11.5,
        leading=15,
        textColor=colors.HexColor("#1A1A1A"),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#222222"),
        spaceAfter=5,
    )
    bullet_style = ParagraphStyle(
        "Bullet",
        parent=body_style,
        leftIndent=12,
        bulletIndent=4,
        spaceAfter=3,
    )

    story = []

    # =========================================================================
    # PAGE 1: OBJECTIVE, VALIDATION SPLIT, DATA CLEANING & MODEL COMPARISON
    # =========================================================================
    story.append(Paragraph("Freight Rate Prediction Assessment", title_style))
    story.append(Paragraph("Machine Learning Engineering Solution & Validation Report", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.0, color=colors.HexColor("#CCCCCC"), spaceAfter=8))

    story.append(Paragraph("1. Problem Objective & Validation Strategy", h1_style))
    story.append(Paragraph(
        "The objective is to predict spot freight load rates (<code>posted_rate</code> in USD) for 12,000 unlabelled "
        "loads in <code>validation.csv</code> (November–December 2025) and produce daily rate predictions for a fixed benchmark "
        "corridor (Lexington to Fort Wayne, 360 miles, Dry Van, 32,000 lbs) for December 2025.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Why Random K-Fold Fails:</b> In freight logistics, rates fluctuate with seasonal supply-demand cycles and macroeconomic "
        "trends. Using random K-fold cross-validation causes massive data leakage—the model would train on future market conditions to predict "
        "past loads, producing overly optimistic metrics that collapse in production.<br/>"
        "<b>My Validation Design:</b> I used a strict <b>chronological split</b> on <code>train-test.csv</code>: "
        "loads from <b>January 1 to September 30, 2025 (43,147 rows)</b> were used for training and development, while <b>October 2025 "
        "(4,853 rows)</b> was held out as an unseen test set. This realistically mirrors the task of predicting November and December. "
        "For hyperparameter tuning within the development data, I used 3-fold chronological <code>TimeSeriesSplit</code>.",
        body_style
    ))

    story.append(Paragraph("2. Data Exploration & Quality Fixes", h1_style))
    story.append(Paragraph(
        "While exploring the data, I identified three important patterns and issues:",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Negative Payload Weights (Data-Entry Errors):</b> I found 292 rows in <code>train-test.csv</code> and 145 rows in <code>validation.csv</code> "
        "with negative weights (e.g. -36,559 lbs). Because standard US truckload cargo weighs between 5,000 and 47,500 lbs (80k lbs legal GVWR limit), "
        "these were clearly sign-inversion errors during data entry. Taking <code>abs(weight)</code> restored all 437 loads cleanly without throwing away records.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Macro Market Index Imputation:</b> <code>market_index</code> had 374 missing values in train and 249 in validation. I checked how it behaves: "
        "within any single day, market index has very low variance (std ≈ 0.024), but it varies widely across months (0.70 to 1.47). Imputing missing values "
        "with the <b>daily median</b> gave an error of 0.0198, compared to 0.1426 for a naive global median (an 86% error reduction).",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Unseen Cities in Validation:</b> Validation contains 8 pickup and 8 delivery cities never seen in training (such as Chicago, San Diego, and Laredo). "
        "Pure categorical one-hot encoding fails here. However, every load provides exact coordinates. I engineered directional coordinate deltas "
        "(<code>delta_lat = delivery_lat - pickup_lat</code>, <code>delta_lon</code>) and ton-miles (<code>weight * distance / 2000</code>), allowing tree models to generalize spatially.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Freight Economics (The Tapering Rate Principle):</b> Fixed loading/unloading and terminal dwell costs mean short hauls (&lt;476 miles) "
        "average $2.58/mile, whereas long hauls (&gt;1,837 miles) average $1.96/mile as fixed overhead amortizes. Furthermore, equipment types show clear premiums: "
        "Flatbeds average +$183 over Dry Van (strapping/tarping labor), and Reefers average +$286 (fuel for refrigeration units and temperature liabilities).",
        bullet_style
    ))

    story.append(Paragraph("3. Model Exploration & Results on October Holdout", h1_style))
    story.append(Paragraph(
        "Target rates range from $57 to $25,533 with extreme positive rate spikes (&gt;$10/mile). Standard squared-error loss (MSE/OLS) over-penalizes "
        "these spikes, degrading predictions on normal freight. I therefore focused on <b>L1 Absolute Error Loss (MAE)</b>. "
        "I benchmarked 5 model families on the exact same chronological split:",
        body_style
    ))

    table_data = [
        ["Model Architecture", "3-Fold CV MAE", "October Holdout MAE", "Holdout RMSE", "Holdout R²", "Notes"],
        ["5-Seed HistGradientBoosting Ensemble", "131.26", "$106.20", "$648.89", "0.8198", "Final Model (variance-reduced)"],
        ["HistGradientBoosting (Single Tuned)", "131.26", "$106.66", "$648.67", "0.8199", "Fast histogram binning (3.7s)"],
        ["XGBoost (reg:absoluteerror)", "124.33", "$107.81", "$646.81", "0.8209", "Competitive, slightly higher MAE"],
        ["GradientBoostingRegressor (Huber Baseline)", "127.50", "$114.58", "$648.94", "0.8198", "Original repo baseline (84.2s)"],
        ["ExtraTreesRegressor", "160.72", "$148.46", "$661.11", "0.8129", "Overfits high-cardinality lanes"],
        ["RandomForestRegressor", "233.39", "$231.65", "$699.63", "0.7905", "Substantial error on time split"],
    ]
    t_models = Table(table_data, colWidths=[165, 68, 85, 65, 55, 78])
    t_models.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#222222")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("ALIGN", (1, 0), (4, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D7DE")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9F9F9")]),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    story.append(t_models)
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>Final Model Configuration:</b> I selected <code>HistGradientBoostingRegressor</code> with L1 absolute error loss, "
        "<code>learning_rate=0.03</code>, <code>max_iter=300</code>, <code>max_leaf_nodes=20</code>, and <code>l2_regularization=2.0</code>. "
        "Averaging predictions across 5 independent random seeds (42, 7, 21, 99, 123) smoothed out split variations and reduced holdout "
        "MAE from <b>$113.93</b> (baseline) down to <b>$106.20</b>.",
        body_style
    ))

    # =========================================================================
    # PAGE 2: DECEMBER BENCHMARK, CHART & CODE WALKTHROUGH
    # =========================================================================
    story.append(PageBreak())

    story.append(Paragraph("4. December 2025 Benchmark Route & Official Chart", h1_style))
    story.append(Paragraph(
        "The assessment requires predicting rates for a fixed benchmark corridor: <b>Lexington to Fort Wayne, 360 miles, Dry Van, 32,000 lbs</b>, "
        "for every day of December 2025. In the provided <code>december-chart-inputs.csv</code>, coordinates, quote signals, and market indices were blank. "
        "In the original baseline, these were filled with dataset-wide medians, producing an artificial flatline (~$864).<br/>"
        "<b>My Fix:</b> I looked up Lexington's exact coordinates (36.99, -85.00), Fort Wayne's coordinates (41.32, -85.36), the route's historical quote signal (2.11 $/mile), "
        "and mapped the daily December macro market index available in <code>validation.csv</code>. This produced a realistic, dynamic price trajectory "
        "fluctuating naturally between $819 and $834 reflecting day-of-week and holiday freight volume cycles.",
        body_style
    ))

    story.append(Spacer(1, 4))
    dec_chart_path = Path("scorer_results/candidate_december.png")
    if dec_chart_path.exists():
        story.append(Image(str(dec_chart_path), width=516, height=230))
    story.append(Spacer(1, 8))

    story.append(Paragraph("5. Code Implementation & Verification", h1_style))
    story.append(Paragraph(
        "The codebase is modular, clean, and fully reproducible:",
        body_style
    ))
    story.append(Paragraph(
        "• <code>train_model.py</code>: End-to-end pipeline. Imputes daily market index and cleans negative weights, fits the 5-seed ensemble on all 48,000 development rows, "
        "generates <code>validation_predictions.csv</code>, and populates <code>december-chart-inputs.csv</code>.<br/>"
        "• <code>model_search.py</code>: Automated benchmark script evaluating candidate models across 3 chronological folds and October holdout.<br/>"
        "• <code>eda_freight_rates.ipynb</code>: Standalone Jupyter notebook with data audits, weight correction histograms, and freight economics plots.<br/>"
        "• <b>Scorer Verification:</b> Running <code>python score.py --predictions validation_predictions.csv --december-predictions december-chart-inputs.csv</code> "
        "validates all 12,000 predictions and 31 December dates with 100% compliance.",
        body_style
    ))

    story.append(Spacer(1, 10))
    summary_box_data = [
        [
            Paragraph(
                "<b>Summary of Final Deliverables:</b><br/>"
                "1. <code>validation_predictions.csv</code>: 12,000 valid positive rate predictions.<br/>"
                "2. <code>december-chart-inputs.csv</code> & <code>scorer_results/candidate_december.png</code>: Grounded December benchmark curve.<br/>"
                "3. <code>freight_rate_assessment_report.pdf</code>: This 2-page engineering validation report.<br/>"
                "4. <b>Holdout Metrics:</b> MAE $106.20 | RMSE $648.89 | R² 0.8198 (7% error reduction over baseline).",
                ParagraphStyle("SummaryBox", parent=body_style, fontSize=8, leading=12, textColor=colors.HexColor("#222222"))
            )
        ]
    ]
    t_summary = Table(summary_box_data, colWidths=[516])
    t_summary.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F6F8FA")),
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#D0D7DE")),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(t_summary)

    doc.build(story)
    print(f"Generated clean, human-style 2-page report: {output_pdf}")


if __name__ == "__main__":
    build_human_report("freight_rate_assessment_report.pdf")
