from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        pass
    def footer(self):
        self.set_y(-15)
        self.set_font("Times", "I", 10)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

def generate_q1_roadmap_pdf():
    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_text_color(0, 0, 0)
    
    lh = 6.0 
    
    # Title
    pdf.set_font("Times", "B", 14)
    pdf.cell(0, 8, "Roadmap for Upgrading Our Parkinson's Research to a Q1 Journal Publication", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(5)

    # Intro
    pdf.set_font("Times", "", 12)
    intro = (
        "Respected Sir, we have already achieved highly promising results using our LightGBM and KAN fusion model on the UCI Parkinson's dataset (achieving 91.45% accuracy). "
        "However, to target a top-tier Q1 medical informatics journal, we need to significantly enhance the novelty, explainability, and statistical rigor of our research. "
        "Below is a proposed phase-by-step roadmap to elevate our work to Q1 standards."
    )
    pdf.multi_cell(0, lh, intro, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # Phase 1
    pdf.set_font("Times", "B", 12)
    pdf.cell(0, lh, "Phase 1: Deep Statistical & Exploratory Analysis (Establishing the Baseline)", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Times", "", 12)
    p1_text = (
        "Instead of just reporting model accuracy, Q1 journals expect rigorous statistical validation of the data.\n"
        "- We will perform statistical tests (e.g., t-tests or ANOVA) on the vocal features to prove which ones are statistically significant in distinguishing between PD and healthy patients.\n"
        "- We will generate feature correlation matrices and distribution plots to demonstrate a profound understanding of the dataset's underlying acoustics."
    )
    pdf.multi_cell(0, lh, p1_text, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # Phase 2
    pdf.set_font("Times", "B", 12)
    pdf.cell(0, lh, "Phase 2: Robust Validation & Eliminating Data Leakage", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Times", "", 12)
    p2_text = (
        "Many existing papers suffer from 'Data Leakage' by using Random Splits (where multiple recordings of the same patient end up in both training and testing sets, artificially inflating accuracy).\n"
        "- We will address this major flaw by implementing Group K-Fold Cross Validation (Patient-Isolated Split) to evaluate true real-world generalization.\n"
        "- We will prove that our Hybrid Architecture (LightGBM for discrete patterns + KAN for continuous spline patterns) remains highly robust even under this strict evaluation setting."
    )
    pdf.multi_cell(0, lh, p2_text, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # Phase 3
    pdf.set_font("Times", "B", 12)
    pdf.cell(0, lh, "Phase 3: Explainable AI & Medical Relevance (The Core Novelty)", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Times", "", 12)
    p3_text = (
        "Medical journals strictly require models to be transparent and clinically interpretable.\n"
        "- We will integrate SHAP (Shapley Additive exPlanations) to visualize exactly which voice features (e.g., Jitter, MFCCs) are driving our model's predictions.\n"
        "- We will extract the symbolic mathematical functions and splines learned by the KAN model and correlate them with physical vocal cord impairments (like hypophonia) seen in Parkinson's, effectively removing the 'black box' nature of deep learning."
    )
    pdf.multi_cell(0, lh, p3_text, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # Phase 4
    pdf.set_font("Times", "B", 12)
    pdf.cell(0, lh, "Phase 4: External Validation & Benchmarking (Proving Generalizability)", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Times", "", 12)
    p4_text = (
        "A Q1 paper must prove that its proposed model works universally, not just on a single dataset.\n"
        "- We will perform Cross-Dataset Validation by testing our pre-trained model on completely unseen external datasets, such as the newly published 'BenSParX' (Bengali Conversational Speech Dataset for PD).\n"
        "- We will present a comprehensive State-of-the-Art (SOTA) comparison, showcasing how our approach outclasses recent publications in both accuracy and interpretability."
    )
    pdf.multi_cell(0, lh, p4_text, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # Conclusion
    pdf.set_font("Times", "B", 12)
    pdf.cell(0, lh, "Conclusion", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Times", "", 12)
    conc_text = (
        "By systematically executing these four phases, we will not only create a highly accurate predictive model but also a medically interpretable and statistically bulletproof framework. This comprehensive methodology will strongly position our research for acceptance in a Q1 journal."
    )
    pdf.multi_cell(0, lh, conc_text, new_x="LMARGIN", new_y="NEXT")

    pdf.output("Q1_Journal_Roadmap.pdf")

if __name__ == "__main__":
    generate_q1_roadmap_pdf()
